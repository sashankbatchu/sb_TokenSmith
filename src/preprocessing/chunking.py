import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer as HFSentenceTransformer
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    HFSentenceTransformer = None

from langchain_text_splitters import RecursiveCharacterTextSplitter

_EMBEDDING_MODEL_CACHE: Dict[str, Any] = {}

# -------------------------- Chunking Configs --------------------------

class ChunkConfig(ABC):
    @abstractmethod
    def validate(self):
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        pass

@dataclass
class SectionRecursiveConfig(ChunkConfig):
    """Configuration for section-based chunking with recursive splitting."""
    recursive_chunk_size: int
    recursive_overlap: int
    
    def to_string(self) -> str:
        return f"chunk_mode=sections+recursive, chunk_size={self.recursive_chunk_size}, overlap={self.recursive_overlap}"

    def validate(self):
        assert self.recursive_chunk_size > 0, "recursive_chunk_size must be > 0"
        assert self.recursive_overlap >= 0, "recursive_overlap must be >= 0"


@dataclass
class SemanticBoundaryConfig(ChunkConfig):
    """
    Configuration for semantic-boundary-aware chunking.
    """
    chunk_size: int
    chunk_overlap: int
    similarity_drop_threshold: float = 0.18
    min_paragraph_chars: int = 80
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    lexical_weight: float = 0.2
    embedding_weight: float = 0.8
    adaptive_threshold_percentile: int = 25

    def to_string(self) -> str:
        return (
            "chunk_mode=semantic_sections, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"similarity_drop_threshold={self.similarity_drop_threshold}, "
            f"min_paragraph_chars={self.min_paragraph_chars}"
        )

    def validate(self):
        assert self.chunk_size > 0, "chunk_size must be > 0"
        assert self.chunk_overlap >= 0, "chunk_overlap must be >= 0"
        assert self.chunk_overlap < self.chunk_size, "chunk_overlap must be < chunk_size"
        assert 0.0 <= self.similarity_drop_threshold <= 1.0, "similarity_drop_threshold must be in [0, 1]"
        assert self.min_paragraph_chars >= 0, "min_paragraph_chars must be >= 0"
        assert 0 <= self.adaptive_threshold_percentile <= 100, "adaptive_threshold_percentile must be in [0, 100]"

# -------------------------- Chunking Strategies --------------------------

class ChunkStrategy(ABC):
    """Abstract base for all chunking strategies."""
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def chunk(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        pass
    
    @abstractmethod
    def artifact_folder_name(self) -> str:
        pass

class SectionRecursiveStrategy(ChunkStrategy):
    """
    Applies recursive character-based splitting to text.
    This is meant to be used on already-extracted sections.
    """

    def __init__(self, config: SectionRecursiveConfig):
        self.config = config
        self.recursive_chunk_size = config.recursive_chunk_size
        self.recursive_overlap = config.recursive_overlap

    def name(self) -> str:
        return f"sections+recursive({self.recursive_chunk_size},{self.recursive_overlap})"

    def artifact_folder_name(self) -> str:
        return "sections"

    @staticmethod
    def _resolve_section_depth(context: Dict[str, Any]) -> int:
        """
        Prefer normalized section_depth if provided by caller.
        Fall back to section_level with 1->0 normalization.
        """
        if "section_depth" in context:
            try:
                return max(0, int(context["section_depth"]))
            except (TypeError, ValueError):
                return 0
        try:
            return max(0, int(context.get("section_level", 1)) - 1)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _depth_size_multiplier(section_depth: int) -> float:
        """
        Hierarchical chunk sizing:
        - shallower sections keep larger chunks
        - deeper subsections use finer chunks
        """
        if section_depth <= 0:
            return 1.2
        if section_depth == 1:
            return 1.0
        if section_depth == 2:
            return 0.9
        return 0.75

    def chunk(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Recursively splits text into smaller chunks based on sentence boundaries.
        If a chunk exceeds recursive_chunk_size, it is further split.
        """
        context = context or {}
        section_depth = self._resolve_section_depth(context)
        size_multiplier = self._depth_size_multiplier(section_depth)
        eff_chunk_size = max(1, int(self.recursive_chunk_size * size_multiplier))
        eff_chunk_overlap = min(
            max(0, int(self.recursive_overlap * size_multiplier)),
            max(0, eff_chunk_size - 1),
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=eff_chunk_size,
            chunk_overlap=eff_chunk_overlap,
            separators=[". "]
        )
        return splitter.split_text(text)


class SemanticBoundaryStrategy(ChunkStrategy):
    """
    Splits text primarily on semantic paragraph boundaries and then packs
    semantically coherent units into size-constrained chunks.
    """

    WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")
    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9#])")
    HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
    PAGE_MARKER_RE = re.compile(r"^\s*---\s*Page\s+\d+\s*---\s*$", re.IGNORECASE)
    ANCHOR_ROLE_PATTERNS = {
        "definition": re.compile(r"^\s*Definition\b", re.IGNORECASE),
        "example": re.compile(r"^\s*Example\b", re.IGNORECASE),
        "theorem": re.compile(r"^\s*(Theorem|Lemma|Proposition|Corollary)\b", re.IGNORECASE),
        "proof": re.compile(r"^\s*Proof\b", re.IGNORECASE),
        "remark": re.compile(r"^\s*Remark\b", re.IGNORECASE),
    }

    def __init__(self, config: SemanticBoundaryConfig):
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.similarity_drop_threshold = config.similarity_drop_threshold
        self.min_paragraph_chars = config.min_paragraph_chars
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embedding_model_name = config.embedding_model_name
        self.lexical_weight = config.lexical_weight
        self.embedding_weight = config.embedding_weight
        self.adaptive_threshold_percentile = config.adaptive_threshold_percentile
        self.embedding_model = self._get_embedding_model(self.embedding_model_name)
        if self.embedding_model is None:
            # Fall back to lexical similarity only if the embedding backend is unavailable.
            self.lexical_weight = 1.0
            self.embedding_weight = 0.0
        self._embedding_cache = {}

    @staticmethod
    def _get_embedding_model(model_name: str):
        if HFSentenceTransformer is None:
            return None
        if model_name not in _EMBEDDING_MODEL_CACHE:
            _EMBEDDING_MODEL_CACHE[model_name] = HFSentenceTransformer(model_name)
        return _EMBEDDING_MODEL_CACHE[model_name]

    def name(self) -> str:
        return (
            "semantic_sections("
            f"{self.chunk_size},{self.chunk_overlap},"
            f"drop={self.similarity_drop_threshold})"
        )

    def artifact_folder_name(self) -> str:
        return "semantic_sections"

    def _sentence_blockify(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        sentences = [s.strip() for s in self.SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if len(sentences) <= 1:
            return [text]

        blocks: List[str] = []
        current = sentences[0]
        for sentence in sentences[1:]:
            if (
                len(current) >= self.min_paragraph_chars
                or self.HEADING_RE.match(sentence)
                or self._paragraph_role(sentence) != "body"
            ):
                blocks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()

        if current.strip():
            blocks.append(current.strip())
        return blocks

    def _split_paragraphs(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        raw_blocks = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        paragraphs: List[str] = []
        for block in raw_blocks:
            if self.PAGE_MARKER_RE.match(block):
                continue
            if "\n" not in block and len(block) > self.chunk_size:
                paragraphs.extend(self._sentence_blockify(block))
            else:
                paragraphs.append(block)

        if not paragraphs and text.strip():
            return self._sentence_blockify(text.strip())
        return paragraphs

    def _token_set(self, text: str) -> Set[str]:
        return set(self.WORD_RE.findall(text.lower()))

    def _cosine_similarity(self, a: str, b: str) -> float:
        if self.embedding_model is None or not a.strip() or not b.strip():
            return 0.0

        if a not in self._embedding_cache:
            self._embedding_cache[a] = self.embedding_model.encode(
                a,
                normalize_embeddings=True
            )

        if b not in self._embedding_cache:
            self._embedding_cache[b] = self.embedding_model.encode(
                b,
                normalize_embeddings=True
            )

        return float(np.dot(self._embedding_cache[a], self._embedding_cache[b]))


    def _jaccard_similarity(self, a: str, b: str) -> float:
        a_toks = self._token_set(a)
        b_toks = self._token_set(b)

        if not a_toks or not b_toks:
            return 0.0

        inter = len(a_toks & b_toks)
        union = len(a_toks | b_toks)

        return inter / union if union else 0.0


    def _similarity(self, a: str, b: str) -> float:
        lexical_sim = self._jaccard_similarity(a, b)
        embedding_sim = self._cosine_similarity(a, b)

        return (
            self.lexical_weight * lexical_sim
            + self.embedding_weight * embedding_sim
        )

    @classmethod
    def _paragraph_role(cls, paragraph: str) -> str:
        if not paragraph or not paragraph.strip():
            return "empty"
        if cls.HEADING_RE.match(paragraph):
            return "heading"
        for role, pattern in cls.ANCHOR_ROLE_PATTERNS.items():
            if pattern.match(paragraph):
                return role
        return "body"

    @staticmethod
    def _is_companion_role(previous_role: str, current_role: str) -> bool:
        if current_role in {"proof", "remark"} and previous_role in {"theorem", "definition", "example"}:
            return True
        if current_role == "example" and previous_role in {"definition", "body", "heading"}:
            return True
        return False

    @staticmethod
    def _tail_excerpt(text: str, size: int = 500) -> str:
        return text[-size:] if len(text) > size else text

    def _compute_adaptive_threshold(self, paragraphs: List[str]) -> float:
        """
        Computes an adaptive split threshold based on the section's own similarity distribution.
        Lower percentile = only split on stronger topic shifts.
        """
        if len(paragraphs) < 3:
            return self.similarity_drop_threshold

        sims = []
        for i in range(1, len(paragraphs)):
            prev_para = paragraphs[i - 1]
            curr_para = paragraphs[i]

            curr_role = self._paragraph_role(curr_para)
            if curr_role == "heading":
                continue

            sims.append(self._similarity(prev_para, curr_para))

        if not sims:
            return self.similarity_drop_threshold

        adaptive = float(np.percentile(sims, self.adaptive_threshold_percentile))
        lower_bound = min(0.12, self.similarity_drop_threshold)
        upper_bound = max(lower_bound, self.similarity_drop_threshold)
        adaptive = max(lower_bound, min(upper_bound, adaptive))

        return adaptive

    def _build_semantic_units(self, paragraphs: List[str]) -> List[str]:
        if not paragraphs:
            return []

        threshold = self._compute_adaptive_threshold(paragraphs)

        units: List[str] = []
        current = paragraphs[0]
        current_role = self._paragraph_role(current)

        for i in range(1, len(paragraphs)):
            para = paragraphs[i]
            para_role = self._paragraph_role(para)
            current_is_substantial = len(current) >= self.min_paragraph_chars
            sim = self._similarity(self._tail_excerpt(current), para)

            should_split = False

            if para_role == "heading" and current.strip():
                should_split = True
            elif self._is_companion_role(current_role, para_role):
                should_split = False
            elif para_role in {"definition", "theorem"} and current.strip():
                should_split = True
            elif para_role == "example" and current_is_substantial and len(current) >= int(self.chunk_size * 0.45):
                should_split = True
            elif current_is_substantial and sim < threshold:
                should_split = True

            if should_split:
                units.append(current.strip())
                current = para
                current_role = para_role
            else:
                current = f"{current}\n\n{para}"
                current_role = para_role if para_role != "body" else current_role

        if current.strip():
            units.append(current.strip())

        return units

    def _paragraph_overlap(self, previous_chunk: str, overlap_chars: int) -> str:
        """
        Uses whole paragraphs for overlap instead of slicing raw characters.
        Avoids starting chunks mid-word or mid-sentence.
        """
        if overlap_chars <= 0:
            return ""

        paras = self._split_paragraphs(previous_chunk)
        selected = []
        total = 0

        for para in reversed(paras):
            selected.insert(0, para)
            total += len(para)

            if total >= overlap_chars:
                break

        return "\n\n".join(selected).strip()

    def _pack_units(
        self,
        units: List[str],
        eff_chunk_size: int,
        eff_chunk_overlap: int
    ) -> List[str]:
        chunks: List[str] = []

        for unit in units:
            if not unit.strip():
                continue

            if len(unit) > eff_chunk_size:
                adaptive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=eff_chunk_size,
                    chunk_overlap=eff_chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks.extend(adaptive_splitter.split_text(unit))
                continue

            if eff_chunk_overlap > 0 and chunks:
                overlap = self._paragraph_overlap(chunks[-1], eff_chunk_overlap)
                chunks.append(f"{overlap}\n\n{unit}".strip() if overlap else unit)
            else:
                chunks.append(unit.strip())

        return chunks

    @staticmethod
    def _resolve_section_depth(context: Dict[str, Any]) -> int:
        """
        Prefer normalized section_depth if provided by caller.
        Fall back to section_level with 1->0 normalization.
        """
        if "section_depth" in context:
            try:
                return max(0, int(context["section_depth"]))
            except (TypeError, ValueError):
                return 0
        try:
            return max(0, int(context.get("section_level", 1)) - 1)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _depth_size_multiplier(section_depth: int) -> float:
        if section_depth <= 0:
            return 1.2
        if section_depth == 1:
            return 1.0
        if section_depth == 2:
            return 0.9
        return 0.75

    def chunk(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        context = context or {}

        if "section_depth" in context or "section_level" in context:
            section_depth = self._resolve_section_depth(context)
            size_multiplier = self._depth_size_multiplier(section_depth)
        else:
            size_multiplier = 1.0

        eff_chunk_size = max(1, int(self.chunk_size * size_multiplier))
        eff_chunk_overlap = min(
            max(0, int(self.chunk_overlap * size_multiplier)),
            max(0, eff_chunk_size - 1),
        )

        paragraphs = self._split_paragraphs(text)

        if not paragraphs:
            return []

        units = self._build_semantic_units(paragraphs)
        chunks = self._pack_units(units, eff_chunk_size, eff_chunk_overlap)

        if not chunks:
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=eff_chunk_size,
                chunk_overlap=eff_chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            return fallback_splitter.split_text(text)

        return chunks

# ----------------------------- Document Chunker ---------------------------------

class DocumentChunker:
    """
    Chunk text via a provided strategy.
    Table blocks (<table>...</table>) are preserved within chunks.
    """

    TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        strategy: Optional[ChunkStrategy],
        keep_tables: bool = True
    ):
        self.strategy = strategy
        self.keep_tables = keep_tables

    def _extract_tables(self, text: str) -> Tuple[str, List[str]]:
        tables = self.TABLE_RE.findall(text)
        for i, t in enumerate(tables):
            text = text.replace(t, f"[TABLE_PLACEHOLDER_{i}]")
        return text, tables

    @staticmethod
    def _restore_tables(chunk: str, tables: List[str]) -> str:
        for i, t in enumerate(tables):
            ph = f"[TABLE_PLACEHOLDER_{i}]"
            if ph in chunk:
                chunk = chunk.replace(ph, t)
        return chunk

    def chunk(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        if not text:
            return []
        work = text
        tables: List[str] = []
        if self.keep_tables:
            work, tables = self._extract_tables(work)

        if self.strategy is None:
            raise ValueError("No chunk strategy provided")
        else:
            chunks = self.strategy.chunk(work, context=context)

        if self.keep_tables and tables:
            chunks = [self._restore_tables(c, tables) for c in chunks]
        return chunks
