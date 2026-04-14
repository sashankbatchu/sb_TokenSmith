import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        eff_chunk_size = max(200, int(self.recursive_chunk_size * size_multiplier))
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

    def name(self) -> str:
        return (
            "semantic_sections("
            f"{self.chunk_size},{self.chunk_overlap},"
            f"drop={self.similarity_drop_threshold})"
        )

    def artifact_folder_name(self) -> str:
        return "semantic_sections"

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if paragraphs:
            return paragraphs
        return [text.strip()] if text.strip() else []

    def _token_set(self, text: str) -> Set[str]:
        return set(self.WORD_RE.findall(text.lower()))

    def _similarity(self, a: str, b: str) -> float:
        a_toks = self._token_set(a)
        b_toks = self._token_set(b)
        if not a_toks or not b_toks:
            return 0.0
        inter = len(a_toks & b_toks)
        union = len(a_toks | b_toks)
        if union == 0:
            return 0.0
        return inter / union

    def _build_semantic_units(self, paragraphs: List[str]) -> List[str]:
        if not paragraphs:
            return []
        units: List[str] = []
        current = paragraphs[0]
        prev_sim = 1.0

        for para in paragraphs[1:]:
            sim = self._similarity(current[-500:], para[:500])
            current_is_substantial = len(current) >= self.min_paragraph_chars
            # Create a boundary when coherence drops significantly.
            if current_is_substantial and sim < self.similarity_drop_threshold and sim < (prev_sim * 0.7):
                units.append(current.strip())
                current = para
            else:
                current = f"{current}\n\n{para}"
            prev_sim = sim

        if current.strip():
            units.append(current.strip())
        return units

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

    def _pack_units(self, units: List[str], eff_chunk_size: int, eff_chunk_overlap: int) -> List[str]:
        chunks: List[str] = []
        current = ""

        for unit in units:
            if not unit.strip():
                continue

            candidate = unit if not current else f"{current}\n\n{unit}"
            if len(candidate) <= eff_chunk_size:
                current = candidate
                continue

            if current.strip():
                chunks.append(current.strip())

            if len(unit) > eff_chunk_size:
                adaptive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=eff_chunk_size,
                    chunk_overlap=eff_chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks.extend(adaptive_splitter.split_text(unit))
                current = ""
                continue

            if eff_chunk_overlap > 0 and chunks:
                tail = chunks[-1][-eff_chunk_overlap:]
                current = f"{tail}\n\n{unit}".strip()
            else:
                current = unit

        if current.strip():
            chunks.append(current.strip())
        return chunks

    def chunk(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        context = context or {}
        section_depth = self._resolve_section_depth(context)
        size_multiplier = self._depth_size_multiplier(section_depth)
        eff_chunk_size = max(200, int(self.chunk_size * size_multiplier))
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
