import pickle
from pathlib import Path

import numpy as np
import pytest


def _make_semantic_strategy(monkeypatch, **config_kwargs):
    from src.preprocessing.chunking import SemanticBoundaryConfig, SemanticBoundaryStrategy

    monkeypatch.setattr(
        SemanticBoundaryStrategy,
        "_get_embedding_model",
        staticmethod(lambda _: None),
    )
    config = SemanticBoundaryConfig(**config_kwargs)
    return SemanticBoundaryStrategy(config)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


class TestSemanticChunkingCorrectness:
    def test_no_text_loss_and_order_preserved_without_overlap(self, monkeypatch):
        strategy = _make_semantic_strategy(
            monkeypatch,
            chunk_size=120,
            chunk_overlap=0,
            min_paragraph_chars=20,
        )
        strategy._similarity = lambda a, b: 0.05

        paragraphs = [
            "Alpha topic introduces balanced trees and their purpose.",
            "Beta topic explains B+ tree leaf linkage for range scans.",
            "Gamma topic describes hash buckets and equality lookups.",
        ]
        text = "\n\n".join(paragraphs)
        chunks = strategy.chunk(text)

        reconstructed = "\n\n".join(chunks)
        assert _normalize_whitespace(reconstructed) == _normalize_whitespace(text)
        positions = [reconstructed.find(paragraph) for paragraph in paragraphs]
        assert positions == sorted(positions)
        assert all(position >= 0 for position in positions)

    def test_max_size_respected(self, monkeypatch):
        strategy = _make_semantic_strategy(
            monkeypatch,
            chunk_size=90,
            chunk_overlap=0,
            min_paragraph_chars=20,
        )
        strategy._similarity = lambda a, b: 0.05

        text = "\n\n".join(
            [
                "This paragraph is deliberately long enough to require chunk splitting while still remaining coherent.",
                "This second paragraph is also long enough to pressure the packer and verify size enforcement.",
            ]
        )
        chunks = strategy.chunk(text)

        assert chunks
        assert all(len(chunk) <= 90 for chunk in chunks)

    def test_semantic_topic_shifts_create_boundaries(self, monkeypatch):
        strategy = _make_semantic_strategy(
            monkeypatch,
            chunk_size=400,
            chunk_overlap=0,
            min_paragraph_chars=20,
        )
        strategy._similarity = lambda a, b: 0.01

        text = (
            "Binary trees support hierarchical search and traversal.\n\n"
            "Transactions guarantee atomicity, consistency, isolation, and durability."
        )
        chunks = strategy.chunk(text)

        assert len(chunks) == 2
        assert "Binary trees" in chunks[0]
        assert "Transactions" in chunks[1]

    def test_same_topic_paragraphs_stay_together(self, monkeypatch):
        strategy = _make_semantic_strategy(
            monkeypatch,
            chunk_size=400,
            chunk_overlap=0,
            min_paragraph_chars=20,
        )
        strategy._similarity = lambda a, b: 0.95

        text = (
            "B+ trees maintain keys in sorted order for efficient search.\n\n"
            "B+ trees also link leaf nodes, making range scans efficient."
        )
        chunks = strategy.chunk(text)

        assert len(chunks) == 1
        assert "range scans efficient" in chunks[0]

    def test_fallback_splitting_handles_single_large_unit(self, monkeypatch):
        strategy = _make_semantic_strategy(
            monkeypatch,
            chunk_size=80,
            chunk_overlap=0,
            min_paragraph_chars=20,
        )
        strategy._similarity = lambda a, b: 0.95

        text = (
            "This is a single giant paragraph with many sentences. "
            "It keeps going so that one semantic unit exceeds the maximum size. "
            "The fallback splitter should break it into multiple chunks safely."
        )
        chunks = strategy.chunk(text)

        assert len(chunks) >= 2
        assert all(len(chunk) <= 80 for chunk in chunks)

    def test_sentence_and_paragraph_boundaries_respected(self, monkeypatch):
        strategy = _make_semantic_strategy(
            monkeypatch,
            chunk_size=120,
            chunk_overlap=0,
            min_paragraph_chars=20,
        )
        strategy._similarity = lambda a, b: 0.01

        text = (
            "Paragraph one ends with a full sentence.\n\n"
            "Paragraph two starts cleanly and should not begin in the middle of a token."
        )
        chunks = strategy.chunk(text)

        assert chunks[0].endswith(".")
        assert chunks[1].startswith("Paragraph two")

    def test_page_markers_are_not_retained_as_content(self, monkeypatch):
        strategy = _make_semantic_strategy(
            monkeypatch,
            chunk_size=200,
            chunk_overlap=0,
            min_paragraph_chars=20,
        )
        strategy._similarity = lambda a, b: 0.05

        text = (
            "First paragraph about indexing.\n\n"
            "--- Page 12 ---\n\n"
            "Second paragraph about transactions."
        )
        chunks = strategy.chunk(text)
        full_text = "\n\n".join(chunks)

        assert "--- Page 12 ---" not in full_text
        assert "First paragraph" in full_text
        assert "Second paragraph" in full_text


class TestChunkingPipelineCorrectness:
    def test_document_chunker_preserves_table_blocks_and_surrounding_text(self):
        from src.preprocessing.chunking import (
            DocumentChunker,
            SectionRecursiveConfig,
            SectionRecursiveStrategy,
        )

        strategy = SectionRecursiveStrategy(
            SectionRecursiveConfig(recursive_chunk_size=500, recursive_overlap=0)
        )
        chunker = DocumentChunker(strategy=strategy, keep_tables=True)

        text = (
            "Before table content.\n\n"
            "<table><tr><td>Cell</td></tr></table>\n\n"
            "After table content."
        )
        chunks = chunker.chunk(text)
        full_text = "\n".join(chunks)

        assert "Before table content." in full_text
        assert "<table><tr><td>Cell</td></tr></table>" in full_text
        assert "After table content." in full_text

    def test_chunk_type_classification_on_simple_examples(self):
        from src.index_builder import classify_chunk_type

        assert classify_chunk_type("Definition: a superkey uniquely identifies tuples.", "") == "definition"
        assert classify_chunk_type("For example, a hash index maps keys to buckets.", "") == "example"
        assert classify_chunk_type("| a | b |\n| 1 | 2 |", "") == "table"
        assert classify_chunk_type("Theorem. Every strict 2PL schedule is conflict serializable.", "") == "theorem"
        assert classify_chunk_type("def insert(node, key):\n    return node", "") == "code"

    def test_build_index_attaches_metadata_and_preserves_chunk_order(self, monkeypatch, tmp_path):
        import src.index_builder as index_builder

        sections = [
            {
                "heading": "Section 14.3 B + -Tree Index Files",
                "content": "Definition. A B+ tree is a balanced index.\n\nExample. Leaves are linked for range queries.",
                "level": 2,
                "chapter": 14,
            }
        ]

        class StubChunker:
            def chunk(self, text, context=None):
                return [
                    "Definition. A B+ tree is a balanced index.",
                    "Example. Leaves are linked for range queries.",
                ]

        class StubEmbedder:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, texts, **kwargs):
                return np.zeros((len(texts), 4), dtype=np.float32)

        class StubFaissIndex:
            def __init__(self, dim):
                self.dim = dim
                self.vectors = None

            def add(self, vectors):
                self.vectors = vectors

        written = {}

        monkeypatch.setattr(index_builder, "extract_sections_from_markdown", lambda *a, **k: sections)
        monkeypatch.setattr(index_builder, "SentenceTransformer", StubEmbedder)
        monkeypatch.setattr(index_builder.faiss, "IndexFlatL2", StubFaissIndex)
        monkeypatch.setattr(index_builder.faiss, "write_index", lambda index, path: written.setdefault("faiss_path", path))

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        class StubConfig:
            def to_string(self):
                return "chunk_mode=semantic_sections"

        index_builder.build_index(
            markdown_file="dummy.md",
            chunker=StubChunker(),
            chunk_config=StubConfig(),
            embedding_model_path="dummy-embedder",
            artifacts_dir=artifacts_dir,
            index_prefix="test_index",
        )

        meta = pickle.loads((artifacts_dir / "test_index_meta.pkl").read_bytes())
        chunks = pickle.loads((artifacts_dir / "test_index_chunks.pkl").read_bytes())

        assert len(meta) == 2
        assert chunks[0].startswith("Definition.")
        assert chunks[1].startswith("Example.")
        assert meta[0]["chapter_num"] == 14
        assert meta[0]["section_depth"] >= 1
        assert meta[0]["section_hierarchy"][0] == "Chapter 14"
        assert meta[0]["chunk_type"] == "definition"
        assert meta[1]["chunk_type"] == "example"
        assert meta[0]["chunk_id"] == 0
        assert meta[1]["chunk_id"] == 1
        assert (artifacts_dir / "test_index_page_to_chunk_map.json").exists()
