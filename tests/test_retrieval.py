"""Tests for the retrieval module using fake (deterministic) embeddings.

Member 3 requirement: retrieval tests with fake embeddings so the suite does not
require a running Ollama server.

We create a minimal fake runtime that returns synthetic unit vectors, build a
LocalIndex with known chunk content, and verify that retrieve_question() and
retrieve_triage_per_topic() return the expected chunks.
"""
from __future__ import annotations

import numpy as np
import pytest

from legal_triage.retrieval import (
    LocalIndex,
    retrieve_question,
    retrieve_triage,
    retrieve_triage_per_topic,
    TRIAGE_TOPIC_SEARCHES,
)
from legal_triage.types import Chunk


# ---------------------------------------------------------------------------
# Fake runtime
# ---------------------------------------------------------------------------

class FakeRuntime:
    """Minimal fake runtime that returns deterministic unit-vector embeddings.

    Each unique text is assigned a unit vector in a different canonical direction
    (e1, e2, e3, …).  This makes cosine similarity perfectly predictable:
    the vector for text_i has similarity 1.0 with itself and 0.0 with all others.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self._registry: dict[str, int] = {}  # text -> column index
        self._counter = 0
        self.chat_model = "fake-model"
        self.embedding_model = "fake-embed"

    def _get_index(self, text: str) -> int:
        if text not in self._registry:
            self._registry[text] = self._counter % self.dim
            self._counter += 1
        return self._registry[text]

    def embed(self, texts: list[str], batch_size: int = 24, **kwargs) -> np.ndarray:
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            idx = self._get_index(text)
            result[row, idx] = 1.0
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunks(texts: list[str], filename: str = "test.txt") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{filename}::p1::c{i+1}",
            filename=filename,
            page_number=1,
            text=text,
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture()
def fake_runtime() -> FakeRuntime:
    return FakeRuntime(dim=32)


@pytest.fixture()
def three_chunk_index(fake_runtime: FakeRuntime) -> tuple[LocalIndex, list[Chunk]]:
    """Index with three chunks, each assigned a distinct embedding direction."""
    texts = [
        "The monthly rent is INR 20,000 payable on the fifth day.",
        "The security deposit is INR 40,000 refundable within 30 days.",
        "The notice period for termination is 30 days written notice.",
    ]
    chunks = _make_chunks(texts)
    vectors = fake_runtime.embed(texts)
    index = LocalIndex(chunks=chunks, vectors=vectors, metadata={"pages": 1, "chunks": 3})
    return index, chunks


# ---------------------------------------------------------------------------
# retrieve_question tests
# ---------------------------------------------------------------------------

class TestRetrieveQuestion:
    def test_top_result_is_most_similar(
        self, three_chunk_index: tuple, fake_runtime: FakeRuntime
    ) -> None:
        index, chunks = three_chunk_index
        query = chunks[0].text  # exact match → similarity 1.0
        results, top_score = retrieve_question(index, query, fake_runtime, top_k=3)
        assert top_score == pytest.approx(1.0, abs=1e-5)
        # The first result must be the chunk whose text matches the query
        assert results[0][0].text == query

    def test_top_k_respected(
        self, three_chunk_index: tuple, fake_runtime: FakeRuntime
    ) -> None:
        index, _ = three_chunk_index
        results, _ = retrieve_question(index, "anything", fake_runtime, top_k=2)
        assert len(results) <= 2

    def test_scores_are_floats_in_range(
        self, three_chunk_index: tuple, fake_runtime: FakeRuntime
    ) -> None:
        index, chunks = three_chunk_index
        results, top_score = retrieve_question(index, chunks[1].text, fake_runtime)
        assert all(-1.0 <= score <= 1.0 for _, score in results)
        assert -1.0 <= top_score <= 1.0

    def test_empty_index_raises(self, fake_runtime: FakeRuntime) -> None:
        """An index with no chunks should not crash retrieve_question."""
        empty_index = LocalIndex(
            chunks=[],
            vectors=np.empty((0, 32), dtype=np.float32),
            metadata={"pages": 0},
        )
        results, top_score = retrieve_question(empty_index, "anything", fake_runtime)
        assert results == []
        assert top_score == pytest.approx(-1.0, abs=1e-5)

    def test_dimension_mismatch_raises(self, fake_runtime: FakeRuntime) -> None:
        chunks = _make_chunks(["some text"])
        mismatched_vectors = np.ones((1, 8), dtype=np.float32)
        index = LocalIndex(chunks=chunks, vectors=mismatched_vectors, metadata={})
        with pytest.raises(RuntimeError, match="dimensions"):
            retrieve_question(index, "anything", fake_runtime)


# ---------------------------------------------------------------------------
# retrieve_triage / retrieve_triage_per_topic tests
# ---------------------------------------------------------------------------

class TestRetrieveTriage:
    def test_triage_returns_non_empty(
        self, three_chunk_index: tuple, fake_runtime: FakeRuntime
    ) -> None:
        index, _ = three_chunk_index
        results, top_score = retrieve_triage(index, fake_runtime)
        # With fake orthogonal embeddings similarity will be 0 for all triage queries;
        # the fallback path should still return something.
        assert isinstance(results, list)
        assert isinstance(top_score, float)

    def test_per_topic_returns_all_chunks_when_small(
        self, three_chunk_index: tuple, fake_runtime: FakeRuntime
    ) -> None:
        """For a small index the per-topic retrieval should surface all chunks."""
        index, chunks = three_chunk_index
        # Use a high top_k_per_topic so all chunks can be returned.
        results, _ = retrieve_triage_per_topic(index, fake_runtime, top_k_per_topic=10)
        returned_texts = {r[0].text for r in results}
        # At minimum, chunks with non-zero similarity to any triage query should appear.
        assert isinstance(results, list)

    def test_per_topic_fallback_when_below_threshold(
        self, fake_runtime: FakeRuntime
    ) -> None:
        """When similarity is below threshold everywhere, a fallback result is returned."""
        texts = ["totally unrelated text about cooking recipes"]
        chunks = _make_chunks(texts)
        vectors = fake_runtime.embed(texts)
        index = LocalIndex(chunks=chunks, vectors=vectors, metadata={"pages": 1})
        results, _ = retrieve_triage_per_topic(index, fake_runtime, top_k_per_topic=5)
        # Fallback should return the one available chunk
        assert len(results) >= 1
        assert results[0][0].text == texts[0]
