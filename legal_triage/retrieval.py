"""Local vector retrieval backed by normalized in-memory NumPy vectors.

Member 1 additions
------------------
* retrieve_triage_per_topic() – one retrieval pass per semantic topic so long documents
  don't lose important clauses that fall outside the top-12 generic chunks.
* Chunk.all_pages support   – chunks that span page breaks cite both pages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SETTINGS
from .ingestion import ingest_files
from .local_runtime import LocalOllama
from .types import Chunk


@dataclass
class LocalIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    metadata: dict[str, object]


def build_index(files: list[tuple[str, bytes]], runtime: LocalOllama) -> LocalIndex:
    chunks, metadata = ingest_files(files)
    vectors = runtime.embed([chunk.text for chunk in chunks])
    if vectors.shape[0] != len(chunks):
        raise RuntimeError("Embedding count does not match the number of document chunks.")
    return LocalIndex(chunks=chunks, vectors=vectors, metadata=metadata)


# ---------------------------------------------------------------------------
# Triage search queries – one per semantic topic (Member 1)
# ---------------------------------------------------------------------------

TRIAGE_TOPIC_SEARCHES: dict[str, str] = {
    "general": "Main purpose, parties, scope, and term of this agreement",
    "obligations": "Obligations duties responsibilities required actions must shall required to",
    "rights": "Rights entitlements permissions may can is entitled to",
    "deadlines": "Dates deadlines notice period due date days within before after schedule",
    "money": "Money rent deposit fees payments late charges penalties damages refunds interest",
    "termination": "Termination cancellation renewal expiry breach default ending the agreement",
    "liability": "Liability indemnity warranties limitation of liability waiver exceptions remedies",
    "ambiguity": "Ambiguous unclear conflicting conditional wording exceptions and potential concerns",
}

# Legacy flat list kept for backward compat with any callers of retrieve_triage()
TRIAGE_SEARCHES = list(TRIAGE_TOPIC_SEARCHES.values())


def _rank_for_query(index: LocalIndex, query: str, runtime: LocalOllama) -> np.ndarray:
    query_vector = runtime.embed([query])
    if query_vector.shape[1] != index.vectors.shape[1]:
        raise RuntimeError("Query embedding dimensions do not match document embeddings.")
    return index.vectors @ query_vector[0]


def retrieve_question(
    index: LocalIndex,
    query: str,
    runtime: LocalOllama,
    top_k: int | None = None,
) -> tuple[list[tuple[Chunk, float]], float]:
    scores = _rank_for_query(index, query, runtime)
    ordered = np.argsort(scores)[::-1][: top_k or SETTINGS.top_k_question]
    top_score = float(scores[ordered[0]]) if len(ordered) else -1.0
    results = [(index.chunks[int(position)], float(scores[position])) for position in ordered]
    return results, top_score


def retrieve_triage(
    index: LocalIndex,
    runtime: LocalOllama,
    top_k: int | None = None,
) -> tuple[list[tuple[Chunk, float]], float]:
    """Legacy multi-query triage (kept for backward compat). Prefer retrieve_triage_per_topic()."""
    query_vectors = runtime.embed(TRIAGE_SEARCHES)
    if query_vectors.shape[1] != index.vectors.shape[1]:
        raise RuntimeError("Triage-query embedding dimensions do not match document embeddings.")
    score_matrix = query_vectors @ index.vectors.T
    best_score = float(np.max(score_matrix)) if score_matrix.size else -1.0

    candidates: dict[int, float] = {}
    for row in score_matrix:
        for position in np.argsort(row)[::-1][:3]:
            score = float(row[position])
            if score >= SETTINGS.min_retrieval_similarity:
                candidates[int(position)] = max(score, candidates.get(int(position), -1.0))
    ordered_positions = sorted(candidates, key=candidates.get, reverse=True)[: top_k or SETTINGS.top_k_triage]
    if not ordered_positions:
        per_chunk = np.max(score_matrix, axis=0)
        ordered_positions = [int(pos) for pos in np.argsort(per_chunk)[::-1][: top_k or SETTINGS.top_k_triage]]
        candidates = {pos: float(per_chunk[pos]) for pos in ordered_positions}
    results = [(index.chunks[position], candidates[position]) for position in ordered_positions]
    return results, best_score


def retrieve_triage_per_topic(
    index: LocalIndex,
    runtime: LocalOllama,
    top_k_per_topic: int = 5,
    max_total: int | None = None,
) -> tuple[list[tuple[Chunk, float]], float]:
    """Run one retrieval pass per semantic topic and merge the top chunks.

    This ensures that a long document (which may have many chunks) still surfaces
    clause categories that would be crowded out by a single bulk search.

    Returns up to *max_total* chunks (default: SETTINGS.top_k_triage × number of topics).
    """
    if max_total is None:
        max_total = SETTINGS.top_k_triage * len(TRIAGE_TOPIC_SEARCHES)

    query_list = list(TRIAGE_TOPIC_SEARCHES.values())
    query_vectors = runtime.embed(query_list)
    if query_vectors.shape[1] != index.vectors.shape[1]:
        raise RuntimeError("Triage-query embedding dimensions do not match document embeddings.")
    score_matrix = query_vectors @ index.vectors.T  # shape: (n_topics, n_chunks)
    global_best = float(np.max(score_matrix)) if score_matrix.size else -1.0

    # Gather top-k_per_topic positions for each topic
    candidates: dict[int, float] = {}
    for row in score_matrix:
        for position in np.argsort(row)[::-1][:top_k_per_topic]:
            score = float(row[position])
            if score >= SETTINGS.min_retrieval_similarity:
                candidates[int(position)] = max(score, candidates.get(int(position), -1.0))

    if not candidates:
        # Fall back: pick the best-scoring chunks even below threshold (for transparency)
        per_chunk = np.max(score_matrix, axis=0)
        fallback_positions = [int(p) for p in np.argsort(per_chunk)[::-1][:max_total]]
        candidates = {p: float(per_chunk[p]) for p in fallback_positions}

    ordered_positions = sorted(candidates, key=lambda p: candidates[p], reverse=True)[:max_total]
    results = [(index.chunks[position], candidates[position]) for position in ordered_positions]
    return results, global_best
