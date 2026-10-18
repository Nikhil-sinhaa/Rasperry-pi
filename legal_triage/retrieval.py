"""Local vector retrieval backed by normalized in-memory NumPy vectors."""
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


TRIAGE_SEARCHES = [
    "Main purpose, parties, scope, and term of this agreement",
    "Obligations duties responsibilities required actions must shall required to",
    "Rights entitlements permissions may can is entitled to",
    "Dates deadlines notice period due date days within before after schedule",
    "Money rent deposit fees payments late charges penalties damages refunds interest",
    "Termination cancellation renewal expiry breach default ending the agreement",
    "Liability indemnity warranties limitation of liability waiver exceptions remedies",
    "Ambiguous unclear conflicting conditional wording exceptions and potential concerns",
]


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
    # Multi-query retrieval gives clause categories a chance to surface in long files.
    query_vectors = runtime.embed(TRIAGE_SEARCHES)
    if query_vectors.shape[1] != index.vectors.shape[1]:
        raise RuntimeError("Triage-query embedding dimensions do not match document embeddings.")
    score_matrix = query_vectors @ index.vectors.T
    best_score = float(np.max(score_matrix)) if score_matrix.size else -1.0

    # Select the best qualifying chunk for each search query, then fill remaining slots
    # using the best per-chunk match. This improves category coverage without a cloud reranker.
    candidates: dict[int, float] = {}
    for row in score_matrix:
        for position in np.argsort(row)[::-1][:3]:
            score = float(row[position])
            if score >= SETTINGS.min_retrieval_similarity:
                candidates[int(position)] = max(score, candidates.get(int(position), -1.0))
    ordered_positions = sorted(candidates, key=candidates.get, reverse=True)[: top_k or SETTINGS.top_k_triage]
    if not ordered_positions:
        # Preserve the closest passages for transparency, but analysis will abstain below threshold.
        per_chunk = np.max(score_matrix, axis=0)
        ordered_positions = [int(pos) for pos in np.argsort(per_chunk)[::-1][: top_k or SETTINGS.top_k_triage]]
        candidates = {pos: float(per_chunk[pos]) for pos in ordered_positions}
    results = [(index.chunks[position], candidates[position]) for position in ordered_positions]
    return results, best_score
