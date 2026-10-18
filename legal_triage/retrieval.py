"""Local vector retrieval backed by normalized in-memory NumPy vectors.

Member 1 additions
------------------
* retrieve_triage_per_topic() – one retrieval pass per semantic topic so long documents
  don't lose important clauses that fall outside the top-12 generic chunks.
* Chunk.all_pages support   – chunks that span page breaks cite both pages.

Phase 4 additions (Member 3)
-----------------------------
* BM25-style keyword scoring alongside embeddings for hybrid retrieval.
  hybrid_retrieve_question() blends 0.7 × embedding cosine + 0.3 × BM25 IDF-weighted
  term score.  This significantly helps lexical legal terms (exact amounts, party names,
  clause numbers, "INR", "days") that dense embeddings sometimes rank lower.
  No new dependencies – pure stdlib + NumPy.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import SETTINGS
from .ingestion import ingest_files
from .local_runtime import LocalOllama
from .types import Chunk

# BM25 hyper-parameters
_BM25_K1: float = 1.5
_BM25_B: float = 0.75
_HYBRID_ALPHA: float = 0.70   # weight for embedding score (1 - alpha goes to BM25)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"[a-z0-9',.-]+", text.casefold())


@dataclass
class LocalIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    metadata: dict[str, object]
    # BM25 pre-computed data (populated by build_index)
    _bm25_df: dict[str, int] = field(default_factory=dict)      # document frequency per term
    _bm25_tf: list[Counter] = field(default_factory=list)       # term freq per chunk
    _bm25_avg_dl: float = field(default=0.0)                    # average doc length


def _compute_bm25_data(chunks: list[Chunk]) -> tuple[dict[str, int], list[Counter], float]:
    """Pre-compute BM25 term statistics over all chunks."""
    tf_list: list[Counter] = []
    df: dict[str, int] = Counter()
    total_len = 0
    for chunk in chunks:
        tokens = _tokenize(chunk.text)
        total_len += len(tokens)
        tf = Counter(tokens)
        tf_list.append(tf)
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    avg_dl = total_len / len(chunks) if chunks else 1.0
    return dict(df), tf_list, avg_dl


def _bm25_scores(query_tokens: list[str], index: LocalIndex) -> np.ndarray:
    """Return a BM25 score array of shape (n_chunks,), min-max normalised to [0, 1]."""
    n = len(index.chunks)
    scores = np.zeros(n, dtype=np.float32)
    N = n or 1
    for term in set(query_tokens):
        df = index._bm25_df.get(term, 0)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        for i, tf_counter in enumerate(index._bm25_tf):
            tf = tf_counter.get(term, 0)
            dl = sum(tf_counter.values()) or 1
            numerator = tf * (_BM25_K1 + 1)
            denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / index._bm25_avg_dl)
            scores[i] += idf * (numerator / denominator)
    # Normalise to [0, 1]
    max_score = scores.max()
    if max_score > 0:
        scores /= max_score
    return scores


def build_index(files: list[tuple[str, bytes]], runtime: LocalOllama) -> LocalIndex:
    chunks, metadata = ingest_files(files)
    # is_query=False (default) → nomic-embed-text uses 'search_document:' prefix for chunks
    vectors = runtime.embed([chunk.text for chunk in chunks], is_query=False)
    if vectors.shape[0] != len(chunks):
        raise RuntimeError("Embedding count does not match the number of document chunks.")
    bm25_df, bm25_tf, bm25_avg_dl = _compute_bm25_data(chunks)
    return LocalIndex(
        chunks=chunks,
        vectors=vectors,
        metadata=metadata,
        _bm25_df=bm25_df,
        _bm25_tf=bm25_tf,
        _bm25_avg_dl=bm25_avg_dl,
    )


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
    # is_query=True → nomic-embed-text uses 'search_query:' prefix for questions/triage queries
    query_vector = runtime.embed([query], is_query=True)
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


def hybrid_retrieve_question(
    index: LocalIndex,
    query: str,
    runtime: LocalOllama,
    top_k: int | None = None,
    alpha: float = _HYBRID_ALPHA,
) -> tuple[list[tuple[Chunk, float]], float]:
    """Blend embedding cosine similarity with BM25 keyword score.

    hybrid_score = alpha * embed_score + (1 - alpha) * bm25_score

    This improves recall for lexical legal terms (exact amounts, clause numbers,
    party names) that dense embeddings sometimes dilute across similar neighbours.
    Falls back to pure embedding retrieval when the index has no BM25 data.
    """
    embed_scores = _rank_for_query(index, query, runtime)  # already cosine, range ~[-1, 1]
    # Normalise embedding scores to [0, 1] for blending
    e_min, e_max = float(embed_scores.min()), float(embed_scores.max())
    if e_max > e_min:
        norm_embed = (embed_scores - e_min) / (e_max - e_min)
    else:
        norm_embed = embed_scores - e_min  # all-zero when all scores equal

    if index._bm25_df:
        query_tokens = _tokenize(query)
        bm25 = _bm25_scores(query_tokens, index)
        combined = alpha * norm_embed + (1.0 - alpha) * bm25
    else:
        combined = norm_embed

    k = top_k or SETTINGS.top_k_question
    ordered = [int(p) for p in np.argsort(combined)[::-1][:k]]

    # BM25 can move an exact keyword hit above the strongest semantic match.
    # Keep the dense winner in the returned evidence set so the retrieval gate
    # measures a passage that the model will actually see, rather than a
    # different (possibly weaker) passage that happened to rank first in the
    # hybrid blend.
    if ordered and len(embed_scores):
        dense_best = int(np.argmax(embed_scores))
        if dense_best not in ordered:
            ordered[-1] = dense_best

    top_score = max((float(embed_scores[p]) for p in ordered), default=-1.0)
    results = [(index.chunks[p], float(embed_scores[p])) for p in ordered]
    return results, top_score


def retrieve_triage(
    index: LocalIndex,
    runtime: LocalOllama,
    top_k: int | None = None,
) -> tuple[list[tuple[Chunk, float]], float]:
    """Legacy multi-query triage (kept for backward compat). Prefer retrieve_triage_per_topic()."""
    # is_query=True → nomic-embed-text uses 'search_query:' prefix for triage queries
    query_vectors = runtime.embed(TRIAGE_SEARCHES, is_query=True)
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
    # is_query=True → nomic-embed-text uses 'search_query:' prefix for triage topic queries
    query_vectors = runtime.embed(query_list, is_query=True)
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


# ---------------------------------------------------------------------------
# Windowed map-reduce triage (B3 fix: prevents silent context overflow)
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 3.5   # conservative estimate
_PROMPT_OVERHEAD_TOKENS = 600   # system prompt + wrapper text


def _estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN)


def _build_window_prompt_text(chunks: list[Chunk]) -> str:
    """Build the source block text for a list of chunks (for token-budget estimation)."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source_id = f"E{i}"
        parts.append(
            f"<{source_id} | file={chunk.filename} | page={chunk.page_number}>\n"
            f"{chunk.text}\n</{source_id}>"
        )
    return "\n\n".join(parts)


def windowed_triage(
    index: LocalIndex,
    runtime: LocalOllama,
    progress_callback: Optional[callable] = None,  # type: ignore[type-arg]
) -> tuple[list[tuple[Chunk, float]], float]:
    """Score chunks in bounded windows and return a prompt-sized candidate set.

    For documents that do not fit in OLLAMA_NUM_CTX the chunk list is split into
    ordered windows that each fit within the token budget.  Topic retrieval is
    run against each window, then candidates are merged, de-duplicated by
    chunk_id, and returned by retrieval score. The final set is capped to the
    same token budget because the caller puts these chunks in one model prompt.

    ``progress_callback(current_window, total_windows)`` is called before each
    window so the UI can display progress ("window 2 of 4").

    Returns the same (results, top_score) tuple as retrieve_triage_per_topic.
    """
    chunks = index.chunks
    if not chunks:
        return [], -1.0

    # Token budget per window: leave room for system prompt, wrapper, and output.
    budget_tokens = int(SETTINGS.num_ctx * 0.75) - _PROMPT_OVERHEAD_TOKENS
    budget_tokens = max(budget_tokens, 512)  # never go below 512

    # Split chunks into windows that fit the budget.
    windows: list[list[Chunk]] = []
    current_window: list[Chunk] = []
    current_tokens = 0
    for chunk in chunks:
        chunk_tokens = _estimate_tokens(chunk.text) + 80  # 80 for tag overhead
        if current_window and current_tokens + chunk_tokens > budget_tokens:
            windows.append(current_window)
            current_window = []
            current_tokens = 0
        current_window.append(chunk)
        current_tokens += chunk_tokens
    if current_window:
        windows.append(current_window)

    if not windows:
        return [], -1.0

    # Run per-topic retrieval scoped to each window's chunks.
    query_list = list(TRIAGE_TOPIC_SEARCHES.values())
    query_vectors = runtime.embed(query_list, is_query=True)

    all_candidates: dict[int, float] = {}
    global_best = -1.0
    total_windows = len(windows)

    for w_idx, window_chunks in enumerate(windows):
        if progress_callback is not None:
            progress_callback(w_idx + 1, total_windows)

        # Build window position map: local_index → global_index
        global_indices = [index.chunks.index(c) for c in window_chunks]

        window_vectors = index.vectors[global_indices]  # shape: (n_window, dim)
        score_matrix = query_vectors @ window_vectors.T  # (n_topics, n_window)

        window_best = float(np.max(score_matrix)) if score_matrix.size else -1.0
        if window_best > global_best:
            global_best = window_best

        top_k_per_topic = max(3, SETTINGS.top_k_triage // len(TRIAGE_TOPIC_SEARCHES))
        for row in score_matrix:
            for local_pos in np.argsort(row)[::-1][:top_k_per_topic]:
                score = float(row[local_pos])
                if score >= SETTINGS.min_retrieval_similarity:
                    global_pos = global_indices[int(local_pos)]
                    all_candidates[global_pos] = max(score, all_candidates.get(global_pos, -1.0))

    if not all_candidates:
        # Fallback: return top-scoring chunks regardless of threshold
        per_chunk = np.max(query_vectors @ index.vectors.T, axis=0)
        global_best = float(np.max(per_chunk)) if per_chunk.size else -1.0
        top_n = SETTINGS.top_k_triage * len(TRIAGE_TOPIC_SEARCHES)
        fallback_positions = [int(p) for p in np.argsort(per_chunk)[::-1][:top_n]]
        all_candidates = {p: float(per_chunk[p]) for p in fallback_positions}

    ordered = sorted(all_candidates, key=lambda p: all_candidates[p], reverse=True)
    results: list[tuple[Chunk, float]] = []
    selected_tokens = 0
    for position in ordered:
        chunk = index.chunks[position]
        chunk_tokens = _estimate_tokens(chunk.text) + 80  # source tag and metadata
        if selected_tokens + chunk_tokens > budget_tokens:
            continue
        results.append((chunk, all_candidates[position]))
        selected_tokens += chunk_tokens
    return results, global_best
