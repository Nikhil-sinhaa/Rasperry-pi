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
    """Tokenizer for BM25 – handles Indian currency/number formatting.

    * Normalises Indian comma-separated numbers (14,75,000 → 1475000)
      so BM25 can match the digits regardless of comma placement.
    * Keeps the rupee symbol ₹ as a standalone token.
    * Strips trailing /- (common Indian currency suffix).
    """
    # Collapse Indian comma-formatted numbers to plain digit strings
    normalised = re.sub(r'(\d(?:[,\d]{0,6}\d))\/-?', lambda m: m.group(1).replace(',', ''), text)
    normalised = re.sub(r'(\d[\d,]*)\/-?', lambda m: m.group(1).replace(',', ''), normalised)
    # Tokenize: include ₹ symbol and standard alphanum tokens
    tokens = re.findall(r'[\u20b9₹]|[a-z0-9]+', normalised.casefold())
    return tokens


# Indian legal term expansions for query-time augmentation
_LEGAL_TERM_EXPANSIONS: dict[str, list[str]] = {
    "cheque":     ["cheque", "check", "draft", "negotiable instrument", "payment instrument"],
    "bounced":    ["bounced", "dishonoured", "dishonored", "returned unpaid", "not honoured"],
    "lakh":       ["lakh", "lac", "100000"],
    "crore":      ["crore", "10000000"],
    "rs":         ["rs", "inr", "rupees", "rupee"],
    "hdfc":       ["hdfc", "bank"],
    "icici":      ["icici", "bank"],
    "sbi":        ["sbi", "state bank", "bank"],
    "invoice":    ["invoice", "bill", "payment"],
    "drawer":     ["drawer", "issuer", "payer"],
    "drawee":     ["drawee", "payee", "recipient"],
    "legal notice": ["legal notice", "demand notice", "notice"],
    "section 138": ["section 138", "negotiable instruments act", "cheque dishonour"],
}


def _expand_legal_query(query: str) -> str:
    """Expand Indian legal shorthand in a query to improve embedding and BM25 recall.

    Returns the original query with expansions appended so that both the original
    terms and their synonyms are embedded together.
    """
    lowered = query.casefold()
    extras: list[str] = []
    for term, synonyms in _LEGAL_TERM_EXPANSIONS.items():
        if term in lowered:
            # Add synonyms that aren't already in the query
            extras.extend(s for s in synonyms if s not in lowered)
    if not extras:
        return query
    # Append unique extras without duplicating existing words
    seen: set[str] = set(lowered.split())
    unique_extras = [e for e in extras if e not in seen]
    if not unique_extras:
        return query
    return query + " " + " ".join(unique_extras)


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

    Indian legal documents use domain-specific shorthand (cheque, Rs., Lakh,
    HDFC, /-, Indian comma-formatted numbers) that dense embeddings may not
    align well with.  The query is expanded with synonyms before embedding and
    BM25 tokenization to improve recall.  The similarity threshold gate in
    agent.py is evaluated against the hybrid combined score so that a strong
    BM25 keyword match is not blocked by a weak embedding cosine alone.
    """
    expanded_query = _expand_legal_query(query)
    embed_scores = _rank_for_query(index, expanded_query, runtime)  # cosine, range ~[-1, 1]
    # Normalise embedding scores to [0, 1] for blending
    e_min, e_max = float(embed_scores.min()), float(embed_scores.max())
    if e_max > e_min:
        norm_embed = (embed_scores - e_min) / (e_max - e_min)
    else:
        norm_embed = embed_scores - e_min  # all-zero when all scores equal

    if index._bm25_df:
        # Tokenize the *expanded* query so BM25 also benefits from synonyms
        query_tokens = _tokenize(expanded_query)
        bm25 = _bm25_scores(query_tokens, index)
        combined = alpha * norm_embed + (1.0 - alpha) * bm25
    else:
        combined = norm_embed

    k = top_k or SETTINGS.top_k_question
    ordered = np.argsort(combined)[::-1][:k]
    # Gate abstention on the *hybrid* score, not just the raw embedding cosine,
    # so that a strong BM25 keyword match is not blocked by a low embedding score.
    top_hybrid_score = float(combined[ordered[0]]) if len(ordered) else -1.0
    results = [(index.chunks[int(p)], float(embed_scores[p])) for p in ordered]
    return results, top_hybrid_score


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
