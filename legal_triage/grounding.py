"""Deterministic validation for source IDs and verbatim quote evidence."""
from __future__ import annotations

import re
from typing import Any

ALLOWED_CATEGORIES = {
    "explicit_statement",
    "potential_concern",
    "ambiguity",
    "jurisdiction_dependency",
}
MIN_QUOTE_CHARS = 24
LEGAL_JUDGMENT_PATTERNS = [
    r"\b(?:is|is not|isn't|would be|will be|may be)\s+(?:un)?enforceable\b",
    r"\b(?:is|is not|isn't|would be|will be|may be)\s+(?:il)?legal\b",
    r"\b(?:is|would be|may be)\s+(?:invalid|valid|void|unlawful)\b",
    r"\bviolates?\s+(?:the\s+)?law\b",
    r"\bcontrary to (?:the )?law\b",
]
ATTRIBUTED_TEXT_PATTERN = re.compile(
    r"^(?:the (?:document|agreement|contract|clause) (?:says|states|provides|specifies|describes)|"
    r"according to the (?:document|agreement|contract|clause))\b",
    flags=re.IGNORECASE,
)


def contains_unqualified_legal_judgment(claim: str, category: str) -> bool:
    """Reject model claims that assert legal validity/enforceability as fact.

    Attributed summaries of what the contract itself says may be retained as explicit statements.
    """
    has_judgment = any(re.search(pattern, claim, flags=re.IGNORECASE) for pattern in LEGAL_JUDGMENT_PATTERNS)
    if not has_judgment:
        return False
    return not (category == "explicit_statement" and ATTRIBUTED_TEXT_PATTERN.search(claim.strip()))


def normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_is_verbatim(quote: str, source_text: str, min_chars: int = MIN_QUOTE_CHARS) -> bool:
    if not isinstance(quote, str) or len(normalize_quote(quote)) < min_chars:
        return False
    return normalize_quote(quote) in normalize_quote(source_text)


def validate_findings(payload: Any, sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only findings with a permitted category and quotes verifiable in retrieved text.

    This rejects fabricated source IDs and non-verbatim quotes. It cannot prove that a model's
    interpretation logically follows from a quote; the UI therefore exposes the quote beside it.
    """
    if not isinstance(payload, dict):
        return []
    raw_findings = payload.get("findings", [])
    if not isinstance(raw_findings, list):
        return []

    clean: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_findings[:30]:
        if not isinstance(raw, dict):
            continue
        category = raw.get("category")
        claim = raw.get("claim")
        raw_evidence = raw.get("evidence")
        if category not in ALLOWED_CATEGORIES or not isinstance(claim, str) or not claim.strip():
            continue
        if contains_unqualified_legal_judgment(claim, str(category)):
            continue
        if not isinstance(raw_evidence, list):
            continue

        evidence: list[dict[str, str]] = []
        seen_sources: set[str] = set()
        for item in raw_evidence[:5]:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            quote = item.get("quote")
            if not isinstance(source_id, str) or source_id not in sources or source_id in seen_sources:
                continue
            if not isinstance(quote, str) or not quote_is_verbatim(quote, str(sources[source_id].get("text", ""))):
                continue
            evidence.append({"source_id": source_id, "quote": quote.strip()})
            seen_sources.add(source_id)
        if not evidence:
            continue

        normalized_claim = re.sub(r"\s+", " ", claim).strip()
        signature = (str(category), normalized_claim.casefold())
        if signature in seen:
            continue
        seen.add(signature)
        lawyer_question = raw.get("lawyer_question", "")
        if not isinstance(lawyer_question, str):
            lawyer_question = ""
        clean.append(
            {
                "category": category,
                "claim": normalized_claim,
                "evidence": evidence,
                "lawyer_question": lawyer_question.strip()[:400],
            }
        )
    return clean
