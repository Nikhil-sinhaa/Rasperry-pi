"""Deterministic validation for source IDs, verbatim quotes, and claim-vs-quote fact checks.

Member 1 additions
------------------
* claim_vs_quote_check()  – drop findings whose numbers / dates / durations cannot be
  traced back to any of the finding's quotes (catches "quote is real, claim is wrong").
* assign_topic()          – rule-based topic tag (obligations / money / deadlines / …)
* assign_attention()      – rule-based attention level (high / medium / low)
* validate_findings() now returns Finding objects with topic, attention and support fields.
"""
from __future__ import annotations

import re
from typing import Any

from .types import ALLOWED_ATTENTION, ALLOWED_SUPPORT, ALLOWED_TOPICS, Finding

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

# ---------------------------------------------------------------------------
# Regex helpers for fact extraction
# ---------------------------------------------------------------------------

# Numbers: plain integers, floats, Indian-comma style (1,00,000), currency amounts
_NUMBER_RE = re.compile(
    r"(?:"
    r"(?:INR|USD|EUR|GBP|Rs\.?)\s*[\d,]+(?:\.\d+)?"  # currency prefixed
    r"|"
    r"[\d,]+(?:\.\d+)?\s*(?:INR|USD|EUR|GBP|lakh|lakhs|crore|crores|%|percent)?"  # number with optional suffix
    r")",
    re.IGNORECASE,
)

# Dates: YYYY-MM-DD, DD/MM/YYYY, DD Month YYYY, Month DD YYYY, relative dates
_DATE_RE = re.compile(
    r"(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}"
    r")",
    re.IGNORECASE,
)

# Durations: "30 days", "11 months", "2 years"
_DURATION_RE = re.compile(
    r"\b\d+\s*(?:day|days|month|months|year|years|week|weeks|hour|hours)\b",
    re.IGNORECASE,
)


def _extract_facts(text: str) -> list[str]:
    """Return a list of normalised number/date/duration tokens found in *text*."""
    facts: list[str] = []
    for pattern in (_NUMBER_RE, _DATE_RE, _DURATION_RE):
        for m in pattern.finditer(text):
            facts.append(re.sub(r"\s+", " ", m.group()).strip().casefold())
    return facts


def _normalize_number(token: str) -> str:
    """Remove formatting separators so 1,00,000 == 100000 == 1,000,00."""
    return re.sub(r"[,\s]", "", token.casefold())


def claim_vs_quote_check(claim: str, quotes: list[str]) -> bool:
    """Return True when every number/date/duration in *claim* appears in at least one quote.

    The check normalises commas in numbers so that formatting differences do not
    cause false drops.  If the claim contains no checkable facts the check passes
    trivially (the model had nothing numeric to misstate).
    """
    claim_facts = _extract_facts(claim)
    if not claim_facts:
        return True  # nothing numeric to verify

    all_quote_text = " ".join(quotes).casefold()
    for fact in claim_facts:
        norm_fact = _normalize_number(fact)
        # Accept if the normalised token appears literally in the combined quotes
        # OR the original token (with punctuation) does.
        if norm_fact not in _normalize_number(all_quote_text) and fact not in all_quote_text:
            return False
    return True


# ---------------------------------------------------------------------------
# Legal judgment filter (unchanged logic, unchanged signatures)
# ---------------------------------------------------------------------------


def contains_unqualified_legal_judgment(claim: str, category: str) -> bool:
    """Reject model claims that assert legal validity/enforceability as fact.

    Attributed summaries of what the contract itself says may be retained as explicit statements.
    """
    has_judgment = any(re.search(pattern, claim, flags=re.IGNORECASE) for pattern in LEGAL_JUDGMENT_PATTERNS)
    if not has_judgment:
        return False
    return not (category == "explicit_statement" and ATTRIBUTED_TEXT_PATTERN.search(claim.strip()))


# ---------------------------------------------------------------------------
# Quote normalisation helpers (public, used by tests)
# ---------------------------------------------------------------------------


def normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_is_verbatim(quote: str, source_text: str, min_chars: int = MIN_QUOTE_CHARS) -> bool:
    if not isinstance(quote, str) or len(normalize_quote(quote)) < min_chars:
        return False
    return normalize_quote(quote) in normalize_quote(source_text)


# ---------------------------------------------------------------------------
# Topic and attention assignment (rule-based, no model confidence used)
# ---------------------------------------------------------------------------

_TOPIC_RULES: list[tuple[str, list[str]]] = [
    ("money", ["inr", "usd", "eur", "gbp", "rs.", "rupee", "amount", "fee", "rent", "deposit",
                "penalty", "fine", "charge", "payment", "refund", "interest", "lakh", "crore",
                "salary", "cost", "price", "compensation", "damages"]),
    ("deadlines", ["days", "months", "years", "deadline", "due date", "notice period",
                   "within", "before", "after", "schedule", "by the", "no later than", "expiry",
                   "expire", "renewal", "term"]),
    ("termination", ["terminat", "cancel", "end the agreement", "end of contract",
                     "expir", "dissolv", "rescind", "exit clause", "break clause"]),
    ("obligations", ["shall", "must", "required to", "obliged", "duty", "responsible",
                     "obligation", "covenant", "undertake"]),
    ("liability", ["liable", "liability", "indemnif", "warrant", "guarantee", "limitation",
                   "waiver", "remedy", "remedies", "damages"]),
    ("jurisdiction", ["jurisdiction", "govern", "law of", "court of", "arbitration",
                      "dispute resolution", "applicable law"]),
    ("ambiguity", ["unclear", "ambiguous", "conflicting", "vague", "uncertain",
                   "not specified", "not defined"]),
]

_ATTENTION_HIGH_PATTERNS = [
    re.compile(r"\b(?:penalty|penalties|fine|damages|liquidated)\b", re.IGNORECASE),
    re.compile(r"\b(?:auto.?renew|automatically renew)\b", re.IGNORECASE),
    re.compile(r"\b(?:one.sided|unilateral)\b", re.IGNORECASE),
    re.compile(r"\b(?:short notice|immediate(?:ly)?)\b", re.IGNORECASE),
    # Money amounts above zero
    re.compile(r"(?:INR|USD|EUR|GBP|Rs\.?)\s*[\d,]+", re.IGNORECASE),
    re.compile(r"\b\d[\d,]*\s*(?:lakh|crore)\b", re.IGNORECASE),
]

_ATTENTION_MEDIUM_PATTERNS = [
    re.compile(r"\b(?:terminat|cancel|exit|expir)\w*\b", re.IGNORECASE),
    re.compile(r"\b(?:notice period|days notice)\b", re.IGNORECASE),
    re.compile(r"\b(?:liable|liability|indemnif)\w*\b", re.IGNORECASE),
]


def assign_topic(claim: str) -> str:
    """Return the best-matching topic for *claim* using keyword rules."""
    lower = claim.casefold()
    for topic, keywords in _TOPIC_RULES:
        if any(kw in lower for kw in keywords):
            return topic
    return "general"


def assign_attention(claim: str, quotes: list[str]) -> str:
    """Return high / medium / low based on content signals in the claim and quotes.

    Never uses the model's own confidence; purely rule-based.
    """
    combined = (claim + " " + " ".join(quotes)).casefold()
    if any(p.search(combined) for p in _ATTENTION_HIGH_PATTERNS):
        return "high"
    if any(p.search(combined) for p in _ATTENTION_MEDIUM_PATTERNS):
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------


def validate_findings(payload: Any, sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only findings with a permitted category, verifiable verbatim quotes, and
    claim-vs-quote fact consistency.

    Returns plain dicts (not Finding objects) so the function signature is backward-
    compatible with the existing agent and UI code.  Each returned dict now contains
    'topic', 'attention', and 'support' keys in addition to the existing fields.
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

        # --- claim-vs-quote number/date/duration check (Member 1) ---
        quotes_text = [e["quote"] for e in evidence]
        numbers_ok = claim_vs_quote_check(claim, quotes_text)
        if not numbers_ok:
            continue  # drop: claim asserts numbers not present in any supporting quote

        normalized_claim = re.sub(r"\s+", " ", claim).strip()
        signature = (str(category), normalized_claim.casefold())
        if signature in seen:
            continue
        seen.add(signature)

        lawyer_question = raw.get("lawyer_question", "")
        if not isinstance(lawyer_question, str):
            lawyer_question = ""

        # Build support set
        support: list[str] = ["quote_valid"]
        if numbers_ok and _extract_facts(claim):
            support.append("numbers_verified")

        topic = assign_topic(normalized_claim)
        attention = assign_attention(normalized_claim, quotes_text)

        clean.append(
            {
                "category": category,
                "claim": normalized_claim,
                "evidence": evidence,
                "lawyer_question": lawyer_question.strip()[:400],
                "topic": topic,
                "attention": attention,
                "support": support,
            }
        )
    return clean
