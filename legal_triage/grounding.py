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
    # Bare copula: "is enforceable", "is not enforceable", "is illegal"
    r"\b(?:is|is not|isn't)\s+(?:un)?enforceable\b",
    r"\b(?:is|is not|isn't)\s+(?:il)?legal\b",
    r"\b(?:is|would be|may be)\s+(?:invalid|valid|void|unlawful)\b",
    # Hedged / modal: "probably unenforceable", "likely illegal", "would not be enforceable"
    r"\b(?:probably|likely|clearly|certainly|arguably|would\s+(?:not\s+)?be|will\s+(?:not\s+)?be|"
    r"may\s+(?:well\s+)?be|might\s+be|could\s+be|appears\s+to\s+be|seems\s+to\s+be)\s+"
    r"(?:un)?enforceable\b",
    r"\b(?:probably|likely|clearly|certainly|arguably|would\s+(?:not\s+)?be|will\s+(?:not\s+)?be|"
    r"may\s+(?:well\s+)?be|might\s+be|could\s+be|appears\s+to\s+be|seems\s+to\s+be)\s+"
    r"(?:il)?legal\b",
    r"\b(?:probably|likely|clearly|certainly|arguably|would\s+(?:not\s+)?be|will\s+(?:not\s+)?be|"
    r"may\s+(?:well\s+)?be|might\s+be|could\s+be|appears\s+to\s+be|seems\s+to\s+be)\s+"
    r"(?:invalid|void|unlawful|unenforceable)\b",
    # "violates law" / "contrary to law"
    r"\bviolates?\s+(?:the\s+)?law\b",
    r"\bcontrary to (?:the )?law\b",
    # "not enforceable under …"
    r"\bnot\s+enforceable\b",
    r"\bnot\s+legally\s+(?:valid|binding|enforceable)\b",
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

# Durations: "30 days", "11 months", "2 years"; also spelled-out: "thirty days"
_WORD_NUMBERS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|lakh|lakhs|crore|crores"
)
_DURATION_RE = re.compile(
    rf"\b(?:\d+|{_WORD_NUMBERS})(?:\s*(?:-|to)\s*(?:\d+|{_WORD_NUMBERS}))?\s*"
    r"(?:day|days|month|months|year|years|week|weeks|hour|hours)\b",
    re.IGNORECASE,
)

# Mapping of spelled-out cardinals → digit strings (for normalisation only)
_WORD_TO_NUM: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
    # Indian number words: kept as normalised suffixes so '5 lakh' == '500000'
    "lakh": "00000", "lakhs": "00000",
    "crore": "0000000", "crores": "0000000",
}


def _normalize_spelled_numbers(text: str) -> str:
    """Replace spelled-out cardinals with their digit equivalents.

    Only whole words are replaced, so 'second' (ordinal) is not touched.
    This is applied before digit extraction so the patterns can match.
    """
    lower = text.casefold()
    for word, digit in _WORD_TO_NUM.items():
        lower = re.sub(rf"\b{re.escape(word)}\b", digit, lower)
    return lower


def _extract_facts(text: str) -> list[str]:
    """Return a list of normalised number/date/duration tokens found in *text*.

    Spelled-out numbers ("thirty days", "two years") are converted to digits
    before pattern matching so that they compare equal to their numeral forms.
    """
    normalised_text = _normalize_spelled_numbers(text)
    facts: list[str] = []
    for pattern in (_NUMBER_RE, _DATE_RE, _DURATION_RE):
        for m in pattern.finditer(normalised_text):
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
# Quote normalisation helpers (public, used by tests and by ingestion callers)
# ---------------------------------------------------------------------------

import unicodedata as _unicodedata


def normalize_text_for_matching(text: str) -> str:
    """Canonical normalisation for quote matching.

    Applied identically to the stored chunk text and to the model-emitted quote
    so that Word/PDF typography differences never cause false abstentions.

    Steps:
    1. NFKC decomposition (ligatures, full-width chars, etc.)
    2. Curly / 'smart' quote → ASCII quote
    3. Em-dash / en-dash → plain hyphen
    4. Soft hyphen (U+00AD) removal
    5. Collapse all whitespace (including non-breaking spaces) to a single space
    6. Strip and casefolding
    """
    # NFKC handles ligatures (ﬁ→fi), full-width digits, etc.
    t = _unicodedata.normalize("NFKC", text)
    # Curly quotes → straight
    t = t.replace("\u2018", "'").replace("\u2019", "'")   # '' → '
    t = t.replace("\u201c", '"').replace("\u201d", '"')   # "" → "
    t = t.replace("\u2032", "'").replace("\u2033", '"')   # prime marks
    # Dashes → plain hyphen (for matching only – display keeps originals)
    t = t.replace("\u2013", "-").replace("\u2014", "-")   # en/em dash
    t = t.replace("\u2012", "-").replace("\u2015", "-")   # figure/horizontal bar
    # Remove soft hyphen (U+00AD) – appears in justified/hyphenated PDFs
    t = t.replace("\u00ad", "")
    # Collapse any whitespace (including \xa0, \u202f, \t, \n) to single space
    t = re.sub(r"[\s\xa0\u202f\u2009\u200b]+", " ", t)
    return t.strip().casefold()


def normalize_quote(text: str) -> str:
    """Legacy alias – delegates to normalize_text_for_matching."""
    return normalize_text_for_matching(text)


def quote_is_verbatim(quote: str, source_text: str, min_chars: int = MIN_QUOTE_CHARS) -> bool:
    if not isinstance(quote, str) or len(normalize_text_for_matching(quote)) < min_chars:
        return False
    return normalize_text_for_matching(quote) in normalize_text_for_matching(source_text)


# ---------------------------------------------------------------------------
# Topic and attention assignment (rule-based, no model confidence used)
# ---------------------------------------------------------------------------

_TOPIC_RULES: list[tuple[str, list[str]]] = [
    ("money", ["inr", "usd", "eur", "gbp", "rs.", "rupee", "amount", "fee", "rent", "deposit",
                "penalty", "fine", "charge", "payment", "refund", "interest", "lakh", "crore",
                "salary", "cost", "price", "compensation", "damages"]),
    # termination must come before deadlines: "terminate"/"terminat" would otherwise match
    # the bare "term" keyword in deadlines via substring match.
    ("termination", ["terminat", "cancel", "end the agreement", "end of contract",
                     "expir", "dissolv", "rescind", "exit clause", "break clause"]),
    ("deadlines", ["days", "months", "years", "deadline", "due date", "notice period",
                   "within", "before", "after", "schedule", "by the", "no later than", "expiry",
                   "expire", "renewal"]),
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
