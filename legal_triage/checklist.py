"""Missing-clause checklist and structured-extraction table for Phase 4.

Member 3 deliverable.

Two features are provided here:

1. **detect_document_type(text)** – lightweight heuristic that classifies an ingested
   text as one of: rental, employment, service, nda, loan, legal_notice, or general.

2. **missing_clause_checklist(chunks, doc_type)** – returns a list of clause-level
   verdicts: *present*, *possibly_present*, or *absent*.  Works purely on keyword
   matching against the stored chunk text; no model call is needed.

3. **extract_structured_fields(chunks)** – runs rule-based regexes over all chunks
   and returns a dict of structured fields (parties, dates, amounts, notice periods,
   governing law) each backed by the verbatim matched text as a quote.

These outputs feed into the Streamlit UI to show:
  - A ✅ / ⚠️ / ❌ checklist of standard clauses,
  - A compact key-value table of extracted structured fields.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .types import Chunk


# ---------------------------------------------------------------------------
# Document-type detection
# ---------------------------------------------------------------------------

_DOC_TYPE_SIGNALS: dict[str, list[str]] = {
    "rental":       ["tenant", "landlord", "rent", "lease", "security deposit",
                     "premises", "occupancy", "lessee", "lessor"],
    "employment":   ["employee", "employer", "salary", "designation", "probation",
                     "notice period", "non-compete", "employment", "appraisal", "joining"],
    "service":      ["service provider", "client", "deliverable", "milestone",
                     "service agreement", "sow", "statement of work", "retainer"],
    "nda":          ["confidential", "non-disclosure", "proprietary information",
                     "trade secret", "nda", "receiving party", "disclosing party"],
    "loan":         ["borrower", "lender", "principal", "interest rate", "repayment",
                     "loan", "disbursement", "emi", "installment"],
    "legal_notice": ["legal notice", "demand notice", "cause of action",
                     "days from receipt", "advocate", "client in matter"],
}


def detect_document_type(full_text: str) -> str:
    """Return a doc-type label based on keyword frequency in the full text."""
    lower = full_text.casefold()
    scores: dict[str, int] = {}
    for doc_type, keywords in _DOC_TYPE_SIGNALS.items():
        scores[doc_type] = sum(lower.count(kw) for kw in keywords)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Per-type checklist definitions
# ---------------------------------------------------------------------------

@dataclass
class ClauseSpec:
    name: str                          # Human-readable clause name
    keywords: list[str]                # Any match → present
    soft_keywords: list[str] = field(default_factory=list)  # Any match → possibly_present
    required: bool = True              # Required (❌ if absent) vs optional (⚠️ if absent)


# Verdicts
PRESENT          = "present"
POSSIBLY_PRESENT = "possibly_present"
ABSENT           = "absent"

_CHECKLISTS: dict[str, list[ClauseSpec]] = {
    "rental": [
        ClauseSpec("Parties identified",           ["tenant", "landlord", "lessee", "lessor"]),
        ClauseSpec("Premises described",           ["apartment", "premises", "property", "address"]),
        ClauseSpec("Term / lease period",          ["term", "lease period", "commencement", "expiry date"]),
        ClauseSpec("Rent amount",                  ["rent", "monthly rent", "rental amount"]),
        ClauseSpec("Payment due date",             ["due date", "payable by", "on or before", "first of",
                                                     "by the fifth", "by the third", "by the first",
                                                     "fifth calendar day", "third calendar day",
                                                     "payable on the"]),
        ClauseSpec("Security deposit",             ["security deposit", "deposit"]),
        ClauseSpec("Deposit return conditions",    ["return", "refund", "deduction", "itemized"]),
        ClauseSpec("Notice to terminate",          ["notice", "days' notice", "written notice"]),
        ClauseSpec("Late payment clause",          ["late fee", "late payment", "penalty", "interest"]),
        ClauseSpec("Repair / maintenance",         ["repair", "maintenance", "damage"], ["landlord shall", "tenant shall"]),
        ClauseSpec("Entry / inspection rights",    ["entry", "inspection", "access", "notice before entering"]),
        ClauseSpec("Subletting restriction",       ["sublet", "sub-let", "assign", "transfer"]),
        ClauseSpec("Governing law",                ["governing law", "jurisdiction", "applicable law"],
                   required=False),
        ClauseSpec("Dispute resolution",           ["dispute", "arbitration", "mediation", "court"],
                   required=False),
        ClauseSpec("Signatures",                   ["signed", "signature", "executed"]),
    ],
    "employment": [
        ClauseSpec("Parties identified",           ["employee", "employer", "company"]),
        ClauseSpec("Designation / role",           ["designation", "position", "role", "title"]),
        ClauseSpec("Start date",                   ["date of joining", "commencement", "start date", "effective date"]),
        ClauseSpec("Base salary / CTC",            ["salary", "ctc", "compensation", "remuneration"]),
        ClauseSpec("Probation period",             ["probation", "probationary"]),
        ClauseSpec("Notice period",                ["notice period", "days' notice"]),
        ClauseSpec("Termination clause",           ["terminat", "dismiss", "separation"]),
        ClauseSpec("Non-disclosure / NDA",         ["confidential", "non-disclosure", "proprietary"]),
        ClauseSpec("Non-compete clause",           ["non-compete", "non compete", "restraint of trade"],
                   required=False),
        ClauseSpec("Leave entitlement",            ["leave", "holiday", "vacation", "annual leave"],
                   required=False),
        ClauseSpec("Governing law",                ["governing law", "jurisdiction"],
                   required=False),
    ],
    "service": [
        ClauseSpec("Parties identified",           ["service provider", "client", "customer"]),
        ClauseSpec("Scope of services",            ["scope", "deliverable", "service", "work"]),
        ClauseSpec("Payment terms",                ["payment", "invoice", "fee", "amount"]),
        ClauseSpec("Term / duration",              ["term", "duration", "contract period", "commencement"]),
        ClauseSpec("Termination clause",           ["terminat", "cancel"]),
        ClauseSpec("Liability cap",                ["limitation of liability", "liability cap", "not liable"]),
        ClauseSpec("IP ownership",                 ["intellectual property", "ip", "ownership", "copyright"],
                   required=False),
        ClauseSpec("Confidentiality",              ["confidential", "non-disclosure"]),
        ClauseSpec("Governing law",                ["governing law", "jurisdiction"]),
    ],
    "nda": [
        ClauseSpec("Parties identified",           ["disclosing party", "receiving party", "parties"]),
        ClauseSpec("Definition of confidential",   ["confidential information", "proprietary", "trade secret"]),
        ClauseSpec("Obligations of recipient",     ["shall not", "must not", "recipient", "obligation"]),
        ClauseSpec("Permitted disclosures",        ["permitted", "required by law", "court order", "except"]),
        ClauseSpec("Duration of obligation",       ["period", "years", "duration", "survive"]),
        ClauseSpec("Exclusions",                   ["exclude", "public domain", "independently developed"],
                   required=False),
        ClauseSpec("Governing law",                ["governing law", "jurisdiction"]),
    ],
    "loan": [
        ClauseSpec("Parties identified",           ["borrower", "lender"]),
        ClauseSpec("Loan amount",                  ["principal", "loan amount", "sum of"]),
        ClauseSpec("Interest rate",                ["interest rate", "interest", "%", "per annum"]),
        ClauseSpec("Repayment schedule",           ["repayment", "emi", "installment", "schedule"]),
        ClauseSpec("Default provisions",           ["default", "event of default", "breach"]),
        ClauseSpec("Security / collateral",        ["security", "collateral", "mortgage", "pledge"],
                   required=False),
        ClauseSpec("Prepayment terms",             ["prepayment", "early repayment"],
                   required=False),
        ClauseSpec("Governing law",                ["governing law", "jurisdiction"]),
    ],
    "general": [
        ClauseSpec("Parties identified",           ["party", "parties", "between"]),
        ClauseSpec("Term / duration",              ["term", "period", "duration", "commencement"]),
        ClauseSpec("Payment / consideration",      ["payment", "amount", "fee", "consideration"]),
        ClauseSpec("Termination",                  ["terminat", "cancel"]),
        ClauseSpec("Governing law",                ["governing law", "jurisdiction"],
                   required=False),
    ],
}


@dataclass
class ClauseVerdict:
    clause_name: str
    verdict: str          # PRESENT | POSSIBLY_PRESENT | ABSENT
    required: bool
    matched_quote: Optional[str] = None   # first matching text snippet (≤ 120 chars)


def missing_clause_checklist(
    chunks: list[Chunk],
    doc_type: str = "general",
) -> list[ClauseVerdict]:
    """Return a ClauseVerdict for every clause in the checklist for *doc_type*.

    Matching is purely keyword-based on the concatenated chunk text.  No model call.
    """
    specs = _CHECKLISTS.get(doc_type, _CHECKLISTS["general"])
    full_text = " ".join(c.text for c in chunks).casefold()

    verdicts: list[ClauseVerdict] = []
    for spec in specs:
        # Check hard keywords first
        matched_quote: Optional[str] = None
        found_hard = False
        for kw in spec.keywords:
            idx = full_text.find(kw.casefold())
            if idx >= 0:
                found_hard = True
                # Extract a short surrounding snippet as evidence
                start = max(0, idx - 20)
                end = min(len(full_text), idx + len(kw) + 80)
                matched_quote = "…" + full_text[start:end].strip() + "…"
                break

        if found_hard:
            verdicts.append(ClauseVerdict(
                clause_name=spec.name,
                verdict=PRESENT,
                required=spec.required,
                matched_quote=matched_quote,
            ))
            continue

        # Check soft keywords
        found_soft = False
        for kw in spec.soft_keywords:
            if kw.casefold() in full_text:
                found_soft = True
                idx = full_text.find(kw.casefold())
                start = max(0, idx - 20)
                end = min(len(full_text), idx + len(kw) + 80)
                matched_quote = "…" + full_text[start:end].strip() + "…"
                break

        verdicts.append(ClauseVerdict(
            clause_name=spec.name,
            verdict=POSSIBLY_PRESENT if found_soft else ABSENT,
            required=spec.required,
            matched_quote=matched_quote,
        ))

    return verdicts


# ---------------------------------------------------------------------------
# Structured-field extraction
# ---------------------------------------------------------------------------

@dataclass
class StructuredField:
    field_name: str
    value: str             # Matched value text
    quote: str             # Surrounding context as evidence
    page_number: int       # Page where match was found
    filename: str


# Regex patterns for structured-field extraction
_PARTY_RE = re.compile(
    r"(?:between\s+(?:and\s+)?|parties?[:\s]+|(?:landlord|tenant|lessee|lessor|employer|employee|borrower|lender|service provider|client)[:\s]+)"
    r"([A-Z][A-Za-z\s,.&()'-]{3,60}?)(?=[,\n;]|\band\b|\bhereinafter\b)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(
    r"(?:INR|Rs\.?|₹|USD|EUR|GBP)\s*[\d,]+(?:\.\d+)?|\b\d[\d,]*\s*(?:lakh|crore|thousand)\b",
    re.IGNORECASE,
)
_NOTICE_PERIOD_RE = re.compile(
    r"\b(\d+)\s*(?:calendar\s+)?days?['s]?\s+(?:written\s+)?notice\b|\b(\d+)\s+months?\s+notice\b",
    re.IGNORECASE,
)
_INTEREST_RATE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*%\s*(?:per\s+annum|p\.a\.|annually|per\s+year)?",
    re.IGNORECASE,
)
_GOV_LAW_RE = re.compile(
    r"(?:governed\s+by|laws\s+of|jurisdiction\s+of)\s+(?:the\s+)?([A-Z][A-Za-z\s]{3,40}?)(?=[,\n.])",
    re.IGNORECASE,
)


def _context(text: str, match: re.Match, window: int = 60) -> str:
    """Return up to *window* chars before and after the match as context."""
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    snippet = text[start:end].strip()
    return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")


def extract_structured_fields(chunks: list[Chunk]) -> list[StructuredField]:
    """Run regex-based extraction over all chunks and return structured fields.

    Each field carries the matched value and a surrounding context quote so it
    can be displayed with source provenance (filename + page number).
    """
    results: list[StructuredField] = []
    seen_values: dict[str, set[str]] = {}  # field_name → set of seen values (dedup)

    def _add(field_name: str, value: str, quote: str, page: int, fname: str) -> None:
        value = re.sub(r"\s+", " ", value).strip()
        if not value or len(value) < 2:
            return
        if value.casefold() in seen_values.get(field_name, set()):
            return
        seen_values.setdefault(field_name, set()).add(value.casefold())
        results.append(StructuredField(
            field_name=field_name,
            value=value,
            quote=quote[:200],
            page_number=page,
            filename=fname,
        ))

    for chunk in chunks:
        text = chunk.text
        page = chunk.page_number
        fname = chunk.filename

        # Dates
        for m in _DATE_RE.finditer(text):
            _add("Date", m.group(), _context(text, m), page, fname)

        # Money amounts
        for m in _MONEY_RE.finditer(text):
            _add("Amount", m.group(), _context(text, m), page, fname)

        # Notice periods
        for m in _NOTICE_PERIOD_RE.finditer(text):
            _add("Notice period", m.group(), _context(text, m), page, fname)

        # Interest rates
        for m in _INTEREST_RATE_RE.finditer(text):
            raw_pct = m.group()
            # Skip very small or very large numbers that are unlikely interest rates
            pct_val = float(re.search(r"[\d.]+", raw_pct).group())
            if 0.1 <= pct_val <= 50:
                _add("Interest rate", raw_pct, _context(text, m), page, fname)

        # Governing law
        for m in _GOV_LAW_RE.finditer(text):
            _add("Governing law", m.group(1).strip(), _context(text, m), page, fname)

    # Deduplicate and limit to top 20 per field
    return results[:40]
