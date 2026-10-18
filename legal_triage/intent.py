"""Layered intent classifier for legal-verdict detection.

Three layers, evaluated in order:
1. Hard-block regex  – unambiguous verdict/advice requests → refuse immediately.
2. Hard-allow regex  – unambiguous "explain the text" requests → always answer.
3. Gemma 3-way JSON classifier (explain_text | legal_verdict | legal_advice) –
   for everything that falls through both hard lists.

When no runtime is available (unit-test mode) the function is permissive on
ambiguous questions so that ordinary "explain the text" questions are never
dropped.

Design notes
------------
* Patterns are anchored with \\b to prevent substring false-positives.
* Injection-style prefixes ("ignore previous rules") are stripped before
  classification and the question is still answered for its text part.
* Devanagari and common Hinglish legal phrases are covered by code-point
  range checks, not transliteration tables (more robust).
* The classifier prompt uses eight labelled few-shot examples (four per class)
  drawn from the B1 probe list to ground the model quickly.
"""
from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .local_runtime import LocalOllama

# ---------------------------------------------------------------------------
# Injection-prefix stripping
# ---------------------------------------------------------------------------

_INJECTION_PREFIXES = re.compile(
    r"^(?:"
    r"ignore (?:previous|all|prior|the above) (?:instructions?|rules?|guidelines?|prompts?)[.,;:!]?\s*"
    r"|you are now\b[^.]*?\.\s*"
    r"|disregard (?:previous|all|prior) (?:instructions?|rules?)[.,;:!]?\s*"
    r"|forget (?:everything|all) (?:you (?:know|learned|were told))[.,;:!]?\s*"
    r"|(?:new|system)\s+prompt[.:]\s*"
    r")+",
    flags=re.IGNORECASE,
)


def strip_injection_prefix(question: str) -> tuple[str, bool]:
    """Return (cleaned_question, had_injection_prefix)."""
    cleaned = _INJECTION_PREFIXES.sub("", question).strip()
    had_prefix = len(cleaned) < len(question.strip())
    return cleaned or question.strip(), had_prefix


# ---------------------------------------------------------------------------
# Unicode / script helpers
# ---------------------------------------------------------------------------

def _contains_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


# ---------------------------------------------------------------------------
# Hard-block patterns (unambiguous legal-verdict / advice requests)
# ---------------------------------------------------------------------------

_HARD_BLOCK: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in [
    # Enforceability
    r"\b(?:is|are|was|were)\s+(?:this|it|the\s+\w+(?:\s+\w+)?)\s+(?:clause|agreement|contract|provision|term)?\s*(?:un)?enforceable\b",
    r"\b(?:is|are|was|were)\s+(?:this|it|the\s+\w+(?:\s+\w+)?)\s+illegal\b",
    r"\b(?:enforceability|legal\s+validity)\b",
    r"\bdoes\s+this\s+(?:violate|break|contravene)\s+(?:the\s+)?law\b",
    # "Is this/it legal/illegal [under …]" — verdict pattern
    r"\bis\s+(?:this|it|the\s+[\w-]+(?:\s+[\w-]+)?)\s+(?:legal|illegal|lawful|unlawful)\b",
    r"\bshould\s+i\s+sign\b",
    r"\bwhether\b(?:\s+[\w-]+){0,4}?\s+(?:is\s+)?(?:enforceable|unenforceable|legal|illegal|lawful|unlawful|void)\b",
    r"\bi\s+should\s+(?:sign|accept|reject|refuse)\b",
    r"\bis\s+(?:this|it)\s+(?:allowed|permitted|prohibited)\s+under\b",
    r"\bwill\s+a\s+court\s+(?:uphold|enforce|reject|void|nullify|strike\s+down)\b",
    r"\b(?:can|could)\s+a\s+court\b",
    # Suing / winning
    r"\bcan\s+i\s+sue\b",
    r"\bwill\s+i\s+win\s+(?:in\s+court|the\s+case)\b",
    r"\b(?:does\s+it|will\s+this|would\s+this)\s+hold\s+up\s+in\s+court\b",
    r"\bstand\s+up\s+in\s+court\b",
    # Sign/accept advice
    r"\b(?:should\s+i|is\s+it\s+(?:safe|okay|ok)\s+to)\s+sign\b",
    r"\bshould\s+i\s+(?:accept|reject|terminate|refuse)\s+(?:this|the)\b",
    r"\bwould\s+you\s+(?:sign|accept)\s+(?:this|it)\b",
    # Eviction / firing by name
    r"\bcan\s+(?:my\s+)?(?:landlord|employer|company|they)\s+(?:legally\s+)?(?:evict|fire|terminate|sue)\s+me\b",
    r"\bcan\s+i\s+be\s+(?:evicted|fired|terminated|sued)\b",
    r"\bwill\s+i\s+be\s+evicted\b",
    # "Is it legal to …" (with infinitive)
    r"\bis\s+it\s+(?:legal|illegal|lawful|unlawful)\s+to\b",
    # "determine whether … legal/enforceable"
    r"\bdetermine\s+whether\b.{0,80}\b(?:legal|illegal|enforceable|void|valid)\b",
    # Scam / fair — verdict framing
    r"\bis\s+(?:this|it)\s+(?:a\s+)?scam\b",
    r"\bam\s+i\s+(?:legally\s+)?(?:required|obligated|liable)\s+to\s+pay\b",
    r"\bcan\s+i\s+walk\s+away\b",
    r"\b(?:legal\s+trouble|in\s+trouble\s+legally)\b",
    # Lawful/legit under … law phrasing
    r"\bunder\s+(?:indian|state|federal|local|\w+)\s+law\b.{0,60}\b(?:legal|illegal|allowed|enforceable|void|valid|lawful)\b",
    r"\b(?:legal|illegal|allowed|enforceable|void|valid|lawful)\b.{0,60}\bunder\s+(?:indian|state|federal|local|\w+)\s+law\b",
    # Non-compete with verdict word
    r"\b(?:non.?compete|restraint\s+of\s+trade)\b.{0,60}\b(?:legal|enforceable|valid|allowed|permissible|applicable)\b",
    r"\b(?:legal|enforceable|valid|allowed|permissible)\b.{0,60}\b(?:non.?compete|restraint\s+of\s+trade)\b",
    # Permissible / legit as verdict marker
    r"\b(?:is|are)\s+(?:this|the)\s+\w+(?:\s+\w+)?\s+(?:clause\s+)?(?:permissible|legit(?:imate)?)\b",
    # Hindi Devanagari legal-verdict phrases (क्या यह कानूनी है, etc.)
    r"क्या\s+यह\s+(?:कानूनी|वैध)(?:\s+है)?\b",
    # Hinglish transliterated
    r"\bkya\s+yeh\s+(?:kaanoon|kanoon|legal|valid|sahi)\s+(?:hai|he)\b",
    r"\b(?:nyayalay|adalat)\s+(?:mein|me)\b",
]]

# ---------------------------------------------------------------------------
# Hard-allow patterns (unambiguous explain-text requests, never block)
# ---------------------------------------------------------------------------

_HARD_ALLOW: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bwhat\s+does\s+(?:this|the)\b",
    r"\bwhat\s+is\s+(?:the\s+)?(?:notice\s+period|rent|deposit|duration|term|penalty)\b",
    r"\bhow\s+(?:long|much|many|often)\b",
    r"\bwhen\s+(?:is|does|can|must|should)\b",
    r"\bexplain\b",
    r"\bsummarise\b",
    r"\bsummarize\b",
    r"\buntil\s+when\b",
    r"\bwhat\s+(?:are|is)\s+(?:the\s+)?(?:terms?|conditions?|clauses?|obligations?|rights?|fees?|dates?|deadlines?)\b",
    r"\bwhat\s+(?:are|is)\s+(?:the\s+)?(?:termination|notice|payment|renewal|expiry)\s+(?:terms?|conditions?|clause|provisions?|period)\b",
    r"\bcan\s+(?:the\s+)?(?:landlord|tenant|employer|employee|either\s+party|both\s+parties)\s+(?:terminate|renew|extend|modify|assign)\b",
    r"\bis\s+(?:the|a)\s+(?:security\s+deposit|notice\s+period|non.?compete)\s+(?:refundable|binding\s+on\s+both|applicable)\b",
    r"\bis\s+(?:subletting|sublease|sub.?letting)\s+(?:allowed|permitted|mentioned|covered)\b",
    r"\bis\s+(?:the|a)\s+(?:security\s+deposit|advance)\s+refundable\b",
    r"\baccording\s+to\s+(?:this|the)\s+(?:document|agreement|contract|clause)\b",
    r"\bwhat\s+(?:notice|warning)\s+(?:is|must|should)\s+(?:be\s+)?given\b",
]]

# ---------------------------------------------------------------------------
# Gemma classifier prompt
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """\
You are a question classifier. Classify the user's question about a legal document into exactly one of:
- "explain_text"  : the user wants to know what the document says or means (facts, dates, amounts, obligations, party rights)
- "legal_verdict" : the user wants you to decide if something is legal, valid, enforceable, or predict a court outcome
- "legal_advice"  : the user wants you to advise them on what to do (sign, walk away, sue, etc.)

Few-shot examples:
Q: "How long is the agreement valid?" → {"intent": "explain_text"}
Q: "Can they terminate with 30 days notice?" → {"intent": "explain_text"}
Q: "How long does the non-compete apply?" → {"intent": "explain_text"}
Q: "Is the security deposit refundable?" → {"intent": "explain_text"}
Q: "Is this clause enforceable?" → {"intent": "legal_verdict"}
Q: "Is this penalty legit under Indian law?" → {"intent": "legal_verdict"}
Q: "Am I legally required to pay the penalty?" → {"intent": "legal_verdict"}
Q: "Is the non-compete clause permissible?" → {"intent": "legal_verdict"}
Q: "Is it okay to sign this?" → {"intent": "legal_advice"}
Q: "Can I walk away from this without legal trouble?" → {"intent": "legal_advice"}

Output ONLY valid JSON: {"intent": "explain_text"|"legal_verdict"|"legal_advice"}"""


def _classify_with_gemma(question: str, runtime: "LocalOllama") -> str:
    """Call Gemma to classify intent. Returns 'legal_verdict', 'legal_advice', or 'explain_text'.

    Defaults to 'explain_text' on any error so that model hiccups never cause
    spurious refusals.
    """
    import json
    try:
        raw = runtime.chat_json(_CLASSIFY_SYSTEM, f'Question: """{question}"""')
        # Strip markdown fences if present
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("```")[1].lstrip("json").strip()
        payload = json.loads(candidate)
        intent = payload.get("intent", "explain_text")
        return intent if intent in {"legal_verdict", "legal_advice", "explain_text"} else "explain_text"
    except Exception:
        return "explain_text"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_intent(
    question: str,
    runtime: "LocalOllama | None" = None,
) -> tuple[str, bool]:
    """Classify question intent with three-layer defence.

    Returns
    -------
    intent : "explain_text" | "legal_verdict" | "legal_advice"
    had_injection : bool — True if an injection prefix was stripped

    Layer 1: hard-allow → explain_text immediately (avoids false blocks).
    Layer 2: hard-block → legal_verdict immediately.
    Layer 3: Gemma classifier (only when runtime is available, else explain_text).
    """
    cleaned, had_injection = strip_injection_prefix(question)

    # Devanagari check — fast-path before regex loops
    if _contains_devanagari(cleaned):
        # Any Devanagari in a legal context is treated as potentially requesting a verdict;
        # send to Gemma for nuance if available, else hard-block.
        if runtime is not None:
            intent = _classify_with_gemma(cleaned, runtime)
            return intent, had_injection
        return "legal_verdict", had_injection

    lower = cleaned.casefold()

    # Layer 1: hard-block first, so "Explain whether this is enforceable" is still a verdict request
    if any(p.search(lower) for p in _HARD_BLOCK):
        return "legal_verdict", had_injection

    # Layer 2: hard-allow (unambiguous explain-the-text questions)
    if any(p.search(lower) for p in _HARD_ALLOW):
        return "explain_text", had_injection

    # Layer 3: Gemma
    if runtime is not None:
        intent = _classify_with_gemma(cleaned, runtime)
        return intent, had_injection

    # No runtime — permissive fallback (avoids false blocks in unit tests)
    return "explain_text", had_injection


def is_definitive_legal_verdict_request(
    question: str,
    runtime: "LocalOllama | None" = None,
) -> bool:
    """Backward-compatible thin wrapper over classify_intent.

    Returns True only for legal_verdict or legal_advice intents.
    """
    intent, _ = classify_intent(question, runtime)
    return intent in {"legal_verdict", "legal_advice"}
