"""Tests for Member 1 tasks.

Covers:
- Verdict filter: all 13 probe questions route correctly (no false positives, correct true positives)
- Grounding: number / date / duration mismatch causes finding to be dropped
- Grounding: topic and attention assignment
- Retrieval: retrieve_triage_per_topic with fake embeddings
- Agent: verifier pass with mocked runtime
- Agent: jurisdiction hints
- Agent: full pipeline with a fake runtime returning canned JSON with invented quote,
         wrong source ID, legal-verdict claim (all must be dropped)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from legal_triage.agent import (
    AnalysisError,
    _full_doc_fits_in_ctx,
    _jurisdiction_hints_for,
    analyze,
    is_definitive_legal_verdict_request,
)
from legal_triage.grounding import (
    assign_attention,
    assign_topic,
    claim_vs_quote_check,
    validate_findings,
)
from legal_triage.retrieval import LocalIndex, retrieve_triage_per_topic
from legal_triage.types import Chunk


# ===========================================================================
# Helpers
# ===========================================================================


def _make_chunk(text: str, page: int = 1, filename: str = "doc.pdf") -> Chunk:
    return Chunk(
        chunk_id=f"{filename}::p{page}::c1",
        filename=filename,
        page_number=page,
        text=text,
    )


def _make_index(texts: list[str], dim: int = 4) -> LocalIndex:
    """Build a LocalIndex with random unit vectors (no Ollama needed)."""
    chunks = [_make_chunk(t, page=i + 1) for i, t in enumerate(texts)]
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((len(chunks), dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / norms
    return LocalIndex(chunks=chunks, vectors=vectors, metadata={"pages": len(chunks), "text_pages": len(chunks)})


def _fake_runtime(embed_dim: int = 4) -> MagicMock:
    """Return a MagicMock LocalOllama that returns unit vectors of *embed_dim*."""
    runtime = MagicMock()
    rng = np.random.default_rng(0)

    def _embed(texts, **kwargs):
        raw = rng.standard_normal((len(texts), embed_dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / norms

    runtime.embed.side_effect = _embed
    return runtime


# ===========================================================================
# 1. Verdict filter – 13 probe questions
# ===========================================================================
#
# Questions that MUST NOT be blocked (false positives in the old filter):
NOT_BLOCKED = [
    "How long is the agreement valid?",
    "Until when is this offer valid?",
    "Is the notice period binding on both parties?",
    "Is the lock-in period valid for 11 months?",
    "Is there a clause that is void if the tenant leaves early?",
    "What does the termination clause say?",
    "When is rent due?",
    "What is a legal notice and when is it sent?",
    "Explain the indemnity clause.",
]

# Questions that MUST be blocked (clear legal verdict / advice requests):
MUST_BLOCK = [
    "Is this clause enforceable?",
    "Can I sue?",
    "Is this safe to sign?",
    "Can I be evicted for this?",
    "Is this late fee legal under Indian law?",
    "Should I sign this contract?",
    "Does this violate the law?",
]


@pytest.mark.parametrize("question", NOT_BLOCKED)
def test_verdict_filter_does_not_block_ordinary_questions(question):
    """Ordinary questions must not trigger the verdict filter (no runtime → permissive)."""
    assert not is_definitive_legal_verdict_request(question, runtime=None), (
        f"False positive: '{question}' was incorrectly blocked"
    )


@pytest.mark.parametrize("question", MUST_BLOCK)
def test_verdict_filter_blocks_clear_verdict_requests(question):
    """Clear verdict/advice requests must always be blocked."""
    assert is_definitive_legal_verdict_request(question, runtime=None), (
        f"False negative: '{question}' was not blocked"
    )


def test_verdict_filter_disambiguates_with_gemma_legal_verdict():
    """When Gemma says 'legal_verdict', an ambiguous question is blocked."""
    runtime = MagicMock()
    runtime.chat_json.return_value = '{"intent": "legal_verdict"}'
    assert is_definitive_legal_verdict_request("Is this clause valid?", runtime=runtime)


def test_verdict_filter_disambiguates_with_gemma_explain_text():
    """When Gemma says 'explain_text', an ambiguous question is allowed through."""
    runtime = MagicMock()
    runtime.chat_json.return_value = '{"intent": "explain_text"}'
    assert not is_definitive_legal_verdict_request("Is the notice period binding on both parties?", runtime=runtime)


def test_verdict_filter_fallback_on_gemma_error():
    """If Gemma errors during disambiguation, the filter falls back to permissive (no block)."""
    runtime = MagicMock()
    runtime.chat_json.side_effect = RuntimeError("connection refused")
    # Ambiguous word 'valid' but Gemma errors → should not block
    assert not is_definitive_legal_verdict_request("How long is the agreement valid?", runtime=runtime)


# ===========================================================================
# 2. Claim-vs-quote number / date / duration check
# ===========================================================================


def test_claim_vs_quote_passes_when_number_present_in_quote():
    claim = "The monthly rent is INR 20,000."
    quotes = ["The tenant must pay rent of INR 20,000 per month on or before the fifth day."]
    assert claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_fails_when_number_not_in_any_quote():
    claim = "The monthly rent is INR 30,000."
    quotes = ["The tenant must pay rent of INR 20,000 per month on or before the fifth day."]
    assert not claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_passes_for_claim_with_no_numbers():
    claim = "The tenant must give notice before vacating."
    quotes = ["The tenant shall inform the landlord in writing before leaving."]
    assert claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_normalises_comma_formatting():
    """1,00,000 and 100000 should be treated as the same number."""
    claim = "The deposit is INR 1,00,000."
    quotes = ["The tenant shall pay a security deposit of INR 100000 at commencement."]
    assert claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_checks_dates():
    claim = "The agreement starts on 01/01/2025."
    quotes = ["This agreement is effective from 01/01/2025 for a period of 12 months."]
    assert claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_fails_wrong_date():
    claim = "The agreement starts on 01/01/2026."
    quotes = ["This agreement is effective from 01/01/2025 for a period of 12 months."]
    assert not claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_checks_duration():
    claim = "The notice period is 30 days."
    quotes = ["Either party may terminate this agreement by giving 30 days written notice."]
    assert claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_fails_wrong_duration():
    claim = "The notice period is 60 days."
    quotes = ["Either party may terminate this agreement by giving 30 days written notice."]
    assert not claim_vs_quote_check(claim, quotes)


# ===========================================================================
# 3. validate_findings drops findings with number mismatches
# ===========================================================================

SOURCE_TEXT = (
    "The tenant must pay a monthly rent of INR 15,000 on or before the fifth day of each month. "
    "A security deposit of INR 30,000 is payable at commencement. "
    "Either party may terminate this agreement by giving 30 days written notice."
)


def _base_finding(claim: str, quote: str) -> dict:
    return {
        "category": "explicit_statement",
        "claim": claim,
        "evidence": [{"source_id": "E1", "quote": quote}],
        "lawyer_question": "",
    }


def test_validate_findings_drops_claim_with_wrong_number():
    sources = {"E1": {"text": SOURCE_TEXT}}
    payload = {
        "findings": [
            _base_finding(
                claim="The monthly rent is INR 25,000.",  # wrong – document says 15,000
                quote="The tenant must pay a monthly rent of INR 15,000 on or before the fifth day of each month.",
            )
        ]
    }
    assert validate_findings(payload, sources) == []


def test_validate_findings_keeps_claim_with_correct_number():
    sources = {"E1": {"text": SOURCE_TEXT}}
    payload = {
        "findings": [
            _base_finding(
                claim="The monthly rent is INR 15,000.",
                quote="The tenant must pay a monthly rent of INR 15,000 on or before the fifth day of each month.",
            )
        ]
    }
    result = validate_findings(payload, sources)
    assert len(result) == 1


def test_validate_findings_drops_claim_with_wrong_duration():
    sources = {"E1": {"text": SOURCE_TEXT}}
    payload = {
        "findings": [
            _base_finding(
                claim="The notice period is 60 days.",  # wrong – document says 30 days
                quote="Either party may terminate this agreement by giving 30 days written notice.",
            )
        ]
    }
    assert validate_findings(payload, sources) == []


# ===========================================================================
# 4. Topic and attention assignment
# ===========================================================================


def test_assign_topic_money():
    assert assign_topic("The rent is INR 20,000 per month.") == "money"


def test_assign_topic_deadlines():
    assert assign_topic("The notice period is 30 days.") == "deadlines"


def test_assign_topic_termination():
    assert assign_topic("Either party may terminate the agreement on breach.") == "termination"


def test_assign_topic_obligations():
    assert assign_topic("The tenant shall maintain the premises in good condition.") == "obligations"


def test_assign_topic_general_fallback():
    assert assign_topic("The parties are ACME Corp and John Doe.") == "general"


def test_assign_attention_high_for_penalty():
    assert assign_attention("A penalty of INR 5,000 is levied for late payment.", []) == "high"


def test_assign_attention_high_for_auto_renewal():
    assert assign_attention("The agreement will auto-renew unless cancelled.", []) == "high"


def test_assign_attention_medium_for_termination():
    assert assign_attention("Either party may terminate on 30 days notice.", []) == "medium"


def test_assign_attention_low_for_plain_obligation():
    assert assign_attention("The tenant shall keep the premises clean.", []) == "low"


def test_validate_findings_includes_topic_and_attention():
    sources = {"E1": {"text": SOURCE_TEXT}}
    payload = {
        "findings": [
            _base_finding(
                claim="A security deposit of INR 30,000 is payable at commencement.",
                quote="A security deposit of INR 30,000 is payable at commencement.",
            )
        ]
    }
    result = validate_findings(payload, sources)
    assert len(result) == 1
    assert result[0]["topic"] == "money"
    assert result[0]["attention"] == "high"  # has a currency amount
    assert "quote_valid" in result[0]["support"]
    assert "numbers_verified" in result[0]["support"]


# ===========================================================================
# 5. retrieve_triage_per_topic with fake embeddings
# ===========================================================================


def test_retrieve_triage_per_topic_returns_results():
    texts = [f"Clause {i}: some obligation to pay INR {i * 1000} monthly." for i in range(1, 20)]
    index = _make_index(texts)
    runtime = _fake_runtime()
    results, top_score = retrieve_triage_per_topic(index, runtime, top_k_per_topic=3)
    assert len(results) > 0
    # All returned items should be (Chunk, float) pairs
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)


def test_retrieve_triage_per_topic_deduplicates_chunks():
    """Same chunk should not appear twice even if it scores well for multiple topics."""
    texts = ["The tenant must pay INR 20,000 rent and terminate with 30 days notice."]
    index = _make_index(texts)
    runtime = _fake_runtime()
    results, _ = retrieve_triage_per_topic(index, runtime, top_k_per_topic=5)
    chunk_ids = [c.chunk_id for c, _ in results]
    assert len(chunk_ids) == len(set(chunk_ids))


# ===========================================================================
# 6. Verifier pass
# ===========================================================================


def _make_runtime_with_verifier(verdict: str = "yes") -> MagicMock:
    runtime = MagicMock()
    runtime.chat_json.return_value = f'{{"verdict": "{verdict}", "reason": "test"}}'
    return runtime


def test_verifier_pass_keeps_finding_on_yes(monkeypatch):
    """With verifier returning 'yes', findings should be kept and 'verifier_agrees' added."""
    from legal_triage.agent import _run_verifier_pass

    finding = {
        "category": "explicit_statement",
        "claim": "The rent is INR 15,000.",
        "evidence": [{"source_id": "E1", "quote": "The tenant must pay a monthly rent of INR 15,000."}],
        "support": ["quote_valid"],
        "topic": "money",
        "attention": "high",
        "lawyer_question": "",
    }
    runtime = _make_runtime_with_verifier("yes")
    result = _run_verifier_pass([finding], runtime)
    assert len(result) == 1
    assert "verifier_agrees" in result[0]["support"]


def test_verifier_pass_drops_finding_on_no():
    """With verifier returning 'no', the finding should be dropped."""
    from legal_triage.agent import _run_verifier_pass

    finding = {
        "category": "explicit_statement",
        "claim": "The rent is INR 15,000.",
        "evidence": [{"source_id": "E1", "quote": "The tenant must pay a monthly rent of INR 15,000."}],
        "support": ["quote_valid"],
        "topic": "money",
        "attention": "high",
        "lawyer_question": "",
    }
    runtime = _make_runtime_with_verifier("no")
    result = _run_verifier_pass([finding], runtime)
    assert result == []


def test_verifier_pass_retains_on_partly():
    """With verifier returning 'partly', the finding should be kept."""
    from legal_triage.agent import _run_verifier_pass

    finding = {
        "category": "potential_concern",
        "claim": "The notice period may be shorter than legally required.",
        "evidence": [{"source_id": "E1", "quote": "Either party may terminate with 7 days notice."}],
        "support": ["quote_valid"],
        "topic": "deadlines",
        "attention": "medium",
        "lawyer_question": "",
    }
    runtime = _make_runtime_with_verifier("partly")
    result = _run_verifier_pass([finding], runtime)
    assert len(result) == 1


def test_verifier_pass_retains_on_runtime_error():
    """If verifier itself errors, the finding is retained (fail-open for verifier only)."""
    from legal_triage.agent import _run_verifier_pass

    finding = {
        "category": "explicit_statement",
        "claim": "The rent is INR 15,000.",
        "evidence": [{"source_id": "E1", "quote": "The tenant must pay a monthly rent of INR 15,000."}],
        "support": ["quote_valid"],
        "topic": "money",
        "attention": "high",
        "lawyer_question": "",
    }
    runtime = MagicMock()
    runtime.chat_json.side_effect = RuntimeError("model unavailable")
    result = _run_verifier_pass([finding], runtime)
    assert len(result) == 1


# ===========================================================================
# 7. Jurisdiction hints
# ===========================================================================


def test_jurisdiction_hints_triggered_by_deposit_keyword():
    notes = _jurisdiction_hints_for("What is the security deposit?", [])
    assert any("deposit" in n.lower() or "Deposit" in n for n in notes)


def test_jurisdiction_hints_triggered_by_finding_claim():
    findings = [{"claim": "The notice period is 7 days.", "evidence": []}]
    notes = _jurisdiction_hints_for("", findings)
    assert any("notice" in n.lower() for n in notes)


def test_jurisdiction_hints_empty_for_irrelevant_question():
    notes = _jurisdiction_hints_for("Who are the parties?", [])
    assert notes == []


# ===========================================================================
# 8. Full pipeline with fake runtime – invented quote, wrong source ID,
#    legal-verdict claim must all be dropped
# ===========================================================================

GOOD_SOURCE_TEXT = (
    "The tenant shall pay a monthly rent of INR 20,000 on or before the fifth day of each month. "
    "A security deposit of three months' rent is payable on commencement of the tenancy."
)

CANNED_JSON_WITH_BAD_FINDINGS = json_str = """{
  "insufficient_information": false,
  "findings": [
    {
      "category": "explicit_statement",
      "claim": "The monthly rent is INR 99,999.",
      "evidence": [{"source_id": "E1", "quote": "INVENTED QUOTE THAT DOES NOT EXIST IN SOURCE TEXT abcdefg"}],
      "lawyer_question": ""
    },
    {
      "category": "explicit_statement",
      "claim": "The tenant must pay INR 20,000.",
      "evidence": [{"source_id": "E999", "quote": "shall pay a monthly rent of INR 20,000 on or before the fifth day"}],
      "lawyer_question": ""
    },
    {
      "category": "explicit_statement",
      "claim": "This clause is unenforceable.",
      "evidence": [{"source_id": "E1", "quote": "shall pay a monthly rent of INR 20,000 on or before the fifth day"}],
      "lawyer_question": ""
    }
  ]
}"""


def test_pipeline_drops_invented_quote_wrong_source_id_and_legal_verdict_claim(monkeypatch):
    """The whole pipeline must drop:
    - findings with quotes not in the source text
    - findings with fabricated source IDs
    - findings that assert legal enforceability
    """
    import json

    from legal_triage import agent as agent_module

    # Patch retrieve_question to return a controlled result
    chunk = _make_chunk(GOOD_SOURCE_TEXT)
    index = _make_index([GOOD_SOURCE_TEXT])
    runtime = MagicMock()

    # embed returns 1-vector matching our index dimension
    dim = index.vectors.shape[1]
    rng = np.random.default_rng(7)

    def _embed(texts, **kwargs):
        raw = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / norms

    runtime.embed.side_effect = _embed
    runtime.chat_json.return_value = CANNED_JSON_WITH_BAD_FINDINGS

    # Ensure we go past the retrieval threshold
    monkeypatch.setattr(agent_module.SETTINGS, "min_retrieval_similarity", -1.0)

    result = analyze(
        "What are the payment terms?",
        index,
        runtime,
        mode="question",
        enable_verifier=False,  # isolate the grounding check
    )
    # All three bad findings must be dropped – abstention or empty findings
    findings = result.get("findings", [])
    assert findings == [], f"Expected no validated findings, got: {findings}"
