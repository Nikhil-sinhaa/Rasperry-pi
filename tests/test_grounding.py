from legal_triage.grounding import normalize_quote, quote_is_verbatim, validate_findings, contains_unqualified_legal_judgment
from legal_triage.grounding import _normalize_number, _extract_facts_from, claim_vs_quote_check


def test_whitespace_normalization_is_stable():
    assert normalize_quote("  The tenant\n must pay  rent. ") == "the tenant must pay rent."


def test_verbatim_quote_accepts_whitespace_variation():
    source = "The tenant must pay rent of INR 20,000 per month on or before the fifth calendar day."
    quote = "The tenant must pay rent of INR 20,000\nper month on or before the fifth calendar day."
    assert quote_is_verbatim(quote, source)


def test_short_or_invented_quotes_are_rejected():
    source = "The tenant must pay rent of INR 20,000 per month on or before the fifth calendar day."
    assert not quote_is_verbatim("The tenant must pay", source)
    assert not quote_is_verbatim("The landlord must pay INR 2,00,000", source)


def test_finding_with_invalid_source_id_is_rejected():
    sources = {"E1": {"text": "The tenant must pay rent of INR 20,000 per month on or before the fifth calendar day."}}
    payload = {"findings": [{
        "category": "explicit_statement",
        "claim": "The rent is due on the fifth day.",
        "evidence": [{"source_id": "E99", "quote": "The tenant must pay rent of INR 20,000 per month."}],
    }]}
    assert validate_findings(payload, sources) == []


def test_finding_with_valid_verbatim_evidence_survives():
    source_text = "The tenant must pay rent of INR 20,000 per month on or before the fifth calendar day."
    sources = {"E1": {"text": source_text}}
    payload = {"findings": [{
        "category": "explicit_statement",
        "claim": "The monthly rent is INR 20,000 and is due by the fifth day.",
        "evidence": [{"source_id": "E1", "quote": "The tenant must pay rent of INR 20,000 per month on or before the fifth calendar day."}],
        "lawyer_question": "How should the payment date be documented?",
    }]}
    clean = validate_findings(payload, sources)
    assert len(clean) == 1
    assert clean[0]["evidence"][0]["source_id"] == "E1"


def test_unqualified_enforceability_claim_is_rejected():
    assert contains_unqualified_legal_judgment("This clause is unenforceable.", "potential_concern")
    assert not contains_unqualified_legal_judgment("The agreement states that this clause is enforceable.", "explicit_statement")
    assert not contains_unqualified_legal_judgment("The tenant must give 30 days' notice.", "explicit_statement")


# ---------------------------------------------------------------------------
# _normalize_number: bracketed duplicate stripping
# ---------------------------------------------------------------------------

def test_normalize_number_strips_bracketed_duplicate():
    assert _normalize_number("thirty (30)") == "thirty30"


def test_normalize_number_strips_only_parens_digit():
    # The spelled form is still present; caller should have run _normalize_spelled_numbers first.
    # After that, thirty → 30, so "30 (30)" → "3030" after casefold and strip.
    assert _normalize_number("30 (30)") == "3030"


def test_normalize_number_removes_commas_and_spaces():
    assert _normalize_number("1,00,000") == "100000"
    assert _normalize_number("20 000") == "20000"


def test_normalize_number_handles_currency_prefix():
    assert _normalize_number("Rs. 1,00,000") == "rs.100000"


# ---------------------------------------------------------------------------
# _extract_facts_from: spelled numbers → digit normalisation
# ---------------------------------------------------------------------------

def test_extract_facts_from_spelled_duration():
    facts = _extract_facts_from("The notice period is thirty days.")
    assert any("30" in f for f in facts), f"Expected '30' in facts, got: {facts}"


def test_extract_facts_from_digit_duration():
    facts = _extract_facts_from("The notice period is 30 days.")
    assert any("30" in f for f in facts), f"Expected '30' in facts, got: {facts}"


def test_extract_facts_from_spelled_and_digit_produce_same_normalized_form():
    spelled_facts = _extract_facts_from("thirty days")
    digit_facts = _extract_facts_from("30 days")
    from legal_triage.grounding import _normalize_number
    norm_spelled = {_normalize_number(f) for f in spelled_facts}
    norm_digit = {_normalize_number(f) for f in digit_facts}
    assert norm_spelled == norm_digit, (
        f"Spelled ({norm_spelled}) and digit ({norm_digit}) forms did not normalize to the same value"
    )


def test_extract_facts_from_empty_text_returns_empty():
    assert _extract_facts_from("") == []


# ---------------------------------------------------------------------------
# claim_vs_quote_check: B2 regression cases
# ---------------------------------------------------------------------------

def test_claim_vs_quote_spelled_number_matches_digit():
    """'30 days' in claim should match 'thirty (30) days' in quote (B2 fix)."""
    claim = "The notice period is 30 days."
    quotes = ["Either party may terminate by giving thirty (30) days written notice."]
    assert claim_vs_quote_check(claim, quotes), (
        "claim_vs_quote_check falsely dropped '30 days' vs 'thirty (30) days'"
    )


def test_claim_vs_quote_digit_matches_spelled():
    """'thirty days' in claim should match '30 days' in quote."""
    claim = "The notice period is thirty days."
    quotes = ["The contract requires 30 days notice."]
    assert claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_wrong_number_fails():
    """A claim with 45 days should NOT pass if only 30 days appears in quotes."""
    claim = "The notice period is 45 days."
    quotes = ["Either party may terminate by giving 30 days written notice."]
    assert not claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_no_numbers_in_claim_passes():
    """A claim with no numbers should always pass (nothing to verify)."""
    claim = "The agreement can be terminated."
    quotes = ["Either party may terminate the agreement."]
    assert claim_vs_quote_check(claim, quotes)


def test_claim_vs_quote_currency_match():
    claim = "The monthly rent is INR 20,000."
    quotes = ["Tenant shall pay INR 20,000 per month as rent."]
    assert claim_vs_quote_check(claim, quotes)

