from legal_triage.grounding import normalize_quote, quote_is_verbatim, validate_findings, contains_unqualified_legal_judgment


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
