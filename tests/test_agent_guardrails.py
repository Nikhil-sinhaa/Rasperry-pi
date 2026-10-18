"""Existing guardrail tests updated for Member 1's verdict filter signature change.

is_definitive_legal_verdict_request() now accepts an optional `runtime` kwarg.
When runtime=None (unit test mode) the function is permissive on ambiguous words,
so all previous tests remain valid.
"""
from legal_triage.agent import is_definitive_legal_verdict_request


def test_enforceability_question_is_detected():
    assert is_definitive_legal_verdict_request("Is this clause enforceable?")
    assert is_definitive_legal_verdict_request("Can I sue?")


def test_plain_language_clause_question_is_not_blocked():
    assert not is_definitive_legal_verdict_request("What does the termination clause mean?")
    assert not is_definitive_legal_verdict_request("When is rent due?")


def test_jurisdiction_specific_legality_question_is_detected():
    assert is_definitive_legal_verdict_request("Is it legal to include this clause?")
    assert is_definitive_legal_verdict_request("Can my landlord legally evict me for this?")


def test_ordinary_question_about_a_legal_notice_is_not_blocked():
    assert not is_definitive_legal_verdict_request("What is a legal notice and when is it sent?")


def test_eviction_and_signing_advice_requests_are_detected():
    assert is_definitive_legal_verdict_request("Can I be evicted for this?")
    assert is_definitive_legal_verdict_request("Is this safe to sign?")
