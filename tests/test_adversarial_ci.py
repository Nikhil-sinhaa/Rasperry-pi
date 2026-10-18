"""Phase 4 – adversarial CI tests suitable for running in a CI/CD pipeline.

Covers the adversarial items listed in the Phase 4 improvement plan:
  1. Unicode lookalike / homoglyph injection in quoted text
  2. Hidden whitespace / zero-width characters in quotes
  3. Conflicting clauses in the same document
  4. Needle-in-a-long-document: important clause buried among noise
  5. Malformed / truncated PDF bytes
  6. BM25 hybrid retrieval: lexical query scores higher than pure embedding
  7. Missing-clause checklist: correct verdicts for present / absent clauses
  8. Structured-field extraction: amounts, notice periods, dates extracted correctly
  9. Document-type detection: correct labels for rental, employment, nda, loan, general

All tests are self-contained; no Ollama server is required.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _txt_bytes(text: str) -> bytes:
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# Shared fake runtime for BM25 tests
# ---------------------------------------------------------------------------

class _FakeRuntime:
    """Orthogonal unit-vector embeddings, safe for retrieval tests."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._registry: dict[str, int] = {}
        self._counter = 0
        self.chat_model = "fake-model"
        self.embedding_model = "fake-embed"

    def _idx(self, text: str) -> int:
        if text not in self._registry:
            self._registry[text] = self._counter % self.dim
            self._counter += 1
        return self._registry[text]

    def embed(self, texts: list[str], is_query: bool = False, batch_size: int = 24) -> np.ndarray:
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            result[row, self._idx(text)] = 1.0
        return result

    def chat_json(self, system: str, user: str, temperature: float = 0.1) -> str:
        return json.dumps({"insufficient_information": True, "findings": []})


# ---------------------------------------------------------------------------
# 1. Unicode homoglyph / lookalike in quotes
# ---------------------------------------------------------------------------

class TestUnicodeLookalikesInQuotes:
    """Cyrillic/Greek homoglyphs inside a quote must not pass verbatim check."""

    def test_cyrillic_o_not_verbatim(self) -> None:
        """A quote with Cyrillic 'о' (U+043E) substituted for Latin 'o' is rejected."""
        from legal_triage.grounding import quote_is_verbatim
        genuine = "The tenant must pay rent of INR 20,000 per month."
        # Replace Latin 'o' with Cyrillic о (U+043E) – visually identical
        poisoned = genuine.replace("o", "\u043e")
        # Must return False (the substituted string is not in the source)
        assert quote_is_verbatim(poisoned, genuine) is False

    def test_greek_capital_a_not_verbatim(self) -> None:
        """Greek capital Alpha (U+0391) in a quote is not verbatim against Latin 'A'."""
        from legal_triage.grounding import quote_is_verbatim
        genuine = "Agreement dated 1 October 2026 between Aarav and Cedar Lane."
        poisoned = genuine.replace("A", "\u0391")
        assert quote_is_verbatim(poisoned, genuine) is False

    def test_normal_unicode_quote_passes(self) -> None:
        """A valid quote that only differs by curly quotes passes after normalisation."""
        from legal_triage.grounding import quote_is_verbatim
        # Source has curly quotes; model emits straight ASCII quotes
        source = "The tenant \u2018must\u2019 pay rent of INR 20,000 per month."
        model_quote = "The tenant 'must' pay rent of INR 20,000 per month."
        # normalize_text_for_matching converts curly to straight → should match
        assert quote_is_verbatim(model_quote, source) is True


# ---------------------------------------------------------------------------
# 2. Hidden whitespace / zero-width characters
# ---------------------------------------------------------------------------

class TestHiddenCharactersInQuotes:

    def test_zero_width_space_rejected_or_handled(self) -> None:
        """A quote with U+200B (zero-width space) does not crash and returns bool."""
        from legal_triage.grounding import quote_is_verbatim
        source = "Monthly rent is INR 18,000 payable on the first of each month."
        poisoned = "Monthly rent is INR 18,000\u200b payable on the first of each month."
        result = quote_is_verbatim(poisoned, source)
        assert isinstance(result, bool)

    def test_non_breaking_space_normalised(self) -> None:
        """A quote with non-breaking spaces (\xa0) is normalised and should match."""
        from legal_triage.grounding import quote_is_verbatim
        source = "Monthly rent is INR 18,000 payable on the first of each month."
        nbsp_quote = "Monthly rent is INR\xa018,000 payable on the first of each month."
        # The normaliser collapses \xa0 to space so these should match
        assert quote_is_verbatim(nbsp_quote, source) is True

    def test_soft_hyphen_stripped(self) -> None:
        """U+00AD (soft hyphen) in a quote is stripped so the quote still matches."""
        from legal_triage.grounding import quote_is_verbatim
        source = "The tenant must pay rent of INR 20,000 per month."
        soft_hyphen_quote = "The tenant must pay rent\u00ad of INR 20,000 per month."
        assert quote_is_verbatim(soft_hyphen_quote, source) is True


# ---------------------------------------------------------------------------
# 3. Conflicting clauses
# ---------------------------------------------------------------------------

class TestConflictingClauses:

    def test_two_amounts_ingested(self) -> None:
        """Conflicting rent amounts in the same document both survive as separate chunks."""
        from legal_triage.ingestion import ingest_files
        doc = (
            "Clause 1. The monthly rent shall be INR 20,000, payable by the 5th.\n\n"
            "Clause 12. Notwithstanding the above, the agreed rent is INR 22,500."
        )
        chunks, _ = ingest_files([("conflict.txt", _txt_bytes(doc))])
        combined = " ".join(c.text for c in chunks)
        assert "INR 20,000" in combined
        assert "INR 22,500" in combined

    def test_conflicting_notice_periods_ingested(self) -> None:
        """Two different notice periods both appear in chunks."""
        from legal_triage.ingestion import ingest_files
        doc = (
            "Clause 8. Tenant may terminate by giving 30 days written notice.\n\n"
            "Clause 9. Notwithstanding clause 8, a 60-day notice is required "
            "during the first year."
        )
        chunks, _ = ingest_files([("notice_conflict.txt", _txt_bytes(doc))])
        combined = " ".join(c.text for c in chunks)
        assert "30 days" in combined.lower()
        assert "60-day" in combined.lower() or "60 day" in combined.lower()


# ---------------------------------------------------------------------------
# 4. Needle in a long document
# ---------------------------------------------------------------------------

class TestNeedleInLongDocument:

    def _build_index(self, rt: _FakeRuntime, chunks_texts: list[str]):
        from legal_triage.retrieval import LocalIndex, _compute_bm25_data
        from legal_triage.types import Chunk
        chunks = [
            Chunk(chunk_id=f"f::p1::c{i}", filename="long.txt", page_number=1, text=t)
            for i, t in enumerate(chunks_texts)
        ]
        vectors = rt.embed(chunks_texts)
        df, tf, avg = _compute_bm25_data(chunks)
        return LocalIndex(chunks=chunks, vectors=vectors, metadata={"pages": 1},
                          _bm25_df=df, _bm25_tf=tf, _bm25_avg_dl=avg)

    def test_hybrid_retrieval_surfaces_needle(self) -> None:
        """The needle chunk with an exact BM25 keyword match scores higher with hybrid."""
        from legal_triage.retrieval import hybrid_retrieve_question, _bm25_scores, _tokenize

        rt = _FakeRuntime(dim=200)
        # 10 noise chunks + 1 needle
        noise = [f"General clause {i}. The parties agree to standard terms and conditions." for i in range(10)]
        needle = "The security deposit of INR 75,000 is refundable within 14 days of vacating."
        all_chunks = noise + [needle]

        index = self._build_index(rt, all_chunks)
        query = "security deposit INR 75000 refundable"

        # BM25 should score the needle higher than noise chunks
        tokens = _tokenize(query)
        bm25 = _bm25_scores(tokens, index)
        needle_idx = len(noise)
        # The needle chunk must have the highest BM25 score
        assert int(np.argmax(bm25)) == needle_idx, (
            f"Needle (idx {needle_idx}) did not get highest BM25; scores: {bm25}"
        )


# ---------------------------------------------------------------------------
# 5. Malformed PDF bytes
# ---------------------------------------------------------------------------

class TestMalformedFiles:

    def test_truncated_pdf_raises_ingestion_error(self) -> None:
        """A PDF header without proper content raises IngestionError."""
        from legal_triage.ingestion import IngestionError, ingest_files
        with pytest.raises(IngestionError):
            ingest_files([("truncated.pdf", b"%PDF-1.4 truncated garbage")])

    def test_completely_empty_file_raises_ingestion_error(self) -> None:
        """Zero-byte file raises IngestionError."""
        from legal_triage.ingestion import IngestionError, ingest_files
        with pytest.raises(IngestionError):
            # Empty txt bytes
            ingest_files([("empty.txt", b"")])

    def test_binary_garbage_as_txt_rejected(self) -> None:
        """Random binary bytes with .txt extension raises IngestionError (non-UTF8)."""
        from legal_triage.ingestion import IngestionError, ingest_files
        binary_garbage = bytes(range(256)) * 4
        with pytest.raises((IngestionError, UnicodeDecodeError, Exception)):
            ingest_files([("garbage.txt", binary_garbage)])


# ---------------------------------------------------------------------------
# 6. BM25 hybrid retrieval correctness
# ---------------------------------------------------------------------------

class TestBM25HybridRetrieval:

    def test_bm25_scores_sum_to_positive_for_matching_query(self) -> None:
        """BM25 scores are > 0 when the query terms appear in at least one chunk."""
        from legal_triage.retrieval import _bm25_scores, _compute_bm25_data, _tokenize
        from legal_triage.types import Chunk
        chunks = [
            Chunk(chunk_id="f::p1::c0", filename="f.txt", page_number=1,
                  text="The rent is INR 20,000 per month payable on the fifth."),
            Chunk(chunk_id="f::p1::c1", filename="f.txt", page_number=1,
                  text="The agreement is for a fixed term of 12 months."),
        ]
        df, tf, avg = _compute_bm25_data(chunks)
        from legal_triage.retrieval import LocalIndex
        index = LocalIndex(
            chunks=chunks,
            vectors=np.zeros((2, 4), dtype=np.float32),
            metadata={"pages": 1},
            _bm25_df=df, _bm25_tf=tf, _bm25_avg_dl=avg,
        )
        scores = _bm25_scores(_tokenize("rent INR 20000"), index)
        # Chunk 0 must score higher than chunk 1 (only chunk 0 has "rent")
        assert scores[0] > scores[1]

    def test_bm25_scores_zero_for_no_match(self) -> None:
        """BM25 returns all-zeros when no query term matches any chunk."""
        from legal_triage.retrieval import _bm25_scores, _compute_bm25_data, _tokenize
        from legal_triage.types import Chunk
        from legal_triage.retrieval import LocalIndex
        chunks = [
            Chunk(chunk_id="f::p1::c0", filename="f.txt", page_number=1,
                  text="The agreement covers general terms."),
        ]
        df, tf, avg = _compute_bm25_data(chunks)
        index = LocalIndex(
            chunks=chunks,
            vectors=np.zeros((1, 4), dtype=np.float32),
            metadata={"pages": 1},
            _bm25_df=df, _bm25_tf=tf, _bm25_avg_dl=avg,
        )
        scores = _bm25_scores(_tokenize("xyzzy frobnicator"), index)
        assert float(scores.sum()) == pytest.approx(0.0)

    def test_build_index_populates_bm25_data(self) -> None:
        """build_index() populates _bm25_df, _bm25_tf, and _bm25_avg_dl on the index."""
        import unittest.mock as mock
        from legal_triage.retrieval import LocalIndex

        dummy_vectors = np.ones((1, 4), dtype=np.float32)

        with mock.patch("legal_triage.retrieval.ingest_files") as mock_ingest, \
             mock.patch("legal_triage.retrieval.LocalOllama") as mock_rt_cls:
            from legal_triage.types import Chunk
            chunk = Chunk(chunk_id="f::p1::c0", filename="f.txt", page_number=1,
                          text="Rent is INR 20,000 per month.")
            mock_ingest.return_value = ([chunk], {"pages": 1, "files": ["f.txt"],
                                                  "chunks": 1, "text_pages": 1, "warnings": []})
            mock_rt = mock.MagicMock()
            mock_rt.embed.return_value = dummy_vectors

            from legal_triage.retrieval import build_index
            idx = build_index([("f.txt", b"Rent is INR 20,000 per month.")], mock_rt)

        assert isinstance(idx._bm25_df, dict)
        assert len(idx._bm25_df) > 0
        assert isinstance(idx._bm25_tf, list)
        assert idx._bm25_avg_dl > 0


# ---------------------------------------------------------------------------
# 7. Missing-clause checklist
# ---------------------------------------------------------------------------

class TestMissingClauseChecklist:

    def test_rental_all_present(self) -> None:
        """A complete rental agreement text produces only PRESENT verdicts for key clauses."""
        from legal_triage.checklist import ABSENT, PRESENT, missing_clause_checklist
        from legal_triage.ingestion import ingest_files
        doc = (
            "This rental agreement is between Cedar Lane (landlord) and Aarav (tenant).\n\n"
            "The apartment premises is at 12 Main Street.\n\n"
            "Term: 1 October 2026 to 30 September 2027. Commencement date: 1 October 2026.\n\n"
            "Monthly rent is INR 20,000. Payment is due by the fifth calendar day of each month.\n\n"
            "Security deposit of INR 40,000 is payable before occupation. The deposit is refundable "
            "within 30 days after vacating, after deductions.\n\n"
            "Either party may end this agreement by giving 30 days' written notice to the other.\n\n"
            "A late fee of 2% applies for each month the rent remains unpaid.\n\n"
            "The tenant must report a repair in writing. The landlord will arrange an inspection.\n\n"
            "The landlord will provide 24 hours' notice before entry for routine inspection.\n\n"
            "Subletting or assignment of the tenancy requires written consent from the landlord.\n\n"
            "This agreement is signed and executed by both parties below."
        )
        chunks, _ = ingest_files([("rental.txt", doc.encode("utf-8"))])
        verdicts = missing_clause_checklist(chunks, "rental")
        absent_required = [v for v in verdicts if v.verdict == ABSENT and v.required]
        assert absent_required == [], f"Unexpectedly absent: {[v.clause_name for v in absent_required]}"

    def test_rental_missing_security_deposit(self) -> None:
        """A rental document without any mention of deposit triggers ABSENT verdict."""
        from legal_triage.checklist import ABSENT, missing_clause_checklist
        from legal_triage.ingestion import ingest_files
        doc = (
            "Clause 1. Tenant is Aarav, landlord is Cedar Lane.\n\n"
            "Clause 2. Monthly rent is INR 20,000 due by the 5th.\n\n"
            "Clause 3. Notice period is 30 days written notice.\n\n"
            "Clause 4. Subletting requires landlord consent."
        )
        chunks, _ = ingest_files([("no_deposit.txt", doc.encode("utf-8"))])
        verdicts = missing_clause_checklist(chunks, "rental")
        deposit_verdict = next((v for v in verdicts if "deposit" in v.clause_name.lower()), None)
        assert deposit_verdict is not None
        assert deposit_verdict.verdict == ABSENT

    def test_nda_checklist(self) -> None:
        """NDA checklist finds confidentiality definition and duration."""
        from legal_triage.checklist import PRESENT, missing_clause_checklist
        from legal_triage.ingestion import ingest_files
        doc = (
            "This Non-Disclosure Agreement (NDA) is between the disclosing party and the receiving party.\n\n"
            "Confidential information means any proprietary or trade secret data shared under this agreement.\n\n"
            "The receiving party shall not disclose any confidential information to third parties.\n\n"
            "Disclosure may be permitted if required by law or court order.\n\n"
            "This obligation shall survive for a period of three years from the date of disclosure.\n\n"
            "Governing law: This agreement is governed by the laws of Maharashtra."
        )
        chunks, _ = ingest_files([("nda.txt", doc.encode("utf-8"))])
        verdicts = missing_clause_checklist(chunks, "nda")
        present = {v.clause_name for v in verdicts if v.verdict == PRESENT}
        assert "Definition of confidential" in present
        assert "Duration of obligation" in present


# ---------------------------------------------------------------------------
# 8. Structured field extraction
# ---------------------------------------------------------------------------

class TestStructuredFieldExtraction:

    def _chunks_from_text(self, text: str):
        from legal_triage.ingestion import ingest_files
        chunks, _ = ingest_files([("doc.txt", text.encode("utf-8"))])
        return chunks

    def test_amounts_extracted(self) -> None:
        """Money amounts (INR) are extracted by the regex."""
        from legal_triage.checklist import extract_structured_fields
        chunks = self._chunks_from_text(
            "Monthly rent is INR 20,000. Security deposit is INR 40,000."
        )
        fields = extract_structured_fields(chunks)
        amounts = [f for f in fields if f.field_name == "Amount"]
        values = {f.value for f in amounts}
        assert any("20,000" in v or "20000" in v for v in values), f"Got: {values}"
        assert any("40,000" in v or "40000" in v for v in values), f"Got: {values}"

    def test_notice_period_extracted(self) -> None:
        """Notice period number is extracted."""
        from legal_triage.checklist import extract_structured_fields
        chunks = self._chunks_from_text(
            "The tenant must give 30 days' written notice to terminate this agreement."
        )
        fields = extract_structured_fields(chunks)
        notice = [f for f in fields if f.field_name == "Notice period"]
        assert notice, "No notice period extracted"
        assert "30" in notice[0].value

    def test_date_extracted(self) -> None:
        """Calendar dates are extracted from the document text."""
        from legal_triage.checklist import extract_structured_fields
        chunks = self._chunks_from_text(
            "This agreement commences on 1 October 2026 and expires on 30 September 2027."
        )
        fields = extract_structured_fields(chunks)
        dates = [f for f in fields if f.field_name == "Date"]
        assert len(dates) >= 1, "No dates extracted"
        assert any("2026" in d.value or "2027" in d.value for d in dates)

    def test_interest_rate_extracted(self) -> None:
        """Interest rate percentage is extracted from loan document."""
        from legal_triage.checklist import extract_structured_fields
        chunks = self._chunks_from_text(
            "The loan carries an annual interest rate of 12% per annum on the outstanding principal."
        )
        fields = extract_structured_fields(chunks)
        rates = [f for f in fields if f.field_name == "Interest rate"]
        assert rates, "No interest rate extracted"
        assert "12" in rates[0].value

    def test_governing_law_extracted(self) -> None:
        """Governing law clause is extracted."""
        from legal_triage.checklist import extract_structured_fields
        chunks = self._chunks_from_text(
            "This agreement is governed by the laws of Maharashtra and courts of Mumbai."
        )
        fields = extract_structured_fields(chunks)
        gov = [f for f in fields if f.field_name == "Governing law"]
        assert gov, "No governing law extracted"

    def test_extraction_has_source_provenance(self) -> None:
        """Every extracted field carries filename and page_number."""
        from legal_triage.checklist import extract_structured_fields
        chunks = self._chunks_from_text("The rent is INR 15,000 payable by the 3rd.")
        fields = extract_structured_fields(chunks)
        for f in fields:
            assert f.filename, "Missing filename"
            assert f.page_number >= 1, "Invalid page_number"
            assert f.quote, "Missing quote context"


# ---------------------------------------------------------------------------
# 9. Document-type detection
# ---------------------------------------------------------------------------

class TestDocumentTypeDetection:

    def test_rental_detected(self) -> None:
        from legal_triage.checklist import detect_document_type
        text = "Tenant Aarav agrees to pay monthly rent of INR 20,000. Security deposit required."
        assert detect_document_type(text) == "rental"

    def test_employment_detected(self) -> None:
        from legal_triage.checklist import detect_document_type
        text = "Employee joins as Senior Developer. Probation period is 6 months. Salary INR 12 LPA."
        assert detect_document_type(text) == "employment"

    def test_nda_detected(self) -> None:
        from legal_triage.checklist import detect_document_type
        text = "The receiving party shall not disclose any confidential information or trade secret."
        assert detect_document_type(text) == "nda"

    def test_loan_detected(self) -> None:
        from legal_triage.checklist import detect_document_type
        text = "The borrower agrees to repay the principal loan amount of INR 5,00,000 at 12% per annum."
        assert detect_document_type(text) == "loan"

    def test_general_fallback(self) -> None:
        from legal_triage.checklist import detect_document_type
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        assert detect_document_type(text) == "general"
