from pathlib import Path

import pytest

from legal_triage.ingestion import IngestionError, ingest_files


def test_demo_text_ingests_with_page_metadata():
    path = Path(__file__).parents[1] / "samples" / "demo_rental_agreement.txt"
    chunks, metadata = ingest_files([(path.name, path.read_bytes())])
    assert chunks
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert any("INR 20,000" in chunk.text for chunk in chunks)
    assert metadata["pages"] == 1


def test_non_pdf_data_disguised_as_pdf_is_rejected():
    with pytest.raises(IngestionError):
        ingest_files([("bad.pdf", b"not actually a PDF")])


def test_empty_text_file_is_rejected():
    with pytest.raises(IngestionError):
        ingest_files([("empty.txt", b"  ")])


def test_selectable_text_pdf_keeps_page_number():
    from io import BytesIO
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "The tenant must pay INR 20,000 per month by the fifth calendar day.")
    pdf.showPage()
    pdf.drawString(72, 720, "The deposit is returned within 30 days.")
    pdf.save()

    chunks, metadata = ingest_files([("rental.pdf", buffer.getvalue())])
    assert metadata["pages"] == 2
    assert metadata["text_pages"] == 2
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert any("INR 20,000" in chunk.text for chunk in chunks)


def test_blank_pdf_explains_ocr_limitation():
    from io import BytesIO
    from pypdf import PdfWriter

    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buffer)
    with pytest.raises(IngestionError, match="OCR"):
        ingest_files([("scan.pdf", buffer.getvalue())])
