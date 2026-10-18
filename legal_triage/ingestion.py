"""In-memory PDF/plain-text ingestion. Uploaded content is never written to disk.

Member 3 additions
------------------
* DOCX support via python-docx (lazy import so missing package is a clear error).
* _normalize_ligatures()  – replaces common PDF ligature codepoints before chunking
  and before quote matching, so a quote extracted from a ligature-bearing PDF still
  matches the verbatim text stored in a chunk.
* _normalize_hyphenation() – strips soft-hyphen line-break artifacts (e.g. "agree-\nment"
  → "agreement") that some PDF extractors produce in multi-column layouts.
"""
from __future__ import annotations

import io
import re
from pathlib import PurePath

from pypdf import PdfReader

from .config import SETTINGS
from .types import Chunk

# ---------------------------------------------------------------------------
# Text normalisation helpers (Member 3)
# ---------------------------------------------------------------------------

# Common ligature substitution table (Unicode Private Use + standard ligature codepoints)
_LIGATURE_TABLE: dict[str, str] = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "st",   # ﬅ
    "\ufb06": "st",   # ﬆ
    "\u00e6": "ae",   # æ
    "\u0153": "oe",   # œ
}
_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_TABLE))


def _normalize_ligatures(text: str) -> str:
    """Replace common PDF ligature codepoints with their ASCII equivalents."""
    return _LIGATURE_RE.sub(lambda m: _LIGATURE_TABLE[m.group()], text)


# Hyphenated line-break pattern: word-\n or word-\r\nword (with optional spaces)
_HYPHEN_LINEBREAK_RE = re.compile(r"-(\r?\n)\s*", re.MULTILINE)


def _normalize_hyphenation(text: str) -> str:
    """Join hyphenated line-break artifacts produced by some PDF extractors.

    For example, "agree-\nment" → "agreement", "multi-\n  column" → "multicolumn".
    Only hyphens immediately before a newline are treated as line-break artifacts;
    standalone hyphens within a line (compound words, ranges) are left unchanged.
    """
    return _HYPHEN_LINEBREAK_RE.sub("", text)


def normalize_text_for_matching(text: str) -> str:
    """Apply all normalisation steps used before quote matching.

    Called by the grounding module (via import) to normalise both the chunk
    source text and a proposed quote before verbatim comparison.
    """
    text = _normalize_ligatures(text)
    text = _normalize_hyphenation(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class IngestionError(ValueError):
    """A file could not be safely parsed or exceeded documented limits."""


def _clean_filename(filename: str) -> str:
    # Do not let user-supplied names contain paths or markup in citations.
    name = PurePath(filename.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._() -]", "_", name).strip(" .")
    return name[:140] or "uploaded_document"


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise IngestionError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Prefer a paragraph/newline/word boundary, but don't make tiny chunks.
            floor = start + int(chunk_size * 0.65)
            candidates = [text.rfind("\n\n", floor, end), text.rfind("\n", floor, end), text.rfind(" ", floor, end)]
            boundary = max(candidates)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= length:
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return chunks


def _page_texts(filename: str, content: bytes) -> tuple[list[tuple[int, str]], list[str], int]:
    lower_name = filename.lower()
    warnings: list[str] = []
    if not content:
        raise IngestionError(f"{filename}: the uploaded file is empty.")

    if lower_name.endswith(".txt"):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError(f"{filename}: text files must be UTF-8 encoded.") from exc
        if not text.strip():
            raise IngestionError(f"{filename}: no readable text was found.")
        text = _normalize_ligatures(_normalize_hyphenation(text))
        return [(1, text)], warnings, 1

    # -----------------------------------------------------------------------
    # DOCX support (Member 3)
    # -----------------------------------------------------------------------
    if lower_name.endswith(".docx"):
        try:
            import docx as _docx  # python-docx; lazy so missing package gives a clear error
        except ImportError as exc:
            raise IngestionError(
                f"{filename}: python-docx is required to read .docx files. "
                "Install it with: pip install python-docx"
            ) from exc
        try:
            doc = _docx.Document(io.BytesIO(content))
        except Exception as exc:
            raise IngestionError(
                f"{filename}: the DOCX file could not be opened ({type(exc).__name__})."
            ) from exc
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        paragraphs.append(cell_text)
        if not paragraphs:
            raise IngestionError(
                f"{filename}: the DOCX file contains no readable text. "
                "Check that the document is not empty or password-protected."
            )
        full_text = "\n\n".join(paragraphs)
        full_text = _normalize_ligatures(_normalize_hyphenation(full_text))
        return [(1, full_text)], warnings, 1

    if not lower_name.endswith(".pdf"):
        raise IngestionError(
            f"{filename}: only PDF, UTF-8 .txt, and .docx files are supported."
        )
    if not content.startswith(b"%PDF"):
        raise IngestionError(f"{filename}: file extension says PDF, but the content is not a PDF.")

    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
    except Exception as exc:
        raise IngestionError(f"{filename}: the PDF could not be parsed ({type(exc).__name__}).") from exc

    if reader.is_encrypted:
        raise IngestionError(f"{filename}: encrypted/password-protected PDFs are not supported. Export an unlocked copy locally.")
    if not reader.pages:
        raise IngestionError(f"{filename}: the PDF contains no pages.")
    physical_page_count = len(reader.pages)
    if physical_page_count > SETTINGS.max_pages:
        raise IngestionError(
            f"{filename}: the PDF has {physical_page_count} pages, exceeding the configured per-upload page limit "
            f"of {SETTINGS.max_pages}. Split it locally and upload a smaller section."
        )

    pages: list[tuple[int, str]] = []
    empty_pages: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        # Apply ligature and hyphenation normalisation at extraction time
        text = _normalize_ligatures(_normalize_hyphenation(text))
        if text.strip():
            pages.append((page_number, text))
        else:
            empty_pages.append(page_number)
    if empty_pages:
        preview = ", ".join(str(x) for x in empty_pages[:12])
        suffix = "…" if len(empty_pages) > 12 else ""
        warnings.append(
            f"{filename}: no extractable text on {len(empty_pages)} page(s) ({preview}{suffix}). "
            "These may be scanned/image-only pages; OCR is not included, so clauses on them could be missed."
        )
    if not pages:
        raise IngestionError(
            f"{filename}: no selectable text was found. This looks scanned or image-only. "
            "This build does not include OCR; use a selectable-text PDF or a UTF-8 .txt file."
        )
    return pages, warnings, physical_page_count


def ingest_files(files: list[tuple[str, bytes]]) -> tuple[list[Chunk], dict[str, object]]:
    """Parse and chunk uploaded documents in memory, preserving page-level citations."""
    if not files:
        raise IngestionError("Upload at least one PDF or UTF-8 text file.")
    if len(files) > SETTINGS.max_files:
        raise IngestionError(f"A maximum of {SETTINGS.max_files} files can be indexed at once.")

    chunks: list[Chunk] = []
    filenames: list[str] = []
    total_pages = 0
    text_pages = 0
    warnings: list[str] = []
    seen_names: dict[str, int] = {}

    for input_name, content in files:
        filename = _clean_filename(input_name)
        seen_names[filename] = seen_names.get(filename, 0) + 1
        if seen_names[filename] > 1:
            stem, dot, suffix = filename.rpartition(".")
            filename = f"{stem or suffix} ({seen_names[filename]}).{suffix}" if dot else f"{filename} ({seen_names[filename]})"
        if len(content) > SETTINGS.max_file_mb * 1024 * 1024:
            raise IngestionError(f"{filename}: file is larger than the {SETTINGS.max_file_mb} MB per-file limit.")

        pages, file_warnings, physical_page_count = _page_texts(filename, content)
        warnings.extend(file_warnings)
        total_pages += physical_page_count
        text_pages += len(pages)
        if total_pages > SETTINGS.max_pages:
            raise IngestionError(f"The upload exceeds the configured limit of {SETTINGS.max_pages} total pages.")
        filenames.append(filename)
        for page_number, page_text in pages:
            for chunk_number, text in enumerate(
                _split_text(page_text, SETTINGS.chunk_size, SETTINGS.chunk_overlap), start=1
            ):
                chunk_id = f"{filename}::p{page_number}::c{chunk_number}"
                chunks.append(Chunk(chunk_id=chunk_id, filename=filename, page_number=page_number, text=text))
                if len(chunks) > SETTINGS.max_chunks:
                    raise IngestionError(f"The upload exceeds the configured limit of {SETTINGS.max_chunks} chunks.")

    if not chunks:
        raise IngestionError("No usable document text was extracted.")
    metadata: dict[str, object] = {
        "files": filenames,
        "pages": total_pages,
        "text_pages": text_pages,
        "chunks": len(chunks),
        "warnings": warnings,
    }
    return chunks, metadata
