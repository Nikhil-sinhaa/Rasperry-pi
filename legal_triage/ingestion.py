"""In-memory PDF/plain-text ingestion. Uploaded content is never written to disk."""
from __future__ import annotations

import io
import re
from pathlib import PurePath

from pypdf import PdfReader

from .config import SETTINGS
from .types import Chunk


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
        return [(1, text)], warnings, 1

    if not lower_name.endswith(".pdf"):
        raise IngestionError(f"{filename}: only PDF and UTF-8 .txt files are supported.")
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
