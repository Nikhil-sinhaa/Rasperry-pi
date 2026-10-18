"""Typed data structures shared by ingestion and retrieval."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    filename: str
    page_number: int
    text: str

    def to_prompt_dict(self, source_id: str | None = None) -> dict[str, object]:
        return {
            "source_id": source_id or self.chunk_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "text": self.text,
        }


@dataclass(frozen=True)
class IndexedDocuments:
    chunks: list[Chunk]
    vectors: object  # numpy.ndarray; kept opaque here to keep pure types lightweight
    filenames: list[str]
    page_count: int
    warnings: list[str]
