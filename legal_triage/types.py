"""Typed data structures shared by ingestion, retrieval, grounding and the UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    filename: str
    page_number: int
    text: str
    # When a chunk spans a page break it may cite two adjacent pages.
    extra_pages: tuple[int, ...] = field(default_factory=tuple)

    def to_prompt_dict(self, source_id: str | None = None) -> dict[str, object]:
        return {
            "source_id": source_id or self.chunk_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "text": self.text,
        }

    @property
    def all_pages(self) -> tuple[int, ...]:
        """Return this chunk's page number plus any extra pages it spans."""
        return (self.page_number,) + self.extra_pages


@dataclass(frozen=True)
class IndexedDocuments:
    chunks: list[Chunk]
    vectors: object  # numpy.ndarray; kept opaque here to keep pure types lightweight
    filenames: list[str]
    page_count: int
    warnings: list[str]


# ---------------------------------------------------------------------------
# Finding schema (published by Member 1; consumed by Member 4's UI)
# ---------------------------------------------------------------------------

ALLOWED_TOPICS = frozenset(
    {
        "obligations",
        "money",
        "deadlines",
        "termination",
        "liability",
        "ambiguity",
        "jurisdiction",
        "general",
    }
)

ALLOWED_ATTENTION = frozenset({"high", "medium", "low"})

ALLOWED_SUPPORT = frozenset({"quote_valid", "numbers_verified", "verifier_agrees", "unverified"})


@dataclass
class Finding:
    """A single validated finding returned to the UI.

    Fields
    ------
    category:   One of the four grounding categories (explicit_statement, etc.)
    claim:      Plain-language description from the model.
    evidence:   List of dicts with source_id and verbatim quote.
    lawyer_question: Optional escalation question.
    topic:      Semantic grouping (obligations / money / deadlines / …).
    attention:  Rule-based importance signal – high / medium / low.
                Never derived from the model's own confidence score.
    support:    Frozenset of checks that passed for this finding.
    """

    category: str
    claim: str
    evidence: list[dict[str, Any]]
    lawyer_question: str = ""
    topic: str = "general"
    attention: str = "low"
    support: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "claim": self.claim,
            "evidence": self.evidence,
            "lawyer_question": self.lawyer_question,
            "topic": self.topic,
            "attention": self.attention,
            "support": self.support,
        }
