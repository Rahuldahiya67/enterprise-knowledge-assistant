"""Core data types shared across ingestion, retrieval and generation."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source file after loading, before chunking."""

    doc_id: str
    title: str
    text: str
    source: str                       # file path or URL
    department: str | None = None
    doc_type: str | None = None
    effective_date: date | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable slice of a document."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    position: int                     # order of the chunk within its document
    department: str | None = None
    doc_type: str | None = None
    effective_date: date | None = None
    embedding: list[float] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "department": self.department,
            "doc_type": self.doc_type,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
        }


class SearchFilters(BaseModel):
    """Optional metadata filters applied before ranking."""

    department: str | None = None
    doc_type: str | None = None
    effective_after: date | None = None

    def matches(self, chunk: Chunk) -> bool:
        if self.department and (chunk.department or "").lower() != self.department.lower():
            return False
        if self.doc_type and (chunk.doc_type or "").lower() != self.doc_type.lower():
            return False
        if self.effective_after and (chunk.effective_date is None or chunk.effective_date < self.effective_after):
            return False
        return True

    def is_empty(self) -> bool:
        return not (self.department or self.doc_type or self.effective_after)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None


class Citation(BaseModel):
    index: int
    title: str
    source: str
    department: str | None = None
    snippet: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[ScoredChunk] = Field(default_factory=list, exclude=True)
    grounded: bool = True             # False when the assistant declined for lack of evidence
