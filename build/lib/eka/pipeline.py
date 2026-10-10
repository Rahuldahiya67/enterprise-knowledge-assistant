"""The RAG pipeline: ingest -> retrieve -> generate -> cite."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .ingest import chunk_document, load_directory, load_file
from .llm import NO_ANSWER, Generator, build_generator
from .models import Answer, Citation, ScoredChunk, SearchFilters
from .retrieval import HybridRetriever
from .stores import Store, build_store

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class IngestStats:
    documents: int
    chunks: int
    total_chunks: int


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        store: Store | None = None,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
    ):
        self.settings = settings or get_settings()
        self.store = store or build_store(self.settings)
        self.embedder = embedder or build_embedder(self.settings)
        self.generator = generator or build_generator(self.settings)
        self.retriever = HybridRetriever(
            self.store, self.embedder, candidate_k=self.settings.candidate_k, rrf_k=self.settings.rrf_k
        )

    # ------------------------------------------------------------------ ingest
    def ingest(self, path: str | Path, batch_size: int = 64) -> IngestStats:
        path = Path(path)
        docs = load_directory(path) if path.is_dir() else [load_file(path)]
        chunks = [
            c
            for d in docs
            for c in chunk_document(d, self.settings.chunk_size, self.settings.chunk_overlap)
        ]
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed([f"{c.title}\n{c.text}" for c in batch])
            self.store.add(batch, vectors)
        return IngestStats(documents=len(docs), chunks=len(chunks), total_chunks=self.store.count())

    # ------------------------------------------------------------------ query
    def retrieve(self, question: str, filters: SearchFilters | None = None, top_k: int | None = None) -> list[ScoredChunk]:
        return self.retriever.retrieve(question, top_k or self.settings.top_k, filters)

    def _is_answerable(self, retrieved: list[ScoredChunk]) -> bool:
        return bool(retrieved) and retrieved[0].score >= self.settings.min_relevance

    @staticmethod
    def build_citations(answer_text: str, retrieved: list[ScoredChunk]) -> list[Citation]:
        cited = sorted({int(n) for n in _CITE.findall(answer_text) if 0 < int(n) <= len(retrieved)})
        citations = []
        for n in cited:
            c = retrieved[n - 1].chunk
            snippet = c.text if len(c.text) <= 240 else c.text[:237].rsplit(" ", 1)[0] + "..."
            citations.append(Citation(index=n, title=c.title, source=c.source, department=c.department, snippet=snippet))
        return citations

    def ask(self, question: str, filters: SearchFilters | None = None) -> Answer:
        retrieved = self.retrieve(question, filters)
        if not self._is_answerable(retrieved):
            return Answer(question=question, answer=NO_ANSWER, citations=[], retrieved=retrieved, grounded=False)
        text = "".join(self.generator.stream(question, retrieved)).strip()
        grounded = NO_ANSWER not in text
        return Answer(
            question=question,
            answer=text,
            citations=self.build_citations(text, retrieved) if grounded else [],
            retrieved=retrieved,
            grounded=grounded,
        )

    def stream(self, question: str, filters: SearchFilters | None = None) -> Iterator[dict]:
        """Yields events: {"type": "token", "text": ...} then a final {"type": "done", ...}."""
        retrieved = self.retrieve(question, filters)
        if not self._is_answerable(retrieved):
            yield {"type": "token", "text": NO_ANSWER}
            yield {"type": "done", "citations": [], "grounded": False}
            return
        parts: list[str] = []
        for piece in self.generator.stream(question, retrieved):
            parts.append(piece)
            yield {"type": "token", "text": piece}
        text = "".join(parts)
        grounded = NO_ANSWER not in text
        citations = self.build_citations(text, retrieved) if grounded else []
        yield {"type": "done", "citations": [c.model_dump() for c in citations], "grounded": grounded}
