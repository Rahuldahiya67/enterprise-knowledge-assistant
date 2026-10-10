"""Vector + keyword stores.

Two backends share one interface:

* ``MemoryStore``: numpy cosine search + BM25, persisted to disk. Zero setup.
* ``MongoAtlasStore``: MongoDB Atlas Vector Search (``$vectorSearch``) for dense
  retrieval and Atlas Search (``$search``) for keyword retrieval, with metadata
  pre-filters pushed down into the database.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Protocol

import math
from collections import Counter

import numpy as np

from .models import Chunk, SearchFilters
from .text import content_tokens


class BM25:
    """Okapi BM25 with the Lucene IDF, log(1 + (N - n + 0.5) / (n + 0.5)), which stays
    positive even when a term appears in most documents (common in small corpora)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tf = [Counter(doc) for doc in corpus]
        self.doc_len = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(corpus) else 0.0
        df = Counter(term for doc in corpus for term in set(doc))
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.doc_tf), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.doc_tf], dtype=np.float32)
            scores += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


class Store(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def vector_search(self, query_vec: np.ndarray, k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]: ...
    def keyword_search(self, query: str, k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


# --------------------------------------------------------------------------- memory
class MemoryStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self._bm25: BM25 | None = None
        if self.path and (self.path / "chunks.json").exists():
            self.load()

    # -- persistence
    def save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        payload = [c.model_dump(mode="json", exclude={"embedding"}) for c in self.chunks]
        (self.path / "chunks.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        if self.vectors is not None:
            np.save(self.path / "vectors.npy", self.vectors)

    def load(self) -> None:
        assert self.path
        data = json.loads((self.path / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk.model_validate(c) for c in data]
        vec_file = self.path / "vectors.npy"
        self.vectors = np.load(vec_file) if vec_file.exists() else None
        self._bm25 = None

    # -- writes
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        existing = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        new_chunks, new_vecs = [], []
        for chunk, vec in zip(chunks, vectors):
            if chunk.chunk_id in existing:           # upsert
                i = existing[chunk.chunk_id]
                self.chunks[i] = chunk
                self.vectors[i] = vec
            else:
                new_chunks.append(chunk)
                new_vecs.append(vec)
        if new_chunks:
            stacked = np.vstack(new_vecs).astype(np.float32)
            self.vectors = stacked if self.vectors is None else np.vstack([self.vectors, stacked])
            self.chunks.extend(new_chunks)
        self._bm25 = None
        self.save()

    def clear(self) -> None:
        self.chunks, self.vectors, self._bm25 = [], None, None
        if self.path:
            for name in ("chunks.json", "vectors.npy"):
                (self.path / name).unlink(missing_ok=True)

    def count(self) -> int:
        return len(self.chunks)

    # -- reads
    def _mask(self, filters: SearchFilters) -> np.ndarray:
        if filters.is_empty():
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([filters.matches(c) for c in self.chunks], dtype=bool)

    def vector_search(self, query_vec: np.ndarray, k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        if not self.chunks or self.vectors is None:
            return []
        scores = self.vectors @ query_vec.astype(np.float32)
        scores = np.where(self._mask(filters), scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def keyword_search(self, query: str, k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        if self._bm25 is None:
            corpus = [content_tokens(f"{c.title} {c.text}") or ["_"] for c in self.chunks]
            self._bm25 = BM25(corpus)
        q = content_tokens(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        scores = np.where(self._mask(filters), scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if np.isfinite(scores[i]) and scores[i] > 0]


# --------------------------------------------------------------------------- mongo
def _to_datetime(d: date | None) -> datetime | None:
    return datetime.combine(d, time.min) if d else None


def vector_index_definition(dims: int) -> dict:
    """Atlas Vector Search index. Filter fields must be declared to be used in $vectorSearch."""
    return {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": dims, "similarity": "cosine"},
            {"type": "filter", "path": "department"},
            {"type": "filter", "path": "doc_type"},
            {"type": "filter", "path": "effective_date"},
        ]
    }


TEXT_INDEX_DEFINITION = {
    "mappings": {
        "dynamic": False,
        "fields": {
            "text": {"type": "string", "analyzer": "lucene.english"},
            "title": {"type": "string", "analyzer": "lucene.english"},
            "department": {"type": "token"},
            "doc_type": {"type": "token"},
            "effective_date": {"type": "date"},
        },
    }
}


class MongoAtlasStore:
    def __init__(self, uri: str, db: str, collection: str, vector_index: str, text_index: str, client=None):
        if client is None:
            from pymongo import MongoClient

            if not uri:
                raise ValueError("EKA_MONGO_URI is required for the mongo vector store")
            client = MongoClient(uri)
        self.collection = client[db][collection]
        self.vector_index = vector_index
        self.text_index = text_index

    # -- setup
    def ensure_indexes(self, dims: int) -> None:
        """Create the Atlas Vector Search and Atlas Search indexes if they are missing."""
        from pymongo.operations import SearchIndexModel

        existing = {ix["name"] for ix in self.collection.list_search_indexes()}
        if self.vector_index not in existing:
            self.collection.create_search_index(
                SearchIndexModel(definition=vector_index_definition(dims), name=self.vector_index, type="vectorSearch")
            )
        if self.text_index not in existing:
            self.collection.create_search_index(SearchIndexModel(definition=TEXT_INDEX_DEFINITION, name=self.text_index))

    # -- writes
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from pymongo import ReplaceOne

        ops = []
        for chunk, vec in zip(chunks, vectors):
            doc = chunk.model_dump(exclude={"embedding", "effective_date"})
            doc["_id"] = chunk.chunk_id
            doc["effective_date"] = _to_datetime(chunk.effective_date)
            doc["embedding"] = [float(x) for x in vec]
            ops.append(ReplaceOne({"_id": chunk.chunk_id}, doc, upsert=True))
        if ops:
            self.collection.bulk_write(ops, ordered=False)

    def clear(self) -> None:
        self.collection.delete_many({})

    def count(self) -> int:
        return self.collection.count_documents({})

    # -- reads
    @staticmethod
    def _doc_to_chunk(doc: dict) -> Chunk:
        doc = dict(doc)
        doc.pop("_id", None)
        doc.pop("embedding", None)
        doc.pop("score", None)
        eff = doc.get("effective_date")
        if isinstance(eff, datetime):
            doc["effective_date"] = eff.date()
        return Chunk.model_validate(doc)

    @staticmethod
    def vector_filter(filters: SearchFilters) -> dict:
        clauses = []
        if filters.department:
            clauses.append({"department": {"$eq": filters.department}})
        if filters.doc_type:
            clauses.append({"doc_type": {"$eq": filters.doc_type}})
        if filters.effective_after:
            clauses.append({"effective_date": {"$gte": _to_datetime(filters.effective_after)}})
        if not clauses:
            return {}
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def vector_pipeline(self, query_vec: np.ndarray, k: int, filters: SearchFilters) -> list[dict]:
        stage = {
            "index": self.vector_index,
            "path": "embedding",
            "queryVector": [float(x) for x in query_vec],
            "numCandidates": max(k * 10, 100),
            "limit": k,
        }
        flt = self.vector_filter(filters)
        if flt:
            stage["filter"] = flt
        return [
            {"$vectorSearch": stage},
            {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
        ]

    def keyword_pipeline(self, query: str, k: int, filters: SearchFilters) -> list[dict]:
        compound: dict = {"must": [{"text": {"query": query, "path": ["text", "title"]}}]}
        flt = []
        if filters.department:
            flt.append({"equals": {"path": "department", "value": filters.department}})
        if filters.doc_type:
            flt.append({"equals": {"path": "doc_type", "value": filters.doc_type}})
        if filters.effective_after:
            flt.append({"range": {"path": "effective_date", "gte": _to_datetime(filters.effective_after)}})
        if flt:
            compound["filter"] = flt
        return [
            {"$search": {"index": self.text_index, "compound": compound}},
            {"$limit": k},
            {"$project": {"embedding": 0, "score": {"$meta": "searchScore"}}},
        ]

    def vector_search(self, query_vec: np.ndarray, k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        docs = self.collection.aggregate(self.vector_pipeline(query_vec, k, filters))
        return [(self._doc_to_chunk(d), float(d.get("score", 0.0))) for d in docs]

    def keyword_search(self, query: str, k: int, filters: SearchFilters) -> list[tuple[Chunk, float]]:
        docs = self.collection.aggregate(self.keyword_pipeline(query, k, filters))
        return [(self._doc_to_chunk(d), float(d.get("score", 0.0))) for d in docs]


def build_store(settings) -> Store:
    if settings.vector_store == "mongo":
        return MongoAtlasStore(
            settings.mongo_uri,
            settings.mongo_db,
            settings.mongo_collection,
            settings.mongo_vector_index,
            settings.mongo_text_index,
        )
    return MemoryStore(settings.index_path)
