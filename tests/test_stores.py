from datetime import date, datetime

import numpy as np

from eka.models import Chunk, SearchFilters
from eka.stores import MemoryStore, MongoAtlasStore, vector_index_definition


def _chunk(i: int, dept: str = "HR", text: str = "leave days") -> Chunk:
    return Chunk(
        chunk_id=f"c{i}", doc_id="d", title="Doc", text=text, source="doc.md", position=i,
        department=dept, effective_date=date(2025, 1, i + 1),
    )


def test_memory_store_persists_and_reloads(tmp_path):
    store = MemoryStore(tmp_path)
    store.add([_chunk(0), _chunk(1, "IT", "password policy")], np.eye(2, dtype=np.float32))
    reloaded = MemoryStore(tmp_path)
    assert reloaded.count() == 2
    hits = reloaded.vector_search(np.array([0, 1], dtype=np.float32), 1, SearchFilters())
    assert hits[0][0].chunk_id == "c1"


def test_memory_store_upserts_by_chunk_id(tmp_path):
    store = MemoryStore(tmp_path)
    store.add([_chunk(0)], np.array([[1, 0]], dtype=np.float32))
    store.add([_chunk(0, text="updated text")], np.array([[0, 1]], dtype=np.float32))
    assert store.count() == 1
    assert store.chunks[0].text == "updated text"


def test_memory_store_keyword_search_respects_filters(tmp_path):
    store = MemoryStore(tmp_path)
    store.add([_chunk(0, "HR", "annual leave"), _chunk(1, "IT", "annual password rotation")], np.eye(2, dtype=np.float32))
    hits = store.keyword_search("annual", 5, SearchFilters(department="IT"))
    assert [c.chunk_id for c, _ in hits] == ["c1"]


def test_memory_store_clear(tmp_path):
    store = MemoryStore(tmp_path)
    store.add([_chunk(0)], np.array([[1.0, 0.0]], dtype=np.float32))
    store.clear()
    assert store.count() == 0 and MemoryStore(tmp_path).count() == 0


class FakeCollection:
    """Just enough of pymongo's Collection to test writes without a database."""

    def __init__(self):
        self.docs: dict[str, dict] = {}

    def bulk_write(self, ops, ordered=True):
        for op in ops:                       # pymongo ReplaceOne(filter, replacement, upsert)
            self.docs[op._filter["_id"]] = op._doc

    def count_documents(self, flt):
        return len(self.docs)

    def delete_many(self, flt):
        self.docs.clear()


class FakeClient(dict):
    def __missing__(self, name):
        db = {}
        db_obj = type("DB", (), {"__getitem__": lambda _self, coll: db.setdefault(coll, FakeCollection())})()
        self[name] = db_obj
        return db_obj


def _mongo_store():
    return MongoAtlasStore("", "db", "chunks", "vec_ix", "text_ix", client=FakeClient())


def test_mongo_add_writes_documents_with_embeddings():
    store = _mongo_store()
    store.add([_chunk(0), _chunk(1)], np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    assert store.count() == 2
    doc = store.collection.docs["c0"]
    assert doc["_id"] == "c0" and len(doc["embedding"]) == 2
    assert isinstance(doc["embedding"][0], float)
    assert isinstance(doc["effective_date"], datetime)
    # upsert, not duplicate
    store.add([_chunk(0)], np.array([[0.5, 0.5]], dtype=np.float32))
    assert store.count() == 2
    store.clear()
    assert store.count() == 0


def test_mongo_vector_pipeline_pushes_filters_into_vectorsearch():
    store = _mongo_store()
    pipeline = store.vector_pipeline(
        np.array([0.1, 0.2]), 5, SearchFilters(department="HR", effective_after=date(2025, 1, 1))
    )
    stage = pipeline[0]["$vectorSearch"]
    assert stage["index"] == "vec_ix" and stage["limit"] == 5 and stage["numCandidates"] >= 50
    assert stage["filter"] == {
        "$and": [{"department": {"$eq": "HR"}}, {"effective_date": {"$gte": datetime(2025, 1, 1)}}]
    }
    assert pipeline[1]["$project"]["score"] == {"$meta": "vectorSearchScore"}


def test_mongo_keyword_pipeline_uses_atlas_search_compound_filter():
    store = _mongo_store()
    pipeline = store.keyword_pipeline("annual leave", 7, SearchFilters(doc_type="policy"))
    search = pipeline[0]["$search"]
    assert search["index"] == "text_ix"
    assert search["compound"]["filter"] == [{"equals": {"path": "doc_type", "value": "policy"}}]
    assert pipeline[1] == {"$limit": 7}


def test_mongo_search_maps_documents_back_to_chunks():
    store = _mongo_store()

    class FakeCollection:
        def aggregate(self, pipeline):
            return [{"_id": "c9", "chunk_id": "c9", "doc_id": "d", "title": "T", "text": "x", "source": "s.md",
                     "position": 0, "department": "HR", "effective_date": datetime(2025, 2, 1), "score": 0.87}]

    store.collection = FakeCollection()
    hits = store.vector_search(np.array([0.1]), 3, SearchFilters())
    assert hits[0][0].chunk_id == "c9" and hits[0][0].effective_date == date(2025, 2, 1)
    assert hits[0][1] == 0.87


def test_vector_index_definition_declares_filter_fields():
    fields = vector_index_definition(1536)["fields"]
    assert fields[0] == {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"}
    assert {f["path"] for f in fields if f["type"] == "filter"} == {"department", "doc_type", "effective_date"}
