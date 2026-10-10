import json

from fastapi.testclient import TestClient

from eka.api import create_app
from eka.pipeline import RAGPipeline


def test_health_reports_index_and_providers(rag):
    client = TestClient(create_app(rag))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0
    assert body["llm"] == "offline-extractive"


def test_ask_returns_answer_with_citations(rag):
    client = TestClient(create_app(rag))
    resp = client.post("/ask", json={"question": "What is the hotel limit per night outside metro cities?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "Rs 4,000" in body["answer"]
    assert body["citations"][0]["source"] == "travel_expense_policy.md"
    assert "retrieved" not in body


def test_ask_validates_input(rag):
    client = TestClient(create_app(rag))
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_stream_endpoint_sends_sse_events(rag):
    client = TestClient(create_app(rag))
    with client.stream("POST", "/ask/stream", json={"question": "Within how many days must the proof of delivery be uploaded?"}) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = "".join(resp.iter_text())
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done" and events[-1]["grounded"]
    assert "3 days" in "".join(e["text"] for e in events if e["type"] == "token")


def test_ingest_upload_then_answer(settings):
    client = TestClient(create_app(RAGPipeline(settings)))
    content = b"---\ntitle: Canteen Guide\ndepartment: Admin\n---\nThe canteen serves lunch from 12:30 to 14:00 on working days."
    resp = client.post("/ingest", files=[("files", ("canteen.md", content, "text/markdown"))])
    assert resp.status_code == 200 and resp.json()["total_chunks"] == 1
    answer = client.post("/ask", json={"question": "When does the canteen serve lunch?"}).json()
    assert "12:30" in answer["answer"]


def test_ingest_rejects_unsupported_files(settings):
    client = TestClient(create_app(RAGPipeline(settings)))
    resp = client.post("/ingest", files=[("files", ("data.exe", b"MZ", "application/octet-stream"))])
    assert resp.status_code == 415


def test_api_key_is_enforced_when_configured(settings, rag):
    rag.settings = settings.model_copy(update={"api_key": "secret"})
    client = TestClient(create_app(rag))
    assert client.post("/ask", json={"question": "How much leave?"}).status_code == 401
    ok = client.post("/ask", json={"question": "How much annual leave?"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200
