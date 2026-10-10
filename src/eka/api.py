"""FastAPI service.

Endpoints
---------
GET  /health          liveness + index size + active providers
POST /ingest          upload one or more .md / .txt / .pdf files
POST /ask             JSON answer with citations
POST /ask/stream      Server-Sent Events: token events, then a final "done" event with citations
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .ingest import SUPPORTED
from .models import Answer, SearchFilters
from .pipeline import RAGPipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    department: str | None = None
    doc_type: str | None = None
    effective_after: date | None = None

    def filters(self) -> SearchFilters:
        return SearchFilters(department=self.department, doc_type=self.doc_type, effective_after=self.effective_after)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(title="Enterprise Knowledge Assistant", version=__version__)
    rag = pipeline or RAGPipeline()
    settings = rag.settings

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "chunks_indexed": rag.store.count(),
            "llm": rag.generator.name,
            "embeddings": type(rag.embedder).__name__,
            "vector_store": settings.vector_store,
        }

    @app.post("/ingest", dependencies=[Depends(check_key)])
    async def ingest(files: list[UploadFile] = File(...)) -> dict:
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for f in files:
            name = Path(f.filename or "").name
            if Path(name).suffix.lower() not in SUPPORTED:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {name}")
            dest = upload_dir / name
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            stats = rag.ingest(dest)
            results.append({"file": name, "chunks": stats.chunks})
        return {"ingested": results, "total_chunks": rag.store.count()}

    @app.post("/ask", response_model=Answer, dependencies=[Depends(check_key)])
    def ask(req: AskRequest) -> Answer:
        return rag.ask(req.question, req.filters())

    @app.post("/ask/stream", dependencies=[Depends(check_key)])
    def ask_stream(req: AskRequest) -> StreamingResponse:
        def events():
            for event in rag.stream(req.question, req.filters()):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def app_factory() -> FastAPI:  # used by `uvicorn eka.api:app_factory --factory`
    return create_app()
