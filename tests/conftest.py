from pathlib import Path

import pytest

from eka.config import Settings
from eka.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "data" / "sample_docs"
EVAL_CASES = ROOT / "eval" / "cases.jsonl"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(index_path=str(tmp_path / "index"), upload_dir=str(tmp_path / "uploads"), _env_file=None)


@pytest.fixture
def rag(settings) -> RAGPipeline:
    pipeline = RAGPipeline(settings)
    pipeline.ingest(SAMPLE_DOCS)
    return pipeline
