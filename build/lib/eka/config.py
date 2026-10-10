"""Runtime configuration, read from environment variables or a .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="EKA_", extra="ignore")

    # --- providers ---------------------------------------------------------
    # "offline" needs no API key: a hashing embedder and an extractive answerer.
    # It keeps tests and demos runnable anywhere; switch to openai / anthropic for real use.
    llm_provider: Literal["offline", "openai", "anthropic"] = "offline"
    embedding_provider: Literal["offline", "openai"] = "offline"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-5"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- storage -----------------------------------------------------------
    vector_store: Literal["memory", "mongo"] = "memory"
    index_path: str = "data/index"
    mongo_uri: str | None = None
    mongo_db: str = "eka"
    mongo_collection: str = "chunks"
    mongo_vector_index: str = "chunk_vector_index"
    mongo_text_index: str = "chunk_text_index"

    # --- chunking & retrieval -----------------------------------------------
    chunk_size: int = 700          # characters
    chunk_overlap: int = 120       # characters
    top_k: int = 5                 # chunks passed to the LLM
    candidate_k: int = 20          # candidates fetched from each retriever before fusion
    rrf_k: int = 60                # reciprocal rank fusion constant
    min_relevance: float = 0.30    # below this, the assistant says it doesn't know

    # --- api ---------------------------------------------------------------
    api_key: str | None = None     # if set, requests must send header X-API-Key
    upload_dir: str = "data/uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
