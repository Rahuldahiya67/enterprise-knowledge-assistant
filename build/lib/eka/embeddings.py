"""Embedding providers.

``HashingEmbedder`` runs fully offline: it hashes word unigrams and bigrams into a
fixed-size vector (sublinear TF, L2-normalised). It is not as good as a neural
model, but it is deterministic, fast and keeps the whole pipeline testable
without an API key. ``OpenAIEmbedder`` is the production choice.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .text import content_tokens


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = content_tokens(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feat in self._features(text):
                idx, sign = self._bucket(feat)
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, value in counts.items():
                out[row, idx] = np.sign(value) * (1.0 + np.log(abs(value))) if value else 0.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class OpenAIEmbedder:
    def __init__(self, api_key: str | None, model: str = "text-embedding-3-small", batch_size: int = 128):
        from openai import OpenAI

        if not api_key:
            raise ValueError("EKA_OPENAI_API_KEY is required for the openai embedding provider")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.dim = 1536 if "small" in model else 3072

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[start : start + self.batch_size]]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        arr = np.asarray(vectors, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def build_embedder(settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model)
    return HashingEmbedder()
