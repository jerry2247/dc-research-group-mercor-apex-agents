"""Embedding client + cosine helper (DC-RS subsystem)."""

from __future__ import annotations

import math
from typing import Protocol


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LiteLLMEmbeddingClient:
    def __init__(self, *, model: str = "text-embedding-3-large") -> None:
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import litellm

        resp = litellm.embedding(model=self.model, input=texts)
        data = getattr(resp, "data", None) or resp["data"]
        out: list[list[float]] = []
        for row in data:
            if hasattr(row, "embedding"):
                out.append(list(row.embedding))
            else:
                out.append(list(row["embedding"]))
        return out


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
