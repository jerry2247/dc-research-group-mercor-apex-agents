"""LiteLLM-backed embedding client + cosine helper. Mirror of
``apex_bench.memory.embeddings`` so the two repos share semantics."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from typing import Protocol


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LiteLLMEmbeddingClient:
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-large",
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import litellm

        backoff = self.initial_backoff_seconds
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = litellm.embedding(
                    model=self.model,
                    input=texts,
                    timeout=self.timeout_seconds,
                )
                data = getattr(resp, "data", None) or resp["data"]
                vectors: list[list[float]] = []
                for item in data:
                    if isinstance(item, dict):
                        vectors.append(list(item["embedding"]))
                    else:
                        vectors.append(list(item.embedding))
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"embedding count mismatch: requested {len(texts)} got {len(vectors)}"
                    )
                return vectors
            except Exception as exc:
                last_exc = exc
                if attempt + 1 == self.max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2.0
        raise RuntimeError(
            f"LiteLLMEmbeddingClient.embed failed after {self.max_retries} attempts: {last_exc}"
        )


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        raise ValueError(f"vector length mismatch: {len(a_list)} vs {len(b_list)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a_list, b_list, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
