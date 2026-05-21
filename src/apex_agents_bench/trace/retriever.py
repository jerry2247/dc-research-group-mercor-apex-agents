"""Dual-axis top-k retrieval from a TraceLedger."""

from __future__ import annotations

from apex_agents_bench.trace.bullet import Bullet, TraceLedger
from apex_agents_bench.trace.embeddings import cosine_similarity


def retrieve(
    store: TraceLedger,
    *,
    query_embedding: list[float],
    k: int,
) -> list[Bullet]:
    """Dual-axis retrieval, source-problem axis first."""
    if k <= 0:
        return []
    active = store.active_bullets()
    if not active:
        return []

    def score(axis_attr: str) -> list[Bullet]:
        scored: list[tuple[float, Bullet]] = []
        for b in active:
            v = getattr(b, axis_attr)
            if not v:
                continue
            s = cosine_similarity(query_embedding, v)
            scored.append((s, b))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [b for _s, b in scored[:k]]

    top_p = score("source_problem_embedding")
    top_c = score("content_embedding")

    out: list[Bullet] = []
    seen: set[str] = set()
    for b in top_p + top_c:
        if b.bullet_id in seen:
            continue
        seen.add(b.bullet_id)
        out.append(b)
    return out
