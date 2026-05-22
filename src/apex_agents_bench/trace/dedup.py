"""Create-time cosine-block dedup against existing bullets."""

from __future__ import annotations

from apex_agents_bench.trace.bullet import Bullet
from apex_agents_bench.trace.embeddings import cosine_similarity


def is_too_similar_to_retrieved(
    *,
    candidate_embedding: list[float],
    retrieved: list[Bullet],
    threshold: float,
) -> tuple[bool, float, str | None]:
    if not retrieved or not candidate_embedding:
        return False, 0.0, None
    best = 0.0
    by: str | None = None
    for b in retrieved:
        if not b.content_embedding:
            continue
        s = cosine_similarity(candidate_embedding, b.content_embedding)
        if s > best:
            best = s
            by = b.bullet_id
    blocked = best > threshold
    return blocked, best, by if blocked else None
