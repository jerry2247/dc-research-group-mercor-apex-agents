"""Create-time dedup against the retrieved subset. Mirror of apex_bench."""

from __future__ import annotations

from apex_agents_bench.dynamic_ledger.embeddings import cosine_similarity
from apex_agents_bench.dynamic_ledger.entry import Entry


def is_too_similar_to_retrieved(
    *,
    candidate_embedding: list[float],
    retrieved: list[Entry],
    threshold: float = 0.85,
) -> tuple[bool, float, str | None]:
    if not retrieved or not candidate_embedding:
        return False, 0.0, None
    best = 0.0
    best_id: str | None = None
    for e in retrieved:
        if not e.content_embedding:
            continue
        s = cosine_similarity(candidate_embedding, e.content_embedding)
        if s > best:
            best = s
            best_id = e.entry_id
    return best > threshold, best, best_id
