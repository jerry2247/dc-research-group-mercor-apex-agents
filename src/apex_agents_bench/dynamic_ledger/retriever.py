"""Dual-embedding top-k retrieval. Mirror of apex_bench.memory.retriever."""

from __future__ import annotations

from apex_agents_bench.dynamic_ledger.embeddings import cosine_similarity
from apex_agents_bench.dynamic_ledger.entry import DynamicLedger, Entry


def _top_k_by(
    entries: list[Entry],
    key_vec: list[float],
    pick_emb,
    k: int,
    similarity_threshold: float,
) -> list[tuple[float, Entry]]:
    scored: list[tuple[float, Entry]] = []
    for e in entries:
        emb = pick_emb(e)
        if not emb:
            continue
        s = cosine_similarity(key_vec, emb)
        if s < similarity_threshold:
            continue
        scored.append((s, e))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]


def retrieve(
    store: DynamicLedger,
    *,
    query_embedding: list[float],
    k: int,
    similarity_threshold: float = 0.0,
) -> list[Entry]:
    """Dual-axis retrieval with optional similarity floor.

    Entries whose best-axis cosine falls below ``similarity_threshold`` are
    dropped, preventing weakly-related notes from being injected when the
    retrieved set is structurally irrelevant. Set ``similarity_threshold=0.0``
    to restore the original "always inject top-k" behaviour.
    """
    if k <= 0:
        return []
    active = store.active_entries()
    if not active:
        return []

    top_p = _top_k_by(active, query_embedding, lambda e: e.source_problem_embedding, k, similarity_threshold)
    top_c = _top_k_by(active, query_embedding, lambda e: e.content_embedding, k, similarity_threshold)

    seen: set[str] = set()
    out: list[Entry] = []
    for _score, e in top_p:
        if e.entry_id in seen:
            continue
        seen.add(e.entry_id)
        out.append(e)
    for _score, e in top_c:
        if e.entry_id in seen:
            continue
        seen.add(e.entry_id)
        out.append(e)
    return out
