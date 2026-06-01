"""Dual-axis top-k retrieval from a ``DLLedger``.

Faithful to the original Dynamic Ledger's dual retrieval: top-k by the
entry's own ``content_embedding`` AND top-k by its
``source_problem_embedding``, unioned by ``entry_id``. This lets a future
task find an entry either through resemblance to the entry's content or
through resemblance to the source problem that produced it.

Like the TRACE retriever, the source-problem axis is taken first so that,
on ties or overlap, the situational match leads the rendered window. The
per-domain ledger is the unit of isolation — the caller passes the
specific domain's ledger, so retrieval never crosses domains.
"""

from __future__ import annotations

from apex_agents_bench.dl.embeddings import cosine_similarity
from apex_agents_bench.dl.entry import DLEntry, DLLedger


def retrieve(
    ledger: DLLedger,
    *,
    query_embedding: list[float],
    k: int = 3,
) -> list[DLEntry]:
    """Dual-axis retrieval, source-problem axis first.

    Returns up to ``2*k`` entries (k per axis, unioned). Empty when the
    ledger has no active entries or ``k <= 0``.
    """
    if k <= 0:
        return []
    active = ledger.active_entries()
    if not active:
        return []

    def score(axis_attr: str) -> list[DLEntry]:
        scored: list[tuple[float, DLEntry]] = []
        for e in active:
            v = getattr(e, axis_attr)
            if not v:
                continue
            s = cosine_similarity(query_embedding, v)
            scored.append((s, e))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [e for _s, e in scored[:k]]

    top_source = score("source_problem_embedding")
    top_content = score("content_embedding")

    out: list[DLEntry] = []
    seen: set[str] = set()
    for e in top_source + top_content:
        if e.entry_id in seen:
            continue
        seen.add(e.entry_id)
        out.append(e)
    return out
