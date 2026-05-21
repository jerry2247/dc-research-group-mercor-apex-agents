"""Unit tests for apex_agents_bench.dynamic_ledger.retriever + dedup."""

from __future__ import annotations

from apex_agents_bench.dynamic_ledger.dedup import is_too_similar_to_retrieved
from apex_agents_bench.dynamic_ledger.entry import DynamicLedger
from apex_agents_bench.dynamic_ledger.retriever import retrieve


def _add(s: DynamicLedger, *, content_emb, source_emb, content="c", problem="p"):
    return s.add(
        section="x",
        content=content,
        source_problem=problem,
        content_embedding=list(content_emb),
        source_problem_embedding=list(source_emb),
        created=1,
    )


def test_retrieve_empty_store_returns_empty() -> None:
    s = DynamicLedger(domain="Investment Banking")
    assert retrieve(s, query_embedding=[1.0, 0.0], k=5) == []


def test_retrieve_dual_axis_dedupes_by_entry_id() -> None:
    s = DynamicLedger(domain="Law")
    a = _add(s, content_emb=[1.0, 0.0], source_emb=[1.0, 0.0])
    b = _add(s, content_emb=[0.9, 0.1], source_emb=[0.1, 0.9])
    out = retrieve(s, query_embedding=[1.0, 0.0], k=2)
    ids = [e.entry_id for e in out]
    assert ids.count(a.entry_id) == 1
    assert ids.count(b.entry_id) == 1


def test_retrieve_source_problem_axis_appears_first() -> None:
    s = DynamicLedger(domain="Management Consulting")
    a = _add(s, content_emb=[0.1, 0.9], source_emb=[1.0, 0.0])
    b = _add(s, content_emb=[1.0, 0.0], source_emb=[0.1, 0.9])
    out = retrieve(s, query_embedding=[1.0, 0.0], k=1)
    assert [e.entry_id for e in out] == [a.entry_id, b.entry_id]


def test_dedup_blocks_above_threshold() -> None:
    s = DynamicLedger(domain="Investment Banking")
    e = _add(s, content_emb=[1.0, 0.0, 0.0], source_emb=[0.0, 1.0, 0.0])
    blocked, best, by = is_too_similar_to_retrieved(
        candidate_embedding=[1.0, 0.0, 0.0],
        retrieved=[e],
        threshold=0.85,
    )
    assert blocked is True
    assert best == 1.0
    assert by == "entry-1"


def test_dedup_passes_under_threshold() -> None:
    s = DynamicLedger(domain="Law")
    e = _add(s, content_emb=[1.0, 0.0, 0.0], source_emb=[0.0, 1.0, 0.0])
    blocked, _best, _by = is_too_similar_to_retrieved(
        candidate_embedding=[0.0, 1.0, 0.0],  # orthogonal
        retrieved=[e],
        threshold=0.85,
    )
    assert blocked is False
