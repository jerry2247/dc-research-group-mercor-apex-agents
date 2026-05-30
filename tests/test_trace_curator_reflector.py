"""Unit tests for the TRACE reflector + curator parsers + apply_ops."""

from __future__ import annotations

from apex_agents_bench.trace.bullet import TraceLedger
from apex_agents_bench.trace.config import TraceConfig
from apex_agents_bench.trace.curator import (
    CuratedOp,
    apply_ops,
    parse_cheatsheet_updates,
)
from apex_agents_bench.trace.reflector import parse_reflector_proposals


class _Embed:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1), 1.0] for i in range(len(texts))]


def test_reflector_parses_well_formed_proposal_block() -> None:
    txt = (
        "<reflector_proposals>"
        '[{"op": "CREATE", "section": "S", "content": "C", "source_problem": "P"},'
        ' {"op": "UPDATE", "bullet_id": "bullet-1", "content": "X"}]'
        "</reflector_proposals>"
    )
    props, err = parse_reflector_proposals(txt)
    assert err is None
    assert [p.op for p in props] == ["CREATE", "UPDATE"]


def test_reflector_drops_bad_ops() -> None:
    txt = (
        "<reflector_proposals>"
        '[{"op": "GARBAGE"},'
        ' {"op": "DELETE", "bullet_id": "bullet-1"}]'
        "</reflector_proposals>"
    )
    props, err = parse_reflector_proposals(txt)
    assert err is None
    assert [p.op for p in props] == ["DELETE"]


def test_curator_parses_cheatsheet_updates_block() -> None:
    txt = (
        "<cheatsheet_updates>"
        '[{"op": "CREATE", "section": "S", "content": "C", "source_problem": "P"},'
        ' {"op": "NO_OP", "reason": "r"}]'
        "</cheatsheet_updates>"
    )
    ops, err = parse_cheatsheet_updates(txt)
    assert err is None
    assert [o.op for o in ops] == ["CREATE", "NO_OP"]


def test_apply_create_commits() -> None:
    store = TraceLedger(domain="Law")
    ops = [CuratedOp(op="CREATE", section="S", content="C", source_problem="P")]
    cfg = TraceConfig(enabled=True)
    stats = apply_ops(
        store=store, ops=ops, retrieved=[], embed=_Embed(),
        cfg=cfg, current_ordinal=1,
    )
    assert stats.create_committed == 1
    assert len(store.bullets) == 1


def test_apply_create_dedups_against_active_store_not_only_retrieved() -> None:
    store = TraceLedger(domain="Law")
    store.add(
        section="S",
        content="existing",
        source_problem="P",
        content_embedding=[1.0, 1.0],
        source_problem_embedding=[0.0, 1.0],
        created=1,
    )
    ops = [CuratedOp(op="CREATE", section="S2", content="duplicate", source_problem="P2")]
    stats = apply_ops(
        store=store,
        ops=ops,
        retrieved=[],
        embed=_Embed(),
        cfg=TraceConfig(enabled=True),
        current_ordinal=2,
    )
    assert stats.create_blocked == 1
    assert stats.create_committed == 0
    assert len(store.bullets) == 1


def test_apply_consolidate_sums_counters() -> None:
    store = TraceLedger(domain="Law")
    b1 = store.add(
        section="S1", content="o1", source_problem="p1",
        content_embedding=[1.0, 0.0], source_problem_embedding=[0.0, 1.0],
        created=1,
    )
    b2 = store.add(
        section="S2", content="o2", source_problem="p2",
        content_embedding=[0.0, 1.0], source_problem_embedding=[1.0, 0.0],
        created=2,
    )
    # Pre-populate counters
    store.record_citation(b1.bullet_id, gt_correct=True)
    store.record_citation(b2.bullet_id, gt_correct=False)

    ops = [
        CuratedOp(
            op="CONSOLIDATE",
            bullet_ids=[b1.bullet_id, b2.bullet_id],
            section="M",
            content="merged",
            source_problem="mp",
        )
    ]
    stats = apply_ops(
        store=store, ops=ops, retrieved=[], embed=_Embed(),
        cfg=TraceConfig(enabled=True), current_ordinal=3,
    )
    assert stats.consolidate == 1
    new_id = "bullet-3"
    new = store.bullets[new_id]
    assert new.usage == 2
    assert new.helpful == 1
    assert new.harmful == 1
    assert store.bullets[b1.bullet_id].active is False
    assert store.bullets[b2.bullet_id].active is False


def test_apply_invalid_id_counted() -> None:
    store = TraceLedger(domain="Law")
    ops = [CuratedOp(op="DELETE", bullet_id="bullet-999")]
    stats = apply_ops(
        store=store, ops=ops, retrieved=[], embed=_Embed(),
        cfg=TraceConfig(enabled=True), current_ordinal=1,
    )
    assert stats.skipped_invalid_bullet_id == 1
