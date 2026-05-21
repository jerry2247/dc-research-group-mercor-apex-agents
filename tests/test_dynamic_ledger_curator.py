"""Unit tests for apex_agents_bench.dynamic_ledger.curator — parser + apply_ops."""

from __future__ import annotations

from apex_agents_bench.dynamic_ledger.config import DynamicLedgerConfig
from apex_agents_bench.dynamic_ledger.curator import (
    VALID_OPS,
    CuratedOp,
    apply_ops,
    parse_memory_updates,
)
from apex_agents_bench.dynamic_ledger.entry import DynamicLedger


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t) % 7), float(len(t) % 11), float(len(t) % 13)] for t in texts]


def test_valid_ops_constant() -> None:
    """Dynamic Ledger has exactly three ops per the DC2 codebase
    reference. NO_OP / CONSOLIDATE are NOT in the DL approach."""
    assert set(VALID_OPS) == {"CREATE", "UPDATE", "DELETE"}


def test_parse_no_block() -> None:
    ops, err = parse_memory_updates("hello")
    assert ops == []
    assert "no <memory_updates>" in (err or "")


def test_parse_empty_array() -> None:
    ops, err = parse_memory_updates("<memory_updates>[]</memory_updates>")
    assert ops == []
    assert err is None


def test_parse_one_create() -> None:
    block = (
        '<memory_updates>[{"op": "CREATE", "section": "s", '
        '"content": "c", "source_problem": "p"}]</memory_updates>'
    )
    ops, err = parse_memory_updates(block)
    assert err is None
    assert len(ops) == 1
    assert ops[0].op == "CREATE"


def test_parse_multi_op_drops_invalid() -> None:
    """The parser drops any op outside the DL set (CREATE / UPDATE /
    DELETE). NO_OP and CONSOLIDATE are not in the DL approach and are
    silently dropped, so the curator's wider prompt cannot poison the
    ledger by emitting them."""
    block = """<memory_updates>
[
  {"op": "CREATE", "section": "t1", "content": "c1", "source_problem": "p1"},
  {"op": "DELETE", "entry_id": "entry-3"},
  {"op": "BOGUS"},
  {"op": "CONSOLIDATE", "entry_ids": ["entry-1","entry-2"]},
  {"op": "NO_OP", "reason": "redundant"}
]
</memory_updates>"""
    ops, err = parse_memory_updates(block)
    assert err is None
    kinds = [o.op for o in ops]
    assert "CONSOLIDATE" not in kinds
    assert "NO_OP" not in kinds
    assert "BOGUS" not in kinds
    assert set(kinds) == {"CREATE", "DELETE"}


def _store() -> DynamicLedger:
    s = DynamicLedger(domain="Investment Banking")
    s.add(
        section="s1",
        content="c1",
        source_problem="p1",
        content_embedding=[1.0, 0.0, 0.0],
        source_problem_embedding=[0.0, 1.0, 0.0],
        created=1,
    )
    s.add(
        section="s2",
        content="c2",
        source_problem="p2",
        content_embedding=[0.0, 1.0, 0.0],
        source_problem_embedding=[0.0, 0.0, 1.0],
        created=1,
    )
    return s


def test_apply_create_commits() -> None:
    s = _store()
    cfg = DynamicLedgerConfig(enabled=True)
    embed = _FakeEmbedder()
    ops = [CuratedOp(op="CREATE", section="new", content="fresh", source_problem="case")]
    stats = apply_ops(store=s, ops=ops, retrieved=[], embed=embed, cfg=cfg, current_ordinal=5)
    assert stats.create_committed == 1
    assert len(s.active_entries()) == 3


def test_apply_create_blocked_by_dedup() -> None:
    s = _store()
    cfg = DynamicLedgerConfig(enabled=True, create_time_similarity_threshold=0.0)
    embed = _FakeEmbedder()
    retrieved = [s.entries["entry-1"]]
    ops = [CuratedOp(op="CREATE", section="new", content="c1", source_problem="p1")]
    stats = apply_ops(
        store=s, ops=ops, retrieved=retrieved, embed=embed, cfg=cfg, current_ordinal=5
    )
    assert stats.create_blocked >= 1
    assert len(s.active_entries()) == 2


def test_apply_delete_invalid_entry_id_counted() -> None:
    s = _store()
    cfg = DynamicLedgerConfig(enabled=True)
    embed = _FakeEmbedder()
    ops = [
        CuratedOp(op="DELETE", entry_id="entry-1"),
        CuratedOp(op="DELETE", entry_id="entry-99999"),
    ]
    stats = apply_ops(store=s, ops=ops, retrieved=[], embed=embed, cfg=cfg, current_ordinal=10)
    assert stats.delete == 1
    assert stats.skipped_invalid_entry_id == 1


def test_apply_only_three_op_types() -> None:
    """ApplyStats covers exactly CREATE / UPDATE / DELETE — there is no
    NO_OP counter or CONSOLIDATE counter; those concepts are not in the
    DL approach."""
    s = _store()
    cfg = DynamicLedgerConfig(enabled=True)
    embed = _FakeEmbedder()
    stats = apply_ops(store=s, ops=[], retrieved=[], embed=embed, cfg=cfg, current_ordinal=5)
    fields = {f for f in stats.__dataclass_fields__}
    assert "no_op" not in fields
    assert "consolidate" not in fields
