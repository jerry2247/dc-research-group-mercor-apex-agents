"""Unit tests for apex_agents_bench.dynamic_ledger.store — snapshot persistence."""

from __future__ import annotations

from pathlib import Path

from apex_agents_bench.dynamic_ledger.entry import DynamicLedger
from apex_agents_bench.dynamic_ledger.store import SnapshotStore


def _store(content="x") -> DynamicLedger:
    s = DynamicLedger(domain="Investment Banking")
    s.add(
        section="x",
        content=content,
        source_problem="p",
        content_embedding=[1.0],
        source_problem_embedding=[0.0],
        created=1,
    )
    return s


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    ss = SnapshotStore.for_domain(tmp_path, "Investment Banking")
    ss.save(_store("hello"), index=1)
    loaded = ss.load(1)
    assert loaded is not None
    assert loaded.active_entries()[0].content == "hello"


def test_load_for_resume_caps_at_completed_csv_rows(tmp_path: Path) -> None:
    ss = SnapshotStore.for_domain(tmp_path, "Law")
    ss.save(_store("t1"), index=1)
    ss.save(_store("t2"), index=2)
    ss.save(_store("t3-ahead"), index=3)
    idx, store = ss.load_for_resume(max_index_allowed=2, domain="Law")
    assert idx == 2
    assert store.active_entries()[0].content == "t2"


def test_resume_empty_returns_fresh_store(tmp_path: Path) -> None:
    ss = SnapshotStore.for_domain(tmp_path, "Management Consulting")
    idx, store = ss.load_for_resume(max_index_allowed=0, domain="Management Consulting")
    assert idx == 0
    assert store.active_entries() == []
