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


def test_load_for_resume_loads_latest_snapshot_on_disk(tmp_path: Path) -> None:
    """``load_for_resume`` must always return the highest snapshot index on disk.

    Snapshots may legitimately exist beyond the CSV's completed-row count
    when the curator emits ops on an agent-failed task (the snapshot is
    saved but no CSV row is written). The snapshot store is the source of
    truth for cheatsheet state; this test pins that contract.
    """
    ss = SnapshotStore.for_domain(tmp_path, "Law")
    ss.save(_store("t1"), index=1)
    ss.save(_store("t2"), index=2)
    ss.save(_store("t3-from-failed-task"), index=3)
    idx, store = ss.load_for_resume(domain="Law")
    assert idx == 3
    assert store.active_entries()[0].content == "t3-from-failed-task"


def test_resume_empty_returns_fresh_store(tmp_path: Path) -> None:
    ss = SnapshotStore.for_domain(tmp_path, "Management Consulting")
    idx, store = ss.load_for_resume(domain="Management Consulting")
    assert idx == 0
    assert store.active_entries() == []
