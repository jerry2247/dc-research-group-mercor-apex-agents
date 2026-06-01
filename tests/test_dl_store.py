"""Unit tests for the DL per-domain snapshot store + resume."""

from __future__ import annotations

from pathlib import Path

from apex_agents_bench.dl.entry import DLLedger
from apex_agents_bench.dl.store import SnapshotStore


def _ledger_with_one(domain: str = "Finance") -> DLLedger:
    led = DLLedger(domain=domain)
    led.add(
        type="snippet",
        content="c",
        source_problem="s",
        content_embedding=[1.0, 0.0],
        source_problem_embedding=[0.0, 1.0],
        created=1,
    )
    return led


def test_save_and_latest_roundtrip(tmp_path: Path) -> None:
    ss = SnapshotStore.for_domain(tmp_path, "Finance")
    led = _ledger_with_one()
    ss.save(led, index=1)
    got = ss.latest()
    assert got is not None
    idx, loaded = got
    assert idx == 1
    assert loaded.domain == "Finance"
    assert len(loaded.active_entries()) == 1
    assert loaded.entries["entry-1"].content == "c"


def test_latest_picks_highest_index(tmp_path: Path) -> None:
    ss = SnapshotStore.for_domain(tmp_path, "Finance")
    led = _ledger_with_one()
    ss.save(led, index=1)
    led.add(
        type="pitfall",
        content="c2",
        source_problem="s2",
        content_embedding=[0.0, 1.0],
        source_problem_embedding=[1.0, 0.0],
        created=2,
    )
    ss.save(led, index=2)
    got = ss.latest()
    assert got is not None
    idx, loaded = got
    assert idx == 2
    assert len(loaded.active_entries()) == 2


def test_latest_none_when_empty(tmp_path: Path) -> None:
    ss = SnapshotStore.for_domain(tmp_path, "Finance")
    assert ss.latest() is None


def test_soft_deleted_entries_persist_through_snapshot(tmp_path: Path) -> None:
    ss = SnapshotStore.for_domain(tmp_path, "Finance")
    led = _ledger_with_one()
    led.soft_delete("entry-1", updated=2)
    ss.save(led, index=2)
    _idx, loaded = ss.latest()
    # the inactive entry is still present (id never reused), just not active
    assert "entry-1" in loaded.entries
    assert loaded.entries["entry-1"].active is False
    assert loaded.active_entries() == []
    assert loaded.next_entry_ord == led.next_entry_ord


def test_curator_log_appends_one_line_per_call(tmp_path: Path) -> None:
    ss = SnapshotStore.for_domain(tmp_path, "Finance")
    ss.append_curator_log({"task_id": "a", "create": 1})
    ss.append_curator_log({"task_id": "b", "delete": 2})
    log_path = tmp_path / "dl" / "Finance" / "curator_log.jsonl"
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


def test_domains_are_isolated_on_disk(tmp_path: Path) -> None:
    ss_fin = SnapshotStore.for_domain(tmp_path, "Finance")
    ss_law = SnapshotStore.for_domain(tmp_path, "Law")
    ss_fin.save(_ledger_with_one("Finance"), index=1)
    # Law has no snapshot yet
    assert ss_law.latest() is None
    assert (tmp_path / "dl" / "Finance").is_dir()
    assert (tmp_path / "dl" / "Law").is_dir()
