"""Unit tests for on-disk persistence + resume semantics (per-domain store).

Agentic port: bank entries store a ``rendered_trajectory`` transcript.
All file IO is under pytest's ``tmp_path`` — no network, no shared state.
"""

from __future__ import annotations

from pathlib import Path

from apex_agents_bench.dc_rs.bank import BankEntry
from apex_agents_bench.dc_rs.store import EMPTY_CHEATSHEET, Store


def _entry(idx: int, *, domain: str = "Finance") -> BankEntry:
    return BankEntry(
        bank_id=f"bank-{idx:05d}",
        task_id=f"t-{idx}",
        domain=domain,
        task_prompt=f"p{idx}",
        rendered_trajectory=f"tr{idx}",
        prompt_embedding=[float(idx), 0.0, 0.0],
        added=idx - 1,
    )


def test_append_and_load_bank_roundtrip_per_domain(tmp_path: Path) -> None:
    store = Store.for_run(tmp_path)
    store.append_bank_entry("Finance", _entry(1, domain="Finance"))
    store.append_bank_entry("Finance", _entry(2, domain="Finance"))
    store.append_bank_entry("Finance", _entry(3, domain="Finance"))

    # Reload from disk into a fresh Store (resume path).
    store2 = Store.for_run(tmp_path)
    bank = store2.load_bank("Finance")
    assert [e.bank_id for e in bank.entries] == ["bank-00001", "bank-00002", "bank-00003"]
    assert [e.task_id for e in bank.entries] == ["t-1", "t-2", "t-3"]
    assert {e.domain for e in bank.entries} == {"Finance"}
    assert [e.rendered_trajectory for e in bank.entries] == ["tr1", "tr2", "tr3"]


def test_cheatsheet_slot_empty_when_no_file(tmp_path: Path) -> None:
    store = Store.for_run(tmp_path)
    assert store.read_cheatsheet("Finance") == EMPTY_CHEATSHEET


def test_cheatsheet_slot_write_read_roundtrip_per_domain(tmp_path: Path) -> None:
    store = Store.for_run(tmp_path)
    store.write_cheatsheet("Finance", "the finance cheatsheet body")
    store.write_cheatsheet("Legal", "the legal cheatsheet body")
    store2 = Store.for_run(tmp_path)
    assert store2.read_cheatsheet("Finance") == "the finance cheatsheet body"
    assert store2.read_cheatsheet("Legal") == "the legal cheatsheet body"


def test_archive_cheatsheet_writes_per_task_file_under_domain(tmp_path: Path) -> None:
    store = Store.for_run(tmp_path)
    p = store.archive_cheatsheet("Finance", "task-abc", "some cheatsheet text")
    assert p.exists()
    assert p.read_text() == "some cheatsheet text"
    assert p.name == "task_task-abc.txt"
    assert p.parent.parent.name == "Finance"


def test_synth_log_append_jsonl_per_domain(tmp_path: Path) -> None:
    import json

    store = Store.for_run(tmp_path)
    store.append_synth_log("Finance", {"task_id": "t-1", "prompt_tokens": 100})
    store.append_synth_log("Finance", {"task_id": "t-2", "prompt_tokens": 200})
    store.append_synth_log("Legal", {"task_id": "t-3", "prompt_tokens": 300})
    fin_lines = store.synth_log_path("Finance").read_text().strip().splitlines()
    legal_lines = store.synth_log_path("Legal").read_text().strip().splitlines()
    assert len(fin_lines) == 2
    assert len(legal_lines) == 1
    parsed = [json.loads(ln) for ln in fin_lines]
    assert parsed[0]["task_id"] == "t-1"
    assert parsed[1]["prompt_tokens"] == 200
    assert json.loads(legal_lines[0])["prompt_tokens"] == 300


def test_store_layout_is_per_domain(tmp_path: Path) -> None:
    """One pool + one cheatsheet slot per domain. The on-disk layout
    is ``runs/<run>/dc_rs/<Domain>/{bank.jsonl,cheatsheet.txt,…}``."""
    store = Store.for_run(tmp_path)
    expected_root = tmp_path / "dc_rs"
    assert store.root == expected_root
    assert expected_root.is_dir()

    store.append_bank_entry("Finance", _entry(1, domain="Finance"))
    store.write_cheatsheet("Finance", "x")
    store.append_bank_entry("Legal", _entry(1, domain="Legal"))
    store.write_cheatsheet("Legal", "y")

    assert (expected_root / "Finance" / "bank.jsonl").is_file()
    assert (expected_root / "Finance" / "cheatsheet.txt").is_file()
    assert (expected_root / "Finance" / "cheatsheets").is_dir()
    assert (expected_root / "Legal" / "bank.jsonl").is_file()
    assert (expected_root / "Legal" / "cheatsheet.txt").is_file()
    assert (expected_root / "Legal" / "cheatsheets").is_dir()


def test_discover_domains_returns_existing_per_domain_dirs(tmp_path: Path) -> None:
    store = Store.for_run(tmp_path)
    assert store.discover_domains() == []
    store.append_bank_entry("Finance", _entry(1, domain="Finance"))
    store.append_bank_entry("Legal", _entry(1, domain="Legal"))
    # Discovery is alphabetical for reproducibility.
    assert store.discover_domains() == ["Finance", "Legal"]


def test_per_domain_state_does_not_cross_contaminate(tmp_path: Path) -> None:
    """Writing to Finance must not affect Legal's on-disk state."""
    store = Store.for_run(tmp_path)
    store.append_bank_entry("Finance", _entry(1, domain="Finance"))
    store.append_bank_entry("Finance", _entry(2, domain="Finance"))
    store.write_cheatsheet("Finance", "finance content")

    legal_bank = store.load_bank("Legal")
    assert legal_bank.entries == []
    assert store.read_cheatsheet("Legal") == EMPTY_CHEATSHEET

    fin_bank = store.load_bank("Finance")
    assert len(fin_bank.entries) == 2
    assert store.read_cheatsheet("Finance") == "finance content"
