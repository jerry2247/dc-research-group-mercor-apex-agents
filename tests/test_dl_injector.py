"""Unit tests for the DL generator injector (typed-entries prepend)."""

from __future__ import annotations

import json
from pathlib import Path

from apex_agents_bench.dl.entry import DLLedger
from apex_agents_bench.dl.injector import augment_initial_messages


def _write_initial(tmp_path: Path) -> Path:
    p = tmp_path / "initial_messages.json"
    p.write_text(
        json.dumps(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "USER_TASK"},
            ]
        ),
        encoding="utf-8",
    )
    return p


def _entries(led: DLLedger, n: int):
    for i in range(1, n + 1):
        led.add(
            type="snippet",
            content=f"BODY{i}",
            source_problem=f"s{i}",
            content_embedding=[1.0, 0.0],
            source_problem_embedding=[0.0, 1.0],
            created=i,
        )
    return list(led.active_entries())


def test_injects_into_user_message_only(tmp_path: Path) -> None:
    p = _write_initial(tmp_path)
    led = DLLedger(domain="D")
    entries = _entries(led, 2)
    prefix = augment_initial_messages(p, entries=entries)
    assert prefix
    data = json.loads(p.read_text(encoding="utf-8"))
    # system message untouched
    assert data[0]["role"] == "system"
    assert data[0]["content"] == "SYS"
    # user message has the block prepended, original preserved at the end
    assert data[1]["role"] == "user"
    assert data[1]["content"].endswith("USER_TASK")
    assert "BODY1" in data[1]["content"]
    assert "BODY2" in data[1]["content"]
    # consult-don't-obey reference framing present, no citation instruction
    assert "reference" in data[1]["content"].lower()
    assert "<citations>" not in data[1]["content"]


def test_empty_entries_is_byte_for_byte_noop(tmp_path: Path) -> None:
    p = _write_initial(tmp_path)
    before = p.read_text(encoding="utf-8")
    prefix = augment_initial_messages(p, entries=[])
    assert prefix == ""
    assert p.read_text(encoding="utf-8") == before


def test_missing_user_message_raises(tmp_path: Path) -> None:
    p = tmp_path / "initial_messages.json"
    p.write_text(json.dumps([{"role": "system", "content": "S"}]), encoding="utf-8")
    led = DLLedger(domain="D")
    entries = _entries(led, 1)
    try:
        augment_initial_messages(p, entries=entries)
    except ValueError as e:
        assert "user message" in str(e)
    else:
        raise AssertionError("expected ValueError when no user message present")
