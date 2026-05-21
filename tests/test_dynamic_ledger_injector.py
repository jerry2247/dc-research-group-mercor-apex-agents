"""Unit tests for apex_agents_bench.dynamic_ledger.injector — initial_messages
augmentation + entries rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex_agents_bench.dynamic_ledger.entry import DynamicLedger
from apex_agents_bench.dynamic_ledger.injector import (
    augment_initial_messages,
    render_entries_block,
)


def _store_with_two() -> DynamicLedger:
    s = DynamicLedger(domain="Investment Banking")
    s.add(
        section="A topic",
        content="alpha workflow",
        source_problem="alpha problem",
        content_embedding=[1.0],
        source_problem_embedding=[0.0],
        created=1,
    )
    s.add(
        section="B topic",
        content="beta note",
        source_problem="beta problem",
        content_embedding=[0.0],
        source_problem_embedding=[1.0],
        created=2,
    )
    return s


def test_render_entries_block_empty_returns_marker() -> None:
    out = render_entries_block([])
    assert "no relevant prior notes" in out


def test_render_entries_block_two_entries() -> None:
    s = _store_with_two()
    out = render_entries_block(list(s.entries.values()))
    assert "<entry entry-1 section=A topic>" in out
    assert "<entry entry-2 section=B topic>" in out
    assert "alpha workflow" in out
    assert "beta note" in out


def test_augment_initial_messages_prepends_to_user_only(tmp_path: Path) -> None:
    """The SYSTEM message must be untouched (fidelity invariant). The
    strategies block is prepended only to the USER message."""
    s = _store_with_two()
    p = tmp_path / "initial_messages.json"
    p.write_text(
        json.dumps(
            [
                {"role": "system", "content": "SYSTEM_PROMPT_VERBATIM"},
                {"role": "user", "content": "the task prompt"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prefix = augment_initial_messages(p, entries=list(s.entries.values()))
    out = json.loads(p.read_text(encoding="utf-8"))
    assert out[0]["content"] == "SYSTEM_PROMPT_VERBATIM"  # untouched
    assert out[1]["content"].startswith(prefix)
    assert out[1]["content"].endswith("the task prompt")
    assert "## Reference cheatsheet" in out[1]["content"]
    assert "<entry entry-1" in out[1]["content"]


def test_augment_raises_when_no_user_message(tmp_path: Path) -> None:
    p = tmp_path / "initial_messages.json"
    p.write_text(
        json.dumps(
            [
                {"role": "system", "content": "no user"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        augment_initial_messages(p, entries=[])


def test_augment_handles_empty_entries(tmp_path: Path) -> None:
    """When entries is empty we still inject the (no relevant prior notes)
    marker — this is intentional: it tells the agent the playbook was
    consulted even if no specific note applied. The runner can choose to
    skip injection entirely if it prefers; the injector itself does what
    it's asked."""
    p = tmp_path / "initial_messages.json"
    p.write_text(
        json.dumps(
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    augment_initial_messages(p, entries=[])
    out = json.loads(p.read_text(encoding="utf-8"))
    assert out[0]["content"] == "S"
    assert "no relevant prior notes" in out[1]["content"]
    assert out[1]["content"].rstrip().endswith("U")
