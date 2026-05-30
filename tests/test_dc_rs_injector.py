"""Unit tests for the DC-RS generator injector (cheatsheet prepend).

Hook A rewrites ``initial_messages.json`` in place: the synthesized
cheatsheet block is prepended to the USER message; the SYSTEM message is
left byte-identical. ``""`` and ``"(empty)"`` are no-ops so the agent
sees the baseline content. Mirrors the TRACE injector test's
``initial_messages.json`` fixture pattern. All IO is under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

from apex_agents_bench.dc_rs.injector import augment_initial_messages


def _write_initial_messages(p: Path, *, system: str, user: str) -> None:
    p.write_text(
        json.dumps(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        ),
        encoding="utf-8",
    )


def test_augment_prepends_cheatsheet_to_user_and_preserves_system(tmp_path: Path) -> None:
    p = tmp_path / "initial_messages.json"
    _write_initial_messages(p, system="VENDOR_SYSTEM", user="the task")
    prefix = augment_initial_messages(p, cheatsheet="LINE A\nLINE B")

    out = json.loads(p.read_text(encoding="utf-8"))
    # System message is untouched.
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "VENDOR_SYSTEM"
    # The cheatsheet text lands in the USER message, before the task body.
    assert out[1]["role"] == "user"
    assert "LINE A" in out[1]["content"]
    assert "LINE B" in out[1]["content"]
    assert out[1]["content"].startswith(prefix)
    assert out[1]["content"].endswith("the task")
    # The returned prefix is the substituted block actually prepended.
    assert prefix != ""
    assert "LINE A" in prefix


def test_augment_empty_cheatsheet_is_noop(tmp_path: Path) -> None:
    p = tmp_path / "initial_messages.json"
    _write_initial_messages(p, system="VENDOR_SYSTEM", user="the task")
    before = p.read_text(encoding="utf-8")

    prefix = augment_initial_messages(p, cheatsheet="")
    assert prefix == ""
    # File is byte-identical to the baseline.
    assert p.read_text(encoding="utf-8") == before


def test_augment_empty_placeholder_is_noop(tmp_path: Path) -> None:
    p = tmp_path / "initial_messages.json"
    _write_initial_messages(p, system="VENDOR_SYSTEM", user="the task")
    before = p.read_text(encoding="utf-8")

    prefix = augment_initial_messages(p, cheatsheet="(empty)")
    assert prefix == ""
    assert p.read_text(encoding="utf-8") == before


def test_augment_whitespace_only_cheatsheet_is_noop(tmp_path: Path) -> None:
    p = tmp_path / "initial_messages.json"
    _write_initial_messages(p, system="VENDOR_SYSTEM", user="the task")
    before = p.read_text(encoding="utf-8")

    prefix = augment_initial_messages(p, cheatsheet="   \n")
    assert prefix == ""
    assert p.read_text(encoding="utf-8") == before
