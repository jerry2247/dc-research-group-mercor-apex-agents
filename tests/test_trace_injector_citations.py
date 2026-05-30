"""Unit tests for TRACE injector + citations extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex_agents_bench.trace.bullet import TraceLedger
from apex_agents_bench.trace.citations import (
    extract_and_strip_citations_from_trajectory,
)
from apex_agents_bench.trace.injector import augment_initial_messages, render_bullets_block


def _two_bullets() -> list:
    s = TraceLedger(domain="Law")
    s.add(
        section="A", content="alpha", source_problem="ap",
        content_embedding=[1.0], source_problem_embedding=[0.0],
        created=1,
    )
    s.add(
        section="B", content="beta", source_problem="bp",
        content_embedding=[0.0], source_problem_embedding=[1.0],
        created=2,
    )
    return list(s.bullets.values())


def test_render_bullets_block_includes_counters() -> None:
    out = render_bullets_block(_two_bullets())
    assert "<bullet bullet-1 section=A" in out
    assert "helpful=0" in out and "harmful=0" in out and "usage=0" in out


def test_render_bullets_block_empty_marker() -> None:
    assert "no relevant strategy bullets" in render_bullets_block([])


def test_augment_initial_messages_preserves_system(tmp_path: Path) -> None:
    p = tmp_path / "im.json"
    p.write_text(
        json.dumps(
            [
                {"role": "system", "content": "VENDOR_SYSTEM"},
                {"role": "user", "content": "the task"},
            ]
        ),
        encoding="utf-8",
    )
    prefix = augment_initial_messages(p, bullets=_two_bullets())
    out = json.loads(p.read_text(encoding="utf-8"))
    assert out[0]["content"] == "VENDOR_SYSTEM"  # untouched
    assert out[1]["content"].startswith(prefix)
    assert out[1]["content"].endswith("the task")
    assert "<bullet bullet-1" in out[1]["content"]


def test_extract_citations_from_trajectory(tmp_path: Path) -> None:
    traj = {
        "messages": [
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "function": {
                            "name": "final_answer",
                            "arguments": json.dumps(
                                {
                                    "answer": "42",
                                    "reasoning": "I did stuff.\n<citations>[bullet-1, bullet-7]</citations>",
                                    "status": "completed",
                                }
                            ),
                        }
                    }
                ],
            }
        ],
        "status": "completed",
    }
    p = tmp_path / "traj.json"
    p.write_text(json.dumps(traj), encoding="utf-8")
    extract, shadow = extract_and_strip_citations_from_trajectory(p)
    assert extract.citations_present
    assert extract.cited_bullet_ids == ["bullet-1", "bullet-7"]
    assert shadow is not None
    shadow_args = json.loads(
        shadow["messages"][0]["tool_calls"][0]["function"]["arguments"]
    )
    assert "<citations>" not in shadow_args["reasoning"]


def test_extract_citations_returns_none_when_absent(tmp_path: Path) -> None:
    traj = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "final_answer",
                            "arguments": json.dumps({"answer": "x", "reasoning": "no tag", "status": "completed"}),
                        }
                    }
                ],
            }
        ],
    }
    p = tmp_path / "traj.json"
    p.write_text(json.dumps(traj), encoding="utf-8")
    extract, shadow = extract_and_strip_citations_from_trajectory(p)
    assert extract.citations_present is False
    assert extract.cited_bullet_ids == []
    assert shadow is None
