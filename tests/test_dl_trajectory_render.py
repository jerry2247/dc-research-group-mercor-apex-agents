"""Unit tests for the DL trajectory renderer (curator's evidence)."""

from __future__ import annotations

from apex_agents_bench.dl.trajectory_render import render_trajectory_for_curator


def test_render_omits_system_and_keeps_roles() -> None:
    traj = {
        "messages": [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "do the task"},
            {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [{"function": {"name": "run_code", "arguments": '{"x": 1}'}}],
            },
            {"role": "tool", "name": "run_code", "content": "result-ok"},
        ]
    }
    out = render_trajectory_for_curator(traj)
    assert "SYSTEM PROMPT" not in out
    assert "do the task" in out
    assert "thinking" in out
    assert "run_code" in out
    assert '{"x": 1}' in out
    assert "result-ok" in out


def test_render_truncates_long_tool_result_only() -> None:
    big = "Z" * 5000
    args = "A" * 5000
    traj = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "t", "arguments": args}}],
            },
            {"role": "tool", "name": "t", "content": big},
        ]
    }
    out = render_trajectory_for_curator(traj, max_chars_per_tool_result=100)
    assert "truncated" in out
    # tool-call arguments are NOT truncated (high-signal)
    assert args in out


def test_render_empty_messages_is_empty_string() -> None:
    assert render_trajectory_for_curator({"messages": []}) == ""
    assert render_trajectory_for_curator({}) == ""
