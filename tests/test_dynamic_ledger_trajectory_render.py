"""Unit tests for apex_agents_bench.dynamic_ledger.trajectory_render."""

from __future__ import annotations

import json

from apex_agents_bench.dynamic_ledger.trajectory_render import render_trajectory_for_curator


def test_empty_trajectory_returns_empty_string() -> None:
    assert render_trajectory_for_curator({}) == ""
    assert render_trajectory_for_curator({"messages": []}) == ""


def test_renders_assistant_reasoning_and_tool_calls() -> None:
    traj = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "the task"},
            {
                "role": "assistant",
                "content": "I will list the tabs first.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "sheets_server_sheets",
                            "arguments": json.dumps({"action": "list_tabs"}),
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "name": "sheets_server_sheets",
                "content": "tabs: [DCF, WACC, Assumptions]",
            },
        ],
    }
    out = render_trajectory_for_curator(traj)
    assert "USER: the task" in out
    assert "ASSISTANT (reasoning): I will list the tabs first." in out
    assert "ASSISTANT (tool_call): sheets_server_sheets" in out
    assert "list_tabs" in out
    assert "TOOL_RESULT" in out
    assert "tabs:" in out


def test_system_messages_are_omitted() -> None:
    traj = {
        "messages": [{"role": "system", "content": "VENDOR SYSTEM PROMPT — should be invisible"}]
    }
    assert render_trajectory_for_curator(traj) == ""


def test_truncates_long_tool_results_only() -> None:
    long_text = "x" * 50_000
    short_text = "short reasoning"
    long_args = json.dumps({"action": "read_tab", "blob": "y" * 50_000})
    traj = {
        "messages": [
            {
                "role": "assistant",
                "content": short_text,
                "tool_calls": [
                    {"function": {"name": "sheets_server_sheets", "arguments": long_args}}
                ],
            },
            {"role": "tool", "name": "sheets_server_sheets", "content": long_text},
        ],
    }
    out = render_trajectory_for_curator(traj, max_chars_per_tool_result=200)
    # Tool result was truncated
    assert "truncated" in out
    # Assistant reasoning was NOT truncated
    assert short_text in out
    # Tool-call ARGUMENTS were NOT truncated (we want the curator to see the
    # full args even when results are long).
    assert "y" * 1000 in out  # arbitrary long substring of args


def test_handles_mcp_style_tool_result_dict() -> None:
    """MCP tool results often arrive as {"type": "text", "text": "..."}."""
    traj = {
        "messages": [
            {
                "role": "tool",
                "name": "filesystem",
                "content": {"type": "text", "text": "/foo\n/bar"},
            },
        ],
    }
    out = render_trajectory_for_curator(traj)
    assert "/foo" in out
    assert "/bar" in out
