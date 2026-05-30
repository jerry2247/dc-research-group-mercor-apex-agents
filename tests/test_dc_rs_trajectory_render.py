"""Unit tests for ``render_trajectory_for_synthesizer`` (agentic-specific).

The renderer walks a trajectory dict's ``messages`` in order, skipping
the system prompt, and emits USER / ASSISTANT(reasoning) /
ASSISTANT(tool_call) / TOOL_RESULT lines. Only tool RESULTS are
truncated to ``max_chars_per_tool_result``; tool-call ARGUMENTS and
assistant reasoning are rendered in full (high-signal portions).
"""

from __future__ import annotations

import json

from apex_agents_bench.dc_rs.trajectory_render import render_trajectory_for_synthesizer


def _synthetic_trajectory(*, long_result: str, long_args: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "UPSTREAM_SYSTEM_PROMPT_SHOULD_BE_SKIPPED"},
            {"role": "user", "content": "THE TASK FOR THE AGENT"},
            {
                "role": "assistant",
                "content": "MY REASONING ABOUT THE PLAN",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_code",
                            "arguments": long_args,
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "name": "run_code",
                "content": long_result,
            },
        ],
        "status": "completed",
    }


def test_render_skips_system_message() -> None:
    traj = _synthetic_trajectory(long_result="ok", long_args="{}")
    out = render_trajectory_for_synthesizer(traj)
    assert "UPSTREAM_SYSTEM_PROMPT_SHOULD_BE_SKIPPED" not in out
    assert "SYSTEM" not in out


def test_render_emits_user_assistant_reasoning_toolcall_and_result() -> None:
    traj = _synthetic_trajectory(long_result="the result", long_args="{}")
    out = render_trajectory_for_synthesizer(traj)
    assert "USER: THE TASK FOR THE AGENT" in out
    assert "ASSISTANT (reasoning): MY REASONING ABOUT THE PLAN" in out
    assert "ASSISTANT (tool_call): run_code(" in out
    assert "TOOL_RESULT[run_code]: the result" in out


def test_render_truncates_long_tool_result_only() -> None:
    cap = 50
    long_result = "R" * (cap + 500)
    long_args = "A" * (cap + 500)  # tool-call args exceed the cap too
    traj = _synthetic_trajectory(long_result=long_result, long_args=long_args)
    out = render_trajectory_for_synthesizer(traj, max_chars_per_tool_result=cap)

    # The tool RESULT is truncated with a marker.
    assert "[truncated" in out
    assert ("R" * (cap + 500)) not in out  # full result body is NOT present

    # The tool-call ARGUMENTS are NOT truncated — the full args survive.
    assert ("A" * (cap + 500)) in out


def test_render_args_full_when_under_default_cap() -> None:
    """A normal-length tool call's args are rendered verbatim (JSON)."""
    args = json.dumps({"path": "/tmp/x.csv", "rows": 5})
    traj = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": args}}],
            },
        ]
    }
    out = render_trajectory_for_synthesizer(traj)
    assert "read_file(" in out
    assert "/tmp/x.csv" in out
    # No reasoning line was emitted for the empty assistant content.
    assert "ASSISTANT (reasoning):" not in out


def test_render_empty_or_malformed_returns_empty_string() -> None:
    assert render_trajectory_for_synthesizer({}) == ""
    assert render_trajectory_for_synthesizer({"messages": "not a list"}) == ""
