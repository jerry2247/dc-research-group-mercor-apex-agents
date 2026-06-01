"""Render a vendor trajectory.json into a curator-readable transcript (DL).

Mirrors the DC-RS / TRACE renderers: each assistant message + tool result
is shown in order with the SYSTEM prompt omitted; long tool-result bodies
are truncated per ``max_chars_per_tool_result`` while tool-call ARGUMENTS
and assistant reasoning text are preserved verbatim (the high-signal
portions — including the error-then-fix transitions the curator mines).

DL runs this on THIS task's trajectory after the agent finishes (the
curator's evidence), not on retrieved past trajectories.
"""

from __future__ import annotations

import json
from typing import Any


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_stringify(c) for c in content)
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _truncate(text: str, cap: int) -> str:
    if cap <= 0 or len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated {len(text) - cap} chars]"


def render_trajectory_for_curator(
    trajectory: dict,
    *,
    max_chars_per_tool_result: int = 8000,
) -> str:
    """Walk the trajectory's messages in order and emit a compact transcript.

    Tool-call args and assistant content are NOT capped; only long tool
    RESULTS are. Returns a single string.
    """
    messages = trajectory.get("messages") or []
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        if role == "system":
            continue
        if role == "user":
            content_s = _stringify(m.get("content", "") or "")
            parts.append(f"[{i}] USER: {content_s}")
        elif role == "assistant":
            content_s = _stringify(m.get("content", "") or "")
            if content_s.strip():
                parts.append(f"[{i}] ASSISTANT (reasoning): {content_s}")
            for tc in m.get("tool_calls") or []:
                fn_obj = tc.get("function") if isinstance(tc, dict) else None
                if not isinstance(fn_obj, dict):
                    continue
                name = fn_obj.get("name", "?")
                args = fn_obj.get("arguments", "")
                args_s = _stringify(args)
                parts.append(f"[{i}] ASSISTANT (tool_call): {name}({args_s})")
        elif role == "tool":
            content_s = _stringify(m.get("content", "") or "")
            name = m.get("name") or m.get("tool_call_name") or "?"
            truncated = _truncate(content_s, max_chars_per_tool_result)
            parts.append(f"[{i}] TOOL_RESULT[{name}]: {truncated}")
    return "\n".join(parts)
