"""Render the agent's trajectory as a curator-readable transcript.

The curator sees every assistant message (with its content + tool_calls)
and every tool result, in turn order. Long tool RESULTS are truncated
to ``trajectory_max_chars_per_tool_result`` — tool-call ARGUMENTS and
assistant reasoning are rendered in full (high-signal portions).
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
        # MCP tool results often arrive as {"type": "text", "text": "..."}; surface text directly.
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
    max_chars_per_tool_result: int = 4000,
) -> str:
    """Walk the trajectory's messages in order and emit a compact transcript.

    Inputs:
      - ``trajectory``: the parsed trajectory.json dict.
      - ``max_chars_per_tool_result``: cap on each individual tool result.
        Tool-call args and assistant content are NOT capped.

    Returns a single string.
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
            # System prompt is the upstream Archipelago one — the curator
            # does not need it re-rendered every turn.
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
        # other roles: skip silently
    return "\n".join(parts)
