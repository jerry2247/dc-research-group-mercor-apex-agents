"""Render a vendor trajectory.json into a curator-readable transcript.

Mirrors the Dynamic-Ledger renderer: each assistant message + tool
result is shown in order with the SYSTEM prompt omitted; long tool
result bodies are truncated per ``max_chars_per_tool_result`` while
tool-call ARGUMENTS and assistant reasoning text are preserved verbatim.
"""

from __future__ import annotations


def _truncate(text: str, cap: int) -> str:
    if cap <= 0 or len(text) <= cap:
        return text
    return text[:cap] + f"\n[... {len(text) - cap} more chars omitted ...]"


def render_trajectory_for_curator(
    trajectory: dict,
    *,
    max_chars_per_tool_result: int = 8000,
) -> str:
    msgs = trajectory.get("messages") or []
    out: list[str] = []
    step = 0
    for m in msgs:
        role = m.get("role", "?")
        if role == "system":
            continue
        if role == "assistant":
            step += 1
            txt = m.get("content") or ""
            if isinstance(txt, list):
                txt = " ".join(str(x) for x in txt)
            txt = str(txt).strip()
            tcs = m.get("tool_calls") or []
            out.append(f"\n--- assistant step {step} ---")
            if txt:
                out.append(f"reasoning:\n{txt}")
            for tc in tcs:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    pass
                else:
                    import json as _j
                    args = _j.dumps(args)
                out.append(f"tool_call {name}: {args}")
        elif role == "tool":
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(x) for x in content)
            content = str(content)
            name = m.get("name", "?")
            out.append(f"tool_result {name}:\n{_truncate(content, max_chars_per_tool_result)}")
        else:
            txt = m.get("content", "")
            if txt:
                out.append(f"\n--- {role} ---\n{txt}")
    return "\n".join(out).strip() + "\n"
