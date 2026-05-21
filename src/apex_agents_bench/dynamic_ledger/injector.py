"""Hook A: render the strategies block + augment initial_messages.json.

The Archipelago agent reads ``initial_messages.json`` to bootstrap its
conversation. We modify the USER message in place, prepending the
Dynamic Ledger strategies block + the injection prefix. The SYSTEM
message is left untouched (a fidelity test asserts this).
"""

from __future__ import annotations

import json
from pathlib import Path

from apex_agents_bench.dynamic_ledger.entry import Entry

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_injection_prefix() -> str:
    return (_PROMPTS_DIR / "generator_injection_block.txt").read_text(encoding="utf-8")


def _truncate_entry_content(content: str, cap: int) -> str:
    """Soft-cap the rendered content. We keep the first ``cap`` characters
    plus a marker telling the agent more exists in the full entry. The
    curator still saves the FULL content on disk; only the at-inject view
    is capped, to avoid blowing the agent's context with 5x13KB entries."""
    if cap <= 0 or len(content) <= cap:
        return content
    # Try to break on a paragraph boundary to keep the truncation clean.
    head = content[:cap]
    last_para = head.rfind("\n\n")
    if last_para > cap * 0.6:
        head = content[:last_para]
    return head + f"\n\n[... {len(content) - len(head):,} more chars in full entry; the head above contains the trigger, critical read, and the workflow opening]"


def render_entries_block(entries: list[Entry], *, max_chars_per_entry: int = 3000) -> str:
    """Render retrieved entries as one ``<entry ...>...</entry>`` block per
    entry, separated by blank lines. PRD §7.1 shape (same on both PRDs).

    ``max_chars_per_entry`` soft-caps each rendered entry. The agent reads
    the strategies block as part of its user prompt; uncapped retrieval of
    several long elaborate entries was observed (2026-05-20) to push the
    user message past 60 KB and induce reasoning-loop failures on
    grok-4.3-high. Setting <= 0 disables the cap.
    """
    if not entries:
        return "(no relevant prior notes)\n"
    parts: list[str] = []
    for e in entries:
        body = _truncate_entry_content(e.content, max_chars_per_entry)
        parts.append(f"<entry {e.entry_id} section={e.section}>\n{body}\n</entry>")
    return "\n\n".join(parts) + "\n"


def augment_initial_messages(
    initial_messages_path: Path,
    *,
    entries: list[Entry],
) -> str:
    """Read ``initial_messages.json``, prepend the strategies block to the
    USER message, write the file back in place. Returns the prefix that
    was prepended (for diagnostics)."""
    raw = json.loads(initial_messages_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"unexpected initial_messages.json shape: expected list, got {type(raw).__name__}"
        )
    prefix_template = _load_injection_prefix()
    block = render_entries_block(entries)
    prefix = prefix_template.replace("{entries_block}", block)
    augmented = False
    for msg in raw:
        if isinstance(msg, dict) and msg.get("role") == "user":
            original = msg.get("content", "") or ""
            if not isinstance(original, str):
                continue
            msg["content"] = prefix + original
            augmented = True
            break
    if not augmented:
        raise ValueError(
            "no user message found in initial_messages.json — cannot inject strategies"
        )
    initial_messages_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return prefix
