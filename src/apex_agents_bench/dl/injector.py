"""Hook A: render the retrieved entries block + augment initial_messages.json.

The Archipelago agent reads ``initial_messages.json`` to bootstrap its
conversation. We modify the USER message in place, prepending the DL
injection block (which wraps the retrieved typed entries). The SYSTEM
message is left untouched (a fidelity test asserts this).

The injection prefix is the DC-RS-style consult-don't-obey framing; DL
adds NO citation instruction (unlike TRACE).
"""

from __future__ import annotations

import json
from pathlib import Path

from apex_agents_bench.dl.entry import DLEntry
from apex_agents_bench.dl.formatting import render_entries_for_generator

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_EMPTY = "(empty)"


def _load_injection_block() -> str:
    return (_PROMPTS_DIR / "generator_injection_block.txt").read_text(encoding="utf-8")


def augment_initial_messages(
    initial_messages_path: Path,
    *,
    entries: list[DLEntry],
) -> str:
    """Read ``initial_messages.json``, prepend the entries block to the USER
    message, write the file back in place. Returns the prefix that was
    prepended (for diagnostics).

    If ``entries`` is empty, the file is left untouched and the empty string
    is returned so the agent sees byte-identical content to the baseline
    path on the first tasks in a domain.
    """
    block = render_entries_for_generator(entries)
    if not block.strip() or block.strip() == _EMPTY:
        return ""

    raw = json.loads(initial_messages_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"unexpected initial_messages.json shape: expected list, got {type(raw).__name__}"
        )
    block_template = _load_injection_block()
    prefix = block_template.replace("{entries_block}", block)
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
        raise ValueError("no user message found in initial_messages.json — cannot inject entries")
    initial_messages_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return prefix
