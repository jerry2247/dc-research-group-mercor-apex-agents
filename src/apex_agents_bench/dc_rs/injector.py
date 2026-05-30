"""Hook A: inject the synthesized cheatsheet into initial_messages.json.

The Archipelago agent reads ``initial_messages.json`` to bootstrap its
conversation. We modify the USER message in place, prepending the DC-RS
injection block (which wraps the synthesized cheatsheet). The SYSTEM
message is left untouched (a fidelity test asserts this).

It substitutes the whole synthesized ``{cheatsheet}`` text into
``prompts/generator_injection_block.txt`` and prepends the result to
the user message.
"""

from __future__ import annotations

import json
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_EMPTY_CHEATSHEET = "(empty)"


def _load_injection_block() -> str:
    return (_PROMPTS_DIR / "generator_injection_block.txt").read_text(encoding="utf-8")


def augment_initial_messages(
    initial_messages_path: Path,
    *,
    cheatsheet: str,
) -> str:
    """Read ``initial_messages.json``, prepend the cheatsheet block to the
    USER message, write the file back in place. Returns the prefix that
    was prepended (for diagnostics).

    If ``cheatsheet`` is empty or the literal ``"(empty)"``, the file is
    left untouched and the empty string is returned so the agent sees
    byte-identical content to the baseline path.
    """
    if not cheatsheet.strip() or cheatsheet.strip() == _EMPTY_CHEATSHEET:
        return ""

    raw = json.loads(initial_messages_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"unexpected initial_messages.json shape: expected list, got {type(raw).__name__}"
        )
    block_template = _load_injection_block()
    prefix = block_template.replace("{cheatsheet}", cheatsheet)
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
            "no user message found in initial_messages.json — cannot inject cheatsheet"
        )
    initial_messages_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return prefix
