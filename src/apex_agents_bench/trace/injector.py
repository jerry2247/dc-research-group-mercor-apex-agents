"""Hook A: render the cheatsheet block + augment initial_messages.json.

The injection prefix is TRACE-specific: it instructs the agent to cite
the bullets it consults on the final line of its ``final_answer``
reasoning field. The citation tag is parsed and stripped before
grading.
"""

from __future__ import annotations

import json
from pathlib import Path

from apex_agents_bench.trace.bullet import Bullet

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_injection_prefix() -> str:
    return (_PROMPTS_DIR / "generator_injection_block.txt").read_text(encoding="utf-8")


def _truncate_bullet_content(content: str, cap: int) -> str:
    if cap <= 0 or len(content) <= cap:
        return content
    head = content[:cap]
    last_para = head.rfind("\n\n")
    if last_para > cap * 0.6:
        head = content[:last_para]
    return head + f"\n\n[... {len(content) - len(head):,} more chars in full bullet]"


def render_bullets_block(bullets: list[Bullet], *, max_chars_per_bullet: int = 3000) -> str:
    if not bullets:
        return "(no relevant strategy bullets yet)\n"
    parts: list[str] = []
    for b in bullets:
        body = _truncate_bullet_content(b.content, max_chars_per_bullet)
        parts.append(
            f"<bullet {b.bullet_id} section={b.section} helpful={b.helpful} "
            f"harmful={b.harmful} usage={b.usage}>\n{body}\n</bullet>"
        )
    return "\n\n".join(parts) + "\n"


def augment_initial_messages(
    initial_messages_path: Path,
    *,
    bullets: list[Bullet],
) -> str:
    """Read ``initial_messages.json``, prepend the cheatsheet block + the
    citation-instruction prefix to the USER message, write the file back."""
    raw = json.loads(initial_messages_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"unexpected initial_messages.json shape: expected list, got {type(raw).__name__}"
        )
    prefix_template = _load_injection_prefix()
    block = render_bullets_block(bullets)
    prefix = prefix_template.replace("{bullets_block}", block)
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
