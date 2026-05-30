"""TRACE citation parsing + shadow-trajectory write.

The generator emits, on the last line of its ``final_answer.reasoning``
field, ``<citations>[bullet-1, bullet-7]</citations>`` (or empty). The
wrapper:

1. Parses out the cited bullet ids and counts malformed tags.
2. Writes a shadow ``trajectory_graded.json`` with the citations tag
   stripped from the reasoning string. The vendor grader reads the
   shadow; the original ``trajectory.json`` is left intact for the
   reflector and curator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_CITATIONS_RE = re.compile(
    r"<citations>\s*\[\s*((?:bullet-\d+(?:\s*,\s*bullet-\d+)*)?)\s*\]\s*</citations>",
    re.IGNORECASE,
)
_LOOSE_CITATIONS_RE = re.compile(
    r"<citations\b[^>]*>(.*?)</citations>", re.IGNORECASE | re.DOTALL
)


@dataclass(frozen=True)
class CitationExtract:
    cited_bullet_ids: list[str]
    citations_present: bool
    citations_malformed_count: int
    stripped_reasoning: str | None


def _extract_from_reasoning(reasoning: str) -> CitationExtract:
    malformed = 0
    for m in _LOOSE_CITATIONS_RE.finditer(reasoning):
        if not _CITATIONS_RE.search(m.group(0)):
            malformed += 1

    matches = list(_CITATIONS_RE.finditer(reasoning))
    if not matches:
        return CitationExtract(
            cited_bullet_ids=[],
            citations_present=False,
            citations_malformed_count=malformed,
            stripped_reasoning=None,
        )
    last = matches[-1]
    inside = last.group(1).strip()
    ids: list[str] = []
    if inside:
        for token in inside.split(","):
            t = token.strip()
            if t and re.fullmatch(r"bullet-\d+", t):
                ids.append(t)

    stripped = (reasoning[: last.start()].rstrip() + reasoning[last.end():]).strip()
    return CitationExtract(
        cited_bullet_ids=ids,
        citations_present=True,
        citations_malformed_count=malformed,
        stripped_reasoning=stripped,
    )


def extract_and_strip_citations_from_trajectory(
    trajectory_path: Path,
) -> tuple[CitationExtract, dict | None]:
    """Read ``trajectory.json``, locate the last ``final_answer`` tool
    call, extract the ``<citations>`` tag from its ``reasoning`` arg, and
    return ``(extract, shadow_trajectory_dict_or_None)``.

    When no citations tag is present, the shadow trajectory is None
    (the grader will read the original trajectory directly).
    """
    traj = json.loads(trajectory_path.read_text(encoding="utf-8"))
    msgs = traj.get("messages") or []
    # Find the last assistant message that calls final_answer
    fa_idx: int | None = None
    fa_call_idx: int | None = None
    for i, m in enumerate(reversed(msgs)):
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        for j, tc in enumerate(tcs):
            fn = (tc.get("function") or {})
            if fn.get("name") == "final_answer":
                fa_idx = len(msgs) - 1 - i
                fa_call_idx = j
                break
        if fa_idx is not None:
            break
    if fa_idx is None or fa_call_idx is None:
        return CitationExtract([], False, 0, None), None

    raw_args = (msgs[fa_idx].get("tool_calls") or [])[fa_call_idx].get("function", {}).get("arguments", "")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return CitationExtract([], False, 0, None), None
    else:
        args = raw_args
    if not isinstance(args, dict):
        return CitationExtract([], False, 0, None), None
    reasoning = str(args.get("reasoning") or "")
    extract = _extract_from_reasoning(reasoning)
    if not extract.citations_present:
        return extract, None

    shadow = json.loads(trajectory_path.read_text(encoding="utf-8"))
    shadow_msgs = shadow["messages"]
    shadow_args = json.loads(
        shadow_msgs[fa_idx]["tool_calls"][fa_call_idx]["function"].get("arguments") or "{}"
    )
    shadow_args["reasoning"] = extract.stripped_reasoning or ""
    shadow_msgs[fa_idx]["tool_calls"][fa_call_idx]["function"]["arguments"] = json.dumps(shadow_args)
    return extract, shadow


def write_shadow_trajectory(shadow: dict, *, out_path: Path) -> Path:
    out_path.write_text(json.dumps(shadow, indent=2) + "\n", encoding="utf-8")
    return out_path
