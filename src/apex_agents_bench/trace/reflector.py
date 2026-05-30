"""Reflector — single LLM call, proposes operations on the cheatsheet.

The TRACE reflector sees the GROUND-TRUTH correctness bit (boolean
``criteria_passed == criteria_total``) and proposes operations the
curator may apply. The reflector is intentionally permissive (it
suggests freely); the curator is conservative (it filters and applies).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from apex_agents_bench.trace.bullet import TraceLedger
from apex_agents_bench.trace.config import TraceConfig

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class ReflectorProposal:
    op: str
    section: str = ""
    content: str = ""
    source_problem: str = ""
    bullet_id: str = ""
    bullet_ids: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ReflectorResult:
    raw_response: str
    proposals: list[ReflectorProposal]
    parse_error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0


_REFLECTOR_RE = re.compile(
    r"<reflector_proposals>\s*(\[.*?\])\s*</reflector_proposals>",
    re.DOTALL | re.IGNORECASE,
)


VALID_OPS = ("CREATE", "UPDATE", "DELETE", "CONSOLIDATE", "NO_OP")


def parse_reflector_proposals(text: str) -> tuple[list[ReflectorProposal], str | None]:
    if not text:
        return [], "empty reflector response"
    matches = list(_REFLECTOR_RE.finditer(text))
    if not matches:
        return [], "no <reflector_proposals> block found"
    block = matches[-1].group(1)
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        return [], f"json parse error: {exc.msg} at line {exc.lineno} col {exc.colno}"
    if not isinstance(data, list):
        return [], "json root is not a list"
    out: list[ReflectorProposal] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("op", "")).upper()
        if op not in VALID_OPS:
            continue
        try:
            if op == "CREATE":
                out.append(
                    ReflectorProposal(
                        op=op,
                        section=str(raw["section"]),
                        content=str(raw["content"]),
                        source_problem=str(raw["source_problem"]),
                    )
                )
            elif op == "UPDATE":
                out.append(
                    ReflectorProposal(
                        op=op, bullet_id=str(raw["bullet_id"]), content=str(raw["content"])
                    )
                )
            elif op == "DELETE":
                out.append(ReflectorProposal(op=op, bullet_id=str(raw["bullet_id"])))
            elif op == "CONSOLIDATE":
                bullet_ids_raw = raw.get("bullet_ids") or []
                if not isinstance(bullet_ids_raw, list):
                    continue
                bullet_ids = [str(x) for x in bullet_ids_raw if isinstance(x, str)]
                if len(bullet_ids) < 2:
                    continue
                out.append(
                    ReflectorProposal(
                        op=op,
                        bullet_ids=bullet_ids,
                        section=str(raw["section"]),
                        content=str(raw["content"]),
                        source_problem=str(raw["source_problem"]),
                    )
                )
            elif op == "NO_OP":
                out.append(ReflectorProposal(op=op, reason=str(raw.get("reason", ""))))
        except (KeyError, TypeError, ValueError):
            continue
    return out, None


def _render_cheatsheet(ledger: TraceLedger) -> str:
    return json.dumps(ledger.serialize_for_llm(), ensure_ascii=False, indent=2)


def _build_reflector_user_message(
    *,
    ledger: TraceLedger,
    task_prompt: str,
    trajectory: str,
    cited_bullet_ids: list[str],
    gt_correct: bool,
) -> str:
    template = _load_prompt("reflector_user_template.txt")
    return (
        template.replace("{rendered_active_cheatsheet}", _render_cheatsheet(ledger))
        .replace("{task_prompt}", task_prompt)
        .replace("{rendered_trajectory}", trajectory)
        .replace("{cited_bullets_json}", json.dumps(cited_bullet_ids))
        .replace("{gt_correct}", "true" if gt_correct else "false")
    )


def reflect(
    ledger: TraceLedger,
    task_prompt: str,
    trajectory: str,
    cited_bullet_ids: list[str],
    gt_correct: bool,
    *,
    cfg: TraceConfig,
) -> ReflectorResult:
    """The reflector LLM call. Sees the GROUND-TRUTH ``gt_correct`` bit.

    The reflector is the TRACE pipeline's first stage; the curator
    consumes its proposals. The boolean ``gt_correct`` is the only
    grading-derived input both calls receive — per TRACE paper. No
    per-criterion scores, no rubric text, no expected answer.
    """
    import litellm

    if cfg.reflector_model is None:
        raise ValueError(
            "TraceConfig.reflector_model is None — the runner must fill it "
            "from the active AgentProfile before calling reflect(). See "
            "docs/TRACE_PRD.md."
        )
    sys_msg = _load_prompt("reflector_system.txt")
    user_msg = _build_reflector_user_message(
        ledger=ledger,
        task_prompt=task_prompt,
        trajectory=trajectory,
        cited_bullet_ids=cited_bullet_ids,
        gt_correct=gt_correct,
    )
    started = time.time()
    kwargs: dict = {
        "model": cfg.reflector_model,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": cfg.reflector_temperature,
        "max_tokens": cfg.reflector_max_tokens,
        "timeout": cfg.reflector_timeout_seconds,
    }
    # The profile's extra args may carry their own temperature / timeout /
    # reasoning_effort etc. Using ``dict.update`` lets the profile override
    # the defaults without raising a kwarg-collision TypeError, which would
    # silently abort the LLM call before LiteLLM is even reached.
    kwargs.update(cfg.model_extra_args or {})
    resp = litellm.completion(**kwargs)
    elapsed = time.time() - started

    choices = getattr(resp, "choices", None) or resp["choices"]
    msg = choices[0].message
    content = getattr(msg, "content", None) or msg["content"]
    if not isinstance(content, str):
        content = str(content)
    usage_obj = getattr(resp, "usage", None) or resp.get("usage", {}) or {}
    prompt_tokens = int(
        getattr(usage_obj, "prompt_tokens", None)
        if getattr(usage_obj, "prompt_tokens", None) is not None
        else (usage_obj.get("prompt_tokens") if isinstance(usage_obj, dict) else 0)
        or 0
    )
    completion_tokens = int(
        getattr(usage_obj, "completion_tokens", None)
        if getattr(usage_obj, "completion_tokens", None) is not None
        else (usage_obj.get("completion_tokens") if isinstance(usage_obj, dict) else 0)
        or 0
    )
    proposals, err = parse_reflector_proposals(content)
    return ReflectorResult(
        raw_response=content,
        proposals=proposals,
        parse_error=err,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_seconds=round(elapsed, 2),
    )
