"""Curator — single LLM call, parses <memory_updates> JSON ops, applies.

The :func:`curate` signature is intentionally narrow:
``(dynamic_ledger, task_prompt, trajectory, *, cfg)``. **NO**
``criteria``, **NO** ``score``, **NO** ``gt_bit``, **NO**
``expected_answer``, **NO** ``judge_rationale``. Load-bearing — see
``test_curator_signature_has_no_outcome``.

The ``<memory_updates>`` XML tag the curator emits is the Dynamic
Ledger's op-list format; the tag name is fixed and is not part of the
runtime's invocation surface.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from apex_agents_bench.dynamic_ledger.config import DynamicLedgerConfig
from apex_agents_bench.dynamic_ledger.dedup import is_too_similar_to_retrieved
from apex_agents_bench.dynamic_ledger.embeddings import EmbeddingClient
from apex_agents_bench.dynamic_ledger.entry import DynamicLedger, Entry

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def system_prompt_text() -> str:
    return _load_prompt("curator_system.txt")


def user_template_text() -> str:
    return _load_prompt("curator_user_template.txt")


VALID_OPS = ("CREATE", "UPDATE", "DELETE")
"""The three item-level operations defined in the Dynamic Ledger
approach (Jerry Gu, Sabrina Yen-Ko, Shurui Liu; mentor: Mirac Suzgun).
No CONSOLIDATE and no NO_OP — neither exists in the DL approach. The
wrapper's `<memory_updates>` parser silently drops any op outside this
set, so the curator can still emit the literal NO_OP / CONSOLIDATE
strings described in the prompt without poisoning the ledger."""


@dataclass
class CuratedOp:
    op: str
    section: str = ""
    content: str = ""
    source_problem: str = ""
    entry_id: str = ""


@dataclass
class CuratorResult:
    raw_response: str
    ops: list[CuratedOp]
    parse_error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0


_MEMORY_UPDATES_RE = re.compile(
    r"<memory_updates>\s*(\[.*?\])\s*</memory_updates>",
    re.DOTALL | re.IGNORECASE,
)


def parse_memory_updates(text: str) -> tuple[list[CuratedOp], str | None]:
    if not text:
        return [], "empty curator response"
    matches = list(_MEMORY_UPDATES_RE.finditer(text))
    if not matches:
        return [], "no <memory_updates> block found"
    block = matches[-1].group(1)
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        return [], f"json parse error: {exc.msg} at line {exc.lineno} col {exc.colno}"
    if not isinstance(data, list):
        return [], "json root is not a list"

    out: list[CuratedOp] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("op", "")).upper()
        if op not in VALID_OPS:
            continue
        try:
            if op == "CREATE":
                out.append(
                    CuratedOp(
                        op=op,
                        section=str(raw["section"]),
                        content=str(raw["content"]),
                        source_problem=str(raw["source_problem"]),
                    )
                )
            elif op == "UPDATE":
                out.append(CuratedOp(op=op, entry_id=str(raw["entry_id"]), content=str(raw["content"])))
            elif op == "DELETE":
                out.append(CuratedOp(op=op, entry_id=str(raw["entry_id"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out, None


@dataclass
class ApplyStats:
    create_committed: int = 0
    create_blocked: int = 0
    update: int = 0
    delete: int = 0
    skipped_invalid_entry_id: int = 0


def apply_ops(
    *,
    store: DynamicLedger,
    ops: list[CuratedOp],
    retrieved: list[Entry],
    embed: EmbeddingClient,
    cfg: DynamicLedgerConfig,
    current_ordinal: int,
) -> ApplyStats:
    """Apply ops in DELETE → UPDATE → CREATE order — the three edits
    defined by the Dynamic Ledger approach in the Dynamic Cheatsheet
    2.0 codebase. No CONSOLIDATE, no NO_OP."""
    stats = ApplyStats()

    deletes = [o for o in ops if o.op == "DELETE"]
    updates = [o for o in ops if o.op == "UPDATE"]
    creates = [o for o in ops if o.op == "CREATE"]

    for o in deletes:
        if store.soft_delete(o.entry_id, updated=current_ordinal):
            stats.delete += 1
        else:
            stats.skipped_invalid_entry_id += 1

    for o in updates:
        existing = store.get(o.entry_id)
        if existing is None or not existing.active:
            stats.skipped_invalid_entry_id += 1
            continue
        try:
            embs = embed.embed([o.content])
        except Exception:
            stats.skipped_invalid_entry_id += 1
            continue
        if store.update_content(
            o.entry_id, content=o.content, content_embedding=embs[0], updated=current_ordinal
        ):
            stats.update += 1

    for o in creates:
        try:
            embs = embed.embed([o.content, o.source_problem])
        except Exception:
            stats.skipped_invalid_entry_id += 1
            continue
        cand_emb, src_emb = embs[0], embs[1]
        blocked, _max, _by = is_too_similar_to_retrieved(
            candidate_embedding=cand_emb,
            retrieved=retrieved,
            threshold=cfg.create_time_similarity_threshold,
        )
        if blocked:
            stats.create_blocked += 1
            continue
        store.add(
            section=o.section,
            content=o.content,
            source_problem=o.source_problem,
            content_embedding=cand_emb,
            source_problem_embedding=src_emb,
            created=current_ordinal,
        )
        stats.create_committed += 1

    return stats


def _render_playbook(ledger: DynamicLedger) -> str:
    return json.dumps(ledger.serialize_for_curator(), ensure_ascii=False, indent=2)


def _build_user_message(
    *,
    dynamic_ledger: DynamicLedger,
    task_prompt: str,
    trajectory: str,
) -> str:
    template = user_template_text()
    return (
        template.replace("{rendered_active_playbook}", _render_playbook(dynamic_ledger))
        .replace("{task_prompt}", task_prompt)
        .replace("{rendered_trajectory}", trajectory)
    )


def curate(
    dynamic_ledger: DynamicLedger,
    task_prompt: str,
    trajectory: str,
    *,
    cfg: DynamicLedgerConfig,
) -> CuratorResult:
    """The single curator call. NO GROUND-TRUTH INPUTS.

    Positional inputs:
      - dynamic_ledger: the per-domain Dynamic Ledger (its active entries
        are rendered into the curator's user message as the playbook)
      - task_prompt: verbatim task prompt the agent saw WITHOUT the
        strategies-block injection
      - trajectory: rendered agent trajectory transcript (already
        truncated per ``cfg.trajectory_max_chars_per_tool_result``)

    Keyword-only:
      - cfg: DynamicLedgerConfig — `cfg.curator_model` is required
        (the runner fills it in from the active :class:`AgentProfile`
        so the curator and the agent share the same model).

    Returns a :class:`CuratorResult`. The load-bearing fidelity test
    ``test_curator_signature_has_no_outcome`` pins this signature.
    """
    import litellm

    if cfg.curator_model is None:
        raise ValueError(
            "DynamicLedgerConfig.curator_model is None — the runner must "
            "fill it from the active AgentProfile before calling curate(). "
            "See docs/DYNAMIC_LEDGER_PRD.md."
        )
    sys_msg = system_prompt_text()
    user_msg = _build_user_message(
        dynamic_ledger=dynamic_ledger,
        task_prompt=task_prompt,
        trajectory=trajectory,
    )
    started = time.time()
    kwargs: dict = {
        "model": cfg.curator_model,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": cfg.curator_temperature,
        "max_tokens": cfg.curator_max_tokens,
        "timeout": cfg.curator_timeout_seconds,
    }
    kwargs.update(cfg.curator_extra_args or {})
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

    ops, err = parse_memory_updates(content)
    return CuratorResult(
        raw_response=content,
        ops=ops,
        parse_error=err,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_seconds=round(elapsed, 2),
    )
