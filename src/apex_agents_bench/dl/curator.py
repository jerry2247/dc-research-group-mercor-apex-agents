"""DL curator — the single LLM call, applies typed CRUD to the ledger.

This is the ONLY LLM call DL makes per task, and it runs AFTER the agent
(faithful to the original Dynamic Ledger's ``observe``). It sees the
retrieved entries (the editable window), the current task, and THIS
task's trajectory. It emits a ``<ledger_updates>`` JSON batch of CREATE /
UPDATE / DELETE operations.

NO ground-truth signal is consumed anywhere in this module — there is no
``score``, ``gt_correct``, ``criteria``, ``rubric``, ``expected_answer``,
``judge_rationale``, or ``task_id`` parameter. This narrow signature is a
load-bearing fidelity invariant of DL, enforced by ``test_dl_fidelity``.

Applying the ops is pure code: DELETE (soft) first so the window settles,
then UPDATE in place (re-embedding the content axis), then CREATE
(embedding both axes). There is NO create-time dedup — every well-formed
CREATE enters the ledger.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from apex_agents_bench.dl.config import DLConfig
from apex_agents_bench.dl.embeddings import EmbeddingClient
from apex_agents_bench.dl.entry import ENTRY_TYPES, DLEntry, DLLedger

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


VALID_OPS = ("CREATE", "UPDATE", "DELETE")


@dataclass
class CuratedOp:
    op: str
    type: str = ""
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


_LEDGER_RE = re.compile(
    r"<ledger_updates>\s*(\[.*\])\s*</ledger_updates>",
    re.DOTALL | re.IGNORECASE,
)


def parse_ledger_updates(text: str) -> tuple[list[CuratedOp], str | None]:
    """Parse the curator's ``<ledger_updates>[...]</ledger_updates>`` block.

    The JSON array must live inside the block; everything outside is
    ignored. Malformed entries are dropped so a partial response cannot
    poison the ledger. A CREATE whose ``type`` is not one of the five
    canonical entry types is dropped.
    """
    if not text:
        return [], "empty curator response"
    matches = list(_LEDGER_RE.finditer(text))
    if not matches:
        return [], "no <ledger_updates> block found"
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
                etype = str(raw["type"]).strip().lower()
                if etype not in ENTRY_TYPES:
                    continue
                out.append(
                    CuratedOp(
                        op=op,
                        type=etype,
                        content=str(raw["content"]),
                        source_problem=str(raw["source_problem"]),
                    )
                )
            elif op == "UPDATE":
                etype_raw = raw.get("type")
                etype = str(etype_raw).strip().lower() if etype_raw is not None else ""
                if etype and etype not in ENTRY_TYPES:
                    etype = ""
                eid = raw.get("entry_id")
                if eid is None:
                    eid = raw.get("id")  # accept the short alias too
                if eid is None:
                    continue
                out.append(
                    CuratedOp(
                        op=op,
                        entry_id=str(eid),
                        content=str(raw["content"]),
                        type=etype,
                    )
                )
            elif op == "DELETE":
                eid = raw.get("entry_id")
                if eid is None:
                    eid = raw.get("id")  # accept the short alias too
                if eid is None:
                    continue
                out.append(CuratedOp(op=op, entry_id=str(eid)))
        except (KeyError, TypeError, ValueError):
            continue
    return out, None


@dataclass
class ApplyStats:
    create: int = 0
    update: int = 0
    delete: int = 0
    skipped_invalid_entry_id: int = 0


def apply_ops(
    *,
    ledger: DLLedger,
    ops: list[CuratedOp],
    embed: EmbeddingClient,
    current_ordinal: int,
) -> ApplyStats:
    """Apply the curator's ops to the ledger deterministically.

    Order: DELETE (soft) → UPDATE (re-embed content) → CREATE (embed both
    axes). No create-time dedup. UPDATE/DELETE against an unknown or
    inactive id are counted as ``skipped_invalid_entry_id`` and not
    applied.
    """
    stats = ApplyStats()
    deletes = [o for o in ops if o.op == "DELETE"]
    updates = [o for o in ops if o.op == "UPDATE"]
    creates = [o for o in ops if o.op == "CREATE"]

    for o in deletes:
        if ledger.soft_delete(o.entry_id, updated=current_ordinal):
            stats.delete += 1
        else:
            stats.skipped_invalid_entry_id += 1

    for o in updates:
        existing = ledger.get(o.entry_id)
        if existing is None or not existing.active:
            stats.skipped_invalid_entry_id += 1
            continue
        try:
            embs = embed.embed([o.content])
        except Exception:
            stats.skipped_invalid_entry_id += 1
            continue
        if ledger.update_content(
            o.entry_id,
            content=o.content,
            content_embedding=embs[0],
            updated=current_ordinal,
            type=(o.type or None),
        ):
            stats.update += 1

    for o in creates:
        try:
            embs = embed.embed([o.content, o.source_problem])
        except Exception:
            stats.skipped_invalid_entry_id += 1
            continue
        ledger.add(
            type=o.type,
            content=o.content,
            source_problem=o.source_problem,
            content_embedding=embs[0],
            source_problem_embedding=embs[1],
            created=current_ordinal,
        )
        stats.create += 1

    return stats


def _render_retrieved(retrieved: list[DLEntry]) -> str:
    from apex_agents_bench.dl.formatting import render_entries_for_curator

    return render_entries_for_curator(retrieved)


def _build_user_message(
    *,
    retrieved: list[DLEntry],
    task_prompt: str,
    trajectory: str,
) -> str:
    template = _load_prompt("curator_prompt.txt")
    return (
        template.replace("{retrieved_entries}", _render_retrieved(retrieved))
        .replace("{task_prompt}", task_prompt)
        .replace("{rendered_trajectory}", trajectory)
    )


def curate(
    ledger: DLLedger,
    retrieved: list[DLEntry],
    task_prompt: str,
    trajectory: str,
    *,
    cfg: DLConfig,
) -> CuratorResult:
    """The DL curator LLM call.

    Sees the retrieved entries, the current task, and THIS task's
    trajectory. Consumes NO ground-truth signal. Returns the parsed op
    list, applied by :func:`apply_ops`.

    Faithful to DC-RS's synthesizer call shape: the single
    ``curator_prompt.txt`` template contains the whole prompt and the
    LiteLLM call uses a single user message (``system=None``).
    """
    import litellm

    if cfg.curator_model is None:
        raise RuntimeError(
            "DLConfig.curator_model is None — the runner must fill it from "
            "the active AgentProfile before calling curate(). See docs/DL_PRD.md."
        )
    user_msg = _build_user_message(
        retrieved=retrieved,
        task_prompt=task_prompt,
        trajectory=trajectory,
    )
    started = time.time()
    kwargs: dict = {
        "model": cfg.curator_model,
        "messages": [
            {"role": "user", "content": user_msg},
        ],
        "temperature": cfg.curator_temperature,
        "max_tokens": cfg.curator_max_tokens,
        "timeout": cfg.curator_timeout_seconds,
    }
    # The profile's extra args may carry their own temperature / timeout /
    # reasoning_effort etc. Using ``dict.update`` lets the profile override
    # the defaults without raising a kwarg-collision TypeError, which would
    # silently abort the LLM call before LiteLLM is even reached.
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
        else (usage_obj.get("prompt_tokens") if isinstance(usage_obj, dict) else 0) or 0
    )
    completion_tokens = int(
        getattr(usage_obj, "completion_tokens", None)
        if getattr(usage_obj, "completion_tokens", None) is not None
        else (usage_obj.get("completion_tokens") if isinstance(usage_obj, dict) else 0) or 0
    )
    ops, err = parse_ledger_updates(content)
    return CuratorResult(
        raw_response=content,
        ops=ops,
        parse_error=err,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_seconds=round(elapsed, 2),
    )
