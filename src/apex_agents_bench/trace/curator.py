"""TRACE curator — second LLM call, applies operations to the cheatsheet.

The TRACE curator receives the reflector's proposals alongside the
cheatsheet, problem, trajectory, and GT bit. It returns a final
``<cheatsheet_updates>`` op list which the wrapper applies to the
ledger.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from apex_agents_bench.trace.bullet import Bullet, TraceLedger
from apex_agents_bench.trace.config import TraceConfig
from apex_agents_bench.trace.dedup import is_too_similar_to_retrieved
from apex_agents_bench.trace.embeddings import EmbeddingClient
from apex_agents_bench.trace.reflector import ReflectorProposal

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


VALID_OPS = ("CREATE", "UPDATE", "DELETE", "CONSOLIDATE", "NO_OP")


@dataclass
class CuratedOp:
    op: str
    section: str = ""
    content: str = ""
    source_problem: str = ""
    bullet_id: str = ""
    bullet_ids: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class CuratorResult:
    raw_response: str
    ops: list[CuratedOp]
    parse_error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0


_CURATOR_RE = re.compile(
    r"<cheatsheet_updates>\s*(\[.*?\])\s*</cheatsheet_updates>",
    re.DOTALL | re.IGNORECASE,
)


def parse_cheatsheet_updates(text: str) -> tuple[list[CuratedOp], str | None]:
    if not text:
        return [], "empty curator response"
    matches = list(_CURATOR_RE.finditer(text))
    if not matches:
        return [], "no <cheatsheet_updates> block found"
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
                out.append(
                    CuratedOp(op=op, bullet_id=str(raw["bullet_id"]), content=str(raw["content"]))
                )
            elif op == "DELETE":
                out.append(CuratedOp(op=op, bullet_id=str(raw["bullet_id"])))
            elif op == "CONSOLIDATE":
                bullet_ids_raw = raw.get("bullet_ids") or []
                if not isinstance(bullet_ids_raw, list):
                    continue
                bullet_ids = [str(x) for x in bullet_ids_raw if isinstance(x, str)]
                if len(bullet_ids) < 2:
                    continue
                out.append(
                    CuratedOp(
                        op=op,
                        bullet_ids=bullet_ids,
                        section=str(raw["section"]),
                        content=str(raw["content"]),
                        source_problem=str(raw["source_problem"]),
                    )
                )
            elif op == "NO_OP":
                out.append(CuratedOp(op=op, reason=str(raw.get("reason", ""))))
        except (KeyError, TypeError, ValueError):
            continue
    return out, None


@dataclass
class ApplyStats:
    create_committed: int = 0
    create_blocked: int = 0
    update: int = 0
    delete: int = 0
    consolidate: int = 0
    no_op: bool = False
    skipped_invalid_bullet_id: int = 0


def apply_ops(
    *,
    store: TraceLedger,
    ops: list[CuratedOp],
    retrieved: list[Bullet],
    embed: EmbeddingClient,
    cfg: TraceConfig,
    current_ordinal: int,
) -> ApplyStats:
    stats = ApplyStats()
    deletes = [o for o in ops if o.op == "DELETE"]
    consolidates = [o for o in ops if o.op == "CONSOLIDATE"]
    updates = [o for o in ops if o.op == "UPDATE"]
    creates = [o for o in ops if o.op == "CREATE"]
    if any(o.op == "NO_OP" for o in ops):
        stats.no_op = True

    for o in deletes:
        if store.soft_delete(o.bullet_id, updated=current_ordinal):
            stats.delete += 1
        else:
            stats.skipped_invalid_bullet_id += 1

    for o in consolidates:
        present: list[Bullet] = []
        for bid in o.bullet_ids:
            b = store.get(bid)
            if b is not None and b.active:
                present.append(b)
        if len(present) < 2:
            stats.skipped_invalid_bullet_id += 1
            continue
        try:
            embs = embed.embed([o.content, o.source_problem])
        except Exception:
            stats.skipped_invalid_bullet_id += 1
            continue
        cand_emb, src_emb = embs[0], embs[1]
        dedup_candidates = [
            b for b in store.active_bullets() if b.bullet_id not in set(o.bullet_ids)
        ]
        blocked, _m, _by = is_too_similar_to_retrieved(
            candidate_embedding=cand_emb,
            retrieved=dedup_candidates,
            threshold=cfg.create_time_similarity_threshold,
        )
        if blocked:
            stats.create_blocked += 1
            continue
        # Sum counters from sources (TRACE convention)
        total_helpful = sum(b.helpful for b in present)
        total_harmful = sum(b.harmful for b in present)
        total_usage = sum(b.usage for b in present)
        for b in present:
            store.soft_delete(b.bullet_id, updated=current_ordinal)
        new = store.add(
            section=o.section,
            content=o.content,
            source_problem=o.source_problem,
            content_embedding=cand_emb,
            source_problem_embedding=src_emb,
            created=current_ordinal,
        )
        store.bullets[new.bullet_id] = new.model_copy(
            update={"helpful": total_helpful, "harmful": total_harmful, "usage": total_usage}
        )
        stats.consolidate += 1

    for o in updates:
        existing = store.get(o.bullet_id)
        if existing is None or not existing.active:
            stats.skipped_invalid_bullet_id += 1
            continue
        try:
            embs = embed.embed([o.content])
        except Exception:
            stats.skipped_invalid_bullet_id += 1
            continue
        if store.update_content(
            o.bullet_id, content=o.content, content_embedding=embs[0], updated=current_ordinal
        ):
            stats.update += 1

    for o in creates:
        try:
            embs = embed.embed([o.content, o.source_problem])
        except Exception:
            stats.skipped_invalid_bullet_id += 1
            continue
        cand_emb, src_emb = embs[0], embs[1]
        blocked, _m, _by = is_too_similar_to_retrieved(
            candidate_embedding=cand_emb,
            retrieved=store.active_bullets(),
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


def _render_cheatsheet(ledger: TraceLedger) -> str:
    return json.dumps(ledger.serialize_for_llm(), ensure_ascii=False, indent=2)


def _render_proposals(proposals: list[ReflectorProposal]) -> str:
    out = []
    for p in proposals:
        d: dict = {"op": p.op}
        if p.bullet_id:
            d["bullet_id"] = p.bullet_id
        if p.bullet_ids:
            d["bullet_ids"] = p.bullet_ids
        if p.section:
            d["section"] = p.section
        if p.content:
            d["content"] = p.content
        if p.source_problem:
            d["source_problem"] = p.source_problem
        if p.reason:
            d["reason"] = p.reason
        out.append(d)
    return json.dumps(out, ensure_ascii=False, indent=2)


def _build_user_message(
    *,
    ledger: TraceLedger,
    task_prompt: str,
    trajectory: str,
    cited_bullet_ids: list[str],
    gt_correct: bool,
    reflector_proposals: list[ReflectorProposal],
) -> str:
    template = _load_prompt("curator_user_template.txt")
    return (
        template.replace("{rendered_active_cheatsheet}", _render_cheatsheet(ledger))
        .replace("{task_prompt}", task_prompt)
        .replace("{rendered_trajectory}", trajectory)
        .replace("{cited_bullets_json}", json.dumps(cited_bullet_ids))
        .replace("{gt_correct}", "true" if gt_correct else "false")
        .replace("{reflector_proposals_json}", _render_proposals(reflector_proposals))
    )


def curate(
    ledger: TraceLedger,
    task_prompt: str,
    trajectory: str,
    cited_bullet_ids: list[str],
    gt_correct: bool,
    reflector_proposals: list[ReflectorProposal],
    *,
    cfg: TraceConfig,
) -> CuratorResult:
    """The TRACE curator LLM call.

    Sees the GROUND-TRUTH ``gt_correct`` bit (boolean
    ``criteria_passed == criteria_total``) AND the reflector's proposed
    operations. Returns the final op list, applied by ``apply_ops``.
    """
    import litellm

    if cfg.curator_model is None:
        raise ValueError(
            "TraceConfig.curator_model is None — the runner must fill it "
            "from the active AgentProfile before calling curate(). See "
            "docs/TRACE_PRD.md."
        )
    sys_msg = _load_prompt("curator_system.txt")
    user_msg = _build_user_message(
        ledger=ledger,
        task_prompt=task_prompt,
        trajectory=trajectory,
        cited_bullet_ids=cited_bullet_ids,
        gt_correct=gt_correct,
        reflector_proposals=reflector_proposals,
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
    ops, err = parse_cheatsheet_updates(content)
    return CuratorResult(
        raw_response=content,
        ops=ops,
        parse_error=err,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_seconds=round(elapsed, 2),
    )
