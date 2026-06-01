"""Integration lifecycle test for DL.

Drives the DL pieces in the SAME order ``runner.run_single_task`` calls
them — Hook A (embed → dual-retrieve → inject) then Hook B (curate →
apply → persist) — across two tasks plus a resume, with fake embeddings
and a fake curator LLM. This is the strongest in-process evidence that
the integrated flow works without a Docker/agent/grading run:

  * task 1 starts from an empty ledger, injects nothing, and the curator
    CREATEs entries;
  * task 2 RETRIEVES the task-1 entries, injects them into the agent's
    user message, and the curator does a MIX of UPDATE + DELETE + CREATE;
  * a fresh runtime over the same run dir RESUMES the exact final state.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from apex_agents_bench.dl import (
    apply_ops as dl_apply_ops,
)
from apex_agents_bench.dl import (
    augment_initial_messages as dl_inject,
)
from apex_agents_bench.dl import (
    curate as dl_curate,
)
from apex_agents_bench.dl import (
    render_trajectory_for_curator as dl_render,
)
from apex_agents_bench.dl import (
    retrieve as dl_retrieve,
)
from apex_agents_bench.dl.config import DLConfig
from apex_agents_bench.dl.runtime import DLRuntime


class _Embed:
    """All-ones vectors → every active entry is equally retrievable, so
    top_k selection is exercised without depending on semantics."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _fake_litellm(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> None:
    """Return canned curator responses in order, one per completion call."""
    seq = iter(responses)

    def fake_completion(**kwargs):
        content = next(seq)
        msg = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=msg)
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=2)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=fake_completion))


def _write_initial(path: Path, task_prompt: str) -> None:
    path.write_text(
        json.dumps(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": task_prompt},
            ]
        ),
        encoding="utf-8",
    )


def _traj(answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "doing it",
                "tool_calls": [{"function": {"name": "run", "arguments": "{}"}}],
            },
            {"role": "tool", "name": "run", "content": answer},
        ]
    }


def _run_one_task(
    rt: DLRuntime,
    *,
    domain: str,
    task_prompt: str,
    trajectory: dict,
    initial_path: Path,
    cfg: DLConfig,
) -> tuple[list[str], str]:
    """Mirror run_single_task's Hook A + Hook B for one task. Returns the
    retrieved entry ids and the (possibly augmented) user message."""
    _write_initial(initial_path, task_prompt)

    # --- Hook A: embed → dual-retrieve → inject ---
    ledger = rt.ledger_for(domain)
    q_emb = rt.embed.embed([task_prompt])[0]
    retrieved = dl_retrieve(ledger, query_embedding=q_emb, k=rt.cfg.top_k)
    if retrieved:
        dl_inject(initial_path, entries=retrieved)
    user_msg = json.loads(initial_path.read_text(encoding="utf-8"))[1]["content"]

    # --- Hook B: curate → apply → persist ---
    ordinal = rt.current_ordinal_for(domain)
    rendered = dl_render(
        trajectory, max_chars_per_tool_result=cfg.trajectory_max_chars_per_tool_result
    )
    cur = dl_curate(ledger, retrieved, task_prompt, rendered, cfg=cfg)
    dl_apply_ops(ledger=ledger, ops=cur.ops, embed=rt.embed, current_ordinal=ordinal)
    rt.persist(domain)

    return [e.entry_id for e in retrieved], user_msg


def test_two_task_lifecycle_plus_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    domain = "Finance"
    cfg = DLConfig(enabled=True, curator_model="openai/gpt-5.5", curator_extra_args={}, top_k=3)

    # Task 1 creates two entries; task 2 updates entry-1, deletes entry-2,
    # and creates a new entry-3.
    task1_ops = (
        "<ledger_updates>["
        '{"op":"CREATE","type":"snippet","content":"<description>snip</description>",'
        '"source_problem":"reading a table"},'
        '{"op":"CREATE","type":"pitfall","content":"<description>trap</description>",'
        '"source_problem":"delivering the answer"}'
        "]</ledger_updates>"
    )
    task2_ops = (
        "<ledger_updates>["
        '{"op":"UPDATE","entry_id":"entry-1","content":"<description>snip v2</description>"},'
        '{"op":"DELETE","entry_id":"entry-2"},'
        '{"op":"CREATE","type":"formula","content":"<description>f</description>",'
        '"source_problem":"a recurring computation"}'
        "]</ledger_updates>"
    )
    _fake_litellm(monkeypatch, [task1_ops, task2_ops])

    rt = DLRuntime.create(cfg=cfg, run_dir=tmp_path, embed=_Embed())

    # --- Task 1 ---
    retr1, msg1 = _run_one_task(
        rt,
        domain=domain,
        task_prompt="TASK ONE",
        trajectory=_traj("ok1"),
        initial_path=tmp_path / "t1_initial.json",
        cfg=cfg,
    )
    assert retr1 == []  # empty ledger on the first task
    assert msg1 == "TASK ONE"  # no injection happened
    led = rt.ledger_for(domain)
    assert {e.entry_id for e in led.active_entries()} == {"entry-1", "entry-2"}

    # --- Task 2 ---
    retr2, msg2 = _run_one_task(
        rt,
        domain=domain,
        task_prompt="TASK TWO",
        trajectory=_traj("ok2"),
        initial_path=tmp_path / "t2_initial.json",
        cfg=cfg,
    )
    # retrieval surfaced the task-1 entries, and they were injected
    assert set(retr2) == {"entry-1", "entry-2"}
    assert msg2.endswith("TASK TWO")
    assert "snip" in msg2  # entry-1 content injected
    assert "reference" in msg2.lower()  # consult-don't-obey framing
    # curator mix applied: entry-1 updated, entry-2 deleted, entry-3 created
    led = rt.ledger_for(domain)
    active = {e.entry_id for e in led.active_entries()}
    assert active == {"entry-1", "entry-3"}
    assert led.get("entry-1").content == "<description>snip v2</description>"
    assert led.get("entry-2").active is False
    assert led.get("entry-3").type == "formula"

    # --- Resume: a fresh runtime reloads the exact final state ---
    rt2 = DLRuntime.create(cfg=cfg, run_dir=tmp_path, embed=_Embed())
    led2 = rt2.ledger_for(domain)
    assert {e.entry_id for e in led2.active_entries()} == {"entry-1", "entry-3"}
    assert led2.get("entry-2") is not None and led2.get("entry-2").active is False
    # next created id does not collide with the soft-deleted entry-2
    assert led2.next_entry_ord == led.next_entry_ord


def test_lifecycle_isolates_domains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DLConfig(enabled=True, curator_model="openai/gpt-5.5", curator_extra_args={}, top_k=3)
    create_op = (
        "<ledger_updates>["
        '{"op":"CREATE","type":"snippet","content":"<description>x</description>",'
        '"source_problem":"sp"}'
        "]</ledger_updates>"
    )
    _fake_litellm(monkeypatch, [create_op])
    rt = DLRuntime.create(cfg=cfg, run_dir=tmp_path, embed=_Embed())

    # Run one task in Finance; Law must remain empty and retrieve nothing.
    _run_one_task(
        rt,
        domain="Finance",
        task_prompt="FIN",
        trajectory=_traj("ok"),
        initial_path=tmp_path / "fin.json",
        cfg=cfg,
    )
    law_ledger = rt.ledger_for("Law")
    law_q = rt.embed.embed(["LAW"])[0]
    assert dl_retrieve(law_ledger, query_embedding=law_q, k=3) == []
    assert len(rt.ledger_for("Finance").active_entries()) == 1
