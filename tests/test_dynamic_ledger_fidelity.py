"""Load-bearing fidelity tests for the apex-agents-bench Dynamic Ledger v3."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) The load-bearing "no GT" signature invariant
# ---------------------------------------------------------------------------


def test_curator_signature_has_no_outcome() -> None:
    from apex_agents_bench.dynamic_ledger import curate

    sig = inspect.signature(curate)
    param_names = list(sig.parameters.keys())
    forbidden = (
        "criteria",
        "score",
        "scores",
        "gt_bit",
        "gt_correct_bit",
        "expected_answer",
        "expected",
        "gold",
        "gold_response",
        "judge_rationale",
        "judge_reason",
        "rubric",
        "autorating",
        "criteria_passed",
        "criteria_total",
        "verifier_result",
        "final_score",
    )
    leaks = [p for p in param_names if any(f in p.lower() for f in forbidden)]
    assert not leaks, f"curate() signature contains GT-leaking param(s): {leaks}"

    assert param_names == [
        "dynamic_ledger",
        "task_prompt",
        "trajectory",
        "cfg",
    ], f"unexpected curate() signature: {param_names}"


# ---------------------------------------------------------------------------
# 2) CSV schema invariants
# ---------------------------------------------------------------------------


def test_dynamic_ledger_off_csv_schema_unchanged() -> None:
    """With the Dynamic Ledger off, ``append_row`` writes the baseline ``CSV_HEADERS``."""
    from apex_agents_bench.runner import CSV_HEADERS

    expected = [
        "task_id",
        "domain",
        "world_id",
        "status",
        "agent_status",
        "final_score",
        "criteria_passed",
        "criteria_total",
        "steps_used",
        "wall_time_seconds",
        "agent_prompt_tokens",
        "agent_completion_tokens",
        "agent_total_tokens",
        "agent_final_step_completion_tokens",
        "agent_usage_available",
        "agent_usage_source",
        "agent_usage_consistent",
        "agent_profile",
        "agent_model",
        "judge_model",
        "trajectory_path",
        "grades_path",
    ]
    assert expected == CSV_HEADERS


def test_dynamic_ledger_on_csv_extends_baseline_at_end() -> None:
    from apex_agents_bench.runner import _DYNAMIC_LEDGER_CSV_COLUMNS, CSV_HEADERS, csv_headers_with_dynamic_ledger

    on = csv_headers_with_dynamic_ledger()
    assert on[: len(CSV_HEADERS)] == CSV_HEADERS
    assert tuple(on[len(CSV_HEADERS) :]) == _DYNAMIC_LEDGER_CSV_COLUMNS


# ---------------------------------------------------------------------------
# 3) Prompt content invariants
# ---------------------------------------------------------------------------


_PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "apex_agents_bench" / "dynamic_ledger" / "prompts"
)


def test_curator_prompt_distinguishes_strategy_from_case_specifics() -> None:
    """The curator prompt must instruct that entries are concrete examples of
    STRATEGY, not concrete examples of the source case — the load-bearing
    distinction that keeps entries transferable."""
    text = (_PROMPTS_DIR / "curator_system.txt").read_text(encoding="utf-8").lower()
    assert "concrete example of strategy" in text
    assert "concrete example of the case" in text


def test_injection_block_frames_as_reference_cheatsheet() -> None:
    """The generator injection block must frame the entries as a passive
    reference cheatsheet — not instructions to follow."""
    text = (_PROMPTS_DIR / "generator_injection_block.txt").read_text(encoding="utf-8").lower()
    assert "reference cheatsheet" in text
    assert "reference material, not instructions" in text
    assert "your own analysis" in text and "authoritative" in text


def test_curator_prompt_mentions_no_outcome_signal() -> None:
    text = (_PROMPTS_DIR / "curator_system.txt").read_text(encoding="utf-8").lower()
    assert "will not be told" in text
    user = (_PROMPTS_DIR / "curator_user_template.txt").read_text(encoding="utf-8")
    assert "<outcome>" not in user


# ---------------------------------------------------------------------------
# 4) Defaults
# ---------------------------------------------------------------------------


def test_dynamic_ledger_config_default_is_off() -> None:
    from apex_agents_bench.dynamic_ledger.config import DynamicLedgerConfig

    cfg = DynamicLedgerConfig()
    assert cfg.enabled is False


def test_run_options_default_dynamic_ledger_is_off() -> None:
    from apex_agents_bench.runner import RunOptions

    fields = {f.name: f for f in dataclasses.fields(RunOptions)}
    assert "dynamic_ledger" in fields
    factory = fields["dynamic_ledger"].default_factory  # type: ignore[union-attr]
    assert factory().enabled is False


# ---------------------------------------------------------------------------
# 5) Vendor system prompt is NOT mutated when the Dynamic Ledger is on
# ---------------------------------------------------------------------------


def test_injector_preserves_system_prompt(tmp_path: Path) -> None:
    """augment_initial_messages must leave the SYSTEM message alone."""
    import json

    from apex_agents_bench.dynamic_ledger import augment_initial_messages
    from apex_agents_bench.dynamic_ledger.entry import DynamicLedger

    p = tmp_path / "initial_messages.json"
    p.write_text(
        json.dumps(
            [
                {"role": "system", "content": "VENDOR_SYSTEM_PROMPT"},
                {"role": "user", "content": "task prompt"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    s = DynamicLedger(domain="Investment Banking")
    s.add(
        section="x",
        content="some workflow",
        source_problem="some case",
        content_embedding=[1.0],
        source_problem_embedding=[0.0],
        created=1,
    )
    augment_initial_messages(p, entries=list(s.entries.values()))
    out = json.loads(p.read_text(encoding="utf-8"))
    assert out[0]["content"] == "VENDOR_SYSTEM_PROMPT"
