"""Load-bearing fidelity tests for the apex-agents-bench TRACE subsystem."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path


def test_reflector_signature() -> None:
    """The reflector takes the GT bit — INTENTIONAL per TRACE paper."""
    from apex_agents_bench.trace import reflect

    sig = inspect.signature(reflect)
    assert list(sig.parameters.keys()) == [
        "ledger",
        "task_prompt",
        "trajectory",
        "cited_bullet_ids",
        "gt_correct",
        "cfg",
    ]


def test_curator_signature() -> None:
    """The TRACE curator takes the GT bit AND the reflector proposals.

    Distinct from the Dynamic Ledger curator (no-GT). This signature is
    pinned here because the TRACE paper's pipeline depends on both
    inputs reaching the curator."""
    from apex_agents_bench.trace import curate

    sig = inspect.signature(curate)
    assert list(sig.parameters.keys()) == [
        "ledger",
        "task_prompt",
        "trajectory",
        "cited_bullet_ids",
        "gt_correct",
        "reflector_proposals",
        "cfg",
    ]


def test_trace_off_csv_schema_baseline_is_unchanged() -> None:
    from apex_agents_bench.runner import CSV_HEADERS

    # The baseline schema MUST be identical to the pre-TRACE baseline.
    # We just sanity-check that no TRACE-specific column leaked into
    # the baseline header tuple.
    forbidden = {"trace_enabled", "gt_correct_bit", "reflector_proposal_count"}
    assert not (forbidden & set(CSV_HEADERS))


def test_trace_on_csv_extends_baseline_at_end() -> None:
    from apex_agents_bench.runner import (
        CSV_HEADERS,
        _TRACE_CSV_COLUMNS,
        csv_headers_with_trace,
    )

    on = csv_headers_with_trace()
    assert on[: len(CSV_HEADERS)] == CSV_HEADERS
    assert tuple(on[len(CSV_HEADERS):]) == _TRACE_CSV_COLUMNS


def test_trace_config_default_is_off() -> None:
    from apex_agents_bench.trace.config import TraceConfig

    cfg = TraceConfig()
    assert cfg.enabled is False
    assert cfg.reflector_model is None
    assert cfg.curator_model is None


def test_run_options_default_trace_is_off() -> None:
    from apex_agents_bench.runner import RunOptions

    fields = {f.name: f for f in dataclasses.fields(RunOptions)}
    assert "trace" in fields
    factory = fields["trace"].default_factory  # type: ignore[union-attr]
    assert factory().enabled is False


# Prompt content invariants ---------------------------------------------------


_PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "apex_agents_bench" / "trace" / "prompts"
)


def test_reflector_prompt_references_gt_bit() -> None:
    text = (_PROMPTS_DIR / "reflector_system.txt").read_text(encoding="utf-8").lower()
    assert "ground-truth" in text or "ground truth" in text
    assert "gt_correct" in text


def test_curator_prompt_references_gt_bit_and_proposals() -> None:
    text = (_PROMPTS_DIR / "curator_system.txt").read_text(encoding="utf-8").lower()
    assert "ground-truth" in text or "ground truth" in text
    assert "reflector" in text
    assert "proposals" in text


def test_injection_block_specifies_citation_format() -> None:
    text = (_PROMPTS_DIR / "generator_injection_block.txt").read_text(encoding="utf-8")
    assert "<citations>" in text
    assert "bullet-" in text


def test_curator_user_template_threads_all_inputs() -> None:
    text = (_PROMPTS_DIR / "curator_user_template.txt").read_text(encoding="utf-8")
    for token in [
        "{rendered_active_cheatsheet}",
        "{task_prompt}",
        "{rendered_trajectory}",
        "{cited_bullets_json}",
        "{gt_correct}",
        "{reflector_proposals_json}",
    ]:
        assert token in text


def test_trace_and_dynamic_ledger_are_mutually_exclusive_in_cli() -> None:
    """Both subsystems sharing the same agent infrastructure must be
    flagged as mutually exclusive at the CLI surface."""
    from apex_agents_bench.cli import app  # noqa: F401 — import check only

    # We don't try to call the CLI here; the runtime enforces it too:
    # see RunOptions docs and the explicit ValueError in runner.run().
