"""Load-bearing fidelity invariants for the apex-agents-bench DL subsystem.

DL is the no-ground-truth, itemised, dual-retrieval, CRUD memory. These
tests pin the invariants that make it faithful and keep it distinct from
DC-RS and TRACE:

  * the curator never sees grading data or task identity (no-GT, like
    DC-RS and like the original DL's observe());
  * the prompts are domain-agnostic and hardcode no tools/lessons;
  * the curator emits a typed CRUD batch (not a wholesale rewrite, not a
    citations mechanism);
  * every entry carries a required type drawn from the five DC-RS
    categories, plus a source_problem retrieval key;
  * there is NO create-time dedup;
  * the CSV schema extends the baseline at the end;
  * the three memory subsystems are mutually exclusive.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "src" / "apex_agents_bench" / "dl" / "prompts"


def _curator_prompt() -> str:
    return (_PROMPTS_DIR / "curator_prompt.txt").read_text(encoding="utf-8")


def _injection_block() -> str:
    return (_PROMPTS_DIR / "generator_injection_block.txt").read_text(encoding="utf-8")


# ---- no ground truth, no identity (the load-bearing DL invariant) ----------


def test_curator_signature_has_no_outcome_or_identity_inputs() -> None:
    from apex_agents_bench.dl import curate

    sig = inspect.signature(curate)
    params = list(sig.parameters.keys())
    assert params == ["ledger", "retrieved", "task_prompt", "trajectory", "cfg"]
    for forbidden in (
        "score",
        "gt_correct",
        "criteria",
        "criteria_passed",
        "rubric",
        "expected_answer",
        "judge_rationale",
        "task_id",
    ):
        assert forbidden not in params, f"curate leaks grading/identity input: {forbidden}"


def test_apply_ops_signature_has_no_dedup_threshold() -> None:
    from apex_agents_bench.dl import apply_ops

    params = list(inspect.signature(apply_ops).parameters.keys())
    # DL does no create-time dedup: no threshold, no retrieved-window filter.
    for forbidden in ("threshold", "retrieved", "create_time_similarity_threshold"):
        assert forbidden not in params, f"apply_ops should not take {forbidden}"


def test_dl_config_has_no_ground_truth_or_dedup_knobs() -> None:
    import dataclasses

    from apex_agents_bench.dl.config import DLConfig

    names = {f.name for f in dataclasses.fields(DLConfig)}
    for forbidden in (
        "gt",
        "ground_truth",
        "create_time_similarity_threshold",
        "dedup_threshold",
    ):
        assert forbidden not in names, f"DLConfig leaks a forbidden knob: {forbidden}"
    assert DLConfig().top_k == 3
    assert DLConfig().embedding_model == "text-embedding-3-large"


# ---- the curator prompt ----------------------------------------------------


def test_curator_prompt_has_three_placeholders_and_no_task_id() -> None:
    body = _curator_prompt()
    assert "{retrieved_entries}" in body
    assert "{task_prompt}" in body
    assert "{rendered_trajectory}" in body
    assert "{task_id}" not in body


def test_curator_prompt_declares_crud_output_contract() -> None:
    body = _curator_prompt()
    assert "<ledger_updates>" in body
    low = body.lower()
    assert "create" in low
    assert "update" in low
    assert "delete" in low
    # the new dual-retrieval key the curator must populate
    assert "source_problem" in body
    assert "entry_id" in body


def test_curator_prompt_names_all_five_types() -> None:
    from apex_agents_bench.dl.entry import ENTRY_TYPES

    body = _curator_prompt()
    for t in ENTRY_TYPES:
        assert t in body, f"curator prompt does not name the required type {t!r}"


def test_curator_prompt_is_not_wholesale_replace() -> None:
    # DL entries persist individually; the curator emits deltas, so it must
    # NOT instruct the DC-RS-style "re-emit every entry verbatim / never wipe"
    # wholesale-replacement behaviour.
    low = _curator_prompt().lower()
    assert "re-emit every" not in low
    assert "anything you do not re-emit is permanently lost" not in low


def test_curator_prompt_has_no_domain_specific_terms() -> None:
    forbidden = (
        r"investment banking",
        r"\bfinance\b",
        r"\bfinancial\b",
        r"\blegal\b",
        r"\blaw\b",
        r"\blawyer\b",
        r"\bconsulting\b",
        r"\bconsultant\b",
        r"\bmedicine\b",
        r"\bmedical\b",
        r"\bDCF\b",
        r"\bEBITDA\b",
        r"\bLBO\b",
    )
    body = _curator_prompt()
    for pat in forbidden:
        hits = re.findall(pat, body, re.IGNORECASE)
        assert not hits, f"curator_prompt.txt leaks domain term {pat!r}: {hits}"


def test_curator_prompt_does_not_hardcode_tools_or_lessons() -> None:
    body = _curator_prompt()
    for token in (
        "python3",
        "tab_index",
        "final_answer",
        "sheets_server",
        "code_execution_server",
    ):
        assert token not in body, f"curator_prompt.txt hardcodes {token!r}"


def test_curator_prompt_keeps_errors_then_fixes_signal() -> None:
    # The DC-RS spine retained: the no-GT curator's most reliable signal.
    low = _curator_prompt().lower()
    assert "error" in low and "correct" in low


# ---- the generator injection block ----------------------------------------


def test_injection_block_has_entries_placeholder() -> None:
    assert "{entries_block}" in _injection_block()


def test_injection_block_is_consult_not_obey() -> None:
    low = _injection_block().lower()
    assert "reference" in low
    assert "do not" in low  # do-not-cite / do-not-follow framing


def test_injection_block_has_no_citation_instruction() -> None:
    # DL has no citations (unlike TRACE).
    body = _injection_block()
    assert "<citations>" not in body
    assert "bullet-" not in body
    assert "citation" not in body.lower()


def test_injection_block_has_no_domain_specific_terms() -> None:
    forbidden = (
        r"investment banking",
        r"\bfinance\b",
        r"\bfinancial\b",
        r"\blegal\b",
        r"\blaw\b",
        r"\bconsulting\b",
        r"\bmedicine\b",
        r"\bDCF\b",
        r"\bEBITDA\b",
    )
    body = _injection_block()
    for pat in forbidden:
        assert not re.findall(pat, body, re.IGNORECASE), f"injection block leaks {pat!r}"


# ---- CSV schema ------------------------------------------------------------


def test_dl_on_csv_extends_baseline_at_end() -> None:
    from apex_agents_bench.runner import (
        _DL_CSV_COLUMNS,
        CSV_HEADERS,
        csv_headers_with_dl,
    )

    on = csv_headers_with_dl()
    assert on == list(CSV_HEADERS) + list(_DL_CSV_COLUMNS)
    assert on[: len(CSV_HEADERS)] == list(CSV_HEADERS)
    assert tuple(on[len(CSV_HEADERS) :]) == _DL_CSV_COLUMNS


def test_baseline_csv_has_no_dl_columns() -> None:
    from apex_agents_bench.runner import _DL_CSV_COLUMNS, CSV_HEADERS

    assert not (set(_DL_CSV_COLUMNS) & set(CSV_HEADERS))


def test_dl_columns_disjoint_from_dc_rs_and_trace() -> None:
    from apex_agents_bench.runner import (
        _DC_RS_CSV_COLUMNS,
        _DL_CSV_COLUMNS,
        _TRACE_CSV_COLUMNS,
    )

    assert not (set(_DL_CSV_COLUMNS) & set(_DC_RS_CSV_COLUMNS))
    assert not (set(_DL_CSV_COLUMNS) & set(_TRACE_CSV_COLUMNS))


# ---- three-way mutual exclusion -------------------------------------------


def test_runner_enforces_three_way_mutex_source() -> None:
    # The runtime guard in run() must reject more than one enabled subsystem.
    from apex_agents_bench import runner

    src = inspect.getsource(runner.run)
    assert "mutually exclusive" in src
    assert "opts.dl.enabled" in src


def test_run_options_default_dl_is_off() -> None:
    import dataclasses

    from apex_agents_bench.runner import RunOptions

    fields = {f.name: f for f in dataclasses.fields(RunOptions)}
    assert "dl" in fields
    factory = fields["dl"].default_factory  # type: ignore[union-attr]
    assert factory().enabled is False
