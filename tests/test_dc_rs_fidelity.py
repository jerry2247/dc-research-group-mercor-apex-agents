"""Load-bearing fidelity invariants for the apex-agents-bench DC-RS subsystem.

Each test pins a system-level property that, if broken, would cause
DC-RS to diverge from Suzgun et al.'s reference, leak grading
information, or hardcode domain / tool assumptions it must not have.

All checks are static (signatures, CSV header tuples, prompt-file
contents) — no network, no API.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

_PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "apex_agents_bench"
    / "dc_rs"
    / "prompts"
)


def _synth_prompt() -> str:
    return (_PROMPTS_DIR / "synthesizer_prompt.txt").read_text(encoding="utf-8")


def _injection_block() -> str:
    return (_PROMPTS_DIR / "generator_injection_block.txt").read_text(encoding="utf-8")


# ---- config default ---------------------------------------------------------


def test_dc_rs_config_default_is_off() -> None:
    from apex_agents_bench.dc_rs.config import DCRSConfig

    cfg = DCRSConfig()
    assert cfg.enabled is False
    assert cfg.synthesizer_model is None
    assert cfg.top_k == 3


# ---- CSV schema -------------------------------------------------------------


def test_dc_rs_on_csv_extends_baseline_at_end() -> None:
    """The DC-RS-on header list is exactly the baseline ``CSV_HEADERS``
    followed by the DC-RS columns — the baseline prefix is preserved."""
    from apex_agents_bench.runner import (
        CSV_HEADERS,
        _DC_RS_CSV_COLUMNS,
        csv_headers_with_dc_rs,
    )

    on = csv_headers_with_dc_rs()
    assert on == list(CSV_HEADERS) + list(_DC_RS_CSV_COLUMNS)
    # Equivalent prefix/suffix split (matches the prose fidelity check).
    assert on[: len(CSV_HEADERS)] == list(CSV_HEADERS)
    assert tuple(on[len(CSV_HEADERS):]) == _DC_RS_CSV_COLUMNS


def test_baseline_csv_has_no_dc_rs_columns() -> None:
    """Off-state is unchanged: none of the DC-RS-specific columns may
    leak into the baseline header tuple."""
    from apex_agents_bench.runner import CSV_HEADERS, _DC_RS_CSV_COLUMNS

    assert not (set(_DC_RS_CSV_COLUMNS) & set(CSV_HEADERS))


# ---- synthesizer signature: no grading / no current-task identifiers --------


def test_synthesizer_signature_has_no_outcome_inputs() -> None:
    """The synthesizer must NOT receive grading data of any kind, and
    must NOT receive the current ``task_id`` either: entries are
    accumulated general knowledge, not per-task tagged."""
    from apex_agents_bench.dc_rs import synthesize

    sig = inspect.signature(synthesize)
    params = list(sig.parameters.keys())
    assert params == [
        "current_cheatsheet",
        "retrieved_cases_block",
        "task_prompt",
        "cfg",
    ]
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
        assert forbidden not in params, f"synthesize leaks grading/identity input: {forbidden}"


# ---- prompt placeholders ----------------------------------------------------


def test_synthesizer_prompt_has_three_placeholders() -> None:
    body = _synth_prompt()
    assert "{current_cheatsheet}" in body
    assert "{retrieved_cases}" in body
    assert "{task_prompt}" in body
    # ``{task_id}`` is intentionally absent: entries must not reference
    # the current task.
    assert "{task_id}" not in body


def test_generator_injection_block_has_cheatsheet_placeholder() -> None:
    assert "{cheatsheet}" in _injection_block()


# ---- prompt is domain-agnostic ----------------------------------------------


def test_synthesizer_prompt_has_no_domain_specific_terms() -> None:
    """The prompt instructions must remain domain-agnostic — no
    benchmark-specific terminology that would prejudice the synthesizer
    toward one line of business. Word boundaries avoid false positives
    like 'the agent consults the cheatsheet'."""
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
    body = _synth_prompt()
    for pat in forbidden:
        hits = re.findall(pat, body, re.IGNORECASE)
        assert not hits, f"synthesizer_prompt.txt leaks domain term {pat!r}: {hits}"


def test_synthesizer_prompt_does_not_hardcode_tools_or_lessons() -> None:
    """The prompt must not hardcode specific tool names or specific
    lessons — those are content the synthesizer derives, not prompt
    boilerplate."""
    body = _synth_prompt()
    for token in (
        "python3",
        "tab_index",
        "final_answer",
        "sheets_server",
        "code_execution_server",
    ):
        assert token not in body, f"synthesizer_prompt.txt hardcodes {token!r}"


# ---- prompt instructs copy-forward + anti-wipe ------------------------------


def test_synthesizer_prompt_instructs_copy_forward() -> None:
    """The load-bearing instruction: re-emit prior entries verbatim by
    default. Without this default the cheatsheet fails to evolve and
    DC-RS regresses to per-case synthesis."""
    body = _synth_prompt().lower()
    assert "verbatim" in body
    assert ("copy" in body) or ("re-emit every" in body)


def test_synthesizer_prompt_instructs_anti_wipe() -> None:
    """The prompt must forbid wiping the cheatsheet to zero entries."""
    body = _synth_prompt().lower()
    assert "never" in body
    assert ("wipe" in body) or ("zero" in body)
