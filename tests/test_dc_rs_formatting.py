"""Unit tests for the retrieved-cases markdown shape (synthesizer input).

Agentic port of the prose ``_format_entries``: empty pool → ``"(empty)"``;
non-empty → preamble + per-case blocks (task + trajectory) in REVERSED
order (most-similar last) + footer.
"""

from __future__ import annotations

from apex_agents_bench.dc_rs.bank import BankEntry
from apex_agents_bench.dc_rs.formatting import format_retrieved_cases
from apex_agents_bench.dc_rs.retriever import Retrieved


def _entry(idx: int) -> BankEntry:
    return BankEntry(
        bank_id=f"bank-{idx:05d}",
        task_id=f"t-{idx}",
        task_prompt=f"PROMPT-{idx}",
        rendered_trajectory=f"TRAJECTORY-{idx}",
        prompt_embedding=[float(idx)],
        added=idx - 1,
    )


def test_format_empty_pool_returns_literal_empty_placeholder() -> None:
    """Matches Suzgun's reference: ``return '(empty)'`` when no entries."""
    assert format_retrieved_cases([]) == "(empty)"


def test_format_three_cases_reversed_so_most_similar_is_last() -> None:
    retrieved = [
        Retrieved(entry=_entry(1), similarity=0.91),  # most similar
        Retrieved(entry=_entry(2), similarity=0.78),
        Retrieved(entry=_entry(3), similarity=0.55),  # least similar
    ]
    out = format_retrieved_cases(retrieved)
    # Reversed order: least similar first, most similar last.
    pos1 = out.find("PROMPT-3")
    pos2 = out.find("PROMPT-2")
    pos3 = out.find("PROMPT-1")
    assert 0 < pos1 < pos2 < pos3
    # Similarities are rendered with 2-decimal format.
    assert "0.91" in out
    assert "0.78" in out
    assert "0.55" in out
    # Each case shows its task prompt AND its agent trajectory.
    for k in (1, 2, 3):
        assert f"TRAJECTORY-{k}" in out
    # Preamble + footer present.
    assert "### PRIOR CASES (START)" in out
    assert "#### PRIOR CASES (END)" in out
    # Preamble cautions against blind copying.
    assert "critical mindset" in out


def test_format_single_case_has_exactly_one_prior_case_header() -> None:
    retrieved = [Retrieved(entry=_entry(1), similarity=0.91)]
    out = format_retrieved_cases(retrieved)
    assert "PROMPT-1" in out
    assert "TRAJECTORY-1" in out
    assert out.count("### PRIOR CASE #") == 1


def test_format_renders_task_and_trajectory_labels() -> None:
    retrieved = [Retrieved(entry=_entry(1), similarity=0.5)]
    out = format_retrieved_cases(retrieved)
    assert "#### Task:" in out
    assert "trajectory" in out.lower()
