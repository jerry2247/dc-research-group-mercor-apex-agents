"""Trajectory + grading JSON parser tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex_agents_bench.trajectory import (
    TrajectoryError,
    parse_grading,
    parse_trajectory,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# parse_trajectory
# -----------------------------------------------------------------------------


def test_parse_trajectory_happy_path(tmp_path: Path) -> None:
    p = _write_json(
        tmp_path / "trajectory.json",
        {
            "status": "completed",
            "time_elapsed": 42.5,
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "thinking..."},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "final"},
            ],
        },
    )
    t = parse_trajectory(p)
    assert t.status == "completed"
    assert t.time_elapsed_seconds == 42.5
    assert t.steps_used == 2
    assert t.total_messages == 5


def test_parse_trajectory_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TrajectoryError, match="missing"):
        parse_trajectory(tmp_path / "no.json")


def test_parse_trajectory_malformed_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(TrajectoryError, match="malformed"):
        parse_trajectory(p)


def test_parse_trajectory_handles_missing_fields(tmp_path: Path) -> None:
    p = _write_json(tmp_path / "thin.json", {})
    t = parse_trajectory(p)
    assert t.status == ""
    assert t.time_elapsed_seconds == 0.0
    assert t.steps_used == 0
    assert t.total_messages == 0
    # Token fields default to 0 when usage is absent.
    assert t.agent_prompt_tokens == 0
    assert t.agent_completion_tokens == 0
    assert t.agent_total_tokens == 0
    assert t.agent_final_step_completion_tokens == 0
    assert t.agent_usage_available is False
    assert t.agent_usage_source == "unavailable"
    assert t.agent_usage_consistent is True


def test_parse_trajectory_extracts_usage_block(tmp_path: Path) -> None:
    """Vendor's UsageTracker.to_dict() writes a flat usage dict on the
    trajectory JSON. We must surface all four fields verbatim, no
    derivation."""
    p = _write_json(
        tmp_path / "with_usage.json",
        {
            "status": "completed",
            "time_elapsed": 100.0,
            "messages": [{"role": "assistant", "content": "x"}],
            "usage": {
                "prompt_tokens": 12345,
                "completion_tokens": 678,
                "total_tokens": 13023,
                "final_answer_tokens": 42,
            },
        },
    )
    t = parse_trajectory(p)
    assert t.agent_prompt_tokens == 12345
    assert t.agent_completion_tokens == 678
    assert t.agent_total_tokens == 13023
    assert t.agent_final_step_completion_tokens == 42
    assert t.agent_usage_available is True
    assert t.agent_usage_source == "trajectory_usage"
    assert t.agent_usage_consistent is True


def test_parse_trajectory_flags_inconsistent_usage_block(tmp_path: Path) -> None:
    p = _write_json(
        tmp_path / "usage_mismatch.json",
        {
            "status": "completed",
            "messages": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 999,
                "final_answer_tokens": 5,
            },
        },
    )
    t = parse_trajectory(p)
    assert t.agent_usage_available is True
    assert t.agent_usage_consistent is False


def test_parse_trajectory_tolerates_malformed_usage(tmp_path: Path) -> None:
    """Defensive: non-numeric or wrong-shape usage must not raise."""
    p = _write_json(
        tmp_path / "bad_usage.json",
        {"status": "completed", "messages": [], "usage": "not a dict"},
    )
    t = parse_trajectory(p)
    assert t.agent_prompt_tokens == 0
    assert t.agent_total_tokens == 0
    assert t.agent_usage_available is False

    p2 = _write_json(
        tmp_path / "bad_usage2.json",
        {"status": "completed", "messages": [], "usage": {"prompt_tokens": "oops"}},
    )
    t2 = parse_trajectory(p2)
    assert t2.agent_prompt_tokens == 0
    assert t2.agent_usage_available is True


def test_parse_trajectory_failed_status(tmp_path: Path) -> None:
    p = _write_json(
        tmp_path / "fail.json",
        {"status": "failed", "time_elapsed": 3600.0, "messages": []},
    )
    t = parse_trajectory(p)
    assert t.status == "failed"
    assert t.time_elapsed_seconds == 3600.0


# -----------------------------------------------------------------------------
# parse_grading
# -----------------------------------------------------------------------------


def test_parse_grading_happy_path(tmp_path: Path) -> None:
    p = _write_json(
        tmp_path / "grades.json",
        {
            "grading_run_id": "gr_x",
            "grading_run_status": "completed",
            "verifier_results": [
                {"verifier_id": "v1", "score": 1.0},
                {"verifier_id": "v2", "score": 0.0},
                {"verifier_id": "v3", "score": 0.7},
            ],
            "scoring_results": {"final_score": 0.5667},
        },
    )
    g = parse_grading(p)
    assert g.grading_run_status == "completed"
    assert g.final_score == pytest.approx(0.5667)
    assert g.criteria_total == 3
    assert g.criteria_passed == 2  # v1 (1.0) and v3 (0.7 >= 0.5)
    assert g.verifier_errors == 0


def test_parse_grading_no_verifiers(tmp_path: Path) -> None:
    p = _write_json(
        tmp_path / "g.json",
        {
            "grading_run_status": "completed",
            "verifier_results": [],
            "scoring_results": {"final_score": 0.0},
        },
    )
    g = parse_grading(p)
    assert g.criteria_total == 0
    assert g.criteria_passed == 0


def test_parse_grading_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TrajectoryError, match="missing"):
        parse_grading(tmp_path / "no.json")


def test_parse_grading_handles_non_numeric_score(tmp_path: Path) -> None:
    p = _write_json(
        tmp_path / "g.json",
        {
            "grading_run_status": "error",
            "verifier_results": [{"verifier_id": "v1", "score": "oops"}],
            "scoring_results": {"final_score": None},
        },
    )
    g = parse_grading(p)
    assert g.final_score == 0.0
    assert g.criteria_passed == 0


def test_parse_grading_counts_verifier_errors(tmp_path: Path) -> None:
    p = _write_json(
        tmp_path / "g_errors.json",
        {
            "grading_run_status": "completed",
            "verifier_results": [
                {"verifier_id": "v1", "score": 1.0, "status": "ok"},
                {"verifier_id": "v2", "score": 0.0, "status": "error"},
            ],
            "scoring_results": {"final_score": 0.5},
        },
    )
    g = parse_grading(p)
    assert g.verifier_errors == 1
