"""Parsers for the vendor's agent trajectory + grading outputs.

The Archipelago agent runner writes a JSON file shaped like ``AgentTrajectoryOutput``
when invoked with ``--output``. The grading runner writes a JSON file with
shape::

    {
        "grading_run_id": str,
        "grading_run_status": str,
        "verifier_results": [ {...}, ... ],
        "scoring_results": { "final_score": float, ... }
    }

Both shapes are stable inputs to our CSV row builder. This module parses
them defensively (the JSON shape is owned by the vendor and can evolve)
and surfaces just the fields we need.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class TrajectoryError(RuntimeError):
    """Raised when a trajectory JSON cannot be read or has missing fields."""


@dataclass(frozen=True)
class TrajectorySummary:
    """The bits of an agent trajectory we surface to the CSV.

    Token counts come straight from the vendor's ``UsageTracker.to_dict()``
    (see ``vendor/archipelago/agents/runner/utils/usage.py``), which sums
    ``prompt_tokens`` and ``completion_tokens`` from every ``ModelResponse``
    the agent receives during the ReAct loop. They are AGENT-SIDE ONLY
    (the orchestrator model that solved the task); they do NOT include
    judge-side tokens, which the grading runner records separately in
    ``grades.json``.
    """

    status: str  # "completed" / "failed" / "error" / "cancelled" / "pending" / "running"
    time_elapsed_seconds: float
    steps_used: int  # count of assistant messages with tool calls or content
    total_messages: int

    # Cumulative across every agent LLM call in this task. Source:
    # trajectory.json's top-level ``usage`` dict (the vendor writes it
    # via ``UsageTracker.to_dict()`` in ``react_toolbelt_agent/main.py``).
    # If the trajectory has no ``usage`` field, all four token counts are 0
    # and ``agent_usage_available`` is False.
    agent_prompt_tokens: int = 0  # vendor: usage.prompt_tokens
    agent_completion_tokens: int = 0  # vendor: usage.completion_tokens
    agent_total_tokens: int = 0  # vendor: usage.total_tokens = prompt + completion
    agent_final_step_completion_tokens: int = 0  # vendor: usage.final_answer_tokens
    agent_usage_available: bool = False
    agent_usage_source: str = "unavailable"
    agent_usage_consistent: bool = False


@dataclass(frozen=True)
class GradingSummary:
    """The bits of a grading output we surface to the CSV."""

    grading_run_status: str  # "completed" / "error" / "cancelled"
    final_score: float
    criteria_passed: int
    criteria_total: int
    verifier_errors: int = 0


# -----------------------------------------------------------------------------


def parse_trajectory(path: Path) -> TrajectorySummary:
    if not path.is_file():
        raise TrajectoryError(f"trajectory JSON missing: {path}")
    with path.open(encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise TrajectoryError(f"trajectory JSON malformed at {path}: {e}") from e

    if not isinstance(data, dict):
        raise TrajectoryError(f"trajectory JSON at {path} is not a JSON object")

    status = str(data.get("status") or "")
    time_elapsed = data.get("time_elapsed") or 0
    try:
        time_elapsed_f = float(time_elapsed)
    except (TypeError, ValueError):
        time_elapsed_f = 0.0

    messages = data.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    steps_used = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")

    # Vendor's UsageTracker writes a flat dict with these four keys.
    # Read them defensively: a malformed or missing ``usage`` block must
    # not raise -- the run already succeeded by the time we parse this.
    raw_usage = data.get("usage")
    usage_available = isinstance(raw_usage, dict)
    usage = raw_usage if usage_available else {}
    if not isinstance(usage, dict):
        usage = {}

    def _as_int(v: object) -> int:
        if v is None:
            return 0
        if isinstance(v, bool):  # bool is a subclass of int in python
            return int(v)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return 0
        return 0

    agent_prompt_tokens = _as_int(usage.get("prompt_tokens"))
    agent_completion_tokens = _as_int(usage.get("completion_tokens"))
    agent_total_tokens = _as_int(usage.get("total_tokens"))
    agent_final_step_completion_tokens = _as_int(usage.get("final_answer_tokens"))
    agent_usage_consistent = agent_total_tokens == agent_prompt_tokens + agent_completion_tokens

    return TrajectorySummary(
        status=status,
        time_elapsed_seconds=time_elapsed_f,
        steps_used=steps_used,
        total_messages=len(messages),
        agent_prompt_tokens=agent_prompt_tokens,
        agent_completion_tokens=agent_completion_tokens,
        agent_total_tokens=agent_total_tokens,
        agent_final_step_completion_tokens=agent_final_step_completion_tokens,
        agent_usage_available=usage_available,
        agent_usage_source="trajectory_usage" if usage_available else "unavailable",
        agent_usage_consistent=agent_usage_consistent,
    )


# -----------------------------------------------------------------------------


def parse_grading(path: Path) -> GradingSummary:
    if not path.is_file():
        raise TrajectoryError(f"grading JSON missing: {path}")
    with path.open(encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise TrajectoryError(f"grading JSON malformed at {path}: {e}") from e

    if not isinstance(data, dict):
        raise TrajectoryError(f"grading JSON at {path} is not a JSON object")

    grading_run_status = str(data.get("grading_run_status") or "")

    scoring_results = data.get("scoring_results") or {}
    if not isinstance(scoring_results, dict):
        scoring_results = {}
    try:
        final_score = float(scoring_results.get("final_score") or 0.0)
    except (TypeError, ValueError):
        final_score = 0.0

    verifier_results = data.get("verifier_results") or []
    if not isinstance(verifier_results, list):
        verifier_results = []
    criteria_total = len(verifier_results)
    criteria_passed = sum(1 for v in verifier_results if isinstance(v, dict) and _interpret_pass(v))
    verifier_errors = sum(
        1
        for v in verifier_results
        if isinstance(v, dict) and str(v.get("status") or "").lower() == "error"
    )

    return GradingSummary(
        grading_run_status=grading_run_status,
        final_score=final_score,
        criteria_passed=criteria_passed,
        criteria_total=criteria_total,
        verifier_errors=verifier_errors,
    )


def _interpret_pass(verifier: dict[str, object]) -> bool:
    """Decide whether one verifier's result counts as a pass.

    The vendor's ``VerifierResult.score`` is in ``[0, 1]`` for the
    ``output_llm`` eval type, with 1.0 == pass and 0.0 == fail. We use
    ``>= 0.5`` to be conservative across future scoring methods.
    """
    score = verifier.get("score")
    if score is None:
        return False
    try:
        return float(score) >= 0.5  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
