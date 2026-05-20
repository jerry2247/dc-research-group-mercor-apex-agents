"""Single-task smoke runner.

A minimal end-to-end run: pick one task, run the agent against it, grade
the result. The point is not to produce a benchmark number -- it is to
fail loudly if any seam in the pipeline is broken, before we spend real
budget on a multi-task run.

The smoke shares its core orchestration with :mod:`apex_agents_bench.runner`
(both call :func:`run_single_task`); the smoke just picks a single task
and surfaces a structured result for human inspection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from apex_agents_bench.agent_profile import AgentProfile
from apex_agents_bench.config import Settings
from apex_agents_bench.dataset import Task, load_tasks
from apex_agents_bench.runner import TaskOutcome, run_single_task

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmokeResult:
    task_id: str
    domain: str
    world_id: str
    profile_name: str
    orchestrator_model: str
    judge_model: str
    agent_status: str
    final_score: float
    criteria_passed: int
    criteria_total: int
    steps_used: int
    wall_time_seconds: float
    output_dir: str


# -----------------------------------------------------------------------------


def pick_smoke_task(
    tasks: list[Task],
    *,
    domain: str | None = None,
    world_id: str | None = None,
    require_no_input_files: bool = True,
) -> Task:
    """Pick a single task to smoke against.

    By default picks the first task that does NOT ship extra task input
    files -- those add an HF download step and inflate smoke wall time.
    About 64% of tasks ship without input files (175 of 480 have inputs
    per the live dataset), so this filter is satisfied trivially.
    """
    pool = tasks
    if domain is not None:
        pool = [t for t in pool if t.domain == domain]
    if world_id is not None:
        pool = [t for t in pool if t.world_id == world_id]
    if require_no_input_files:
        no_inputs = [t for t in pool if not t.task_input_files]
        if no_inputs:
            pool = no_inputs
    if not pool:
        raise RuntimeError(
            f"No tasks match smoke criteria "
            f"(domain={domain!r}, world_id={world_id!r}, "
            f"require_no_input_files={require_no_input_files}). "
            "Run `apex-agents-bench catalog` first to see what's available."
        )
    return pool[0]


# -----------------------------------------------------------------------------


def run_smoke(
    settings: Settings,
    *,
    profile: AgentProfile,
    domain: str | None = None,
    world_id: str | None = None,
    require_no_input_files: bool = True,
    output_dir: Path | None = None,
    hf_token: str | None = None,
) -> SmokeResult:
    """End-to-end smoke run for a single task."""
    tasks = load_tasks(settings.dataset_dir)
    task = pick_smoke_task(
        tasks,
        domain=domain,
        world_id=world_id,
        require_no_input_files=require_no_input_files,
    )
    log.info(
        "smoke task selected: task_id=%s domain=%s world=%s prompt_chars=%d",
        task.task_id,
        task.domain,
        task.world_id,
        task.prompt_chars,
    )
    log.info(
        "smoke agent profile: name=%s orchestrator_model=%s",
        profile.name,
        profile.orchestrator_model,
    )

    if output_dir is None:
        output_dir = settings.runs_dir / "smoke" / f"{profile.name}__{task.task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    outcome: TaskOutcome = run_single_task(
        settings=settings,
        profile=profile,
        task=task,
        output_dir=output_dir,
        hf_token=hf_token,
    )

    return SmokeResult(
        task_id=outcome.task_id,
        domain=outcome.domain,
        world_id=outcome.world_id,
        profile_name=profile.name,
        orchestrator_model=profile.orchestrator_model,
        judge_model=settings.judge.model_id,
        agent_status=outcome.agent_status,
        final_score=outcome.final_score,
        criteria_passed=outcome.criteria_passed,
        criteria_total=outcome.criteria_total,
        steps_used=outcome.steps_used,
        wall_time_seconds=outcome.wall_time_seconds,
        output_dir=str(output_dir),
    )


# -----------------------------------------------------------------------------


def render_result(result: SmokeResult) -> str:
    return json.dumps(
        {
            "task_id": result.task_id,
            "domain": result.domain,
            "world_id": result.world_id,
            "profile": result.profile_name,
            "orchestrator_model": result.orchestrator_model,
            "judge_model": result.judge_model,
            "agent_status": result.agent_status,
            "final_score": round(result.final_score, 4),
            "criteria": f"{result.criteria_passed}/{result.criteria_total}",
            "steps_used": result.steps_used,
            "wall_time_seconds": round(result.wall_time_seconds, 1),
            "output_dir": result.output_dir,
        },
        indent=2,
    )
