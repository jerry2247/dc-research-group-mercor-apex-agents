"""Dataset characterization for APEX-Agents.

Produces a deterministic JSON snapshot of the public split's properties:
task counts per domain / world, criteria distribution, task-input-file
prevalence. Output is a stable bytes-comparable artifact (modulo
``generated_at``, which can be omitted with ``--no-timestamp``).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from apex_agents_bench.dataset import Task, load_tasks, load_worlds


@dataclass(frozen=True)
class LengthStats:
    n: int
    min: int
    p25: int
    median: int
    p75: int
    max: int
    mean: float

    @classmethod
    def from_ints(cls, xs: list[int]) -> LengthStats:
        if not xs:
            return cls(n=0, min=0, p25=0, median=0, p75=0, max=0, mean=0.0)
        sorted_xs = sorted(xs)
        return cls(
            n=len(xs),
            min=sorted_xs[0],
            p25=sorted_xs[len(xs) // 4],
            median=sorted_xs[len(xs) // 2],
            p75=sorted_xs[(3 * len(xs)) // 4],
            max=sorted_xs[-1],
            mean=round(statistics.mean(xs), 2),
        )


@dataclass(frozen=True)
class CatalogReport:
    dataset_dir: str
    generated_at: str | None
    total_tasks: int
    total_worlds: int
    domains_by_task_count: dict[str, int]
    domains_by_world_count: dict[str, int]
    worlds_by_task_count: dict[str, int]
    prompt_chars: LengthStats
    criteria_per_task: LengthStats
    tasks_with_input_files: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# -----------------------------------------------------------------------------


def build_report(dataset_dir: Path, *, include_timestamp: bool = True) -> CatalogReport:
    tasks = load_tasks(dataset_dir)
    worlds = load_worlds(dataset_dir)
    return _build_from(tasks, worlds, dataset_dir, include_timestamp=include_timestamp)


def _build_from(
    tasks: list[Task],
    worlds: list,
    dataset_dir: Path,
    *,
    include_timestamp: bool,
) -> CatalogReport:
    prompt_chars = [t.prompt_chars for t in tasks]
    criteria_counts = [t.n_criteria for t in tasks]
    tasks_per_world = Counter(t.world_id for t in tasks)
    domains_per_task = Counter(t.domain for t in tasks)
    domains_per_world = Counter(w.domain for w in worlds)
    with_inputs = sum(1 for t in tasks if t.task_input_files)

    return CatalogReport(
        dataset_dir=str(dataset_dir),
        generated_at=(
            datetime.now(UTC).isoformat(timespec="seconds") if include_timestamp else None
        ),
        total_tasks=len(tasks),
        total_worlds=len(worlds),
        domains_by_task_count=dict(sorted(domains_per_task.items())),
        domains_by_world_count=dict(sorted(domains_per_world.items())),
        worlds_by_task_count=dict(sorted(tasks_per_world.items())),
        prompt_chars=LengthStats.from_ints(prompt_chars),
        criteria_per_task=LengthStats.from_ints(criteria_counts),
        tasks_with_input_files=with_inputs,
    )


def write_report(report: CatalogReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_json() + "\n", encoding="utf-8")
