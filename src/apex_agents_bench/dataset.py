"""APEX-Agents dataset loader.

The dataset ships as two JSON files plus per-task / per-world archives:

  tasks_and_rubrics.json    -- list of 480 task records
  world_descriptions.json   -- list of 33 world descriptions
  world_files_zipped/<world_id>.zip  -- the world snapshot, fetched per-task
  task_files/<task_id>/...           -- optional per-task starter files

This module loads the two JSON files into typed records. It does NOT
download anything from HuggingFace -- that is :mod:`apex_agents_bench.world`.
It does NOT call any model. Its only job is to surface the rows as typed
records so the runner can iterate.

The task schema (per the HF dataset card and the
``examples/hugging_face_task/main.py`` reference runner):

  task_id              str    -- e.g. "task_9ba58a6197114140877a1df1754d2993"
  task_name            str    -- short title
  domain               str    -- "Investment Banking" | "Law" | "Management Consulting"
                                 (verbatim, case + space-preserving)
  world_id             str    -- foreign key into world_descriptions.json
  prompt               str    -- the agent's first user message
  rubric               list[dict]  -- one entry per binary verifier criterion
                                     each dict has: verifier_id, criteria
  task_input_files     bool   -- whether per-task starter files exist
                                 under task_files/<task_id>/  (~36% of tasks)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class DatasetError(RuntimeError):
    """Raised when the dataset index is missing, malformed, or unreadable."""


@dataclass(frozen=True)
class Criterion:
    """One binary verifier criterion. Pass / Fail at grading time."""

    verifier_id: str
    criteria: str  # the natural-language criterion text

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Criterion:
        return cls(
            verifier_id=str(d.get("verifier_id") or ""),
            criteria=str(d.get("criteria") or ""),
        )


@dataclass(frozen=True)
class Task:
    """One APEX-Agents task."""

    task_id: str
    task_name: str
    domain: str
    world_id: str
    prompt: str
    rubric: tuple[Criterion, ...]
    task_input_files: bool  # True if task_files/<task_id>/ has starter files

    # Reference metadata shipped with the dataset but not used by the published
    # `output_llm` verifier path (which grades agent artifacts against rubric
    # criteria, not against gold strings). Surfaced for the `show` command and
    # for any downstream analysis; passing them to the grading runner would be
    # a fidelity break, so we deliberately do NOT thread them into verifiers.
    expected_output: str = ""
    gold_response: str = ""
    gold_response_type: str = ""

    @property
    def n_criteria(self) -> int:
        return len(self.rubric)

    @property
    def prompt_chars(self) -> int:
        return len(self.prompt)


@dataclass(frozen=True)
class World:
    """One APEX-Agents world."""

    world_id: str
    world_name: str
    domain: str

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> World:
        return cls(
            world_id=str(d.get("world_id") or ""),
            world_name=str(d.get("world_name") or ""),
            domain=str(d.get("domain") or ""),
        )


# -----------------------------------------------------------------------------


def tasks_path(dataset_dir: Path) -> Path:
    return dataset_dir / "tasks_and_rubrics.json"


def worlds_path(dataset_dir: Path) -> Path:
    return dataset_dir / "world_descriptions.json"


def validate(dataset_dir: Path) -> None:
    """Raise DatasetError if the dataset index is unusable."""
    tp = tasks_path(dataset_dir)
    if not tp.is_file():
        raise DatasetError(
            f"Expected APEX-Agents task index at {tp}. "
            "Run `make fetch-dataset` (or `bash scripts/fetch_dataset.sh`) first."
        )
    wp = worlds_path(dataset_dir)
    if not wp.is_file():
        raise DatasetError(
            f"Expected APEX-Agents world index at {wp}. "
            "Run `make fetch-dataset` (or `bash scripts/fetch_dataset.sh`) first."
        )


def load_tasks(dataset_dir: Path) -> list[Task]:
    """Read tasks_and_rubrics.json into a list of Task records.

    Preserves the on-disk order (which is the dataset's published order).
    """
    validate(dataset_dir)
    with tasks_path(dataset_dir).open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise DatasetError(
            f"{tasks_path(dataset_dir)}: expected a JSON list at the top level, got {type(raw).__name__}"
        )
    return [_row_to_task(r) for r in raw]


def load_worlds(dataset_dir: Path) -> list[World]:
    """Read world_descriptions.json into a list of World records."""
    validate(dataset_dir)
    with worlds_path(dataset_dir).open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise DatasetError(
            f"{worlds_path(dataset_dir)}: expected a JSON list at the top level, got {type(raw).__name__}"
        )
    return [World.from_dict(r) for r in raw if isinstance(r, dict)]


def load_worlds_by_id(dataset_dir: Path) -> dict[str, World]:
    return {w.world_id: w for w in load_worlds(dataset_dir)}


def get_task(dataset_dir: Path, task_id: str) -> Task:
    """Fetch a single task by id. Raises if absent."""
    for t in load_tasks(dataset_dir):
        if t.task_id == task_id:
            return t
    raise DatasetError(f"task_id {task_id!r} not found in {tasks_path(dataset_dir)}")


# -----------------------------------------------------------------------------


def _row_to_task(row: object) -> Task:
    if not isinstance(row, dict):
        raise DatasetError(f"expected each task to be a dict, got {type(row).__name__}")
    rubric_raw = row.get("rubric") or []
    if not isinstance(rubric_raw, list):
        raise DatasetError(
            f"task {row.get('task_id', '?')!r}: rubric must be a list, got {type(rubric_raw).__name__}"
        )
    rubric = tuple(Criterion.from_dict(item) for item in rubric_raw if isinstance(item, dict))
    return Task(
        task_id=str(row.get("task_id") or ""),
        task_name=str(row.get("task_name") or ""),
        domain=str(row.get("domain") or ""),
        world_id=str(row.get("world_id") or ""),
        prompt=str(row.get("prompt") or ""),
        rubric=rubric,
        task_input_files=bool(row.get("task_input_files")),
        expected_output=str(row.get("expected_output") or ""),
        gold_response=str(row.get("gold_response") or ""),
        gold_response_type=str(row.get("gold_response_type") or ""),
    )
