"""Browseable index of APEX-Agents tasks.

One row per task, summarized: id, name, domain, world, criteria count,
first-sentence preview of the prompt. Pure read-only -- no model calls,
no network. Safe to invoke before credentials are configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from apex_agents_bench.dataset import Task, load_tasks


@dataclass(frozen=True)
class TaskSummary:
    task_id: str
    task_name: str
    domain: str
    world_id: str
    first_sentence: str
    prompt_chars: int
    n_criteria: int
    has_input_files: bool


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(prompt: str, max_chars: int = 140) -> str:
    """Return the first sentence of ``prompt``, capped at max_chars."""
    prompt = re.sub(r"\s+", " ", prompt).strip()
    if not prompt:
        return ""
    parts = _SENTENCE_END.split(prompt, maxsplit=1)
    first = parts[0]
    if len(first) > max_chars:
        first = first[: max_chars - 1].rstrip() + "…"
    return first


def summarize(task: Task) -> TaskSummary:
    return TaskSummary(
        task_id=task.task_id,
        task_name=task.task_name,
        domain=task.domain,
        world_id=task.world_id,
        first_sentence=_first_sentence(task.prompt),
        prompt_chars=task.prompt_chars,
        n_criteria=task.n_criteria,
        has_input_files=task.task_input_files,
    )


def build_index(dataset_dir: Path) -> list[TaskSummary]:
    """Read the dataset and return one summary per task. On-disk order preserved."""
    return [summarize(t) for t in load_tasks(dataset_dir)]
