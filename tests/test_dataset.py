"""Dataset-loader tests with a synthetic mini-dataset.

We do NOT depend on the real APEX-Agents dataset being fetched. A tmp_path
fixture is enough to exercise every code path in :mod:`apex_agents_bench.dataset`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex_agents_bench.dataset import (
    DatasetError,
    World,
    get_task,
    load_tasks,
    load_worlds,
    load_worlds_by_id,
    validate,
)


def _write_tasks(dataset_dir: Path, rows: list[dict]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "tasks_and_rubrics.json").write_text(json.dumps(rows), encoding="utf-8")


def _write_worlds(dataset_dir: Path, rows: list[dict]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "world_descriptions.json").write_text(json.dumps(rows), encoding="utf-8")


def test_validate_missing_tasks_index(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="task index"):
        validate(tmp_path)


def test_validate_missing_worlds_index(tmp_path: Path) -> None:
    _write_tasks(tmp_path, [])
    with pytest.raises(DatasetError, match="world index"):
        validate(tmp_path)


def test_load_tasks_minimal(tmp_path: Path) -> None:
    _write_worlds(tmp_path, [])
    _write_tasks(
        tmp_path,
        [
            {
                "task_id": "task_abc",
                "task_name": "Do a thing",
                "domain": "banking",
                "world_id": "world_1",
                "prompt": "compute X",
                "rubric": [
                    {"verifier_id": "v1", "criteria": "X is correct"},
                    {"verifier_id": "v2", "criteria": "X is well-formatted"},
                ],
                "task_input_files": False,
            }
        ],
    )
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.task_id == "task_abc"
    assert t.domain == "banking"
    assert t.world_id == "world_1"
    assert t.n_criteria == 2
    assert t.rubric[0].verifier_id == "v1"
    assert t.task_input_files is False


def test_load_tasks_preserves_order(tmp_path: Path) -> None:
    _write_worlds(tmp_path, [])
    _write_tasks(
        tmp_path,
        [
            {
                "task_id": f"task_{i}",
                "task_name": f"t{i}",
                "domain": "law",
                "world_id": "w1",
                "prompt": "p",
                "rubric": [],
                "task_input_files": False,
            }
            for i in range(5)
        ],
    )
    tasks = load_tasks(tmp_path)
    assert [t.task_id for t in tasks] == [f"task_{i}" for i in range(5)]


def test_load_worlds(tmp_path: Path) -> None:
    _write_tasks(tmp_path, [])
    _write_worlds(
        tmp_path,
        [
            {"world_id": "world_1", "world_name": "BBDC/TVPG", "domain": "banking"},
            {"world_id": "world_2", "world_name": "Case Study", "domain": "consulting"},
        ],
    )
    worlds = load_worlds(tmp_path)
    assert len(worlds) == 2
    assert worlds[0] == World(world_id="world_1", world_name="BBDC/TVPG", domain="banking")
    by_id = load_worlds_by_id(tmp_path)
    assert by_id["world_2"].domain == "consulting"


def test_get_task_by_id(tmp_path: Path) -> None:
    _write_worlds(tmp_path, [])
    _write_tasks(
        tmp_path,
        [
            {
                "task_id": "task_a",
                "task_name": "A",
                "domain": "law",
                "world_id": "w1",
                "prompt": "x",
                "rubric": [],
                "task_input_files": False,
            },
            {
                "task_id": "task_b",
                "task_name": "B",
                "domain": "law",
                "world_id": "w1",
                "prompt": "y",
                "rubric": [],
                "task_input_files": True,
            },
        ],
    )
    t = get_task(tmp_path, "task_b")
    assert t.task_name == "B"
    assert t.task_input_files is True
    with pytest.raises(DatasetError, match="not found"):
        get_task(tmp_path, "task_missing")


def test_malformed_tasks_index(tmp_path: Path) -> None:
    _write_worlds(tmp_path, [])
    (tmp_path / "tasks_and_rubrics.json").write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(DatasetError, match="expected a JSON list"):
        load_tasks(tmp_path)


def test_rubric_with_non_dict_entries_filtered(tmp_path: Path) -> None:
    """Defensive: the loader should silently drop non-dict rubric entries."""
    _write_worlds(tmp_path, [])
    _write_tasks(
        tmp_path,
        [
            {
                "task_id": "task_x",
                "task_name": "X",
                "domain": "banking",
                "world_id": "w1",
                "prompt": "p",
                "rubric": [
                    {"verifier_id": "v1", "criteria": "ok"},
                    "junk",  # not a dict
                    {"verifier_id": "v2", "criteria": "ok2"},
                ],
                "task_input_files": False,
            },
        ],
    )
    [t] = load_tasks(tmp_path)
    assert t.n_criteria == 2
    assert [c.verifier_id for c in t.rubric] == ["v1", "v2"]
