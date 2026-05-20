"""Catalog tool tests."""

from __future__ import annotations

import json
from pathlib import Path

from apex_agents_bench.catalog import LengthStats, build_report


def _write_tasks(dataset_dir: Path, rows: list[dict]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "tasks_and_rubrics.json").write_text(json.dumps(rows), encoding="utf-8")


def _write_worlds(dataset_dir: Path, rows: list[dict]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "world_descriptions.json").write_text(json.dumps(rows), encoding="utf-8")


def test_length_stats_on_empty_input() -> None:
    s = LengthStats.from_ints([])
    assert s.n == 0 and s.min == 0 and s.max == 0 and s.mean == 0.0


def test_length_stats_basic() -> None:
    s = LengthStats.from_ints([1, 2, 3, 4, 5])
    assert s.n == 5
    assert s.min == 1
    assert s.max == 5
    assert s.median == 3
    assert s.mean == 3.0


def test_build_report_smoke(tmp_path: Path) -> None:
    _write_worlds(
        tmp_path,
        [
            {"world_id": "w1", "world_name": "World 1", "domain": "banking"},
            {"world_id": "w2", "world_name": "World 2", "domain": "consulting"},
        ],
    )
    _write_tasks(
        tmp_path,
        [
            {
                "task_id": "t1",
                "task_name": "T1",
                "domain": "banking",
                "world_id": "w1",
                "prompt": "p1",
                "rubric": [{"verifier_id": "v1", "criteria": "x"}],
                "task_input_files": False,
            },
            {
                "task_id": "t2",
                "task_name": "T2",
                "domain": "banking",
                "world_id": "w1",
                "prompt": "longerprompt",
                "rubric": [
                    {"verifier_id": "v1", "criteria": "x"},
                    {"verifier_id": "v2", "criteria": "y"},
                ],
                "task_input_files": True,
            },
            {
                "task_id": "t3",
                "task_name": "T3",
                "domain": "consulting",
                "world_id": "w2",
                "prompt": "p3",
                "rubric": [],
                "task_input_files": False,
            },
        ],
    )
    rep = build_report(tmp_path, include_timestamp=False)
    assert rep.total_tasks == 3
    assert rep.total_worlds == 2
    assert rep.domains_by_task_count == {"banking": 2, "consulting": 1}
    assert rep.domains_by_world_count == {"banking": 1, "consulting": 1}
    assert rep.worlds_by_task_count == {"w1": 2, "w2": 1}
    assert rep.prompt_chars.min == 2  # "p1" / "p3"
    assert rep.tasks_with_input_files == 1
    # JSON round-trips.
    parsed = json.loads(rep.to_json())
    assert parsed["total_tasks"] == 3
    assert parsed["generated_at"] is None
