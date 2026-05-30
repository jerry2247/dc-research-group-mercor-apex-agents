"""Task-index browsing tests."""

from __future__ import annotations

import json
from pathlib import Path

from apex_agents_bench.dataset import Criterion, Task
from apex_agents_bench.task_index import _first_sentence, build_index, summarize


def _make_task(prompt: str = "What is X? Compute it.") -> Task:
    return Task(
        task_id="t1",
        task_name="T1",
        domain="banking",
        world_id="w1",
        prompt=prompt,
        rubric=(Criterion(verifier_id="v1", criteria="ok"),),
        task_input_files=False,
    )


def test_first_sentence_basic() -> None:
    assert _first_sentence("This is one. This is two.") == "This is one."


def test_first_sentence_truncates_long_sentences() -> None:
    s = _first_sentence("a" * 200, max_chars=50)
    assert len(s) == 50  # 49 chars + 1 ellipsis
    assert s.endswith("…")


def test_first_sentence_handles_whitespace() -> None:
    assert _first_sentence("\n\n  Hello.   ") == "Hello."


def test_summarize_extracts_fields() -> None:
    t = _make_task("Compute the IRR. Then write a memo.")
    s = summarize(t)
    assert s.task_id == "t1"
    assert s.domain == "banking"
    assert s.first_sentence == "Compute the IRR."
    assert s.n_criteria == 1
    assert s.has_input_files is False


def test_build_index_preserves_order(tmp_path: Path) -> None:
    (tmp_path / "world_descriptions.json").write_text("[]", encoding="utf-8")
    rows = [
        {
            "task_id": f"t{i}",
            "task_name": f"T{i}",
            "domain": "law",
            "world_id": "w1",
            "prompt": f"p{i}",
            "rubric": [],
            "task_input_files": False,
        }
        for i in range(4)
    ]
    (tmp_path / "tasks_and_rubrics.json").write_text(json.dumps(rows), encoding="utf-8")
    out = build_index(tmp_path)
    assert [s.task_id for s in out] == ["t0", "t1", "t2", "t3"]
