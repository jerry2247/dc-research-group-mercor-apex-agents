"""Smoke task selection tests (no network, no Docker)."""

from __future__ import annotations

import pytest

from apex_agents_bench.dataset import Criterion, Task
from apex_agents_bench.smoke import pick_smoke_task


def _t(tid: str, *, domain: str = "banking", world: str = "w1", inputs: bool = False) -> Task:
    return Task(
        task_id=tid,
        task_name=tid,
        domain=domain,
        world_id=world,
        prompt="p",
        rubric=(Criterion(verifier_id="v1", criteria="ok"),),
        task_input_files=inputs,
    )


def test_picks_first_no_input_files() -> None:
    tasks = [
        _t("a", inputs=True),
        _t("b", inputs=False),
        _t("c", inputs=False),
    ]
    picked = pick_smoke_task(tasks)
    assert picked.task_id == "b"


def test_falls_back_to_first_if_all_have_inputs() -> None:
    tasks = [
        _t("a", inputs=True),
        _t("b", inputs=True),
    ]
    picked = pick_smoke_task(tasks)
    assert picked.task_id == "a"


def test_respects_domain_filter() -> None:
    tasks = [
        _t("a", domain="banking"),
        _t("b", domain="consulting"),
        _t("c", domain="law"),
    ]
    picked = pick_smoke_task(tasks, domain="consulting")
    assert picked.task_id == "b"


def test_respects_world_filter() -> None:
    tasks = [
        _t("a", world="w1"),
        _t("b", world="w2"),
        _t("c", world="w2"),
    ]
    picked = pick_smoke_task(tasks, world_id="w2")
    assert picked.task_id == "b"


def test_raises_when_no_match() -> None:
    tasks = [_t("a", domain="banking")]
    with pytest.raises(RuntimeError, match="No tasks match"):
        pick_smoke_task(tasks, domain="medicine")


def test_allow_input_files_off_filter() -> None:
    tasks = [
        _t("a", inputs=True),
        _t("b", inputs=True),
    ]
    picked = pick_smoke_task(tasks, require_no_input_files=False)
    assert picked.task_id == "a"
