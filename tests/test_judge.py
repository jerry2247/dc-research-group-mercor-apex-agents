"""Judge-config builder tests."""

from __future__ import annotations

import json
from pathlib import Path

from apex_agents_bench.config import JudgeConfig
from apex_agents_bench.dataset import Criterion, Task
from apex_agents_bench.judge import (
    DEFAULT_EVAL_CONFIG_ID,
    DEFAULT_SCORING_CONFIG_ID,
    DEFAULT_SCORING_DEFN_ID,
    build_eval_configs,
    build_grading_settings,
    build_scoring_config,
    build_verifiers,
    write_eval_configs,
    write_grading_settings,
    write_scoring_config,
    write_verifiers,
)


def _make_task(criteria: list[tuple[str, str]]) -> Task:
    return Task(
        task_id="task_x",
        task_name="X",
        domain="banking",
        world_id="world_1",
        prompt="p",
        rubric=tuple(Criterion(verifier_id=vid, criteria=ct) for vid, ct in criteria),
        task_input_files=False,
    )


# -----------------------------------------------------------------------------


def test_grading_settings_default_is_gpt55() -> None:
    s = build_grading_settings(JudgeConfig())
    assert s["llm_judge_model"] == "openai/gpt-5.5"
    assert s["llm_judge_extra_args"] in (None, {})


def test_grading_settings_passes_through_extra_args() -> None:
    s = build_grading_settings(JudgeConfig(extra_args={"reasoning_effort": "high"}))
    assert s["llm_judge_extra_args"] == {"reasoning_effort": "high"}


def test_grading_settings_write(tmp_path: Path) -> None:
    out = write_grading_settings(JudgeConfig(), tmp_path / "grading_settings.json")
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["llm_judge_model"] == "openai/gpt-5.5"


# -----------------------------------------------------------------------------


def test_build_verifiers_one_per_criterion_in_order() -> None:
    task = _make_task([("v1", "criterion one"), ("v2", "criterion two"), ("v3", "criterion three")])
    out = build_verifiers(task)
    assert len(out) == 3
    assert [v["verifier_id"] for v in out] == ["v1", "v2", "v3"]
    assert [v["verifier_index"] for v in out] == [0, 1, 2]


def test_build_verifiers_first_is_primary_objective() -> None:
    task = _make_task([("v1", "first"), ("v2", "second")])
    out = build_verifiers(task)
    assert out[0]["verifier_values"]["is_primary_objective"] is True
    assert out[1]["verifier_values"]["is_primary_objective"] is False


def test_build_verifiers_carries_world_and_task_ids() -> None:
    task = _make_task([("v1", "x")])
    [v] = build_verifiers(task)
    assert v["world_id"] == "world_1"
    assert v["task_id"] == "task_x"
    assert v["eval_config_id"] == DEFAULT_EVAL_CONFIG_ID
    assert v["verifier_dependencies"] is None


def test_build_verifiers_criteria_passed_verbatim() -> None:
    """No rewriting / normalization of the criterion text."""
    task = _make_task([("v1", "  CASE-SENSITIVE TEXT 0.30% \n with whitespace  ")])
    [v] = build_verifiers(task)
    assert v["verifier_values"]["criteria"] == "  CASE-SENSITIVE TEXT 0.30% \n with whitespace  "


def test_write_verifiers_round_trip(tmp_path: Path) -> None:
    task = _make_task([("v1", "first"), ("v2", "second")])
    out = write_verifiers(task, tmp_path / "verifiers.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert len(parsed) == 2
    assert parsed[0]["verifier_id"] == "v1"


# -----------------------------------------------------------------------------


def test_eval_configs_default_shape() -> None:
    cfg = build_eval_configs()
    assert len(cfg) == 1
    assert cfg[0]["eval_config_id"] == DEFAULT_EVAL_CONFIG_ID
    assert cfg[0]["eval_defn_id"] == "output_llm"


def test_eval_configs_write(tmp_path: Path) -> None:
    out = write_eval_configs(tmp_path / "eval_configs.json")
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body[0]["eval_defn_id"] == "output_llm"


# -----------------------------------------------------------------------------


def test_scoring_config_default_shape() -> None:
    cfg = build_scoring_config()
    assert cfg["scoring_config_id"] == DEFAULT_SCORING_CONFIG_ID
    assert cfg["scoring_defn_id"] == DEFAULT_SCORING_DEFN_ID


def test_scoring_config_write(tmp_path: Path) -> None:
    out = write_scoring_config(tmp_path / "scoring_config.json")
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["scoring_defn_id"] == "template"
