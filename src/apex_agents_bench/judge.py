"""Grading-side wrapper: build the vendor's grading-config files.

The Archipelago grading runner takes five JSON files on its CLI:

  --grading-settings  -- ``{"llm_judge_model": str, "llm_judge_extra_args": dict|None}``
  --verifiers         -- ``[Verifier, ...]`` derived from the task's rubric
  --eval-configs      -- list of EvalConfig (we use the default ``ec_output_llm``)
  --scoring-config    -- ``ScoringConfig`` (we use ``template``)
  --trajectory        -- the agent's trajectory output

This module owns generating the first four. The trajectory comes straight
from the agent runner's ``--output``.

Our only deliberate divergence from Mercor's published example is the
``llm_judge_model`` -- they ship ``gemini/gemini-2.5-flash``; we ship
``openai/gpt-5.5`` as our single fixed judge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apex_agents_bench.config import JudgeConfig
from apex_agents_bench.dataset import Task

# Eval config id matching the example's ``ec_output_llm``. This is the only
# eval surface APEX-Agents currently uses -- output-LLM verifiers grade
# binary criteria against the final-state snapshot.
DEFAULT_EVAL_CONFIG_ID = "ec_output_llm"

# Scoring config id matching the example's ``sc_default`` (which uses
# ``scoring_defn_id="template"``). The template scoring method returns one
# averaged score across all verifiers.
DEFAULT_SCORING_CONFIG_ID = "sc_default"
DEFAULT_SCORING_DEFN_ID = "template"


# -----------------------------------------------------------------------------


def build_grading_settings(judge: JudgeConfig) -> dict[str, Any]:
    """Build the JSON object written to ``grading_settings.json``.

    Shape comes from vendor's ``GradingSettings`` pydantic model. The
    vendor's grading runner reads exactly two fields::

        llm_judge_model       -- LiteLLM-routable model name
        llm_judge_extra_args  -- dict | None, splatted into litellm.acompletion

    We use ``None`` (or an empty dict) for extra_args by default so OpenAI's
    medium reasoning effort applies to gpt-5.5.
    """
    payload: dict[str, Any] = {
        "llm_judge_model": judge.model_id,
        "llm_judge_extra_args": dict(judge.extra_args) if judge.extra_args else None,
    }
    return payload


def write_grading_settings(judge: JudgeConfig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(build_grading_settings(judge), indent=2) + "\n", encoding="utf-8"
    )
    return out_path


# -----------------------------------------------------------------------------


def build_verifiers(task: Task) -> list[dict[str, Any]]:
    """Build the list[Verifier] JSON shape the grading runner expects.

    Each entry is shaped identically to the reference example's per-criterion
    builder -- the first criterion is marked ``is_primary_objective``, all
    point at ``eval_config_id=ec_output_llm``, no dependencies. We do not
    rewrite, re-order, or filter criteria.
    """
    out: list[dict[str, Any]] = []
    for i, c in enumerate(task.rubric):
        out.append(
            {
                "verifier_id": c.verifier_id,
                "verifier_version": 1,
                "world_id": task.world_id,
                "task_id": task.task_id,
                "eval_config_id": DEFAULT_EVAL_CONFIG_ID,
                "verifier_values": {
                    "criteria": c.criteria,
                    "is_primary_objective": i == 0,
                },
                "verifier_index": i,
                "verifier_dependencies": None,
            }
        )
    return out


def write_verifiers(task: Task, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_verifiers(task), indent=2) + "\n", encoding="utf-8")
    return out_path


# -----------------------------------------------------------------------------


def build_eval_configs() -> list[dict[str, Any]]:
    """Verbatim copy of Mercor's published ``eval_configs.json``."""
    return [
        {
            "eval_config_id": DEFAULT_EVAL_CONFIG_ID,
            "eval_config_name": "Output LLM Verifier",
            "eval_defn_id": "output_llm",
            "eval_config_values": {},
        }
    ]


def write_eval_configs(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_eval_configs(), indent=2) + "\n", encoding="utf-8")
    return out_path


# -----------------------------------------------------------------------------


def build_scoring_config() -> dict[str, Any]:
    """Verbatim copy of Mercor's published ``scoring_config.json``."""
    return {
        "scoring_config_id": DEFAULT_SCORING_CONFIG_ID,
        "scoring_config_name": "Default Scoring",
        "scoring_defn_id": DEFAULT_SCORING_DEFN_ID,
        "scoring_config_values": {},
    }


def write_scoring_config(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_scoring_config(), indent=2) + "\n", encoding="utf-8")
    return out_path
