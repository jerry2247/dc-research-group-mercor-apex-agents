"""Runner logic tests (resume / select / stats).

These tests exercise the runner's pure-Python machinery; they do NOT
spawn Docker, do NOT call any model, do NOT touch HF.
"""

from __future__ import annotations

import csv
from pathlib import Path

from apex_agents_bench.agent_profile import get_profile
from apex_agents_bench.azure_routing import AzureConfig
from apex_agents_bench.config import Settings
from apex_agents_bench.dataset import Criterion, Task
from apex_agents_bench.runner import (
    CSV_HEADERS,
    RunOptions,
    _preflight_credentials,
    _select_tasks,
    append_failure,
    append_row,
    build_run_manifest,
    calculate_stats,
    failure_log_path,
    load_completed_task_ids,
    manifest_path,
    write_manifest,
)


def _t(
    tid: str,
    domain: str = "banking",
    world: str = "w1",
    *,
    task_input_files: bool = False,
) -> Task:
    return Task(
        task_id=tid,
        task_name=tid,
        domain=domain,
        world_id=world,
        prompt="p",
        rubric=(Criterion(verifier_id="v1", criteria="ok"),),
        task_input_files=task_input_files,
    )


def _opts(**kwargs) -> RunOptions:
    base: dict = {
        "profile": get_profile("gpt-5.5-medium"),
        "settings": Settings.defaults(),
        "output_csv": Path("/tmp/_unused.csv"),
        "output_dir": Path("/tmp/_unused"),
    }
    base.update(kwargs)
    return RunOptions(**base)


# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------


def test_csv_headers_have_expected_columns() -> None:
    assert CSV_HEADERS[:4] == ["task_id", "domain", "world_id", "status"]
    for col in (
        "final_score",
        "criteria_passed",
        "criteria_total",
        "agent_profile",
        "agent_model",
        "judge_model",
        "agent_prompt_tokens",
        "agent_completion_tokens",
        "agent_total_tokens",
        "agent_final_step_completion_tokens",
        "agent_usage_available",
        "agent_usage_source",
        "agent_usage_consistent",
    ):
        assert col in CSV_HEADERS
    for col in (
        "judge_prompt_tokens",
        "judge_completion_tokens",
        "judge_total_tokens",
        "cost_usd",
        "agent_cost_usd",
        "judge_cost_usd",
    ):
        assert col not in CSV_HEADERS


def test_append_row_writes_header_only_once(tmp_path: Path) -> None:
    p = tmp_path / "r.csv"
    for tid in ("a", "b", "c"):
        append_row(
            p,
            {
                "task_id": tid,
                "domain": "banking",
                "world_id": "w1",
                "status": "completed",
                "final_score": 0.5,
                "criteria_passed": 1,
                "criteria_total": 2,
                "agent_profile": "gpt-5.5-medium",
                "agent_model": "openai/gpt-5.5",
                "judge_model": "openai/gpt-5.5",
                "agent_prompt_tokens": 10,
                "agent_completion_tokens": 2,
                "agent_total_tokens": 12,
                "agent_final_step_completion_tokens": 2,
                "agent_usage_available": True,
                "agent_usage_source": "trajectory_usage",
                "agent_usage_consistent": True,
            },
        )
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["agent_total_tokens"] == "12"


def test_load_completed_task_ids_empty(tmp_path: Path) -> None:
    assert load_completed_task_ids(tmp_path / "no.csv") == set()


def test_append_then_resume(tmp_path: Path) -> None:
    p = tmp_path / "r.csv"
    for tid, status in (("a", "completed"), ("b", "completed"), ("c", "pending")):
        append_row(
            p,
            {
                "task_id": tid,
                "domain": "banking",
                "world_id": "w1",
                "status": status,
                "final_score": 0.0,
                "criteria_passed": 0,
                "criteria_total": 1,
                "agent_profile": "gpt-5.5-medium",
                "agent_model": "openai/gpt-5.5",
                "judge_model": "openai/gpt-5.5",
                "agent_prompt_tokens": 0,
                "agent_completion_tokens": 0,
                "agent_total_tokens": 0,
                "agent_final_step_completion_tokens": 0,
                "agent_usage_available": False,
                "agent_usage_source": "unavailable",
                "agent_usage_consistent": True,
            },
        )
    done = load_completed_task_ids(p)
    assert done == {"a", "b"}


# -----------------------------------------------------------------------------
# _select_tasks
# -----------------------------------------------------------------------------


def test_select_tasks_by_domain() -> None:
    tasks = [
        _t("1", "banking"),
        _t("2", "consulting"),
        _t("3", "banking"),
        _t("4", "law"),
    ]
    out = _select_tasks(tasks, _opts(domain="banking"))
    assert [t.task_id for t in out] == ["1", "3"]


def test_select_tasks_by_world() -> None:
    tasks = [
        _t("1", "banking", "w1"),
        _t("2", "banking", "w2"),
        _t("3", "banking", "w1"),
    ]
    out = _select_tasks(tasks, _opts(world_id="w1"))
    assert [t.task_id for t in out] == ["1", "3"]


def test_select_tasks_by_task_ids_overrides_filters() -> None:
    tasks = [
        _t("1", "banking", "w1"),
        _t("2", "consulting", "w2"),
        _t("3", "law", "w3"),
    ]
    out = _select_tasks(tasks, _opts(domain="banking", task_ids=("2", "3")))
    assert [t.task_id for t in out] == ["2", "3"]


def test_select_tasks_start_and_limit() -> None:
    tasks = [_t(str(i), "banking") for i in range(10)]
    out = _select_tasks(tasks, _opts(domain="banking", start_index=3, limit=4))
    assert [t.task_id for t in out] == ["3", "4", "5", "6"]


def test_select_tasks_empty_filter_returns_all() -> None:
    tasks = [_t(str(i)) for i in range(3)]
    out = _select_tasks(tasks, _opts())
    assert [t.task_id for t in out] == ["0", "1", "2"]


# -----------------------------------------------------------------------------
# calculate_stats
# -----------------------------------------------------------------------------


def test_calculate_stats_basic(tmp_path: Path) -> None:
    p = tmp_path / "r.csv"
    for tid, domain, score in (
        ("a", "banking", 0.8),
        ("b", "banking", 0.6),
        ("c", "consulting", 0.4),
    ):
        append_row(
            p,
            {
                "task_id": tid,
                "domain": domain,
                "world_id": "w1",
                "status": "completed",
                "final_score": score,
                "criteria_passed": 1,
                "criteria_total": 2,
                "agent_profile": "gpt-5.5-medium",
                "agent_model": "openai/gpt-5.5",
                "judge_model": "openai/gpt-5.5",
                "agent_prompt_tokens": 0,
                "agent_completion_tokens": 0,
                "agent_total_tokens": 0,
                "agent_final_step_completion_tokens": 0,
                "agent_usage_available": False,
                "agent_usage_source": "unavailable",
                "agent_usage_consistent": True,
            },
        )
    stats = calculate_stats(p)
    assert stats["total_completed"] == 3
    assert stats["overall_mean"] == 0.6
    assert stats["by_domain"]["banking"] == {"n": 2, "mean": 0.7}
    assert stats["by_domain"]["consulting"] == {"n": 1, "mean": 0.4}


def test_run_options_is_frozen() -> None:
    """RunOptions must be immutable so a running runner can't be mutated mid-flight."""
    from dataclasses import FrozenInstanceError

    import pytest

    o = _opts()
    with pytest.raises(FrozenInstanceError):
        o.limit = 5  # type: ignore[misc]


# -----------------------------------------------------------------------------
# sidecars
# -----------------------------------------------------------------------------


def test_failure_and_manifest_sidecar_paths(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    assert failure_log_path(out) == tmp_path / "results.failures.jsonl"
    assert manifest_path(out) == tmp_path / "results.run_manifest.json"


def test_append_failure_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    append_failure(
        out,
        {
            "scope": "task",
            "status": "skipped",
            "task_id": "t1",
            "error": "grading failed",
        },
    )
    path = failure_log_path(out)
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert '"task_id": "t1"' in body
    assert '"output_csv":' in body


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    write_manifest(out, {"schema_version": 1, "status": "starting"})
    parsed = manifest_path(out).read_text(encoding="utf-8")
    assert '"schema_version": 1' in parsed
    assert '"status": "starting"' in parsed


def test_build_run_manifest_contains_nonsecret_run_state(tmp_path: Path) -> None:
    tasks = [_t("a", "banking", "w1"), _t("b", "law", "w2")]
    opts = _opts(output_csv=tmp_path / "r.csv", output_dir=tmp_path)
    manifest = build_run_manifest(opts, tasks, tasks[1:], status="starting")
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "starting"
    assert manifest["profile"]["name"] == "gpt-5.5-medium"
    assert manifest["judge"]["model_id"] == "openai/gpt-5.5"
    assert manifest["agent"]["max_steps"] == 50
    assert manifest["dataset"]["task_ids"] == ["a", "b"]
    assert "OPENAI_API_KEY" not in str(manifest)


def test_build_run_manifest_records_nonsecret_azure_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AZURE_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_API_BASE", "https://unit-test.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_API_VERSION", "2025-07-01-preview")

    tasks = [_t("a", "banking", "w1")]
    opts = _opts(
        output_csv=tmp_path / "r.csv",
        output_dir=tmp_path,
        azure=AzureConfig(enabled=True, deployment_name="gpt-5.5-test"),
    )
    manifest = build_run_manifest(opts, tasks, tasks, status="starting")

    assert manifest["azure"] == {
        "enabled": True,
        "route": "openai-compatible-/openai/v1",
        "deployment_name": "gpt-5.5-test",
        "api_base_present": True,
        "api_base_scheme": "https",
        "api_base_host": "unit-test.services.ai.azure.com",
        "api_base_path": "/openai/v1",
        "api_key_present": True,
        "api_version_present": True,
        "api_version_used_by_route": False,
        "effective_agent_model": "openai/gpt-5.5-test",
        "effective_judge_model": "openai/gpt-5.5-test",
        "embeddings_provider": "openai",
    }
    assert "azure-secret" not in str(manifest)


def test_preflight_credentials_fails_before_docker_when_keys_missing(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    for key in ("OPENAI_API_KEY", "XAI_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    settings = Settings.defaults().with_dataset_dir(tmp_path)
    # Cache the world zip so this assertion is about model/judge keys, not HF.
    world_dir = tmp_path / "world_files_zipped"
    world_dir.mkdir(parents=True)
    (world_dir / "w1.zip").write_bytes(b"not-a-real-zip-but-preflight-only")
    opts = _opts(settings=settings)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _preflight_credentials(opts, [_t("a", "banking", "w1")], hf_token=None)


def test_preflight_requires_deepseek_key_for_deepseek_profile(tmp_path: Path, monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    settings = Settings.defaults().with_dataset_dir(tmp_path)
    world_dir = tmp_path / "world_files_zipped"
    world_dir.mkdir(parents=True)
    (world_dir / "w1.zip").write_bytes(b"not-a-real-zip-but-preflight-only")
    opts = _opts(settings=settings, profile=get_profile("deepseek-v4-pro-max"))

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        _preflight_credentials(opts, [_t("a", "banking", "w1")], hf_token=None)


def test_preflight_azure_requires_key_and_base_not_api_version(tmp_path: Path, monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("AZURE_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_API_BASE", "https://unit-test.services.ai.azure.com/openai/v1")
    monkeypatch.delenv("AZURE_API_VERSION", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    settings = Settings.defaults().with_dataset_dir(tmp_path)
    world_dir = tmp_path / "world_files_zipped"
    world_dir.mkdir(parents=True)
    (world_dir / "w1.zip").write_bytes(b"not-a-real-zip-but-preflight-only")
    opts = _opts(settings=settings, azure=AzureConfig(enabled=True))

    _preflight_credentials(opts, [_t("a", "banking", "w1")], hf_token=None)

    monkeypatch.delenv("AZURE_API_BASE", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_API_BASE"):
        _preflight_credentials(opts, [_t("a", "banking", "w1")], hf_token=None)


def test_preflight_requires_hf_for_uncached_task_input_subsystems(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    settings = Settings.defaults().with_dataset_dir(tmp_path)
    world_dir = tmp_path / "world_files_zipped"
    world_dir.mkdir(parents=True)
    (world_dir / "w1.zip").write_bytes(b"not-a-real-zip-but-preflight-only")
    random_dir = tmp_path / "task_files" / "a" / "random"
    random_dir.mkdir(parents=True)
    (random_dir / "ignored.txt").write_text("ignored", encoding="utf-8")
    opts = _opts(settings=settings)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        _preflight_credentials(
            opts,
            [_t("a", "banking", "w1", task_input_files=True)],
            hf_token=None,
        )


def test_preflight_accepts_cached_task_input_subsystem(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    settings = Settings.defaults().with_dataset_dir(tmp_path)
    world_dir = tmp_path / "world_files_zipped"
    world_dir.mkdir(parents=True)
    (world_dir / "w1.zip").write_bytes(b"not-a-real-zip-but-preflight-only")
    filesystem_dir = tmp_path / "task_files" / "a" / "filesystem"
    filesystem_dir.mkdir(parents=True)
    (filesystem_dir / "starter.txt").write_text("used", encoding="utf-8")
    opts = _opts(settings=settings)

    _preflight_credentials(
        opts,
        [_t("a", "banking", "w1", task_input_files=True)],
        hf_token=None,
    )
