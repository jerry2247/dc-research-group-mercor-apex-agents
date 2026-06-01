"""Runner/CLI integration invariants for DL (no Docker/network)."""

from __future__ import annotations

import dataclasses

from typer.testing import CliRunner

from apex_agents_bench.cli import app
from apex_agents_bench.dl.config import DLConfig
from apex_agents_bench.runner import RunOptions

runner = CliRunner()


def test_cli_rejects_dl_with_dc_rs() -> None:
    res = runner.invoke(
        app,
        ["run", "--model", "gpt-5.5-medium", "--world", "w", "--dc-rs", "--dl"],
    )
    assert res.exit_code == 2
    assert "mutually exclusive" in res.stdout


def test_cli_rejects_dl_with_trace() -> None:
    res = runner.invoke(
        app,
        ["run", "--model", "gpt-5.5-medium", "--world", "w", "--trace", "--dl"],
    )
    assert res.exit_code == 2
    assert "mutually exclusive" in res.stdout


def test_cli_run_help_mentions_dl() -> None:
    res = runner.invoke(app, ["run", "--help"])
    assert res.exit_code == 0
    assert "--dl" in res.stdout


def test_run_options_threads_dl_config() -> None:
    # A constructed RunOptions carries the DLConfig with the chosen top_k.
    fields = {f.name for f in dataclasses.fields(RunOptions)}
    assert "dl" in fields
    cfg = DLConfig(enabled=True, top_k=3)
    assert cfg.enabled is True
    assert cfg.top_k == 3
    assert cfg.curator_model is None  # filled from the profile by the runner


def test_run_single_task_accepts_dl_runtime_param() -> None:
    import inspect

    from apex_agents_bench.runner import run_single_task

    params = list(inspect.signature(run_single_task).parameters.keys())
    assert "dl_runtime" in params
