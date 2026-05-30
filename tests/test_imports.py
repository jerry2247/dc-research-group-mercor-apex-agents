"""Structural tests -- verify the package is importable and policies hold.

These tests do not call any API and do not require Docker. They run on a
developer machine with the venv set up and no ``.env``. If any of these
fail, ``make setup`` did not complete correctly.
"""

from __future__ import annotations


def test_apex_agents_bench_imports() -> None:
    import apex_agents_bench

    assert apex_agents_bench.__version__


def test_apex_agents_bench_submodules_import() -> None:
    """Every wrapper module must be importable without side effects."""
    from apex_agents_bench import (  # noqa: F401
        agent_profile,
        catalog,
        cli,
        config,
        dataset,
        docker_env,
        judge,
        paths,
        runner,
        smoke,
        task_index,
        trajectory,
        world,
    )


def test_settings_defaults_match_policy() -> None:
    from apex_agents_bench.config import (
        AGENT_MAX_STEPS,
        AGENT_TIMEOUT_SECONDS,
        DEFAULT_JUDGE_MODEL,
        RUNS_PER_TASK,
        Settings,
    )

    s = Settings.defaults()
    assert s.judge.model_id == DEFAULT_JUDGE_MODEL
    assert s.judge.model_id == "openai/gpt-5.5", (
        "Project policy: judge is gpt-5.5 at OpenAI's default reasoning_effort "
        "(medium), routed via LiteLLM's openai/ prefix."
    )
    assert s.agent.max_steps == AGENT_MAX_STEPS == 50, (
        "Project policy: max_steps=50, matching Mercor's published example."
    )
    assert s.agent.timeout_seconds == AGENT_TIMEOUT_SECONDS == 3600, (
        "Project policy: timeout=3600s, matching Mercor's published example."
    )
    assert RUNS_PER_TASK == 1, "Project policy: one run per (task, model)."


def test_repo_root_resolves() -> None:
    from apex_agents_bench.paths import (
        archipelago_agents_dir,
        archipelago_environment_dir,
        archipelago_example_dir,
        archipelago_grading_dir,
        repo_root,
        vendor_dir,
    )

    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert vendor_dir().is_dir()
    assert (vendor_dir() / "UPSTREAM.md").is_file()
    assert (vendor_dir() / "PATCHES.md").is_file()
    assert (vendor_dir() / "LICENSE_UPSTREAM").is_file()
    assert archipelago_agents_dir().is_dir()
    assert archipelago_grading_dir().is_dir()
    assert archipelago_environment_dir().is_dir()
    assert archipelago_example_dir().is_dir()


def test_vendor_mcp_config_present_and_has_nine_servers() -> None:
    """The 9-server MCP config Mercor ships must be untouched."""
    import json

    from apex_agents_bench.paths import archipelago_example_dir

    p = archipelago_example_dir() / "mcp_config_all_oss_servers.json"
    assert p.is_file()
    cfg = json.loads(p.read_text(encoding="utf-8"))
    assert set(cfg.get("mcpServers", {}).keys()) == {
        "calendar_server",
        "chat_server",
        "code_execution_server",
        "sheets_server",
        "filesystem_server",
        "mail_server",
        "pdf_server",
        "slides_server",
        "docs_server",
    }


def test_cli_help_runs() -> None:
    """`apex-agents-bench --help` must work without any of the heavy deps."""
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-m", "apex_agents_bench", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "apex-agents-bench" in r.stdout
