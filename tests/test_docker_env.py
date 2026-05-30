"""Docker-env orchestration tests.

All subprocess + httpx calls are mocked. These tests run on any machine,
with or without Docker. The integration test that actually starts a
container is marked ``docker`` and is skipped by default (see
``pyproject.toml::pytest.ini_options.markers``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apex_agents_bench.docker_env import (
    DockerEnvError,
    EnvContainer,
    docker_available,
    docker_compose_command,
    start_env,
    stop_env,
    wait_for_health,
)


def test_docker_compose_command_uses_v2() -> None:
    assert docker_compose_command() == ["docker", "compose"]


def test_docker_available_returns_false_when_docker_binary_missing() -> None:
    with patch("apex_agents_bench.docker_env.shutil.which", return_value=None):
        assert docker_available() is False


def test_docker_available_returns_true_when_info_exits_zero() -> None:
    with (
        patch("apex_agents_bench.docker_env.shutil.which", return_value="/usr/local/bin/docker"),
        patch("apex_agents_bench.docker_env.subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        assert docker_available() is True


def test_docker_available_returns_false_when_info_fails() -> None:
    with (
        patch("apex_agents_bench.docker_env.shutil.which", return_value="/usr/local/bin/docker"),
        patch("apex_agents_bench.docker_env.subprocess.run") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="daemon not running"
        )
        assert docker_available() is False


# -----------------------------------------------------------------------------


def test_start_env_refuses_when_daemon_missing(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")
    with (
        patch("apex_agents_bench.docker_env.docker_available", return_value=False),
        pytest.raises(DockerEnvError, match="Docker daemon"),
    ):
        start_env(tmp_path, 8080)


def test_start_env_refuses_when_compose_dir_missing(tmp_path: Path) -> None:
    with (
        patch("apex_agents_bench.docker_env.docker_available", return_value=True),
        pytest.raises(DockerEnvError, match="Compose directory"),
    ):
        start_env(tmp_path / "nope", 8080)


def test_start_env_refuses_when_no_compose_yaml(tmp_path: Path) -> None:
    with (
        patch("apex_agents_bench.docker_env.docker_available", return_value=True),
        pytest.raises(DockerEnvError, match="No docker-compose"),
    ):
        start_env(tmp_path, 8080)


def test_start_env_happy_path_calls_down_then_up(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")

    with (
        patch("apex_agents_bench.docker_env.docker_available", return_value=True),
        patch("apex_agents_bench.docker_env.subprocess.run") as mock_run,
        patch("apex_agents_bench.docker_env.wait_for_health", return_value=True),
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        container = start_env(tmp_path, 8080)

    # First call: `docker compose down -v`
    first = mock_run.call_args_list[0]
    assert first.args[0][:3] == ["docker", "compose", "down"]
    # Second call: `docker compose up -d --build`
    second = mock_run.call_args_list[1]
    assert second.args[0][:3] == ["docker", "compose", "up"]
    assert container.host_port == 8080
    assert container.base_url == "http://localhost:8080"


def test_start_env_propagates_up_failure(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")
    with (
        patch("apex_agents_bench.docker_env.docker_available", return_value=True),
        patch("apex_agents_bench.docker_env.subprocess.run") as mock_run,
    ):
        # down -v succeeds, up fails
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="port in use"),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="logs", stderr=""),
        ]
        with pytest.raises(DockerEnvError, match="exit code 1"):
            start_env(tmp_path, 8080)


def test_start_env_propagates_health_timeout(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")
    with (
        patch("apex_agents_bench.docker_env.docker_available", return_value=True),
        patch("apex_agents_bench.docker_env.subprocess.run") as mock_run,
        patch("apex_agents_bench.docker_env.wait_for_health", return_value=False),
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with pytest.raises(DockerEnvError, match="not become healthy"):
            start_env(tmp_path, 8080)


# -----------------------------------------------------------------------------


def test_wait_for_health_returns_true_on_200() -> None:
    container = EnvContainer(compose_dir=Path("/tmp/x"), host_port=8080)
    response_ok = MagicMock(status_code=200)
    with patch("httpx.get", return_value=response_ok):
        assert wait_for_health(container, timeout_seconds=5) is True


def test_wait_for_health_returns_false_on_timeout() -> None:
    import httpx as httpx_mod

    container = EnvContainer(compose_dir=Path("/tmp/x"), host_port=8080)
    with (
        patch("httpx.get", side_effect=httpx_mod.ConnectError("refused")),
        patch("apex_agents_bench.docker_env.HEALTH_POLL_INTERVAL_SECONDS", 0.0),
    ):
        assert wait_for_health(container, timeout_seconds=0) is False


# -----------------------------------------------------------------------------


def test_stop_env_invokes_down_v() -> None:
    container = EnvContainer(compose_dir=Path("/tmp/x"), host_port=8080)
    with patch("apex_agents_bench.docker_env.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        stop_env(container)
    args = mock_run.call_args_list[0].args[0]
    assert args[:3] == ["docker", "compose", "down"]
    assert "-v" in args
