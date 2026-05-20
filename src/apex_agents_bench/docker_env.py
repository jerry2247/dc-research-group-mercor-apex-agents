"""Environment-container orchestration.

The Archipelago environment is a Docker container that serves the MCP
gateway at ``http://<host>:<port>``. Our runner is responsible for the full
container lifecycle for each task:

  1. ``docker compose down -v`` -- destroy any leftover state
  2. ``docker compose up -d --build`` -- start a fresh container
  3. Poll ``GET /health`` until 200
  4. POST world + task files to ``/data/populate``
  5. POST MCP server config to ``/apps``
  6. Run the agent against ``/mcp/``
  7. POST nothing; stream the final state out via ``/data/snapshot``
  8. ``docker compose down -v`` -- clean up

This module owns steps 1-3 and 8. Steps 4-7 are coordinated by
:mod:`apex_agents_bench.runner`.

All Docker invocations are subprocess calls -- one fewer dep than the
docker-py SDK, and the reference example uses ``subprocess.run`` directly,
so we stay shape-compatible.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

log = logging.getLogger(__name__)


class DockerEnvError(RuntimeError):
    """Raised when the environment container cannot be brought up."""


HEALTH_PATH = "/health"
HEALTH_TIMEOUT_SECONDS = 180
HEALTH_POLL_INTERVAL_SECONDS = 1.0


# -----------------------------------------------------------------------------


def docker_available() -> bool:
    """Quick probe: is the Docker daemon reachable?"""
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0


def docker_compose_command() -> list[str]:
    """Whichever compose CLI is available. Prefer ``docker compose`` (V2)
    over the legacy ``docker-compose``."""
    return ["docker", "compose"]


# -----------------------------------------------------------------------------


@dataclass
class EnvContainer:
    """A live (or about-to-be-live) environment container.

    The container's listening host port + the project-compose-directory are
    captured here so callers can talk to it (and we can tear it down on
    SIGINT / process exit).
    """

    compose_dir: Path
    host_port: int
    _started: bool = False

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.host_port}"


# -----------------------------------------------------------------------------


_active_containers: list[EnvContainer] = []


def _atexit_cleanup() -> None:
    """Force-down any containers still tracked at process exit."""
    for c in list(_active_containers):
        try:
            stop_env(c)
        except Exception as e:
            log.warning("atexit cleanup failed for %s: %s", c.compose_dir, e)


def _sigint_handler(signum: int, frame: FrameType | None) -> None:
    """Tear down any active containers, then re-raise the default behavior."""
    _atexit_cleanup()
    # Restore default handler and re-deliver so the parent gets SIGINT semantics.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    raise KeyboardInterrupt


# Registered once per process import. Idempotent re-registration is fine.
atexit.register(_atexit_cleanup)
# Some environments (notebooks, threading edge cases) reject signal()
# outside the main thread. Cleanup via atexit() still applies in that case.
with contextlib.suppress(ValueError, OSError):
    signal.signal(signal.SIGINT, _sigint_handler)


# -----------------------------------------------------------------------------


def start_env(
    compose_dir: Path,
    host_port: int,
    *,
    extra_env: dict[str, str] | None = None,
) -> EnvContainer:
    """Bring up a fresh environment container at ``host_port``.

    Always runs ``docker compose down -v`` first to discard leftover state.
    Blocks until the container reports healthy.
    """
    if not docker_available():
        raise DockerEnvError(
            "Docker daemon is not reachable. Is Docker Desktop running? Run `make docker-check`."
        )
    if not compose_dir.is_dir():
        raise DockerEnvError(f"Compose directory does not exist: {compose_dir}")
    if (
        not (compose_dir / "docker-compose.yaml").is_file()
        and not (compose_dir / "docker-compose.yml").is_file()
    ):
        raise DockerEnvError(
            f"No docker-compose.yaml in {compose_dir}; expected the vendor's environment dir."
        )

    container = EnvContainer(compose_dir=compose_dir, host_port=host_port)
    _active_containers.append(container)

    env = {"PATH": _path_env(), **(extra_env or {})}
    # The vendor's compose file binds the container's 8080 to host 8080 by
    # default. To use a different host port, pass it via PORT env var; the
    # vendor compose file (per upstream README) respects ``$PORT``.
    if host_port != 8080:
        env["PORT"] = str(host_port)

    log.info("docker compose down -v  (in %s)", compose_dir)
    subprocess.run(
        [*docker_compose_command(), "down", "-v"],
        cwd=str(compose_dir),
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )

    log.info("docker compose up -d --build  (in %s, host_port=%d)", compose_dir, host_port)
    r = subprocess.run(
        [*docker_compose_command(), "up", "-d", "--build"],
        cwd=str(compose_dir),
        env=env,
        timeout=900,
        check=False,
    )
    if r.returncode != 0:
        # Best-effort log tail to surface the actual error to the user.
        _dump_logs(compose_dir, env)
        _safe_remove(container)
        raise DockerEnvError(
            f"docker compose up failed with exit code {r.returncode}; check logs above."
        )
    container._started = True

    if not wait_for_health(container, timeout_seconds=HEALTH_TIMEOUT_SECONDS):
        _dump_logs(compose_dir, env)
        _safe_remove(container)
        raise DockerEnvError(
            f"environment container at {container.base_url} did not become healthy "
            f"within {HEALTH_TIMEOUT_SECONDS}s. See logs above."
        )

    log.info("environment ready at %s", container.base_url)
    return container


def wait_for_health(container: EnvContainer, *, timeout_seconds: int) -> bool:
    """Poll ``GET /health`` until 200 or until timeout. Returns success."""
    # Lazy import so module import doesn't pull httpx if it's not used.
    import httpx

    deadline = time.time() + timeout_seconds
    url = container.base_url + HEALTH_PATH
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200:
                return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
    return False


def stop_env(container: EnvContainer) -> None:
    """Destroy the container. Idempotent."""
    env = {"PATH": _path_env()}
    if container.host_port != 8080:
        env["PORT"] = str(container.host_port)
    try:
        subprocess.run(
            [*docker_compose_command(), "down", "-v"],
            cwd=str(container.compose_dir),
            capture_output=True,
            env=env,
            timeout=120,
            check=False,
        )
    finally:
        _safe_remove(container)


# -----------------------------------------------------------------------------


def _safe_remove(container: EnvContainer) -> None:
    if container in _active_containers:
        _active_containers.remove(container)


def _dump_logs(compose_dir: Path, env: dict[str, str]) -> None:
    try:
        r = subprocess.run(
            [*docker_compose_command(), "logs", "--no-color", "--tail", "200"],
            cwd=str(compose_dir),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        if r.stdout:
            log.error("=== docker compose logs (tail 200) ===\n%s", r.stdout)
        if r.stderr:
            log.error("=== docker compose stderr ===\n%s", r.stderr)
    except Exception as e:
        log.error("could not collect docker logs: %s", e)


def _path_env() -> str:
    """Return ``$PATH`` so we never inherit a bare environment when calling
    subprocess.run with our own env dict."""
    import os

    return os.environ.get("PATH", "")
