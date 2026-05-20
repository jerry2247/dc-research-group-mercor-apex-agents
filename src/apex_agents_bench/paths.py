"""Path resolution for apex-agents-bench.

The repo root is whichever directory contains both ``pyproject.toml`` and
``vendor/archipelago/``. We resolve it once, lazily, and key every other
path off it. Tests and CLI callers can override via the env var
``APEX_AGENTS_BENCH_ROOT``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class RepoLayoutError(RuntimeError):
    """Raised when the repo layout cannot be resolved."""


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the absolute path to the apex-agents-bench repo root.

    Resolution order:
      1. ``$APEX_AGENTS_BENCH_ROOT`` env var, if set and pointing at a valid root.
      2. Walk parents of this file until we find ``pyproject.toml`` AND
         ``vendor/archipelago``.
    """
    override = os.environ.get("APEX_AGENTS_BENCH_ROOT")
    if override:
        p = Path(override).resolve()
        if _looks_like_root(p):
            return p
        raise RepoLayoutError(
            f"APEX_AGENTS_BENCH_ROOT={override!r} is not a valid apex-agents-bench root "
            "(missing pyproject.toml or vendor/archipelago)."
        )

    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if _looks_like_root(candidate):
            return candidate
    raise RepoLayoutError(
        "Could not locate apex-agents-bench repo root. "
        "Set APEX_AGENTS_BENCH_ROOT or run from inside the repo."
    )


def _looks_like_root(p: Path) -> bool:
    return (p / "pyproject.toml").is_file() and (p / "vendor" / "archipelago").is_dir()


def vendor_dir() -> Path:
    """The vendored Archipelago tree."""
    return repo_root() / "vendor" / "archipelago"


def archipelago_agents_dir() -> Path:
    """The Archipelago agents package (installed editable in our venv)."""
    return vendor_dir() / "agents"


def archipelago_grading_dir() -> Path:
    """The Archipelago grading package (installed editable in our venv)."""
    return vendor_dir() / "grading"


def archipelago_environment_dir() -> Path:
    """The Archipelago environment dir (Dockerfile + docker-compose lives here)."""
    return vendor_dir() / "environment"


def archipelago_example_dir() -> Path:
    """The hugging_face_task example -- canonical reference for our runner."""
    return vendor_dir() / "examples" / "hugging_face_task"


def data_dir() -> Path:
    return repo_root() / "data"


def runs_dir() -> Path:
    return repo_root() / "runs"


def default_dataset_dir() -> Path:
    """Default path where the APEX-Agents dataset index + cache lives."""
    return data_dir() / "apex-agents"


def default_tasks_path() -> Path:
    """Default path to the cached ``tasks_and_rubrics.json``."""
    return default_dataset_dir() / "tasks_and_rubrics.json"


def default_worlds_path() -> Path:
    """Default path to the cached ``world_descriptions.json``."""
    return default_dataset_dir() / "world_descriptions.json"
