"""World snapshot management.

Each APEX-Agents task runs inside a "world" -- a starter filesystem +
``.apps_data`` directory representing a simulated professional environment.
Worlds ship as zip archives in the HF dataset under
``world_files_zipped/<world_id>.zip``. There are 33 worlds shared across
480 tasks.

This module is responsible for:
  - fetching a world zip from HF on demand (cached locally)
  - fetching per-task input files (some tasks ship extra starter files)
  - extracting the world into a temporary scratch directory for one task
  - cleaning up the scratch directory afterwards

We do NOT pre-download all 33 worlds at setup -- that would be 18.7 GB
of disk for no benefit if the user only runs a few tasks. HF's
``hf_hub_download`` is request-deduplicated and uses content-addressed
caching, so the second task that needs a world just hits the local cache.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


class WorldError(RuntimeError):
    """Raised when a world snapshot is missing or malformed."""


# Two subsystems that get populated into the environment container.
# These match Mercor's reference example at
# ``vendor/archipelago/examples/hugging_face_task/main.py``.
SUBSYSTEMS: tuple[str, ...] = ("filesystem", ".apps_data")


def world_zip_path(dataset_dir: Path, world_id: str) -> Path:
    """Where the cached world zip lives. May or may not exist."""
    return dataset_dir / "world_files_zipped" / f"{world_id}.zip"


def has_populatable_subsystem(root: Path) -> bool:
    """Return True if ``root`` has task data the environment will consume."""
    if not root.is_dir():
        return False
    for subsystem in SUBSYSTEMS:
        sub_dir = root / subsystem
        if sub_dir.exists() and any(sub_dir.rglob("*")):
            return True
    return False


# -----------------------------------------------------------------------------


def fetch_world_zip(dataset_dir: Path, world_id: str, *, hf_token: str | None = None) -> Path:
    """Download the world zip from HF if not already present, return its path.

    Uses ``huggingface_hub.hf_hub_download`` which caches inside
    ``~/.cache/huggingface`` and then symlinks into our dataset dir. If the
    user has already downloaded the world (e.g. via ``hf download``), this
    is a no-op.
    """
    cached = world_zip_path(dataset_dir, world_id)
    if cached.is_file():
        return cached

    # Lazy import so `--help` doesn't require huggingface_hub.
    from huggingface_hub import hf_hub_download

    cached.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching world zip from HF: %s", world_id)
    src = hf_hub_download(
        repo_id="mercor/apex-agents",
        repo_type="dataset",
        filename=f"world_files_zipped/{world_id}.zip",
        token=hf_token,
    )
    # Copy (not symlink) so users can ``rm -rf data/`` without breaking the HF cache.
    shutil.copy(src, cached)
    return cached


def fetch_task_files(
    dataset_dir: Path,
    task_id: str,
    *,
    hf_token: str | None = None,
) -> Path | None:
    """Download per-task starter files into ``data/apex-agents/task_files/<task_id>/``.

    Returns the directory containing ``filesystem/`` and ``.apps_data/`` (if any),
    or ``None`` if the task ships no extra files.
    """
    from huggingface_hub import snapshot_download

    task_root = dataset_dir / "task_files" / task_id
    if has_populatable_subsystem(task_root):
        return task_root

    pattern = f"task_files/{task_id}/**"
    log.info("fetching task input files from HF: %s", task_id)
    snapshot_dir = snapshot_download(
        repo_id="mercor/apex-agents",
        repo_type="dataset",
        allow_patterns=[pattern],
        token=hf_token,
    )
    src = Path(snapshot_dir) / "task_files" / task_id
    if not has_populatable_subsystem(src):
        return None
    task_root.parent.mkdir(parents=True, exist_ok=True)
    if task_root.exists():
        shutil.rmtree(task_root)
    shutil.copytree(src, task_root, symlinks=False)
    return task_root


# -----------------------------------------------------------------------------


@contextmanager
def materialize_world(zip_path: Path) -> Iterator[Path]:
    """Extract a world zip into a fresh temporary directory.

    Use as::

        with materialize_world(zip_path) as scratch:
            # scratch/filesystem/ and scratch/.apps_data/ are populated
            ...
        # scratch is gone on exit
    """
    if not zip_path.is_file():
        raise WorldError(f"World zip does not exist: {zip_path}")

    tmp = Path(tempfile.mkdtemp(prefix="apex_agents_world_"))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_subsystem_tarball(scratch_dir: Path, subsystem: str, out_dir: Path) -> Path | None:
    """Build a tar.gz of one subsystem (``filesystem`` or ``.apps_data``) for
    POSTing to the environment's ``/data/populate`` endpoint.

    Returns the tarball path, or ``None`` if the subsystem is absent or empty.
    The shape (dereference=True, arcnames relative to the subsystem root)
    matches the reference example's ``populate_subsystems`` in
    ``vendor/archipelago/examples/hugging_face_task/main.py`` byte-for-byte.
    """
    sub_dir = scratch_dir / subsystem
    if not sub_dir.exists():
        return None
    entries = list(sub_dir.rglob("*"))
    if not entries:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"{subsystem}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.dereference = True  # HF stores files as symlinks to blobs
        for entry in entries:
            tar.add(entry, arcname=str(entry.relative_to(sub_dir)), recursive=False)
    return tar_path


def tar_gz_to_zip(tar_gz_path: Path) -> Path:
    """Convert a tar.gz to a zip in the same directory.

    Mercor's grading runner accepts zip snapshots, while the environment
    container streams snapshots out as tar.gz -- so we convert. Logic
    matches the reference example's helper of the same name.
    """
    stem = tar_gz_path.stem
    if stem.endswith(".tar"):
        stem = stem[:-4]
    zip_path = tar_gz_path.parent / f"{stem}.zip"
    with (
        tarfile.open(tar_gz_path, "r:gz") as tar,
        zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf,
    ):
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                if f is not None:
                    zf.writestr(member.name, f.read())
    return zip_path
