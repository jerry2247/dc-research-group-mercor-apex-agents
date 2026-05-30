"""World materialization tests.

We test the local code paths only (subsystem tarball building, tar-to-zip
conversion, scratch-dir lifecycle). HF download is exercised separately
under the ``network`` marker.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from apex_agents_bench.world import (
    SUBSYSTEMS,
    has_populatable_subsystem,
    make_subsystem_tarball,
    materialize_world,
    tar_gz_to_zip,
    world_zip_path,
)


def test_subsystem_set() -> None:
    """Matches the published reference example."""
    assert SUBSYSTEMS == ("filesystem", ".apps_data")


def test_world_zip_path_shape(tmp_path: Path) -> None:
    p = world_zip_path(tmp_path, "world_42")
    assert p == tmp_path / "world_files_zipped" / "world_42.zip"


def test_has_populatable_subsystem_requires_consumed_subsystem(tmp_path: Path) -> None:
    assert has_populatable_subsystem(tmp_path / "missing") is False

    (tmp_path / "random").mkdir()
    (tmp_path / "random" / "note.txt").write_text("ignored", encoding="utf-8")
    assert has_populatable_subsystem(tmp_path) is False

    (tmp_path / "filesystem").mkdir()
    (tmp_path / "filesystem" / "starter.txt").write_text("used", encoding="utf-8")
    assert has_populatable_subsystem(tmp_path) is True


# -----------------------------------------------------------------------------


def _make_zip_world(zip_dest: Path) -> None:
    """Build a tiny world zip with both subsystems populated."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "filesystem").mkdir()
        (root / "filesystem" / "hello.txt").write_text("hi", encoding="utf-8")
        (root / ".apps_data").mkdir()
        (root / ".apps_data" / "calendar").mkdir()
        (root / ".apps_data" / "calendar" / "events.json").write_text("[]", encoding="utf-8")
        with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in root.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(root)))


def test_materialize_world_then_cleanup(tmp_path: Path) -> None:
    zip_dest = tmp_path / "world_1.zip"
    _make_zip_world(zip_dest)

    captured: Path | None = None
    with materialize_world(zip_dest) as scratch:
        captured = scratch
        assert (scratch / "filesystem" / "hello.txt").is_file()
        assert (scratch / ".apps_data" / "calendar" / "events.json").is_file()
    # Scratch is gone after exit.
    assert captured is not None
    assert not captured.exists()


def test_make_subsystem_tarball_skips_empty(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    # No subsystem directory present.
    assert make_subsystem_tarball(tmp_path, "filesystem", out_dir) is None


def test_make_subsystem_tarball_includes_files(tmp_path: Path) -> None:
    sub_root = tmp_path / "filesystem"
    sub_root.mkdir()
    (sub_root / "a.txt").write_text("a", encoding="utf-8")
    (sub_root / "nested").mkdir()
    (sub_root / "nested" / "b.txt").write_text("b", encoding="utf-8")
    out_dir = tmp_path / "out"
    tar = make_subsystem_tarball(tmp_path, "filesystem", out_dir)
    assert tar is not None and tar.is_file()
    with tarfile.open(tar, "r:gz") as tf:
        names = {m.name for m in tf.getmembers()}
    # Arcnames are relative to the subsystem root.
    assert "a.txt" in names
    assert "nested/b.txt" in names


def test_tar_gz_to_zip_roundtrip(tmp_path: Path) -> None:
    # Build a tar.gz with one file.
    src = tmp_path / "snap.tar.gz"
    payload = tmp_path / "x.txt"
    payload.write_text("payload", encoding="utf-8")
    with tarfile.open(src, "w:gz") as tf:
        tf.add(payload, arcname="x.txt")

    z = tar_gz_to_zip(src)
    assert z.exists()
    with zipfile.ZipFile(z) as zf:
        assert "x.txt" in zf.namelist()
        assert zf.read("x.txt") == b"payload"
