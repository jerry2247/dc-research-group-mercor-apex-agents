"""Per-domain JSON snapshot persistence. Mirror of apex_bench.memory.store."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from apex_agents_bench.dynamic_ledger.entry import DynamicLedger

_SNAPSHOT_PREFIX = "snapshot_"
_SNAPSHOT_SUFFIX = ".json"
_LOG_NAME = "curator_log.jsonl"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


@dataclass(frozen=True)
class SnapshotStore:
    domain_dir: Path

    @classmethod
    def for_domain(cls, run_dir: Path, domain: str) -> SnapshotStore:
        return cls(domain_dir=run_dir / "dynamic_ledger" / domain)

    def snapshot_path(self, index: int) -> Path:
        return self.domain_dir / f"{_SNAPSHOT_PREFIX}{index:04d}{_SNAPSHOT_SUFFIX}"

    def log_path(self) -> Path:
        return self.domain_dir / _LOG_NAME

    def list_snapshot_indices(self) -> list[int]:
        if not self.domain_dir.is_dir():
            return []
        out: list[int] = []
        for p in self.domain_dir.iterdir():
            name = p.name
            if name.startswith(_SNAPSHOT_PREFIX) and name.endswith(_SNAPSHOT_SUFFIX):
                stem = name[len(_SNAPSHOT_PREFIX) : -len(_SNAPSHOT_SUFFIX)]
                try:
                    out.append(int(stem))
                except ValueError:
                    continue
        out.sort()
        return out

    def save(self, store: DynamicLedger, *, index: int) -> Path:
        path = self.snapshot_path(index)
        payload = store.model_dump(mode="json")
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
        return path

    def load(self, index: int) -> DynamicLedger | None:
        path = self.snapshot_path(index)
        if not path.is_file():
            return None
        return DynamicLedger.model_validate_json(path.read_text(encoding="utf-8"))

    def latest(self) -> tuple[int, DynamicLedger] | None:
        indices = self.list_snapshot_indices()
        if not indices:
            return None
        i = indices[-1]
        store = self.load(i)
        if store is None:
            return None
        return i, store

    def load_for_resume(self, *, max_index_allowed: int, domain: str) -> tuple[int, DynamicLedger]:
        for i in reversed(self.list_snapshot_indices()):
            if i <= max_index_allowed:
                loaded = self.load(i)
                if loaded is not None:
                    return i, loaded
        return 0, DynamicLedger(domain=domain)

    def append_curator_log(self, record: dict) -> None:
        self.domain_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
