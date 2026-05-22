"""Per-domain on-disk snapshot store + resume reconciliation.

Snapshots live at
``<run_dir>/trace/<Domain>/snapshot_<NNNN>.json``. A sidecar
``curator_log.jsonl`` records one line per (reflector, curator) cycle.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from apex_agents_bench.trace.bullet import TraceLedger

log = logging.getLogger(__name__)

_SNAPSHOT_RE = re.compile(r"^snapshot_(\d{4,})\.json$")


@dataclass
class SnapshotStore:
    domain_dir: Path

    @classmethod
    def for_domain(cls, run_dir: Path, domain: str) -> "SnapshotStore":
        d = run_dir / "trace" / domain
        d.mkdir(parents=True, exist_ok=True)
        return cls(domain_dir=d)

    def snapshot_path(self, index: int) -> Path:
        return self.domain_dir / f"snapshot_{index:04d}.json"

    def save(self, store: TraceLedger, *, index: int) -> Path:
        p = self.snapshot_path(index)
        p.write_text(store.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return p

    def latest(self) -> tuple[int, TraceLedger] | None:
        idxs: list[int] = []
        for f in self.domain_dir.iterdir():
            m = _SNAPSHOT_RE.match(f.name)
            if m:
                idxs.append(int(m.group(1)))
        if not idxs:
            return None
        idx = max(idxs)
        return idx, TraceLedger.model_validate_json(self.snapshot_path(idx).read_text(encoding="utf-8"))

    def load_for_resume(self, *, domain: str) -> tuple[int, TraceLedger]:
        """Load the highest snapshot on disk for this domain.

        Returns ``(0, empty-store)`` when no snapshots exist. The snapshot
        store is the source of truth for cheatsheet state; the results CSV
        is only the source of truth for which tasks have been completed.
        Loading the latest snapshot ensures curator emissions from
        agent-failed tasks are preserved across resumes.
        """
        candidates: list[int] = []
        for f in self.domain_dir.iterdir():
            m = _SNAPSHOT_RE.match(f.name)
            if m:
                candidates.append(int(m.group(1)))
        if not candidates:
            return 0, TraceLedger(domain=domain)
        idx = max(candidates)
        store = TraceLedger.model_validate_json(
            self.snapshot_path(idx).read_text(encoding="utf-8")
        )
        return idx, store

    def append_curator_log(self, record: dict) -> None:
        p = self.domain_dir / "curator_log.jsonl"
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
