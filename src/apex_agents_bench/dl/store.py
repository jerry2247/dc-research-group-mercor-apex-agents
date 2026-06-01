"""Per-domain on-disk snapshot store + resume reconciliation (DL).

Snapshots live at ``<run_dir>/dl/<Domain>/snapshot_<NNNN>.json``. A
sidecar ``curator_log.jsonl`` records one line per curator call. The
layout and resume semantics mirror the TRACE ``SnapshotStore``: the
latest snapshot on disk is the source of truth for ledger state; the
results CSV is the source of truth only for which tasks completed.
Soft-deleted entries are preserved in the snapshot so a replay is exact
and ids are never reused.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from apex_agents_bench.dl.entry import DLLedger

log = logging.getLogger(__name__)

_SNAPSHOT_RE = re.compile(r"^snapshot_(\d{4,})\.json$")


@dataclass
class SnapshotStore:
    domain_dir: Path

    @classmethod
    def for_domain(cls, run_dir: Path, domain: str) -> SnapshotStore:
        d = run_dir / "dl" / domain
        d.mkdir(parents=True, exist_ok=True)
        return cls(domain_dir=d)

    def snapshot_path(self, index: int) -> Path:
        return self.domain_dir / f"snapshot_{index:04d}.json"

    def save(self, ledger: DLLedger, *, index: int) -> Path:
        p = self.snapshot_path(index)
        p.write_text(ledger.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return p

    def latest(self) -> tuple[int, DLLedger] | None:
        idxs: list[int] = []
        for f in self.domain_dir.iterdir():
            m = _SNAPSHOT_RE.match(f.name)
            if m:
                idxs.append(int(m.group(1)))
        if not idxs:
            return None
        idx = max(idxs)
        return idx, DLLedger.model_validate_json(
            self.snapshot_path(idx).read_text(encoding="utf-8")
        )

    def append_curator_log(self, record: dict) -> None:
        p = self.domain_dir / "curator_log.jsonl"
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
