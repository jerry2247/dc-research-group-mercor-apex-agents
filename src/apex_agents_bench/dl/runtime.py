"""Per-run in-flight DL state.

Holds one ledger per benchmark domain, the embedding client, and the run
directory where snapshots are persisted. Each domain is fully isolated:
retrieval for a Finance task never sees Legal entries, and the Legal
ledger does not carry over to Finance. Constructed once by the runner at
the start of a run; mutated as tasks complete. On resume the latest
snapshot for every domain with an on-disk directory is pre-loaded.

Mirrors ``TraceRuntime`` (snapshot-based) rather than ``DCRSRuntime``
(append-only bank), because DL stores itemised, individually-mutable
entries with soft-delete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from apex_agents_bench.dl.config import DLConfig
from apex_agents_bench.dl.embeddings import EmbeddingClient, LiteLLMEmbeddingClient
from apex_agents_bench.dl.entry import DLLedger
from apex_agents_bench.dl.store import SnapshotStore

log = logging.getLogger(__name__)


@dataclass
class DLRuntime:
    cfg: DLConfig
    run_dir: Path
    embed: EmbeddingClient
    ledgers: dict[str, DLLedger] = field(default_factory=dict)
    snapshot_stores: dict[str, SnapshotStore] = field(default_factory=dict)
    next_ordinal: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        cfg: DLConfig,
        run_dir: Path,
        embed: EmbeddingClient | None = None,
    ) -> DLRuntime:
        """Build a runtime, pre-loading the latest snapshot for every domain
        that already has a snapshot directory on disk (resume).

        The snapshot store is the source of truth for ledger state; the CSV
        is only the source of truth for which task_ids have been completed.
        Loading the latest snapshot regardless of the CSV count means curator
        emissions from agent-failed tasks are preserved across resumes.
        """
        if embed is None:
            embed = LiteLLMEmbeddingClient(model=cfg.embedding_model)
        rt = cls(cfg=cfg, run_dir=run_dir, embed=embed)
        dl_root = run_dir / "dl"
        if dl_root.is_dir():
            for sub in dl_root.iterdir():
                if not sub.is_dir():
                    continue
                rt.ledger_for(sub.name)
        return rt

    def ledger_for(self, domain: str) -> DLLedger:
        if domain not in self.ledgers:
            ss = self.snapshot_stores.get(domain) or SnapshotStore.for_domain(self.run_dir, domain)
            self.snapshot_stores[domain] = ss
            latest = ss.latest()
            if latest is None:
                self.ledgers[domain] = DLLedger(domain=domain)
                self.next_ordinal[domain] = 0
            else:
                idx, loaded = latest
                self.ledgers[domain] = loaded
                self.next_ordinal[domain] = idx
        return self.ledgers[domain]

    def current_ordinal_for(self, domain: str) -> int:
        if domain not in self.next_ordinal:
            self.ledger_for(domain)
        self.next_ordinal[domain] += 1
        return self.next_ordinal[domain]

    def persist(self, domain: str) -> Path:
        ledger = self.ledgers[domain]
        index = self.next_ordinal[domain]
        ss = self.snapshot_stores[domain]
        return ss.save(ledger, index=index)


def dl_csv_fragment_empty() -> dict:
    """Default fragment for the DL CSV columns when the run is on but a
    per-task error prevents filling individual fields."""
    return {
        "dl_enabled": True,
        "dl_snapshot_index_before": 0,
        "dl_retrieved_count": 0,
        "dl_retrieved_entry_ids": "[]",
        "dl_curator_create_count": 0,
        "dl_curator_update_count": 0,
        "dl_curator_delete_count": 0,
        "dl_curator_skipped_invalid_id_count": 0,
        "dl_curator_parse_error": "",
        "dl_active_entry_count_after": 0,
        "dl_total_entry_count_after": 0,
        "dl_total_active_chars_after": 0,
        "dl_curator_prompt_tokens": 0,
        "dl_curator_completion_tokens": 0,
        "dl_curator_wall_seconds": 0.0,
        "dl_trajectory_chars_seen_by_curator": 0,
    }
