"""Per-run in-flight Dynamic Ledger state for apex-agents-bench.

Mirror of apex_bench's runtime.py, sharing the same shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from apex_agents_bench.dynamic_ledger.config import DynamicLedgerConfig
from apex_agents_bench.dynamic_ledger.embeddings import EmbeddingClient, LiteLLMEmbeddingClient
from apex_agents_bench.dynamic_ledger.entry import DynamicLedger
from apex_agents_bench.dynamic_ledger.store import SnapshotStore

log = logging.getLogger(__name__)


@dataclass
class DynamicLedgerRuntime:
    cfg: DynamicLedgerConfig
    run_dir: Path
    embed: EmbeddingClient
    stores: dict[str, DynamicLedger] = field(default_factory=dict)
    snapshot_stores: dict[str, SnapshotStore] = field(default_factory=dict)
    next_ordinal: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        cfg: DynamicLedgerConfig,
        run_dir: Path,
        completed_per_domain: dict[str, int] | None = None,
        embed: EmbeddingClient | None = None,
    ) -> DynamicLedgerRuntime:
        """Build a runtime, pre-loading the latest snapshot for every domain that
        already has a snapshot directory on disk.

        ``completed_per_domain`` is accepted for backward compatibility with the
        runner's resume bookkeeping but is intentionally NOT used to gate which
        snapshot loads: the snapshot store is the source of truth for ledger
        state, the CSV is only the source of truth for which task_ids have been
        completed (and the runner uses the CSV separately to skip already-run
        tasks). Loading the latest snapshot regardless of the CSV count means
        curator emissions from agent-failed tasks are preserved across resumes.
        """
        if embed is None:
            embed = LiteLLMEmbeddingClient(model=cfg.embedding_model)
        rt = cls(cfg=cfg, run_dir=run_dir, embed=embed)
        # Pre-warm: scan disk for any per-domain snapshot directory and load the
        # latest snapshot from each. Domains that have a snapshot directory but
        # no completed CSV rows (every task failed before grading) are still
        # discovered here so their ledgers persist.
        dl_root = run_dir / "dynamic_ledger"
        if dl_root.is_dir():
            for sub in dl_root.iterdir():
                if not sub.is_dir():
                    continue
                rt.store_for(sub.name)
        return rt

    def store_for(self, domain: str) -> DynamicLedger:
        if domain not in self.stores:
            ss = self.snapshot_stores.get(domain) or SnapshotStore.for_domain(self.run_dir, domain)
            self.snapshot_stores[domain] = ss
            latest = ss.latest()
            if latest is None:
                self.stores[domain] = DynamicLedger(domain=domain)
                self.next_ordinal[domain] = 0
            else:
                idx, loaded = latest
                self.stores[domain] = loaded
                self.next_ordinal[domain] = idx
        return self.stores[domain]

    def current_ordinal_for(self, domain: str) -> int:
        if domain not in self.next_ordinal:
            self.store_for(domain)
        self.next_ordinal[domain] += 1
        return self.next_ordinal[domain]

    def persist(self, domain: str) -> Path:
        store = self.stores[domain]
        index = self.next_ordinal[domain]
        ss = self.snapshot_stores[domain]
        return ss.save(store, index=index)


def dynamic_ledger_csv_fragment_empty() -> dict:
    return {
        "dynamic_ledger_enabled": True,
        "dynamic_ledger_snapshot_index_before": 0,
        "retrieved_entry_count": 0,
        "retrieved_entry_ids": "[]",
        "curator_create_count": 0,
        "curator_create_blocked_count": 0,
        "curator_update_count": 0,
        "curator_delete_count": 0,
        "dynamic_ledger_active_entry_count_after": 0,
        "dynamic_ledger_total_entry_count_after": 0,
        "dynamic_ledger_total_active_chars_after": 0,
        "curator_prompt_tokens": 0,
        "curator_completion_tokens": 0,
        "curator_wall_seconds": 0.0,
        "trajectory_chars_seen_by_curator": 0,
    }
