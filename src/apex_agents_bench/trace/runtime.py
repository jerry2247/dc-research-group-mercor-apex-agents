"""Per-run in-flight TRACE state."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from apex_agents_bench.trace.bullet import TraceLedger
from apex_agents_bench.trace.config import TraceConfig
from apex_agents_bench.trace.embeddings import EmbeddingClient, LiteLLMEmbeddingClient
from apex_agents_bench.trace.store import SnapshotStore

log = logging.getLogger(__name__)


@dataclass
class TraceRuntime:
    cfg: TraceConfig
    run_dir: Path
    embed: EmbeddingClient
    stores: dict[str, TraceLedger] = field(default_factory=dict)
    snapshot_stores: dict[str, SnapshotStore] = field(default_factory=dict)
    next_ordinal: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        cfg: TraceConfig,
        run_dir: Path,
        completed_per_domain: dict[str, int] | None = None,
        embed: EmbeddingClient | None = None,
    ) -> "TraceRuntime":
        """Build a runtime, pre-loading the latest snapshot for every domain that
        already has a snapshot directory on disk.

        ``completed_per_domain`` is accepted for backward compatibility with the
        runner's resume bookkeeping but is intentionally NOT used to gate which
        snapshot loads: the snapshot store is the source of truth for cheatsheet
        state, the CSV is only the source of truth for which task_ids have been
        completed. Loading the latest snapshot regardless of the CSV count means
        curator emissions from agent-failed tasks are preserved across resumes.
        """
        if embed is None:
            embed = LiteLLMEmbeddingClient(model=cfg.embedding_model)
        rt = cls(cfg=cfg, run_dir=run_dir, embed=embed)
        trace_root = run_dir / "trace"
        if trace_root.is_dir():
            for sub in trace_root.iterdir():
                if not sub.is_dir():
                    continue
                rt.store_for(sub.name)
        return rt

    def store_for(self, domain: str) -> TraceLedger:
        if domain not in self.stores:
            ss = self.snapshot_stores.get(domain) or SnapshotStore.for_domain(self.run_dir, domain)
            self.snapshot_stores[domain] = ss
            latest = ss.latest()
            if latest is None:
                self.stores[domain] = TraceLedger(domain=domain)
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

    def record_citations(self, domain: str, *, cited: list[str], gt_correct: bool) -> int:
        store = self.store_for(domain)
        n = 0
        for bid in cited:
            if store.record_citation(bid, gt_correct=gt_correct):
                n += 1
        return n


def trace_csv_fragment_empty() -> dict:
    return {
        "trace_enabled": True,
        "trace_snapshot_index_before": 0,
        "retrieved_bullet_count": 0,
        "retrieved_bullet_ids": "[]",
        "citations_present": False,
        "citations_count": 0,
        "citations_malformed_count": 0,
        "gt_correct_bit": False,
        "reflector_proposal_count": 0,
        "curator_create_count": 0,
        "curator_create_blocked_count": 0,
        "curator_update_count": 0,
        "curator_delete_count": 0,
        "curator_consolidate_count": 0,
        "curator_no_op": False,
        "trace_active_bullet_count_after": 0,
        "trace_total_bullet_count_after": 0,
        "trace_total_active_chars_after": 0,
        "reflector_prompt_tokens": 0,
        "reflector_completion_tokens": 0,
        "reflector_wall_seconds": 0.0,
        "curator_prompt_tokens": 0,
        "curator_completion_tokens": 0,
        "curator_wall_seconds": 0.0,
        "trajectory_chars_seen_by_curator": 0,
    }
