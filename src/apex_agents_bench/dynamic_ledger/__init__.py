"""Dynamic Ledger — no-ground-truth subsystem for apex-agents-bench.

Per ``docs/DYNAMIC_LEDGER_PRD.md`` The Dynamic Ledger is gated by
``DynamicLedgerConfig.enabled``; when off, importing this package has no
effect on the baseline pipeline and the CSV schema is byte-identical to
the no-ledger path.

The curator function signature is intentionally narrow — see
``curator.curate`` and the load-bearing fidelity test
``test_curator_signature_has_no_outcome``. No ground-truth signal
(per-criterion score, aggregate pass/fail bit, expected answer, judge
rationale) is part of any function in this package.
"""

from __future__ import annotations

from apex_agents_bench.dynamic_ledger.config import DynamicLedgerConfig
from apex_agents_bench.dynamic_ledger.curator import (
    CuratedOp,
    CuratorResult,
    apply_ops,
    curate,
    parse_memory_updates,
)
from apex_agents_bench.dynamic_ledger.dedup import is_too_similar_to_retrieved
from apex_agents_bench.dynamic_ledger.embeddings import (
    EmbeddingClient,
    LiteLLMEmbeddingClient,
    cosine_similarity,
)
from apex_agents_bench.dynamic_ledger.entry import DynamicLedger, Entry
from apex_agents_bench.dynamic_ledger.injector import (
    augment_initial_messages,
    render_entries_block,
)
from apex_agents_bench.dynamic_ledger.retriever import retrieve
from apex_agents_bench.dynamic_ledger.runtime import DynamicLedgerRuntime
from apex_agents_bench.dynamic_ledger.store import SnapshotStore
from apex_agents_bench.dynamic_ledger.trajectory_render import render_trajectory_for_curator

__all__ = [
    "CuratedOp",
    "CuratorResult",
    "DynamicLedger",
    "DynamicLedgerConfig",
    "DynamicLedgerRuntime",
    "EmbeddingClient",
    "Entry",
    "LiteLLMEmbeddingClient",
    "SnapshotStore",
    "apply_ops",
    "augment_initial_messages",
    "cosine_similarity",
    "curate",
    "is_too_similar_to_retrieved",
    "parse_memory_updates",
    "render_entries_block",
    "render_trajectory_for_curator",
    "retrieve",
]
