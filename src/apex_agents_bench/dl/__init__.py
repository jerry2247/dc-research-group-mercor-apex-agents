"""DL — Dynamic Ledger subsystem for apex-agents-bench.

An adaptation of the original Dynamic Ledger (Dynamic Cheatsheet 2.0): an
itemised memory of individual TYPED entries, indexed for DUAL retrieval
(entry content AND the source problem that produced it), edited by a
curator through typed CRUD operations. DL consumes NO ground-truth signal
— like DC-RS, and like the original DL's ``observe`` which ignores the
score. It is therefore the no-GT, itemised counterpart that sits between
DC-RS (no GT, one monolithic cheatsheet) and TRACE (uses GT, itemised
bullets with counters + citations).

Per task there is exactly ONE LLM call — the curator — and it runs AFTER
the agent:

  1. (Hook A, before the agent) embed the current task prompt → DUAL
     top-k=3 retrieval (content axis + source-problem axis, unioned) from
     THIS DOMAIN's ledger → render the retrieved typed entries, grouped
     under their five category headers, into the generator injection block
     → prepend it to the agent's initial messages. NO LLM call.
  2. (Hook B, after the agent) ONE curator LLM call reads the retrieved
     entries, the current task, and THIS task's trajectory, and emits a
     CREATE / UPDATE / DELETE batch → apply deterministically (no dedup) →
     persist a per-domain snapshot. NO ground-truth signal is consumed.

Entry quality and the curator's voice follow this repo's DC-RS subsystem;
the framework (typed entries, dual retrieval, CRUD) follows the original
Dynamic Ledger. Per-domain isolation: each benchmark domain keeps its own
ledger, so a Finance task never retrieves from a Legal ledger.
"""

from __future__ import annotations

from apex_agents_bench.dl.config import DLConfig
from apex_agents_bench.dl.curator import (
    CuratedOp,
    CuratorResult,
    apply_ops,
    curate,
    parse_ledger_updates,
)
from apex_agents_bench.dl.embeddings import (
    EmbeddingClient,
    LiteLLMEmbeddingClient,
    cosine_similarity,
)
from apex_agents_bench.dl.entry import (
    ENTRY_TYPES,
    TYPE_TO_SECTION,
    DLEntry,
    DLLedger,
)
from apex_agents_bench.dl.formatting import (
    render_entries_for_curator,
    render_entries_for_generator,
)
from apex_agents_bench.dl.injector import augment_initial_messages
from apex_agents_bench.dl.retriever import retrieve
from apex_agents_bench.dl.runtime import DLRuntime, dl_csv_fragment_empty
from apex_agents_bench.dl.store import SnapshotStore
from apex_agents_bench.dl.trajectory_render import render_trajectory_for_curator

__all__ = [
    "ENTRY_TYPES",
    "TYPE_TO_SECTION",
    "CuratedOp",
    "CuratorResult",
    "DLConfig",
    "DLEntry",
    "DLLedger",
    "DLRuntime",
    "EmbeddingClient",
    "LiteLLMEmbeddingClient",
    "SnapshotStore",
    "apply_ops",
    "augment_initial_messages",
    "cosine_similarity",
    "curate",
    "dl_csv_fragment_empty",
    "parse_ledger_updates",
    "render_entries_for_curator",
    "render_entries_for_generator",
    "render_trajectory_for_curator",
    "retrieve",
]
