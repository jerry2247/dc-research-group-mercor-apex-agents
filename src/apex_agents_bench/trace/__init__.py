"""TRACE — Tool-augmented Reasoning via Atomic Cheatsheet Editing
(Liao, Nair, Yang; Stanford CS224N).

A test-time-learning subsystem that USES ground-truth signal (the
boolean ``criteria_passed == criteria_total``) — intentionally,
following the TRACE paper. Distinct from the Dynamic Ledger
subsystem which is no-GT by design.

Pipeline:
  - RETRIEVE     dual top-k cosine into the per-domain cheatsheet
  - INJECT       prepend the cheatsheet block + citation instruction
                 to the agent's user message
  - GENERATE     vendor agent (unmodified), emits citations tag on
                 the last line of final_answer.reasoning
  - CITE         parse + strip the citations tag; bump bullets'
                 usage / helpful / harmful counters based on gt_bit
  - GRADE        vendor grader reads the SHADOW trajectory (citations
                 stripped); produces gt_bit = (criteria_passed ==
                 criteria_total)
  - REFLECT      single LLM call (same model as agent) reads
                 cheatsheet + problem + trajectory + cited_bullets +
                 gt_bit; emits proposed ops
  - CURATE       single LLM call (same model as agent) reads the
                 above + reflector_proposals; emits final ops,
                 applied to the ledger

Per project goal: components mirror the Dynamic Ledger where possible
(embeddings, retrieval, dedup, store), the prompts differ (TRACE-
specific reflector + curator), the reflector and curator BOTH receive
the GT bit, and bullets carry helpful/harmful/usage counters that
condition the reflector + curator's behavior.
"""

from __future__ import annotations

from apex_agents_bench.trace.bullet import Bullet, TraceLedger
from apex_agents_bench.trace.citations import (
    CitationExtract,
    extract_and_strip_citations_from_trajectory,
    write_shadow_trajectory,
)
from apex_agents_bench.trace.config import TraceConfig
from apex_agents_bench.trace.curator import (
    CuratedOp,
    CuratorResult,
    apply_ops,
    curate,
    parse_cheatsheet_updates,
)
from apex_agents_bench.trace.dedup import is_too_similar_to_retrieved
from apex_agents_bench.trace.embeddings import (
    EmbeddingClient,
    LiteLLMEmbeddingClient,
    cosine_similarity,
)
from apex_agents_bench.trace.injector import (
    augment_initial_messages,
    render_bullets_block,
)
from apex_agents_bench.trace.reflector import (
    ReflectorProposal,
    ReflectorResult,
    parse_reflector_proposals,
    reflect,
)
from apex_agents_bench.trace.retriever import retrieve
from apex_agents_bench.trace.runtime import TraceRuntime
from apex_agents_bench.trace.store import SnapshotStore
from apex_agents_bench.trace.trajectory_render import render_trajectory_for_curator

__all__ = [
    "Bullet",
    "CitationExtract",
    "CuratedOp",
    "CuratorResult",
    "EmbeddingClient",
    "LiteLLMEmbeddingClient",
    "ReflectorProposal",
    "ReflectorResult",
    "SnapshotStore",
    "TraceConfig",
    "TraceLedger",
    "TraceRuntime",
    "apply_ops",
    "augment_initial_messages",
    "cosine_similarity",
    "curate",
    "extract_and_strip_citations_from_trajectory",
    "is_too_similar_to_retrieved",
    "parse_cheatsheet_updates",
    "parse_reflector_proposals",
    "reflect",
    "render_bullets_block",
    "render_trajectory_for_curator",
    "retrieve",
    "write_shadow_trajectory",
]
