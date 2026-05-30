"""DC-RS (Dynamic Cheatsheet — Retrieval Synthesis) subsystem for apex-agents-bench.

A faithful port of Suzgun et al.'s DC-RS (arXiv:2504.07952) adapted to
the agentic, tool-use, code-execution apex-agents-bench harness, with
per-domain isolation: each benchmark domain keeps its own pool and its
own cheatsheet slot, so a Finance task never retrieves from a Legal pool
and the Legal cheatsheet does not bleed into the Finance run.

Per task there is exactly ONE LLM call — the synthesizer — and it runs
BEFORE the agent:

  1. (Hook A, before the agent) embed the current task prompt → retrieve
     top-k=3 most similar ``(task_prompt, rendered_trajectory)`` pairs
     from THIS DOMAIN's pool → one synthesizer LLM call produces a fresh
     ``<cheatsheet>`` (informed by the domain's previous cheatsheet and
     the retrieved past trajectories) → apply the anti-wipe guard →
     write/archive the cheatsheet → inject it into the agent's initial
     messages.
  2. (Hook B, after the agent) append the new
     ``(task_prompt, rendered_trajectory, embedding)`` triple to the
     domain's pool. NO LLM call here.

No ground-truth signal reaches the synthesizer. There is no post-task
curator and no ops list: the per-domain cheatsheet is replaced whole
each task (copy-forward) and the per-domain pool is append-only.
"""

from __future__ import annotations

from apex_agents_bench.dc_rs.bank import Bank, BankEntry
from apex_agents_bench.dc_rs.config import DCRSConfig
from apex_agents_bench.dc_rs.curation import apply_wipe_guard
from apex_agents_bench.dc_rs.extract import extract_cheatsheet
from apex_agents_bench.dc_rs.formatting import format_retrieved_cases
from apex_agents_bench.dc_rs.injector import augment_initial_messages
from apex_agents_bench.dc_rs.retriever import Retrieved, retrieve
from apex_agents_bench.dc_rs.synthesizer import SynthesizerResult, synthesize
from apex_agents_bench.dc_rs.trajectory_render import render_trajectory_for_synthesizer

__all__ = [
    "Bank",
    "BankEntry",
    "DCRSConfig",
    "Retrieved",
    "SynthesizerResult",
    "apply_wipe_guard",
    "augment_initial_messages",
    "extract_cheatsheet",
    "format_retrieved_cases",
    "render_trajectory_for_synthesizer",
    "retrieve",
    "synthesize",
]
