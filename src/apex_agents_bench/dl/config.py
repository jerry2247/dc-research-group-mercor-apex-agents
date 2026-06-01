"""``DLConfig`` — knobs for the Dynamic Ledger (DL) subsystem.

DL is the third memory architecture in apex-agents-bench, alongside DC-RS
(no ground truth, one monolithic cheatsheet, single synthesizer call
before the agent) and TRACE (uses ground truth, itemised bullets with
counters, reflector + curator after grading, citations).

DL keeps the original Dynamic Ledger mechanics — individual typed entries
indexed for DUAL retrieval (entry content AND source problem), edited by a
curator through typed CRUD operations — but consumes NO ground-truth
signal (like DC-RS, and like the original DL's ``observe`` which ignores
``score``). Per task there is exactly ONE LLM call — the curator — and it
runs AFTER the agent. No grading outcome is threaded into it.

See ``docs/DL_PRD.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DLConfig:
    """Dynamic Ledger configuration.

    No ground-truth bit is threaded into the curator — DL does not consume
    grading outcomes. There is no create-time dedup threshold: every entry
    the curator creates enters the ledger.
    """

    enabled: bool = False
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    top_k: int = 3

    # The curator runs on the SAME model as the active AgentProfile under
    # test; only the judge model is fixed (gpt-5.5 medium). The runner fills
    # these in from the active AgentProfile before the first call. Set
    # explicitly only for ablation studies.
    curator_model: str | None = None
    curator_extra_args: dict | None = None

    curator_temperature: float = 1.0
    curator_max_tokens: int = 24_000
    curator_timeout_seconds: int = 1800

    # apex-agents-bench-specific: cap per-tool-result rendering when building
    # the trajectory transcript the curator reads. Tool-call ARGUMENTS and
    # assistant reasoning text are NOT capped — those are the high-signal
    # portions.
    trajectory_max_chars_per_tool_result: int = 8000
