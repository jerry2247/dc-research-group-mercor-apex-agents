"""``TraceConfig`` — knobs for the TRACE subsystem.

TRACE (Tool-augmented Reasoning via Atomic Cheatsheet Editing,
Liao/Nair/Yang) is a memory mechanism where:

  - the generator is told it may cite atomic strategy bullets from a
    shared cheatsheet,
  - a **reflector** then reads (cheatsheet, problem, work, GROUND TRUTH
    correctness bit) and proposes operations on the cheatsheet,
  - a **curator** reads the same inputs plus the reflector's proposals
    and applies the final operations.

We keep everything the paper recommends EXCEPT the SFT step. Per the
project goal:
  * GT is used (the boolean ``criteria_passed == criteria_total``).
  * Atomic bullets are free-form — no length cap.
  * OpenAI embeddings.
  * Reflector and curator both run on the same model as the agent
    profile under test; only the judge model is fixed at gpt-5.5
    medium.

See ``docs/TRACE_PRD.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceConfig:
    """TRACE configuration. The GROUND-TRUTH bit (boolean
    correctness) IS threaded into the reflector and curator —
    intentionally, per the TRACE paper."""

    enabled: bool = False
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    top_k_per_axis: int = 8

    # Reflector + curator both run on the agent profile's model. The
    # runner fills these in from the active AgentProfile before the
    # first call. Set explicitly only for ablation studies.
    reflector_model: str | None = None
    curator_model: str | None = None
    model_extra_args: dict | None = None

    reflector_temperature: float = 1.0
    curator_temperature: float = 1.0
    reflector_max_tokens: int = 24_000
    curator_max_tokens: int = 24_000
    reflector_timeout_seconds: int = 1800
    curator_timeout_seconds: int = 1800

    create_time_similarity_threshold: float = 0.85
    per_domain_ledger: bool = True
    snapshot_every_problem: bool = True

    # Trajectory-render cap for the agentic-trajectory transcript. Tool-
    # call arguments and assistant reasoning are NOT truncated; only
    # long tool-result bodies are.
    trajectory_max_chars_per_tool_result: int = 8000
