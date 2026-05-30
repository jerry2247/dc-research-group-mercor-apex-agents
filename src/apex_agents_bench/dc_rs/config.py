"""``DCRSConfig`` — knobs for the DC-RS subsystem (apex-agents-bench).

DC-RS makes exactly one synthesizer LLM call per task. The synthesizer
runs on the same model as the active AgentProfile under test; only the
judge model is fixed (gpt-5.5 medium). No ground-truth bit is threaded
into the synthesizer — DC-RS does not consume grading outcomes.

This is the agentic port of the sibling apex-bench ``DCRSConfig``; it
adds ``trajectory_max_chars_per_tool_result`` to bound how much of each
tool result is rendered into the past-case transcripts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DCRSConfig:
    """DC-RS configuration. No ground-truth bit is threaded into the
    synthesizer — DC-RS does not consume grading outcomes."""

    enabled: bool = False
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    top_k: int = 3

    # Filled in from the active AgentProfile by the runner.
    synthesizer_model: str | None = None
    synthesizer_extra_args: dict | None = None

    synthesizer_temperature: float = 1.0
    synthesizer_max_tokens: int = 24_000
    synthesizer_timeout_seconds: int = 1800

    # apex-agents-bench-specific: cap per-tool-result rendering when
    # building the past-case transcripts the synthesizer reads. Tool-call
    # ARGUMENTS and assistant reasoning text are NOT capped — those are the
    # high-signal portions.
    trajectory_max_chars_per_tool_result: int = 4000
