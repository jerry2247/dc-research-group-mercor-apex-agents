"""``DynamicLedgerConfig`` — knobs for the Dynamic Ledger subsystem.

See ``docs/DYNAMIC_LEDGER_PRD.md``. The shared fields mirror the sister
apex-bench config; this file adds ``trajectory_max_chars_per_tool_result``
to bound the curator's view of long tool results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicLedgerConfig:
    """Dynamic Ledger configuration. No GT signal anywhere in this struct.

    The curator runs on the SAME model as the agent profile under test
    (and uses the same thinking effort): if the agent is grok-4.3-high
    the curator is also grok-4.3-high, if the agent is gpt-5.5-medium
    the curator is also gpt-5.5-medium. Only the judge model is fixed
    (gpt-5.5 medium). Leave ``curator_model`` and
    ``curator_extra_args`` ``None``; the runner fills them in from the
    selected :class:`AgentProfile` before calling the curator.

    Setting them explicitly is allowed for experiments (e.g., curator-
    ablation studies that hold the curator model fixed across agent
    profiles). The CLI does not surface those knobs.
    """

    enabled: bool = False
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    top_k_per_axis: int = 5
    # Minimum cosine similarity for an entry to actually be injected into the
    # generator prompt. Top-k still selects up to ``top_k_per_axis`` candidates
    # per axis, but any candidate below this floor is dropped. Set to 0.0 to
    # restore the original "always inject top-k" behaviour. The default 0.40
    # was chosen empirically to suppress weakly-related entries that were
    # destabilising the generator on tasks where the retrieved entries were
    # only superficially relevant.
    retrieval_similarity_threshold: float = 0.40
    # The curator model is the agent profile's orchestrator model by
    # default. Set explicitly only for curator-ablation experiments.
    curator_model: str | None = None
    curator_extra_args: dict | None = None
    curator_max_tokens: int = 16_000
    curator_temperature: float = 1.0
    curator_timeout_seconds: int = 1800
    create_time_similarity_threshold: float = 0.85
    per_domain_ledger: bool = True
    snapshot_every_problem: bool = True

    # apex-agents-bench-specific: cap per-tool-result rendering when
    # building the curator's view of the trajectory. Tool-call ARGUMENTS
    # and assistant reasoning text are NOT capped — those are the
    # high-signal portions.
    trajectory_max_chars_per_tool_result: int = 4000


