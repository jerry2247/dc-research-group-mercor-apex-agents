"""Registry of agent profiles for APEX-Agents runs.

A *profile* is a named (LiteLLM model name, ``orchestrator_extra_args``)
combination. We use named profiles instead of free-form CLI flags so that:

  1. The set of models we run against is explicit and reviewable.
  2. Per-provider config quirks (``reasoning_effort`` vs ``thinking_tokens``,
     OpenAI's ``verbosity``, Grok's ``temperature``, DeepSeek's ``max``
     effort) live in one place.
  3. A run's record can persist a single profile name and be reproduced
     exactly later.

Picking a model: ``apex-agents-bench smoke --model gpt-5.5-medium``.
Listing all profiles: ``apex-agents-bench models``.

Adding a new profile is a code change here, not a CLI flag -- by design.

Why no ``max_tokens`` / ``max_input_tokens`` here (unlike apex-bench's
``test_models.py``): Archipelago's agent runner does not pass these into
``ModelConfig`` because there is no ``ModelConfig`` -- it calls
``litellm.acompletion(model=..., messages=..., **extra_args)`` directly. The
provider's defaults apply unless we explicitly include them in
``extra_args``. We do not, because Mercor's published example does not
either -- token caps are provider-side and we inherit them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentProfile:
    """One named (test-model, mode) combination.

    ``orchestrator_model`` is the LiteLLM-routable string passed as
    ``--orchestrator-model`` to the vendored agent runner.

    ``orchestrator_extra_args`` is the dict passed as the
    ``--orchestrator-extra-args`` JSON file, splatted directly into
    ``litellm.acompletion(**extra_args)`` by the vendor's ``generate_response``.
    """

    name: str
    """Slug the user types: ``gpt-5.5-medium``, ``grok-4.3-high``, ..."""

    family: str
    """Display family: ``gpt-5.5``, ``grok-4.3``, ..."""

    provider: str
    """``openai`` | ``xai`` | ``deepseek`` | ``anthropic-bedrock``. Affects
    which API key must be set."""

    orchestrator_model: str
    """LiteLLM-routable string. E.g. ``openai/gpt-5.5``."""

    orchestrator_extra_args: dict[str, Any] = field(default_factory=dict)
    """Extra LiteLLM kwargs splatted at the call site."""

    notes: str = ""
    """One-line summary printed by ``apex-agents-bench models``."""

    def to_extra_args_json(self) -> dict[str, Any]:
        """Return the kwargs that go into ``orchestrator_extra_args.json``."""
        return dict(self.orchestrator_extra_args)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
#
# Per-provider notes (justifying the values below):
#
# OpenAI GPT-5.5
#   - reasoning_effort: low | medium | high | xhigh. Default is medium.
#   - verbosity: low | medium | high. We pin verbosity=medium to match
#     apex-bench's sister profile registry.
#   - temperature: GPT-5.5 in reasoning mode rejects custom temperature; we
#     omit it.
#   - LiteLLM routing: ``openai/gpt-5.5``.
#
# xAI Grok 4.3
#   - reasoning_effort: low | medium | high (three tiers).
#   - temperature: accepted; we use 0.8 to mirror apex-bench's choice (which
#     mirrors Mercor's apex-evals upstream grok-4-0709 entry).
#   - LiteLLM routing: ``xai/grok-4.3``.
#
# DeepSeek V4 Pro
#   - reasoning_effort: high | max. We register ONLY max, by policy.
#   - DeepSeek documents max effort as `thinking.reasoning_effort=max`.
#     LiteLLM 1.83.x collapses top-level `reasoning_effort` into
#     `thinking: {"type": "enabled"}` for the native deepseek/ adapter, so
#     we send the exact DeepSeek body through `extra_body`.
#   - temperature/top_p are omitted; DeepSeek's thinking-mode docs say they
#     have no effect in thinking mode.
#   - LiteLLM routing: ``deepseek/deepseek-v4-pro`` with the official
#     OpenAI-compatible base URL.
#
# Anthropic Claude on AWS Bedrock -- DEFERRED.
#   The Archipelago call path passes the model string through to LiteLLM
#   verbatim, so technically `bedrock/us.anthropic.claude-opus-4-6-v1:0`
#   would work AS SOON AS AWS credentials are configured. We still defer
#   because apex-bench's sister project defers (no Bedrock plumbing on the
#   apex-evals side yet) and we want the two repos' active model surface to
#   stay in lockstep until Bedrock lands jointly. See
#   ``docs/IMPLEMENTATION_PLAN.md`` Phase 2.


_REGISTRY: dict[str, AgentProfile] = {}


def _add(p: AgentProfile) -> None:
    if p.name in _REGISTRY:
        raise RuntimeError(f"duplicate profile name: {p.name}")
    _REGISTRY[p.name] = p


# --- OpenAI GPT-5.5 ---------------------------------------------------------

for _effort in ("low", "medium", "high", "xhigh"):
    _add(
        AgentProfile(
            name=f"gpt-5.5-{_effort}",
            family="gpt-5.5",
            provider="openai",
            orchestrator_model="openai/gpt-5.5",
            orchestrator_extra_args={
                "reasoning_effort": _effort,
                "verbosity": "medium",
                # 1800s (30 min) per LLM call. Splatted into
                # litellm.acompletion(**extra_args). Without this LiteLLM
                # applies its 600s default, which is too short for
                # reasoning_effort=high on long agent contexts (post-ReSum
                # summaries + the full toolbelt schema can run into tens of
                # thousands of tokens). The vendor's AGENT_TIMEOUT_SECONDS
                # (3600s, set in config.py) still caps the whole agent loop.
                "timeout": 1800,
            },
            notes=f"OpenAI GPT-5.5, reasoning_effort={_effort}, verbosity=medium.",
        )
    )

# --- xAI Grok 4.3 -----------------------------------------------------------

for _effort in ("low", "medium", "high"):
    _add(
        AgentProfile(
            name=f"grok-4.3-{_effort}",
            family="grok-4.3",
            provider="xai",
            orchestrator_model="xai/grok-4.3",
            orchestrator_extra_args={
                "reasoning_effort": _effort,
                "temperature": 0.8,
                # 1800s (30 min) per LLM call. See the gpt-5.5 block above
                # for rationale; same LiteLLM 600s default is too short on
                # long agent contexts.
                "timeout": 1800,
            },
            notes=f"xAI Grok 4.3, reasoning_effort={_effort}, temperature=0.8.",
        )
    )

# --- DeepSeek V4 Pro --------------------------------------------------------

_add(
    AgentProfile(
        name="deepseek-v4-pro-max",
        family="deepseek-v4-pro",
        provider="deepseek",
        orchestrator_model="deepseek/deepseek-v4-pro",
        orchestrator_extra_args={
            "api_base": "https://api.deepseek.com",
            "extra_body": {
                "thinking": {
                    "type": "enabled",
                    "reasoning_effort": "max",
                }
            },
            # 1800s (30 min) per LLM call. See the gpt-5.5 block above
            # for rationale; same LiteLLM 600s default is too short on
            # long agent contexts.
            "timeout": 1800,
        },
        notes="DeepSeek V4 Pro, thinking.reasoning_effort=max.",
    )
)

# --- Anthropic Claude on AWS Bedrock -- DEFERRED ----------------------------
# See module docstring for rationale. Sketched here for the future:
_DEFERRED_CLAUDE_PROFILES_NOTE = """
Claude profile shape (to be enabled once apex-bench's Bedrock plumbing lands):

  claude-opus-4.6:   bedrock/us.anthropic.claude-opus-4-6-v1:0
  claude-sonnet-4.6: bedrock/us.anthropic.claude-sonnet-4-6-v1:0
  claude-haiku-4.5:  bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0

Each with three thinking tiers (off / medium / high).
"""


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def all_profiles() -> list[AgentProfile]:
    """All registered profiles, in insertion order (= display order)."""
    return list(_REGISTRY.values())


def profile_names() -> list[str]:
    return list(_REGISTRY.keys())


def get_profile(name: str) -> AgentProfile:
    """Look up a profile by name; raise KeyError with a helpful message."""
    if name not in _REGISTRY:
        suggestions = [n for n in _REGISTRY if name.split("-")[0] in n]
        suggestion_text = f" Did you mean one of: {suggestions}?" if suggestions else ""
        raise KeyError(
            f"Unknown agent profile {name!r}. "
            f"Use `apex-agents-bench models` to list available profiles.{suggestion_text}"
        )
    return _REGISTRY[name]


def profiles_by_family() -> dict[str, list[AgentProfile]]:
    out: dict[str, list[AgentProfile]] = {}
    for p in _REGISTRY.values():
        out.setdefault(p.family, []).append(p)
    return out


__all__ = [
    "AgentProfile",
    "all_profiles",
    "get_profile",
    "profile_names",
    "profiles_by_family",
]
