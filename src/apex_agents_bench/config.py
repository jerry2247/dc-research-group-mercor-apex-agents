"""Typed run configuration for apex-agents-bench.

A ``Settings`` instance is the single source of truth for a run: judge model,
agent caps, MCP server list, paths. CLI flags override; env vars do not (env
is only for credentials).

Project policies (these are NOT knobs):
  - RUNS_PER_TASK = 1, always. Mirror of apex-bench. Every reported number in
    this project uses one run per (task, model). Variance signal comes from
    per-domain bins (n=160) and per-criterion granularity.
  - JUDGE = gpt-5.5 at OpenAI's default medium reasoning effort. Same judge
    as apex-bench keeps cross-benchmark numbers comparable. Deliberate diff
    from Archipelago's example default (gemini/gemini-2.5-flash).
  - AGENT_MAX_STEPS = 50, AGENT_TIMEOUT_SECONDS = 3600. These match Mercor's
    published ``examples/hugging_face_task/agent_config.json`` exactly, NOT
    the agent registry's higher defaults (250 / 10800) which the published
    example overrides.
  - MCP_SERVERS = the 9-server set from ``mcp_config_all_oss_servers.json``.
    Subsetting tools would change task semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from apex_agents_bench.paths import (
    archipelago_example_dir,
    default_dataset_dir,
    runs_dir,
)

# --- Policy defaults (do not change casually) --------------------------------

DEFAULT_JUDGE_MODEL = "openai/gpt-5.5"
"""GPT-5.5 routed through LiteLLM's openai/ prefix. No reasoning_effort
override => OpenAI's default of ``medium`` applies. This is the project judge
across every run; only the agent model varies.

Note the leading ``openai/`` -- Archipelago's grading runner passes this
string verbatim to ``litellm.acompletion(model=...)``, and the example uses
the same provider-prefix form (``gemini/gemini-2.5-flash``)."""

DEFAULT_JUDGE_TIMEOUT_SECONDS = 600
"""LiteLLM request timeout for a single judge call. Matches the vendor's
default in ``grading/runner/utils/llm.py``."""

RUNS_PER_TASK = 1
"""Project policy: one run per (task, model). Not a knob."""

AGENT_CONFIG_ID = "react_toolbelt_agent"
"""The vendor's published agent, used unchanged. The DC-RS and TRACE
subsystems operate via pre/post-agent runner hooks and prompt injection;
neither substitutes the agent or registers a different ``agent_config_id``."""

AGENT_MAX_STEPS = 50
"""Maximum ReAct steps per task. Matches Mercor's published example value,
NOT the agent registry's default of 250."""

AGENT_TIMEOUT_SECONDS = 3600
"""Per-task wall-clock cap. Matches Mercor's published example value, NOT
the agent registry's default of 10800."""

# Verbatim copy of vendor/archipelago/examples/hugging_face_task/mcp_config_all_oss_servers.json
# The 9-server set every task runs with. Subset would change benchmark semantics.
MCP_SERVERS: tuple[str, ...] = (
    "calendar_server",
    "chat_server",
    "code_execution_server",
    "sheets_server",
    "filesystem_server",
    "mail_server",
    "pdf_server",
    "slides_server",
    "docs_server",
)

VALID_DOMAINS: tuple[str, ...] = ("Investment Banking", "Law", "Management Consulting")
"""Exact domain strings used by the dataset (mercor/apex-agents). Verified
against the published tasks_and_rubrics.json index. The CLI's --domain
filter expects one of these literal strings (case-sensitive, including
the space). 160 tasks per domain; total 480."""

# Default port we expose the environment container on. Override via env var
# ``APEX_AGENTS_HOST_PORT`` if 8080 is taken.
DEFAULT_HOST_PORT = 8080


# --- Settings ----------------------------------------------------------------


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration of the grading judge.

    ``model_id`` is the LiteLLM-routable name (e.g. ``openai/gpt-5.5``).
    ``extra_args`` is passed verbatim into LiteLLM ``acompletion`` by the
    vendor's grading runner; we leave it empty by default so OpenAI's default
    ``reasoning_effort=medium`` applies.
    """

    model_id: str = DEFAULT_JUDGE_MODEL
    timeout_seconds: int = DEFAULT_JUDGE_TIMEOUT_SECONDS
    extra_args: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRunConfig:
    """Caps on a single agent's execution. These match Mercor's published
    example exactly; deviating from them changes the benchmark and is gated
    behind a fidelity-override flag in the CLI."""

    max_steps: int = AGENT_MAX_STEPS
    timeout_seconds: int = AGENT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class Settings:
    """Run-time policy for an apex-agents-bench invocation.

    All paths are absolute. CLI commands construct one of these from defaults
    + flags; library functions accept it as their config.
    """

    dataset_dir: Path
    runs_dir: Path
    judge: JudgeConfig
    agent: AgentRunConfig
    mcp_config_path: Path
    host_port: int = DEFAULT_HOST_PORT

    @classmethod
    def defaults(cls) -> Settings:
        return cls(
            dataset_dir=default_dataset_dir(),
            runs_dir=runs_dir(),
            judge=JudgeConfig(),
            agent=AgentRunConfig(),
            mcp_config_path=archipelago_example_dir() / "mcp_config_all_oss_servers.json",
        )

    def with_dataset_dir(self, p: Path) -> Settings:
        return Settings(
            dataset_dir=p,
            runs_dir=self.runs_dir,
            judge=self.judge,
            agent=self.agent,
            mcp_config_path=self.mcp_config_path,
            host_port=self.host_port,
        )

    def with_judge(self, judge: JudgeConfig) -> Settings:
        return Settings(
            dataset_dir=self.dataset_dir,
            runs_dir=self.runs_dir,
            judge=judge,
            agent=self.agent,
            mcp_config_path=self.mcp_config_path,
            host_port=self.host_port,
        )

    def with_agent(self, agent: AgentRunConfig) -> Settings:
        return Settings(
            dataset_dir=self.dataset_dir,
            runs_dir=self.runs_dir,
            judge=self.judge,
            agent=agent,
            mcp_config_path=self.mcp_config_path,
            host_port=self.host_port,
        )

    def with_host_port(self, port: int) -> Settings:
        return Settings(
            dataset_dir=self.dataset_dir,
            runs_dir=self.runs_dir,
            judge=self.judge,
            agent=self.agent,
            mcp_config_path=self.mcp_config_path,
            host_port=port,
        )
