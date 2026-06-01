"""Behavioral-fidelity tests: confirm our wrapper does not diverge from the
vendored Archipelago harness in ways that would alter what reaches the model
or the judge.

These tests do NOT call any API. They assert structural and code-level
invariants that protect against accidental drift.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from apex_agents_bench.agent_profile import all_profiles
from apex_agents_bench.config import (
    AGENT_CONFIG_ID,
    AGENT_MAX_STEPS,
    AGENT_TIMEOUT_SECONDS,
    MCP_SERVERS,
)
from apex_agents_bench.paths import (
    archipelago_agents_dir,
    archipelago_example_dir,
    archipelago_grading_dir,
    vendor_dir,
)

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "apex_agents_bench"


# -----------------------------------------------------------------------------
# Audit A1 -- vendor source integrity (zero active patches)
# -----------------------------------------------------------------------------


def test_vendored_source_patches_are_only_the_documented_ones() -> None:
    """The ONLY vendored Python-source patch is Patch 002 (gpt-5.5 Responses-API
    bridge in agents/runner/utils/llm.py). Any other `# PATCH (apex-agents-bench)`
    marker in vendored .py means an undocumented change snuck in -- update
    PATCHES.md AND this test if you add another."""
    marked: list[str] = []
    for path in vendor_dir().rglob("*.py"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "PATCH (apex-agents-bench)" in text or "# vendored-patch:" in text:
                marked.append(str(path.relative_to(vendor_dir())))
    assert marked == ["agents/runner/utils/llm.py"], (
        f"unexpected vendored-source patch markers: {marked}. "
        "If you added a patch, record it in vendor/archipelago/PATCHES.md and "
        "update this test."
    )


def test_patches_md_records_the_active_patches() -> None:
    patches_md = (vendor_dir() / "PATCHES.md").read_text(encoding="utf-8")
    # Both active patches must be recorded by number.
    assert "Patch 001" in patches_md  # Dockerfile sandbox_fs.so compile step
    assert "Patch 002" in patches_md  # gpt-5.5 Responses-API tool-calling bridge


def test_upstream_md_records_pinned_commit() -> None:
    upstream_md = (vendor_dir() / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "3f4a8234a27d71a17e0aa19e60f116440bcd1481" in upstream_md


# -----------------------------------------------------------------------------
# Audit A2 -- runner loop matches the published example's per-task lifecycle.
# -----------------------------------------------------------------------------


def test_runner_uses_fresh_container_per_task() -> None:
    """Runner must call start_env / stop_env once per task."""
    src = (SRC / "runner.py").read_text(encoding="utf-8")
    assert "start_env(" in src
    assert "stop_env(" in src


def test_runner_skips_grading_when_agent_not_completed() -> None:
    """Mirror of upstream: grading only runs when the trajectory.status == 'completed'."""
    src = (SRC / "runner.py").read_text(encoding="utf-8")
    assert 'if traj.status == "completed":' in src


def test_runner_refuses_to_save_failed_grading_as_zero_score() -> None:
    """A completed agent with failed grading is not a completed benchmark row."""
    src = (SRC / "runner.py").read_text(encoding="utf-8")
    assert "grading subprocess exited with code" in src
    assert "grading did not complete" in src
    assert "grading result criterion count mismatch" in src


def test_runner_refuses_to_save_nonzero_agent_subprocess_as_completed() -> None:
    """A nonzero vendor agent runner exit is a failed task, even with artifacts."""
    src = (SRC / "runner.py").read_text(encoding="utf-8")
    assert "agent subprocess exited with code" in src
    assert 'raise RuntimeError(f"agent subprocess exited with code {rc}")' in src


# -----------------------------------------------------------------------------
# Audit A5 -- agent config matches the published example exactly.
# -----------------------------------------------------------------------------


def test_agent_config_id_is_react_toolbelt_agent() -> None:
    assert AGENT_CONFIG_ID == "react_toolbelt_agent"
    # Also verified at the writing site:
    src = (SRC / "runner.py").read_text(encoding="utf-8")
    assert '"agent_config_id": AGENT_CONFIG_ID' in src


def test_agent_max_steps_matches_published_example() -> None:
    """The published example sets max_steps=50; our policy mirrors it."""
    assert AGENT_MAX_STEPS == 50
    example_path = archipelago_example_dir() / "agent_config.json"
    example = json.loads(example_path.read_text(encoding="utf-8"))
    assert example["agent_config_values"]["max_steps"] == AGENT_MAX_STEPS


def test_agent_timeout_matches_published_example() -> None:
    assert AGENT_TIMEOUT_SECONDS == 3600
    example_path = archipelago_example_dir() / "agent_config.json"
    example = json.loads(example_path.read_text(encoding="utf-8"))
    assert example["agent_config_values"]["timeout"] == AGENT_TIMEOUT_SECONDS


# -----------------------------------------------------------------------------
# Audit A6 -- MCP server set matches the published 9-server config.
# -----------------------------------------------------------------------------


def test_mcp_servers_match_published_example() -> None:
    example_path = archipelago_example_dir() / "mcp_config_all_oss_servers.json"
    cfg = json.loads(example_path.read_text(encoding="utf-8"))
    declared = set((cfg.get("mcpServers") or {}).keys())
    assert declared == set(MCP_SERVERS), (
        f"MCP server set drift: example says {sorted(declared)}, "
        f"config.py says {sorted(MCP_SERVERS)}"
    )


def test_mcp_servers_size_is_nine() -> None:
    assert len(MCP_SERVERS) == 9


def test_runner_enforces_mcp_set_at_call_time() -> None:
    """If someone hands us a subsetted MCP config the runner must refuse."""
    src = (SRC / "runner.py").read_text(encoding="utf-8")
    assert "MCP config drift" in src


# -----------------------------------------------------------------------------
# Audit A7 -- rubric criteria passed verbatim (no rewriting, no filtering).
# -----------------------------------------------------------------------------


def test_verifiers_built_from_rubric_verbatim() -> None:
    """The judge module must build verifiers straight from task.rubric without
    rewriting the ``criteria`` text or skipping entries."""
    src = (SRC / "judge.py").read_text(encoding="utf-8")
    # Each rubric criterion -> one verifier, in order; criteria string passed through.
    assert "for i, c in enumerate(task.rubric)" in src
    assert '"criteria": c.criteria' in src


# -----------------------------------------------------------------------------
# Audit A8 -- agent profiles use only LiteLLM-routable strings + extra_args.
# -----------------------------------------------------------------------------


_ALLOWED_EXTRA_ARG_KEYS = frozenset(
    {
        "reasoning_effort",
        "verbosity",
        "temperature",
        # Provider endpoint plumbing, not a behavioral model knob. We pin the
        # official DeepSeek OpenAI-compatible base URL instead of relying on a
        # LiteLLM default alias.
        "api_base",
        # Provider endpoint plumbing for native DeepSeek max effort. Pinned
        # LiteLLM preserves `thinking.reasoning_effort=max` through extra_body;
        # top-level reasoning_effort is collapsed to plain thinking-enabled.
        "extra_body",
        # HTTP plumbing, not a model capability. Splatted into
        # litellm.acompletion(timeout=...) by the vendor's generate_response;
        # bounds how long to wait for one LLM response. Required because
        # LiteLLM's 600s default is too short for long agent contexts at
        # reasoning_effort=high. Does NOT change sampling, tools, or the
        # model's behavior. The agent loop's wall-clock cap
        # (AGENT_TIMEOUT_SECONDS=3600) is still the upper bound on a whole
        # task.
        "timeout",
    }
)


def test_no_profile_uses_disallowed_extra_args() -> None:
    """We only set keys Mercor's published example sets in
    ``orchestrator_config.json`` extra_args (effectively none) plus our
    per-family knobs (reasoning_effort, verbosity, temperature) and provider
    plumbing (api_base, extra_body, timeout). Anything
    else implies a hidden capability (tools, system prompt, top_p, ...) and
    fails the audit."""
    for p in all_profiles():
        extras = set(p.orchestrator_extra_args.keys()) - _ALLOWED_EXTRA_ARG_KEYS
        assert not extras, (
            f"profile {p.name!r} sets disallowed extra_args keys: {extras}. "
            "If this is intentional, update _ALLOWED_EXTRA_ARG_KEYS and confirm "
            "the key is provider-documented and does not add hidden tools or data."
        )


def test_no_profile_sets_top_p() -> None:
    for p in all_profiles():
        assert "top_p" not in p.orchestrator_extra_args, p.name


def test_no_profile_sets_max_tokens() -> None:
    """Token caps are provider-side; the upstream example does not set them."""
    for p in all_profiles():
        assert "max_tokens" not in p.orchestrator_extra_args, p.name
        assert "max_input_tokens" not in p.orchestrator_extra_args, p.name


def test_gpt55_profiles_expected_shape() -> None:
    """gpt-5.5-* profiles: reasoning_effort + verbosity=medium, no temperature.

    Archipelago has no ModelConfig default temperature to neutralize;
    omitting temperature sends no custom temperature to LiteLLM/OpenAI.
    """
    for p in (pp for pp in all_profiles() if pp.family == "gpt-5.5"):
        ea = p.orchestrator_extra_args
        assert ea.get("verbosity") == "medium", p.name
        assert "temperature" not in ea, p.name
        assert ea["reasoning_effort"] in {"low", "medium", "high", "xhigh"}, p.name
        assert p.orchestrator_model == "openai/gpt-5.5", p.name


def test_grok43_profiles_expected_shape() -> None:
    """grok-4.3-* profiles: reasoning_effort + temperature=0.8."""
    for p in (pp for pp in all_profiles() if pp.family == "grok-4.3"):
        ea = p.orchestrator_extra_args
        assert ea.get("temperature") == 0.8, p.name
        assert "verbosity" not in ea, p.name
        assert ea["reasoning_effort"] in {"low", "medium", "high"}, p.name
        assert p.orchestrator_model == "xai/grok-4.3", p.name


def test_deepseek_v4_pro_profile_matches_docs_shape() -> None:
    """deepseek-v4-pro: a single max-effort profile, no sampling knobs."""
    profiles = [p for p in all_profiles() if p.family == "deepseek-v4-pro"]
    assert len(profiles) == 1
    p = profiles[0]
    ea = p.orchestrator_extra_args
    assert p.name == "deepseek-v4-pro-max"
    assert p.provider == "deepseek"
    assert p.orchestrator_model == "deepseek/deepseek-v4-pro"
    assert ea == {
        "api_base": "https://api.deepseek.com",
        "extra_body": {
            "thinking": {
                "type": "enabled",
                "reasoning_effort": "max",
            }
        },
        "timeout": 1800,
    }


def test_no_claude_profile_registered() -> None:
    """Claude/Bedrock is deferred."""
    for p in all_profiles():
        assert p.provider != "anthropic-bedrock", (
            f"Claude profile {p.name!r} should be deferred -- see agent_profile.py "
            "module docstring."
        )


# -----------------------------------------------------------------------------
# Audit A_LITELLM -- the vendor passes the model string verbatim to LiteLLM.
# -----------------------------------------------------------------------------


def test_archipelago_passes_model_string_verbatim() -> None:
    """Both vendor LiteLLM call sites must pass the model name string straight
    through to ``litellm.acompletion(model=...)`` -- this is the load-bearing
    invariant that lets us register gpt-5.5 / grok-4.3 / deepseek-v4-pro
    with zero vendor patches."""
    agents_llm = (archipelago_agents_dir() / "runner" / "utils" / "llm.py").read_text(
        encoding="utf-8"
    )
    grading_llm = (archipelago_grading_dir() / "runner" / "utils" / "llm.py").read_text(
        encoding="utf-8"
    )

    # Agents call site -- kwargs assembled with `"model": model`, then acompletion.
    assert re.search(r'"model"\s*:\s*model', agents_llm)
    assert "acompletion(**kwargs)" in agents_llm

    # Grading call site -- same shape via litellm.acompletion.
    assert re.search(r'"model"\s*:\s*model', grading_llm)
    assert "litellm.acompletion(**kwargs)" in grading_llm


def test_vendor_gpt55_responses_bridge_patch_present() -> None:
    """Patch 002: gpt-5.5 (a reasoning model) drops tool_calls over chat
    completions, so the vendored agent llm.py must route it through the
    Responses API. This test pins that the bridge is present, is GATED to the
    gpt-5.5 family (grok and others keep the chat path), and that the chat
    path itself is untouched."""
    agents_llm = (archipelago_agents_dir() / "runner" / "utils" / "llm.py").read_text(
        encoding="utf-8"
    )
    # The patch marker + the gated Responses-API branch are present.
    assert "PATCH (apex-agents-bench)" in agents_llm
    assert "_is_reasoning_bridge_model" in agents_llm
    assert "await aresponses(" in agents_llm
    assert "_responses_output_to_model_response" in agents_llm
    # Multi-turn correctness: the chat history must be translated into Responses
    # API input items (assistant tool_calls -> function_call,
    # role=tool -> function_call_output); passing raw chat dicts breaks turn 2+.
    assert "_chat_messages_to_responses_input" in agents_llm
    assert '"type": "function_call"' in agents_llm
    assert '"type": "function_call_output"' in agents_llm
    # The branch is gated on the gpt-5.5 family AND tools, and feeds the
    # TRANSLATED input (not the raw chat messages) to the Responses API.
    assert re.search(r"if\s+_is_reasoning_bridge_model\(model\)\s+and\s+tools\s*:", agents_llm)
    assert re.search(r'"input"\s*:\s*_chat_messages_to_responses_input\(messages\)', agents_llm)
    # The unmodified chat path still exists for every other model.
    assert "acompletion(**kwargs)" in agents_llm

    # Gating: the predicate matches ONLY the gpt-5.5 family, so grok/others
    # never take the bridge. Asserted on the source so no cross-venv import of
    # the vendored module (which needs its own deps) is required.
    body = re.search(r"def _is_reasoning_bridge_model\(.*?\n(.*?)\n\n", agents_llm, re.DOTALL)
    assert body is not None, "could not locate _is_reasoning_bridge_model body"
    pred = body.group(1)
    assert '"gpt-5.5"' in pred and "in m" in pred
    # Nothing in the predicate widens the gate to grok/deepseek/other families.
    assert "grok" not in pred.lower()
    assert "deepseek" not in pred.lower()


# -----------------------------------------------------------------------------
# Audit -- system prompt is verbatim from the published example.
# -----------------------------------------------------------------------------


def test_system_prompt_matches_published_example() -> None:
    """The system prompt our runner injects MUST match Mercor's example
    byte-for-byte. We hold a copy in our runner.py because the example
    inlines it as a string literal; if the example ever moves the prompt to
    a separate file, drop the in-repo copy and read from disk."""
    from apex_agents_bench.runner import UPSTREAM_SYSTEM_PROMPT

    example_main = (archipelago_example_dir() / "main.py").read_text(encoding="utf-8")
    # Pull the literal between the triple-quoted system_prompt assignment.
    m = re.search(
        r'system_prompt\s*=\s*"""(.+?)"""',
        example_main,
        flags=re.DOTALL,
    )
    assert m, "could not locate the system_prompt assignment in the published example main.py"
    upstream_literal = m.group(1)
    assert upstream_literal == UPSTREAM_SYSTEM_PROMPT, (
        "system prompt drift: our copy in runner.py differs from the "
        "published example -- update UPSTREAM_SYSTEM_PROMPT."
    )


# -----------------------------------------------------------------------------
# Audit -- judge model swap is the ONLY deliberate diff from grading_settings.
# -----------------------------------------------------------------------------


def test_judge_model_default_is_gpt55() -> None:
    from apex_agents_bench.config import JudgeConfig

    assert JudgeConfig().model_id == "openai/gpt-5.5"


def test_docs_do_not_claim_gpt55_agent_temperature() -> None:
    """The agent profiles intentionally omit temperature for GPT-5.5."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "temperature=1.0" not in readme


def test_published_example_judge_is_gemini_documented_as_diff() -> None:
    """Sanity: the upstream example judge is gemini-2.5-flash. Our docs and
    config declare gpt-5.5 as the deliberate swap; this test confirms the
    upstream value hasn't moved (if it did, our AUDIT doc would mis-describe
    the diff)."""
    example = json.loads(
        (archipelago_example_dir() / "grading_settings.json").read_text(encoding="utf-8")
    )
    assert example["llm_judge_model"] == "gemini/gemini-2.5-flash", (
        f"upstream judge model changed from gemini/gemini-2.5-flash to "
        f"{example['llm_judge_model']!r}; update docs/AUDIT.md."
    )
