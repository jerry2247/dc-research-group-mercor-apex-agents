# Behavioral fidelity audit

> **Read this if you're asking:** "Are we actually running Mercor's
> Archipelago harness, or did we silently change something? Same agent?
> Same MCP servers? Same step cap? Same scoring?"
>
> **TL;DR:** 12-component audit comparing our wrapper to upstream behavior
> with code-level evidence and regression tests for every claim. If
> `make check` is green, every claim in this document still holds.

**Last reviewed: 2026-05-19 against upstream commit `3f4a8234a27d71a17e0aa19e60f116440bcd1481`.**

This document records every check that confirms our wrapper does not
change what reaches the agent or the judge relative to Mercor's
reference runner (`vendor/archipelago/examples/hugging_face_task/main.py`).
The intent is to *match Archipelago's behavior exactly, excluding the
deliberate judge-model substitution documented in
`docs/REPRODUCIBILITY.md`*.

This is a public-harness fidelity claim, not a byte-identical output
claim. Provider APIs are stochastic, model backends can drift, and Mercor's
private leaderboard infrastructure is not fully published.

Every audit item has a corresponding regression test under `tests/`
that must continue to pass on every CI run. If an audit assertion ever
fails, treat it as a halt-the-line event.

---

## A1. Vendor source integrity

**Claim**: The only changes to `vendor/archipelago/` are the three
provenance files we added (`LICENSE_UPSTREAM`, `PATCHES.md`,
`UPSTREAM.md`) plus a stripped `.git`. There are **zero active code
patches** — Archipelago's call sites accept arbitrary LiteLLM model
strings (see A_LITELLM below), so the GPT-5.5 / Grok 4.3 surface works
on unmodified vendor source.

**Verification** (re-runnable):
```bash
diff -rq vendor/archipelago <fresh-upstream-checkout> | grep -v '^Only in vendor.*: LICENSE_UPSTREAM\|^Only in vendor.*: PATCHES.md\|^Only in vendor.*: UPSTREAM.md\|^Only in <upstream>.*: .git\|^Only in <upstream>.*: .github'
```

**Recorded result** on 2026-05-19: empty output (no logic / source
diffs).

**Tests**:
- `tests/test_fidelity.py::test_no_vendored_patch_markers_present`
- `tests/test_fidelity.py::test_patches_md_records_zero_active_patches`
- `tests/test_fidelity.py::test_upstream_md_records_pinned_commit`

---

## A2. Per-task lifecycle

**Claim**: `apex_agents_bench.runner.run_single_task` replicates the
upstream `examples/hugging_face_task/main.py::main` for a single task,
step-for-step:

| Upstream `main.py` step | Our `runner.py` |
|---|---|
| `start_environment()` (down -v + up + health-poll) | `start_env(compose_dir, host_port)` |
| `populate_subsystems(world)` via POST `/data/populate` | `_populate_environment(..., "world")` |
| `populate_subsystems(task)` if `task.task_input_files` | `_populate_environment(..., "task")` if `task.task_input_files` |
| POST `/apps` with mcp_config_all_oss_servers.json | `_configure_mcp(..., settings.mcp_config_path)` |
| `subprocess.run(["uv","run","python","-m","runner.main", ...], cwd=AGENTS_DIR)` | `_run_agent_subprocess(...)` (identical) |
| `httpx.stream("POST", "/data/snapshot")` | identical |
| `tar_gz_to_zip(final_tar_gz)` | identical |
| Skip grading if `agent_status != "completed"` | identical (`if traj.status == "completed":`) |
| `subprocess.run(["uv","run","python","-m","runner.main", ...], cwd=GRADING_DIR)` | `_run_grading_subprocess(...)` (identical) |

We add four things on top, all centralized in `runner.py` and
documented in `docs/REPRODUCIBILITY.md` as deliberate diffs:

1. **Profile registry** -- the upstream's hardcoded
   `gemini/gemini-3-pro-preview` is replaced by the user-chosen profile.
2. **Judge override** -- the upstream's hardcoded
   `gemini/gemini-2.5-flash` is replaced by `openai/gpt-5.5` (medium).
3. **CSV resume** -- not present upstream; checks `results.csv` for
   `status="completed"` rows and skips them.
4. **Failure sidecars** -- skipped, preflight, and grading-failure cases
   are written to `results.failures.jsonl`; they are not written as
   completed CSV rows.

**Tests**:
- `tests/test_fidelity.py::test_runner_uses_fresh_container_per_task`
- `tests/test_fidelity.py::test_runner_skips_grading_when_agent_not_completed`
- `tests/test_fidelity.py::test_runner_refuses_to_save_failed_grading_as_zero_score`
- `tests/test_fidelity.py::test_runner_refuses_to_save_nonzero_agent_subprocess_as_completed`
- `tests/test_runner.py::test_append_then_resume`

---

## A3. Agent identity (`react_toolbelt_agent`)

**Claim**: The agent is exactly the vendor's registered
`react_toolbelt_agent`. We do not register a custom agent and we do
not swap the implementation.

**Source of truth**: `vendor/archipelago/agents/runner/agents/registry.py`
defines the agent's `agent_config_id` value as the enum
`AgentConfigIds.REACT_TOOLBELT_AGENT`. Our `agent_config.json` writer
emits the literal `"react_toolbelt_agent"` string, matching the enum.

**Tests**:
- `tests/test_fidelity.py::test_agent_config_id_is_react_toolbelt_agent`

---

## A4. Agent step / time caps

**Claim**: We run the agent with `max_steps=50` and `timeout=3600`
seconds. These match Mercor's published reference config in
`vendor/archipelago/examples/hugging_face_task/agent_config.json`
verbatim. We deliberately do NOT use the agent registry's higher
defaults (`max_steps=250, timeout=10800`) because the published
example overrides them, and our fidelity claim is to the published
example.

**Source of truth**: read
`vendor/archipelago/examples/hugging_face_task/agent_config.json`.

**Tests**:
- `tests/test_fidelity.py::test_agent_max_steps_matches_published_example`
- `tests/test_fidelity.py::test_agent_timeout_matches_published_example`

---

## A5. MCP server set

**Claim**: Every task runs against the full 9-server set declared in
`vendor/archipelago/examples/hugging_face_task/mcp_config_all_oss_servers.json`:
`calendar_server`, `chat_server`, `code_execution_server`,
`sheets_server`, `filesystem_server`, `mail_server`, `pdf_server`,
`slides_server`, `docs_server`. Subsetting the toolbelt would change
task semantics and is refused at run time.

**Source of truth**: the JSON file linked above + our
`apex_agents_bench.config.MCP_SERVERS` tuple. The runner asserts
equality before configuring the gateway, so a hand-edited config that
adds or drops a server fails fast.

**Tests**:
- `tests/test_fidelity.py::test_mcp_servers_match_published_example`
- `tests/test_fidelity.py::test_mcp_servers_size_is_nine`
- `tests/test_fidelity.py::test_runner_enforces_mcp_set_at_call_time`

---

## A6. Rubric pass-through

**Claim**: Every criterion in the task's rubric becomes one verifier,
in the same order, with the criterion text passed verbatim. The
first criterion gets `is_primary_objective=true` (matching the
upstream `verifier_index == 0`). No criteria are rewritten, normalized,
or dropped.

**Source of truth**: read `apex_agents_bench.judge.build_verifiers`.

**Tests**:
- `tests/test_fidelity.py::test_verifiers_built_from_rubric_verbatim`
- `tests/test_judge.py::test_build_verifiers_criteria_passed_verbatim`
- `tests/test_judge.py::test_build_verifiers_first_is_primary_objective`

---

## A7. Eval + scoring config

**Claim**: We use exactly the upstream's `ec_output_llm` eval config
and `template` scoring method. We do not subclass, override, or pass
custom values into either.

**Source of truth**:
- `vendor/archipelago/examples/hugging_face_task/eval_configs.json`
- `vendor/archipelago/examples/hugging_face_task/scoring_config.json`

Our `judge.build_eval_configs()` and `judge.build_scoring_config()`
emit byte-for-byte identical JSON.

**Tests**:
- `tests/test_judge.py::test_eval_configs_default_shape`
- `tests/test_judge.py::test_scoring_config_default_shape`

---

## A8. System prompt

**Claim**: The system prompt prepended to the agent's initial messages
is the exact same string as Mercor's published example. We copy it
into `apex_agents_bench.runner.UPSTREAM_SYSTEM_PROMPT` because the
upstream inlines it as a literal in `main.py`; a fidelity test
re-extracts the literal from the upstream file at test time and
asserts equality.

**Tests**:
- `tests/test_fidelity.py::test_system_prompt_matches_published_example`

---

## A9. Agent profile knobs (no hidden capabilities)

**Claim**: Every active agent profile sets ONLY model-mode knobs plus
the per-call HTTP timeout in `orchestrator_extra_args`:
`{reasoning_effort, verbosity, temperature, timeout}`. None of these is set:
- `tools=` — agent's tool surface comes from the MCP gateway only.
- `top_p`, `max_tokens`, `max_input_tokens` — provider-side defaults;
  upstream example does not set them either.
- `system_prompt` — we use the upstream literal (A8); a profile-side
  `system_prompt` would override it and is refused.
- `web_search`, `code_interpreter`, `image_input` — none of these is a
  knob we surface.

**Tests**:
- `tests/test_fidelity.py::test_no_profile_uses_disallowed_extra_args`
- `tests/test_fidelity.py::test_no_profile_sets_top_p`
- `tests/test_fidelity.py::test_no_profile_sets_max_tokens`
- `tests/test_fidelity.py::test_gpt55_profiles_match_apex_bench_shape`
- `tests/test_fidelity.py::test_grok43_profiles_match_apex_bench_shape`

---

## A10. Agent-token telemetry only

**Claim**: CSV token telemetry is sourced only from the agent
trajectory's top-level `usage` block, which Archipelago writes via
`vendor/archipelago/agents/runner/utils/usage.py::UsageTracker.to_dict()`.
The runner does not synthesize cost, does not flatten judge usage into
the CSV, and records whether usage was present and internally consistent.

**Source of truth**:
- `vendor/archipelago/agents/runner/utils/usage.py` accumulates
  `prompt_tokens`, `completion_tokens`, `total_tokens`, and
  `final_answer_tokens` from LiteLLM `ModelResponse.usage`.
- `apex_agents_bench.trajectory.parse_trajectory` copies those four
  fields into `TrajectorySummary`, plus `agent_usage_available`,
  `agent_usage_source`, and `agent_usage_consistent`.
- `apex_agents_bench.runner.CSV_HEADERS` includes only agent usage
  columns; it intentionally excludes judge token and cost columns.

**Tests**:
- `tests/test_trajectory.py::test_parse_trajectory_extracts_usage_block`
- `tests/test_trajectory.py::test_parse_trajectory_flags_inconsistent_usage_block`
- `tests/test_runner.py::test_csv_headers_have_expected_columns`

---

## A11. Claude/Bedrock deferred

**Claim**: No Claude profiles are registered. This mirrors apex-bench's
deferral. Bedrock plumbing is structurally absent from the apex-evals
sister harness; we wait until both repos can enable Claude jointly.

**Tests**:
- `tests/test_fidelity.py::test_no_claude_profile_registered`

---

## A_LITELLM. Model string passed to LiteLLM verbatim

**Claim**: Both vendor LiteLLM call sites (`agents/runner/utils/llm.py`
and `grading/runner/utils/llm.py`) pass the model name string directly
to `litellm.acompletion(model=<string>)`. There is no allow-list,
no `MODEL_MAPPINGS` dict, no validate-then-call sequence. This is the
load-bearing invariant that lets `gpt-5.5` and `grok-4.3` work with
**zero vendor patches** — they are forwarded to LiteLLM as
`openai/gpt-5.5` and `xai/grok-4.3` and routed by prefix.

**Source of truth**: the two `llm.py` files above. We grep for both
`"model": model` and `acompletion(**kwargs)` patterns in each file.

**Tests**:
- `tests/test_fidelity.py::test_archipelago_passes_model_string_verbatim`

---

## Deliberate, documented differences from Mercor's example

These are **policy choices** with reasons recorded in
`docs/REPRODUCIBILITY.md` and `docs/IMPLEMENTATION_PLAN.md`. They are
not fidelity bugs; they are scope choices.

| Aspect | Mercor's example | Ours | Reason |
|---|---|---|---|
| Judge model | `gemini/gemini-2.5-flash` (open example); Gemini 2.5 Pro Thinking=On (leaderboard) | `openai/gpt-5.5` (medium-by-default) | Cross-benchmark consistency with apex-bench; see REPRODUCIBILITY.md. |
| Agent model | `gemini/gemini-3-pro-preview` (open example) | profile-driven (`gpt-5.5-*`, `grok-4.3-*`) | We compare DC variants against a fixed agent surface. |
| Number of runs per task | 1 | 1 | Same. |
| Test models | one example task | profile registry (7 profiles), one profile per invocation | Parity with apex-bench. |
| Claude/Bedrock | (would route via direct Anthropic API) | deferred | Joint deferral with apex-bench. |
| Resume | none | per-CSV `status="completed"` resume | Operational ergonomics; does not change per-task behavior. |

Every other behavioral knob — agent type, step cap, timeout, MCP
servers, system prompt, rubric-to-verifier translation, eval/scoring
config, per-task fresh container, scoring formula — matches the
upstream example exactly.

---

## Re-running the audit

```bash
make check     # all fidelity + structural tests, non-docker
```

If any `tests/test_fidelity.py::*` test fails, do not commit; we have
drifted. If `diff -rq vendor/archipelago <upstream>` shows
differences other than the three provenance files, do not commit; we
have drifted.
