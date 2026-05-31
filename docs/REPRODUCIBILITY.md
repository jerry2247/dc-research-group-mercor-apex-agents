# Reproducibility

> **Read this if you're asking:** "Why one run per task? Why gpt-5.5
> instead of Mercor's Gemini? What gets recorded per run? Is the
> dataset license a problem?"

## Project policies (and why)

### 1. One run per (task, model). Always.

The standard APEX-Agents leaderboard reports pass@1 (one run) and
sometimes pass@8 (eight runs). We use pass@1 for every reported number.

**Why one run, not eight?** Eight runs multiplies our cost-per-cell by
8x. Across 8 agent profiles × 480 tasks, eight runs would be ~$1k/cell.
Variance signal is recovered from per-domain bins (160 tasks each) and
per-criterion granularity (mean 4.06 criteria/task -> ~649 binary
judgments per per-domain cell, ~1,949 per full run). We accept that
single-run pass@1 numbers will not match
Mercor's published leaderboard pass@8 numbers exactly; the comparison
we care about is *cross-method* (memory-equipped DC vs no-memory
baseline) on the same surface.

**Enforced at**: `apex_agents_bench.config.RUNS_PER_TASK`, asserted in
`tests/test_imports.py::test_settings_defaults_match_policy`.

### 2. Judge = `openai/gpt-5.5` at OpenAI default `reasoning_effort=medium`.

Mercor's open example ships `gemini/gemini-2.5-flash`; their
leaderboard uses Gemini 2.5 Pro Thinking=On. **We use GPT-5.5 instead.**

**Why GPT-5.5?**

- *Cost*: GPT-5.5 medium is cheaper per judgment than Gemini 2.5 Pro
  Thinking.
- *Reasoning ability*: medium-effort GPT-5.5 is closer to Gemini Pro
  Thinking than to Gemini Flash, so the judge quality stays above the
  open example's default.

**Self-enhancement caveat**: when the *agent* is also `gpt-5.5-*`, the
judge and agent are the same model family. This introduces a known
self-enhancement bias (judge LLMs grade their own family more
generously). We accept this as a known caveat -- documented in this
file and in `docs/AUDIT.md` -- and compute it side-by-side with the
non-family pairs (`grok-4.3-*` / `deepseek-v4-pro-max` agents +
`gpt-5.5` judge) as a
sanity-check.

**Enforced at**: `apex_agents_bench.config.DEFAULT_JUDGE_MODEL`,
asserted in `tests/test_fidelity.py::test_judge_model_default_is_gpt55`.

### 3. Agent = `react_toolbelt_agent`, `max_steps=50`, `timeout=3600s`.

Verbatim from Mercor's published
`examples/hugging_face_task/agent_config.json`. The agent registry's
defaults are higher (`max_steps=250, timeout=10800`) but the *published
example* overrides them; our fidelity claim is to the example.

You can raise these via `--max-steps` / `--timeout-seconds` flags on
`apex-agents-bench run`, but doing so diverges from published numbers
and the CLI surfaces a warning.

**Enforced at**: `apex_agents_bench.config.AGENT_MAX_STEPS` /
`AGENT_TIMEOUT_SECONDS`, asserted in
`tests/test_fidelity.py::test_agent_max_steps_matches_published_example`
and `test_agent_timeout_matches_published_example`.

### 4. All 9 MCP servers, every run.

Mercor's published `mcp_config_all_oss_servers.json` declares 9
servers. We use all 9. Subsetting (e.g. dropping `code_execution_server`
for tasks that don't seem to need it) would change task semantics
because the agent's exploration is shaped by which tools are visible.

**Enforced at**: `apex_agents_bench.config.MCP_SERVERS`, asserted in
`tests/test_fidelity.py::test_mcp_servers_match_published_example`. The
runner refuses a hand-edited MCP config that subsets at run time.

### 5. Minimal vendor source modifications (one build patch).

Archipelago's LiteLLM call sites pass model names verbatim, so the
GPT-5.5 / Grok 4.3 / DeepSeek surface needs **no** source patch. The one
vendored change is a build-time patch to `environment/Dockerfile` that
compiles `sandbox_fs.so` for the code-execution server (Patch 001 in
`PATCHES.md`); there are zero patches to the Python runners.

**Enforced at**: `vendor/archipelago/PATCHES.md`, asserted in
`tests/test_fidelity.py::test_no_vendored_patch_markers_present` (which
guards against any `# vendored-patch:` marker in the Python source).

### 6. Per-task fresh container.

Each task gets a fresh `docker compose up -d --build`. World state
(filesystem, calendar, mail, chat) does NOT leak between tasks. This
matches the published example's `docker compose down -v` + `up`
pattern at the start of every run.

**Enforced at**: `apex_agents_bench.runner.run_single_task` (start_env
+ stop_env wrap each task body), asserted in
`tests/test_fidelity.py::test_runner_uses_fresh_container_per_task`.

---

## What gets recorded per run

A single run writes one CSV row per completed task (resume-skippable)
plus run sidecars and per-task artifacts:

```
runs/<UTC-timestamp>__<profile>__<scope>/
├── results.csv                  one row per task (the "answer" file)
├── results.run_manifest.json    non-secret run config, hashes, selected tasks
├── results.failures.jsonl       skipped/preflight/grading failure records
└── <task_id>/
    ├── initial_messages.json    system prompt + task prompt (what the agent saw)
    ├── agent_config.json        max_steps=50, timeout=3600
    ├── orchestrator_extra_args.json    profile-specific knobs
    ├── grading_settings.json    judge model + extra_args
    ├── verifiers.json           one entry per rubric criterion
    ├── eval_configs.json        ec_output_llm
    ├── scoring_config.json      template
    ├── trajectory.json          agent's full message history + tool calls
    ├── final_snapshot.tar.gz    raw container snapshot (from /data/snapshot)
    ├── final_snapshot.zip       same snapshot for the grading runner
    └── grades.json              verifier-level pass/fail + judge rationales
```

The CSV row records:
- `task_id`, `domain`, `world_id`
- `status="completed"` (skipped tasks are NOT written, matching upstream)
- `agent_status` (`completed`/`failed`/`error`)
- `final_score` (0.0-1.0), `criteria_passed`, `criteria_total`
- `steps_used`, `wall_time_seconds`
- agent-side token telemetry from `trajectory.json.usage`:
  `agent_prompt_tokens`, `agent_completion_tokens`, `agent_total_tokens`,
  `agent_final_step_completion_tokens`, `agent_usage_available`,
  `agent_usage_source`, `agent_usage_consistent`
- `agent_profile`, `agent_model`, `judge_model`
- pointers to `trajectory.json` and `grades.json`

The CSV intentionally does **not** include judge tokens or cost columns.
Judge usage remains in `grades.json`; costs should be computed separately
from provider invoices or an explicit post-run script.

**Provenance for full reproducibility**: pair the CSV with the vendor
SHA (`vendor/archipelago/UPSTREAM.md`) + dataset revision + `Settings`
dump + (provider-side) model snapshot date.

---

## Dataset license

The APEX-Agents dataset (`mercor/apex-agents` on HuggingFace) is
**CC-BY-4.0 with an additional eval-only clause**:

> APEX-Agents is intended exclusively for model evaluation. Any use of
> this dataset for training, fine-tuning, or parameter fitting is
> forbidden. Crawling or scraping the dataset is also forbidden.

We honor both clauses:

- **Eval-only.** We use the dataset *only* to grade agent outputs;
  we never use task prompts, rubrics, or trajectories as supervision
  for any training pipeline. The DC-RS memory mechanism is a
  *test-time* facility (the synthesizer reads the per-domain pool and
  cheatsheet, and the pool is appended to, at inference time); it does
  not consume APEX-Agents data as training signal, and it consumes no
  ground-truth/grading signal at all.
- **No scraping.** The dataset is gated; we download via the official
  HuggingFace API after accepting the eval-only terms in the dataset
  card UI. `scripts/fetch_dataset.sh` checks `hf auth whoami` and
  refuses to proceed without a logged-in token.

The dataset is **not redistributed** in this repository. It is fetched
at run time into `data/apex-agents/`, which is gitignored.

---

## What a "comparable run" looks like

If you re-run the same `apex-agents-bench run` invocation on a different
machine, the following should match within provider stochasticity:

1. `Settings` dump (judge, agent caps, dataset_dir, host_port).
2. Vendor SHA from `vendor/archipelago/UPSTREAM.md`.
3. Dataset revision (use `hf_hub_download`'s revision lock if you need
   bit-exact replay).
4. CSV row count (per the same selection filters).
5. `criteria_passed` / `criteria_total` per task (modulo stochasticity).
6. `final_score` per task (within ~1-2% noise; binary criteria + LLM
   judge make the per-task score relatively stable but not
   deterministic).

What WON'T match without further effort: exact `trajectory.json` byte
content (tool-call order and tokens drift), `wall_time_seconds`,
provider-side request ids.

## No exact Mercor-output parity claim

Even if we used Mercor's exact public-example model and judge, this repo
cannot guarantee byte-identical outputs to Mercor. Provider APIs are
stochastic, model backends and aliases can drift, and Mercor's private
leaderboard run environment is not fully published. The claim this repo
makes is narrower and testable: for the public Archipelago example, the
task lifecycle, agent, MCP server set, prompt, verifier construction,
scoring config, and saved telemetry match the published harness except
for the documented model/judge substitutions and local resume/sidecar
behavior.

---

## A note on Reducto (visual artifact extraction)

The vendor's grading runner can optionally extract images from PDFs /
spreadsheets / slides / docs via Reducto, to give the judge LLM visual
context. Without a Reducto key, grading still works -- the judge just
doesn't see embedded charts. For tasks whose deliverable is text, Reducto
is unnecessary; only chart-heavy artifacts benefit from it. If you run a task that
produces a chart-heavy artifact and care about visual fidelity, set
`REDUCTO_API_KEY` in `.env`. We treat Reducto as an optional capability
matching what the vendor ships, not a requirement.
