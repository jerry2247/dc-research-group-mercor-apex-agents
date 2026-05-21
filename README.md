# apex-agents-bench

This repository is the **evaluation harness** our research group uses
to test our test-time memory mechanism — **Dynamic Ledger**, our
extension of Dynamic Cheatsheet (Suzgun et al.) — on **Mercor's
APEX-Agents** benchmark.

> **Authors.** Jerry Gu, Kyleen Liao, Shurui Liu, Roshen Nair, Arnold Yang.
> **In collaboration with.** Mirac Suzgun (Stanford SAIL NLP).
> **Research focus.** Extension of Dynamic Cheatsheet to study agent test-time learning on long-horizon professional-services tasks.

**The benchmark is Mercor's, not ours.** We vendor their official
Archipelago harness at a pinned commit (`vendor/archipelago/`, commit
`3f4a8234`) with zero modifications and add a thin policy/runner layer
(judge selection, profile registry, reproducible CSV output) so we can
evaluate our framework against Mercor's published evaluation surface.

What lives where:

- **Mercor's Archipelago harness** (vendored, byte-equivalent to the
  pinned upstream commit): `vendor/archipelago/`
- **Mercor's benchmark dataset**: not redistributed; fetched at run
  time from `mercor/apex-agents` on HuggingFace
- **Our evaluation harness** (the policy + runner + audit + CSV
  schema): `src/apex_agents_bench/`

Behavioral fidelity to Mercor's published evaluation behavior is
enforced by 110 pytest assertions and an 11-component code-level
audit; see [`docs/AUDIT.md`](docs/AUDIT.md).

> **Sister repo.** [`apex-bench`](https://github.com/jerry2247/dc-research-group-mercor-apex)
> targets the **APEX-v1-extended** benchmark — the single-shot
> text-deliverable surface (100 tasks, no agent, no code execution)
> via the `apex_evals` harness. Same judge, same project policies,
> different evaluation surface.

---

## Memory subsystems — Dynamic Ledger & TRACE

This repository ships two test-time-learning subsystems layered on the
baseline Mercor / Archipelago harness. Both are off by default; pick
at most one per run.

| Subsystem            | CLI flag             | Uses GT? | Spec                                                |
|----------------------|----------------------|----------|-----------------------------------------------------|
| **Dynamic Ledger**   | `--dynamic-ledger`   | No       | [`docs/DYNAMIC_LEDGER_PRD.md`](docs/DYNAMIC_LEDGER_PRD.md) |
| **TRACE**            | `--trace`            | Yes (boolean `criteria_passed == criteria_total` bit) | [`docs/TRACE_PRD.md`](docs/TRACE_PRD.md) |

Both subsystems share the same retrieval (dual top-k cosine on
`text-embedding-3-large`), the same per-domain isolation, the same
create-time cosine-block dedup, and the same per-task snapshot
discipline. They differ in:

- **TRACE** uses two LLM calls (reflector → curator) and threads the
  boolean correctness bit into both — per the TRACE paper. The
  generator cites bullets in its `final_answer.reasoning`; citations
  bump per-bullet `helpful` / `harmful` / `usage` counters that
  condition future edits.
- **Dynamic Ledger** uses a single curator call with NO ground-truth
  signal. The generator does not cite anything; the curator critiques
  the trajectory and prescribes domain standard practice.

Curator (and TRACE's reflector) run on the **same model** as the
selected agent profile — only the judge model is fixed at gpt-5.5.

## Dynamic Ledger — no-ground-truth extension

The contribution this repository carries beyond the baseline harness
is **Dynamic Ledger**: a no-ground-truth, per-domain, dual-retrieval
playbook of workflows the agent accumulates *during* an evaluation run.
After each task the curator (same model as the agent under test; only
the judge is fixed at gpt-5.5) reads the agent's full trajectory
— with no access to the rubric, the score, or the expected answer — and
emits a single JSON array of edit operations against the per-domain
ledger. Future tasks in the same domain retrieve from the updated
ledger and inject the most-relevant entries into the user prompt before
the agent starts. The ledger is off by default; enable with
`--dynamic-ledger` (the baseline pipeline is byte-identical when off).
See [`docs/DYNAMIC_LEDGER_PRD.md`](docs/DYNAMIC_LEDGER_PRD.md) for the
full specification.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DYNAMIC LEDGER — per-task lifecycle  (NO ground-truth signal anywhere)     │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│   Per-domain ledger                       task.prompt                       │
│   (active entries only;                       │                             │
│    persisted between tasks)                   │  embed via                  │
│            │                                  │  text-embedding-3-large     │
│            ▼                                  ▼                             │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │  ❶  RETRIEVE      dual top-k cosine, k=5 per axis    │                  │
│   │       top-k( content_embedding )                     │                  │
│   │          ∪                                           │                  │
│   │       top-k( source_problem_embedding )              │                  │
│   │       → deduped subset (≤ 2k entries)                │                  │
│   └────────────────────────┬────────────────────────────┘                   │
│                            ▼                                                │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │  ❷  INJECT       render strategies block;            │                  │
│   │                  prepend to USER message of          │                  │
│   │                  initial_messages.json               │                  │
│   │                  (vendor system prompt UNCHANGED)    │                  │
│   └────────────────────────┬────────────────────────────┘                   │
│                            ▼                                                │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │  ❸  AGENT        Archipelago react_toolbelt          │                  │
│   │                  (vendored, byte-equivalent)         │                  │
│   │                  9 MCP servers · multi-turn ReAct    │                  │
│   │                  → final_answer{answer, reasoning}   │                  │
│   └────────────────────────┬────────────────────────────┘                   │
│                            │  trajectory.json                               │
│                            ▼                                                │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │  ❹  GRADE  (vendored grader subprocess)              │                  │
│   │      reads trajectory.json                           │                  │
│   │      writes grades.json + results.csv row            │                  │
│   └────────────────────────┬────────────────────────────┘                   │
│                            │                                                │
│                            ▼     (NO arrow from GRADE to CURATE)            │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  ❺  CURATE     single LLM call (same model as the agent profile)│       │
│   │                                                                 │       │
│   │   Inputs:                                                       │       │
│   │     · active playbook (Dynamic Ledger for this domain)          │       │
│   │     · task prompt (no injection prefix)                         │       │
│   │     · full trajectory (tool results truncated, calls + reason   │       │
│   │       in full)                                                  │       │
│   │     ╳ no criteria · no score · no expected answer · no rubric   │       │
│   │                                                                 │       │
│   │   Output  (single JSON array of ops):                           │       │
│   │     <memory_updates>[                                           │       │
│   │       { "op":"CREATE",  "section","content","source_problem" },│       │
│   │       { "op":"UPDATE",  "entry_id","content" },                │       │
│   │       { "op":"CONSOLIDATE", "entry_ids":[…], … },               │       │
│   │       { "op":"DELETE",  "entry_id" },                           │       │
│   │       { "op":"NO_OP",   "reason" }                              │       │
│   │     ]</memory_updates>                                          │       │
│   │                                                                 │       │
│   │   Apply order:   DELETE → CONSOLIDATE → UPDATE → CREATE         │       │
│   │   CREATE is gated: cosine-block against the retrieved subset    │       │
│   │   prevents near-duplicate entries.                              │       │
│   └────────────────────────────────┬────────────────────────────────┘       │
│                                    │ updated ledger                         │
│                                    ▼                                        │
│         runs/<run>/dynamic_ledger/<Domain>/snapshot_NNNN.json               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The five-stage shape and its tag-name (`<memory_updates>`) follow
Suzgun et al.'s published Dynamic Ledger spec.** Our extensions, all
documented in [`docs/DYNAMIC_LEDGER_PRD.md`](docs/DYNAMIC_LEDGER_PRD.md):

- **No ground truth** to the curator (load-bearing fidelity test:
  `tests/test_dynamic_ledger_fidelity.py::test_curator_signature_has_no_outcome`).
- **Critical-diagnosis curator framing** — the curator acts as a senior
  reviewer that critiques the colleague's work, not a chronicler.
- **Multi-op output by default** (the curator emits 2–4 ops per session
  covering distinct lessons of varied scope).
- **Two entry shapes** — elaborate playbook entries (multi-paragraph
  tool-call workflows) and focused action notes (one-paragraph
  quirk/recovery notes).
- **Per-domain isolation** — each domain has its own ledger and snapshot
  history; never cross-domain retrieval.

### Results

We report **Pass@1** (final score = 100) and **mean score** (Mercor's
0/100 per-criterion average), one run per (task, model). The
Dynamic-Ledger-on column is intentionally empty pending the planned
sweep; the baseline column is what's currently in
`runs/ib-world219-grok43high/`.

| Method | IB World 219 (14-task pilot) Pass@1 | IB World 219 mean | Full IB (160 tasks) | Full Law (160 tasks) | Full MC (160 tasks) |
|---|---|---|---|---|---|
| Baseline · grok-4.3-high · no memory | **0 / 14 = 0.0 %** | **0.00** | _not yet run_ | _not yet run_ | _not yet run_ |
| Dynamic Ledger · grok-4.3-high · `--dynamic-ledger` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| TRACE · grok-4.3-high · `--trace` | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| _Public reference_ — GPT-5.5 (xHigh), Mercor leaderboard | n/a | n/a | _pending fetch_ | _pending fetch_ | _pending fetch_ |

Method spec: [`docs/DYNAMIC_LEDGER_PRD.md`](docs/DYNAMIC_LEDGER_PRD.md)
(no-GT) and [`docs/TRACE_PRD.md`](docs/TRACE_PRD.md) (uses-GT).
Project-wide context: [`docs/PROJECT.md`](docs/PROJECT.md).

> **Reading the baseline.** Three of the 14 World-219 IB tasks ended
> with `agent_status='failed'` after the agent exhausted `max_steps=50`
> in a `sheets_server` retry loop; the remaining 11 produced a wrong
> `final_answer`. Grading was correct — the vendor judge applies
> rounding-to-criterion-precision tolerance and rejects anything else.
> See `runs/ib-world219-grok43high/results.failures.jsonl` for the
> per-task failure log.

### Enabling the Dynamic Ledger

```bash
apex-agents-bench run --model grok-4.3-high \
    --world world_1e4d4288e63f4a08851a3cc441eb3ccb \
    --dynamic-ledger \
    --output runs/ib-w219-grok43high-ledger/results.csv
```

The CLI surface is `--dynamic-ledger / --no-dynamic-ledger`. Default is
off (the baseline path); when off, the CSV schema is byte-identical to
the no-ledger shape. When on, snapshots of the ledger and a curator
audit log are written to `runs/<run>/dynamic_ledger/<Domain>/`.

---

## TL;DR

A one-task pilot on Investment Banking at the cheapest profile,
~10 minutes end-to-end (Docker required):

```bash
make setup                                                 # one-time: venv + uv-sync vendor + .env
$EDITOR .env                                               # paste OPENAI_API_KEY, XAI_API_KEY, HF_TOKEN
make docker-check                                          # daemon up, image buildable, port free
make fetch-dataset                                         # one-time: download task + world index (~1.2 MB)

source .venv/bin/activate                                  # per-session
set -a; source .env; set +a                                # per-session: load keys into shell

apex-agents-bench run --model grok-4.3-low \
    --domain "Investment Banking" --limit 1 \
    --output runs/ib-pilot/results.csv
```

The judge is fixed (`gpt-5.5`); you never specify it. The agent runs
in a Docker container with the published 9-MCP-server toolbelt
(calendar, chat, code execution, sheets, filesystem, mail, pdfs,
slides, docs). Re-running the same `--output` resumes from where it
left off; completed tasks are never re-paid.

---

## Table of contents

0. [Dynamic Ledger — our extension](#dynamic-ledger--our-extension)  ←  architecture diagram + results table
1. [What this is, in five lines](#1-what-this-is-in-five-lines)
2. [First-time setup](#2-first-time-setup)
3. [Running the benchmark](#3-running-the-benchmark)  ←  the section most readers want
4. [Reading the results](#4-reading-the-results)
5. [Browsing the dataset](#5-browsing-the-dataset)
6. [Troubleshooting](#6-troubleshooting)
7. [Project policies](#7-project-policies)
8. [Repo layout](#8-repo-layout)
9. [Documentation index](#9-documentation-index)
10. [Citation, license, contact](#10-citation-license-contact)

---

## 1. What this is, in five lines

- **APEX-Agents** is a Mercor benchmark of 480 tasks across 33
  simulated "worlds" in three domains: **Investment Banking** (10
  worlds, 160 tasks), **Management Consulting** (11 worlds, 160 tasks),
  **Law** (12 worlds, 160 tasks). Each world ships a starter
  filesystem, calendar, inbox, chat history, and spreadsheets.
- An agent (default: Archipelago's `react_toolbelt_agent`) runs in a
  Docker container against one world for up to 50 steps / 1 hour with
  access to 9 MCP tools (calendar, chat, code execution, sheets,
  filesystem, mail, pdfs, slides, docs).
- A judge LLM grades each binary criterion Pass/Fail by comparing
  before/after snapshots and reading the agent-produced artifacts.
- We use **Mercor's official Archipelago harness** vendored at commit
  `3f4a8234` with **zero active vendor patches** — Archipelago accepts
  arbitrary LiteLLM model strings, so our test models work on
  unmodified vendor source. See
  [`vendor/archipelago/PATCHES.md`](vendor/archipelago/PATCHES.md).
- The **judge** is fixed: `gpt-5.5` at OpenAI's default
  `reasoning_effort=medium`. The **agent models** are `gpt-5.5-*` and
  `grok-4.3-*` profiles. **One run per (task, model), always.** See
  [§7](#7-project-policies).

---

## 2. First-time setup

Run in order; each step verifies the previous.

### 2.1 — Prerequisites

| Requirement | Why | Quick check |
|---|---|---|
| **Python 3.13** | Archipelago pins `>=3.13,<3.14`. | `python3.13 --version` |
| **Docker Desktop** (or Docker Engine) | The environment container hosts the MCP gateway + 9 servers. | `docker info` |
| **`uv`** | Vendor packages are uv-managed (avoids the `runner` module name collision). | `uv --version` |
| **~20 GB free disk** | World snapshots (~18.7 GB if you pre-fetch all 33) + the env container (~2 GB). | `df -h .` |
| **HF account, terms accepted** | `mercor/apex-agents` is gated; accept eval-only terms at https://huggingface.co/datasets/mercor/apex-agents while signed in. | `hf auth whoami` |

### 2.2 — Install the project

```bash
git clone https://github.com/jerry2247/dc-research-group-mercor-apex-agents.git apex-agents-bench
cd apex-agents-bench
make setup
```

`make setup` creates `.venv` (Python 3.13), pip-installs the wrapper
and dev tools, **uv-syncs** the vendored Archipelago `agents/` and
`grading/` packages into their own uv-managed venvs (both ship
top-level `runner` packages and can't coexist in one venv), registers
pre-commit hooks, and copies `.env.example` to `.env`.

### 2.3 — Fill in API keys

Open `.env` and set the keys you need. See
[`.env.example`](.env.example) for the full annotated list.

| Variable | Required for | Where to get one |
|---|---|---|
| `OPENAI_API_KEY` | the judge (`gpt-5.5`) AND `gpt-5.5-*` agent profiles | platform.openai.com |
| `XAI_API_KEY` | `grok-4.3-*` agent profiles | console.x.ai |
| `HF_TOKEN` | dataset download (gated) | huggingface.co/settings/tokens |

Optional:

| Variable | When you need it |
|---|---|
| `REDUCTO_API_KEY` | only matters when the agent produces chart-heavy artifacts (PDFs / spreadsheets) and the judge needs visual extraction. Many tasks produce text answers and grade fine without it. |

> **Whitespace matters.** `python-dotenv` does not strip whitespace.
> `HF_TOKEN= hf_...` (note the space) is silently invalid.

### 2.4 — Verify Docker + dataset access

```bash
make docker-check                  # daemon up; image buildable; port 8080 free
make fetch-dataset                 # downloads task + world index (~1.2 MB)
apex-agents-bench info             # paths + judge + vendor probe + HF/Docker checks
apex-agents-bench catalog          # writes data/catalog.json with dataset stats
apex-agents-bench models           # 7 active agent profiles
make check                         # 110 tests + ruff + mypy (non-docker; no API calls)
```

If any of those fail, fix the failure before spending budget.

World snapshots are **not** pre-downloaded by default. The runner
fetches each world's zip on demand (~200 MB – 1 GB per world) and
caches it. To pre-fetch all 33 worlds in advance:

```bash
hf download mercor/apex-agents --repo-type dataset \
    --include "world_files_zipped/*.zip" --local-dir data/apex-agents
```

(That's ~18.7 GB; usually unnecessary if you're running per-domain
pilots first.)

### 2.5 — Activate the venv and load `.env` (every new terminal session)

```bash
source .venv/bin/activate
set -a; source .env; set +a
```

Activation puts the `apex-agents-bench` command on `PATH`; the second
line exports the API keys into your shell.

---

## 3. Running the benchmark

### 3.1 — Recommended flow: pilot first, then scale

To verify the full pipeline before committing to a full domain, run
**one task first** with the output path you want for the eventual
full domain. The runner resumes by `task_id` on the same CSV, so the
pilot row is reused (never re-paid) when you scale.

```bash
# Step 1 — one-task pilot, ~$1–$5 depending on profile and world complexity
apex-agents-bench run --model grok-4.3-high \
    --domain "Investment Banking" --limit 1 \
    --output runs/ib-grok43high/results.csv

# Inspect: results.csv has 1 row with status=completed and a non-zero
# final_score; <task_id>/trajectory.json shows the agent's full
# message + tool-call history; <task_id>/grades.json has the judge's
# per-criterion rationales.

# Step 2 — finish the domain (remaining 159 IB tasks; runs sequentially)
apex-agents-bench run --model grok-4.3-high \
    --domain "Investment Banking" \
    --output runs/ib-grok43high/results.csv
```

### 3.2 — Command shape

```
apex-agents-bench run --model <profile> [--domain <d>] [--world <id>] [--limit <N>] [--output <path>]
```

| Flag | Values | Notes |
|---|---|---|
| `--model` | a profile from `apex-agents-bench models` | required |
| `--domain` | `"Investment Banking"` / `"Law"` / `"Management Consulting"` | optional; **case-sensitive, exact string including the space** |
| `--world` | a world id from `apex-agents-bench worlds` | optional |
| `--limit` | integer | optional; full domain = 160 |
| `--task-ids` | comma-separated ids | overrides `--domain` / `--world` / `--limit` |
| `--output` | path | default: `runs/<UTC>__<profile>__<scope>/results.csv` |

The judge is **fixed**: `gpt-5.5` at OpenAI's default
`reasoning_effort=medium`. You do not configure it.

### 3.3 — Agent profiles

`apex-agents-bench models` lists the current set:

| Family | Profiles | LiteLLM routing notes |
|---|---|---|
| OpenAI GPT-5.5 | `gpt-5.5-low`, `gpt-5.5-medium`, `gpt-5.5-high`, `gpt-5.5-xhigh` | `openai/gpt-5.5`; `reasoning_effort` per profile; `verbosity=medium`; no custom temperature is sent. |
| xAI Grok 4.3 | `grok-4.3-low`, `grok-4.3-medium`, `grok-4.3-high` | `xai/grok-4.3`; `reasoning_effort` per profile; `temperature=0.8`. |
| Anthropic Claude 4.6 (Bedrock) | **deferred** | mirror of apex-bench; will enable once Bedrock plumbing lands |

### 3.4 — More example invocations

```bash
# A single world end-to-end
apex-agents-bench run --model gpt-5.5-medium --world <world_id>

# A single task by id (overrides --domain / --world)
apex-agents-bench run --model gpt-5.5-medium \
    --task-ids task_9ba58a6197114140877a1df1754d2993

# Full Management Consulting domain
apex-agents-bench run --model gpt-5.5-high \
    --domain "Management Consulting"
```

### 3.5 — Output, resume, and provenance

```
runs/<UTC-timestamp>__<profile>__<scope>/
├── results.csv                              one row per completed task
├── results.run_manifest.json                run config, hashes, selected task ids
├── results.failures.jsonl                   skipped/preflight/grading failures
└── <task_id>/
    ├── trajectory.json                      agent's message history + tool calls
    ├── grades.json                          per-criterion judge rationales
    ├── final_snapshot.tar.gz                raw final container snapshot
    └── final_snapshot.zip                   snapshot passed to the grader
```

World snapshots are downloaded into `data/apex-agents/` and cached
(only fetched once per world). To resume a partial run, re-run with
the same `--output`; the runner reads existing rows and skips
`task_id`s whose `status == "completed"`. Default output paths include
a UTC timestamp, so explicit `--output` is required for resumability.

---

## 4. Reading the results

Each completed task writes one row to `results.csv`. The schema is
fixed in `src/apex_agents_bench/runner.py::CSV_HEADERS` and protected
by `tests/test_runner.py`.

### 4.1 — Score and provenance columns

| Column | Meaning |
|---|---|
| `task_id`, `domain`, `world_id` | dataset identifiers |
| `status` | always `completed` for a written row; the runner writes a row only when both the agent and the grader produced usable output |
| `agent_status` | the vendor's `AgentStatus` enum: `completed` / `failed` / `error` / `cancelled` |
| `final_score` | scoring method = `template`; `[0.0, 1.0]`; equals `criteria_passed / criteria_total` for our setup |
| `criteria_passed`, `criteria_total` | binary judge verdicts; see `grades.json` for per-criterion rationales |
| `steps_used` | ReAct steps the agent consumed (cap = 50 per project policy) |
| `wall_time_seconds` | task end-to-end, including container start, snapshot, and grading |
| `agent_profile`, `agent_model`, `judge_model` | provenance |
| `trajectory_path`, `grades_path` | filesystem pointers to the per-task JSON artifacts |

### 4.2 — Token telemetry columns (agent-side only)

The token count columns are sourced **verbatim** from the vendor's
`UsageTracker.to_dict()` output written to `trajectory.json` (see
`vendor/archipelago/agents/runner/utils/usage.py`). They are
cumulative across every LLM call the agent makes during the ReAct
loop — typically one call per step, for up to 50 steps.

| Column | Vendor field | Definition |
|---|---|---|
| `agent_prompt_tokens` | `usage.prompt_tokens` | Sum of `response.usage.prompt_tokens` across every agent LLM call in this task. |
| `agent_completion_tokens` | `usage.completion_tokens` | Sum of `response.usage.completion_tokens` across every agent LLM call. For reasoning models that don't expose reasoning tokens as a separate field, reasoning is included in this number. |
| `agent_total_tokens` | `usage.total_tokens` | `agent_prompt_tokens + agent_completion_tokens`. |
| `agent_final_step_completion_tokens` | `usage.final_answer_tokens` | The completion tokens of the *last* LLM call in the loop (the one that produced `final_answer`). Useful for distinguishing tasks where the agent typed a long final reply from those that just modified files and exited tersely. |
| `agent_usage_available` | top-level `usage` present | Whether the vendor trajectory included an agent usage block. |
| `agent_usage_source` | wrapper provenance | `trajectory_usage` when present, otherwise `unavailable`. |
| `agent_usage_consistent` | wrapper check | Whether `usage.total_tokens == prompt_tokens + completion_tokens`. |

These are **agent-side only**. The judge's token usage is recorded
separately in each task's `grades.json` and is intentionally NOT
rolled into the CSV — it's shared evaluation overhead, not a
model-output metric for cross-method comparisons.

No cost columns are written. Provider billing is not reconstructed from
LiteLLM pricing maps; use the saved provider-reported token counts for
post-run cost analysis if needed.

### 4.3 — End-of-run summary

At the end of a run the CLI prints a Rich table: tasks completed,
overall mean score, per-domain mean score, and the CSV path. The
table is recomputed from the CSV, so it reflects every successful
task on disk (including ones that landed during a previous resume).

---

## 5. Browsing the dataset

```bash
apex-agents-bench worlds                                       # all 33 worlds
apex-agents-bench worlds --domain "Investment Banking"         # 10 IB worlds
apex-agents-bench tasks                                        # all 480 tasks
apex-agents-bench tasks --domain "Law" -n 5                    # first 5 Law tasks
apex-agents-bench tasks --world <world_id>                     # tasks in one world
apex-agents-bench show <task_id>                               # full prompt + rubric + reference metadata
apex-agents-bench catalog                                      # JSON summary
```

Per-domain task characterization is in
[`docs/BENCHMARK_STRUCTURE.md`](docs/BENCHMARK_STRUCTURE.md).

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: apex-agents-bench` | venv not activated | `source .venv/bin/activate` |
| `error: Docker daemon is not running` | Docker Desktop quit | start Docker Desktop, then `make docker-check` |
| `error: port 8080 is in use` | something else on 8080 | `lsof -i :8080`; stop it, or set `APEX_AGENTS_HOST_PORT=8090` in `.env` |
| `HF dataset access denied` | eval-only terms not accepted | sign in at https://huggingface.co/datasets/mercor/apex-agents and click Agree |
| `Authentication failed: invalid api key` | key has whitespace or was revoked | check `.env`; no spaces around `=` |
| `litellm.BadRequestError: Unsupported value: 'temperature'` | shouldn't occur with the bundled profiles; if it does, a profile is overriding the reasoning-model temperature defaults | file an issue with `apex-agents-bench info` output |
| Agent hits step cap (50) without finalizing | task is harder than its 50-step budget | this matches Mercor's published config; logged as `agent_status=failed`, not a wrapper bug |
| First task per world is slow | world zip (~200 MB – 1 GB) is being fetched from HF and cached | subsequent tasks in that world reuse the cache |
| `make check` is slow on first run | uv-syncing two vendor venvs + installing the wrapper | only the first run; subsequent `make check` is < 5 s |

If something is genuinely broken in the harness, file an issue with
the `agent_status` of the failing task and `apex-agents-bench info`
output.

---

## 7. Project policies

These are NOT knobs. Each is enforced in code and protected by a test.

| Policy | Rationale | Enforced at |
|---|---|---|
| **1 run per (task, model)** | Mirror of apex-bench. Trades leaderboard-parity for cross-method comparability within budget. Variance signal from per-domain (n=160) and per-criterion bins. | `apex_agents_bench.config.RUNS_PER_TASK`; `tests/test_imports.py` |
| **Judge = `gpt-5.5`** at OpenAI default `reasoning_effort=medium` | Same judge as apex-bench keeps cross-benchmark numbers comparable. Deliberate diff from Archipelago's example default (`gemini/gemini-2.5-flash`). | `apex_agents_bench.config.DEFAULT_JUDGE_MODEL`; `tests/test_imports.py` |
| **Agent = `react_toolbelt_agent`** with `max_steps=50, timeout=3600s` | Exactly matches Mercor's published `examples/hugging_face_task/agent_config.json`. NOT the agent-registry defaults (250 / 10800), which Mercor's own example overrides. | `apex_agents_bench.config.AGENT_MAX_STEPS / AGENT_TIMEOUT_SECONDS`; `tests/test_fidelity.py` |
| **All 9 MCP servers, every run** | Same as upstream `mcp_config_all_oss_servers.json`. Subsetting tools would change task semantics; the runner refuses a subsetted config. | `apex_agents_bench.config.MCP_SERVERS`; `tests/test_fidelity.py` |
| **Zero vendor source modifications** | Archipelago accepts arbitrary LiteLLM model strings, so no `MODEL_MAPPINGS` patch is needed. `PATCHES.md` lists zero active patches. | `vendor/archipelago/PATCHES.md`; `tests/test_fidelity.py` |
| **Per-task fresh container** | World state leaks between agent runs in the same container; per-task isolation matches the published example's `docker compose down -v` + `up` pattern. | `apex_agents_bench.docker_env`; `tests/test_runner.py` |

Full discussion: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
Line-by-line fidelity audit: [`docs/AUDIT.md`](docs/AUDIT.md).

---

## 8. Repo layout

```
apex-agents-bench/
├── README.md                       <- this file
├── pyproject.toml                  PEP 621; Python 3.13; ruff / mypy / pytest config
├── Makefile                        `make help` lists every target
├── .env.example                    template for API keys (copied to .env on setup)
├── .gitattributes                  marks vendor/** as linguist-vendored
├── docs/                           9 docs (INDEX / ARCHITECTURE / AUDIT / ...)
├── scripts/                        setup.sh, docker_check.sh, fetch_dataset.sh, smoke_test.sh
├── src/apex_agents_bench/          wrapper (CLI, config, runner, agent profiles, ...)
├── vendor/archipelago/             pristine Mercor Archipelago + 3 provenance files
├── tests/                          110 tests; all pass on a clean tree
├── data/                           gitignored — dataset index + world snapshot cache
└── runs/                           gitignored — per-run CSV + trajectories + grades + snapshots
```

Why the wrapper is a separate package: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 9. Documentation index

| Doc | What it answers |
|---|---|
| [`docs/INDEX.md`](docs/INDEX.md) | full doc index with section pointers |
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | 4-phase project plan (Phase 3 = Dynamic Ledger integration) |
| [`docs/AUDIT.md`](docs/AUDIT.md) | line-by-line confirmation we match Archipelago's behavior |
| [`docs/BENCHMARK_STRUCTURE.md`](docs/BENCHMARK_STRUCTURE.md) | what the 480 tasks across 33 worlds look like |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | 1 run/task policy, dataset license, what gets recorded per run |
| [`docs/HARNESS_NOTES.md`](docs/HARNESS_NOTES.md) | how Archipelago works internally (ReAct loop, MCP gateway, ReSum) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | wrapper-vs-vendor split + diff policy |
| [`docs/COST.md`](docs/COST.md) | per-task and full-run budgets across profiles |
| [`docs/DOCKER.md`](docs/DOCKER.md) | Docker prereqs, troubleshooting, image build |
| [`vendor/archipelago/UPSTREAM.md`](vendor/archipelago/UPSTREAM.md) | upstream commit SHA + resync procedure |
| [`vendor/archipelago/PATCHES.md`](vendor/archipelago/PATCHES.md) | every vendor edit (currently zero) |

---

## 10. Citation, license, contact

**Citation.** Cite the underlying benchmark first — it is Mercor's
work, not ours:

> Vidgen, B. et al. *APEX-Agents.* 2026. arXiv:2601.14242.

If you build on this evaluation harness specifically (the wrapper, the
agent-profile registry, the reproducibility schema), a suggested form is:

> Gu, J., Liao, K., Liu, S., Nair, R., Yang, A., in collaboration with
> M. Suzgun (Stanford SAIL NLP). *apex-agents-bench: evaluation harness
> for Dynamic Ledger on Mercor's APEX-Agents.* 2026.
> https://github.com/jerry2247/dc-research-group-mercor-apex-agents

The research contribution under evaluation here is **Dynamic Ledger**
(our group's extension of Dynamic Cheatsheet, adapted to a multi-turn
agent surface); this repository is the engineering harness that lets
us evaluate it reproducibly against Mercor's published benchmark.

**License.** This repository is MIT (see `LICENSE`). The vendored
Archipelago harness is Apache-2.0 (see
`vendor/archipelago/LICENSE_UPSTREAM`). The APEX-Agents dataset is not
redistributed; it is fetched at run time under its own CC-BY-4.0 +
eval-only terms (see `docs/REPRODUCIBILITY.md`).

**Contact.** File an issue on the GitHub repository. For substantive
research questions, contact the authors directly.

---

## Related

- [`apex-bench`](https://github.com/jerry2247/dc-research-group-mercor-apex)
  — sister repo targeting the **APEX-v1-extended** benchmark
  (single-shot text-deliverable surface, vendored `apex_evals`
  harness, same `gpt-5.5` judge). Use that repo for the text-only
  benchmark; use this repo for the agentic benchmark.
