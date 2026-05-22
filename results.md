# Results — apex-agents-bench

Detailed results for the **Dynamic Ledger** subsystem evaluated on Mercor's **APEX-Agents** benchmark. Numbers are headline-only here; the source of truth is the per-task CSV (`runs/<run>/results.csv`) and the per-task ledger snapshots (`runs/<run>/dynamic_ledger/<Domain>/`).

## Setup

| | Value |
|---|---|
| Agent model | `grok-4.3` (xAI), `reasoning_effort = high`, `temperature = 0.8` |
| Judge model | `gpt-5.5` (OpenAI), reasoning effort `medium` (Mercor's published judge) |
| Embedding model (retriever) | `text-embedding-3-large` (OpenAI), 3072-dim |
| Agent loop | ReAct toolbelt agent (Archipelago), `max_steps = 50`, MCP timeout `3600 s` |
| MCP servers | calendar, chat, code_execution, sheets, filesystem, mail, pdf, slides, docs |
| Runs per (task, agent profile) | 1 |
| Rollout slice run end-to-end so far | Investment Banking, world `1e4d4288…` (n = 14 selected; 11 graded; 3 agent-runtime failures excluded from aggregates) |
| Rollout order for remaining cells | see [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md) |

## Headline — Pass@1 (`criteria_passed == criteria_total`) · mean `final_score`

| Method | Investment Banking | Law | Management Consulting | Macro mean |
|---|:---:|:---:|:---:|:---:|
| `grok-4.3-high` (baseline) | 0/11 · 0.000 *(world 1e4d4288, n=11)* | TBD | TBD | TBD |
| + Dynamic Ledger | **1/11 · 0.190** *(world 1e4d4288, n=11)* | TBD | TBD | TBD |
| + TRACE | TBD | TBD | TBD | TBD |

Each cell will accumulate worlds as the rollout proceeds. Investment Banking has 10 worlds (160 tasks total); Law has 12 (160); Management Consulting has 11 (160). The per-domain Dynamic Ledger persists across worlds inside a domain (snapshot store under `runs/grok43high-dl/dynamic_ledger/<Domain>/`), so a single IB Pass@1 / mean over all 160 IB tasks is the eventual headline.

## Did the curator learn something?

Yes. Concrete worked example.

On **task `bb48b8b…`** (multi-method DCF implied share-price computation), the baseline scored **0.000** (0 of 2 criteria met). With Dynamic Ledger, the same model on the same task scored **1.000** (2 of 2 criteria met).

The retriever surfaced three entries from the running ledger at task start. All three were emitted by the curator on *earlier* IB tasks (different DCF cases that the agent had not solved cleanly), and all three are in formula / convention shape — passive reference content, not procedures:

```
Midyear convention DCF discount periods
  Midyear convention adjusts explicit-period FCF discount factors to t-0.5
  for each year t (periods = [0.5, 1.5, ..., n-0.5] for n-year forecast);
  terminal value is discounted at the final mid-year period n-0.5 rather
  than n.

Implied FCF yield from DCF terminal methods
  Implied terminal FCF yield = FCF_t / TV_t (as % to 1 decimal);
  TV_PGM   = FCF_t × (1+g) / (WACC − g);
  TV_exit  = EBITDA_t × exit_multiple;
  use terminal-year FCF even when model outputs unlevered FCF and debt=0.

Revenue projection with segment-specific growth rates in DCF models
  Revenue_t,s = revenue_{t-1,s} × (1 + g_s) for each segment s;
  total_revenue_t = sum over s of revenue_t,s; use total_revenue_t as
  denominator for all operating expense, capex and ΔNWC percentages and
  as base for NOPAT and UFCF.
```

The generator-side injection block frames the entries as a passive formula sheet ("treat the cheatsheet the way you'd treat a formula sheet during an exam — do not follow it"), so the agent's own analysis and tool usage stayed authoritative. The cheatsheet entries spared it from re-deriving the terminal-value relations and the per-segment revenue projection structure; the deliverable passed both criteria.

## Per-task breakdown

Rows are listed **in execution order** (`ord`) — the order in which the runner processed each task and the curator's ordinal at the time of its call. This is the order in which the ledger grew, so the `retrieved` column reads coherently across rows.

**Column meanings:**
- `final_score` — rubric score on this task (0 – 1), as graded by the gpt-5.5 judge against Mercor's per-criterion rubric. `N/A` when the underlying agent loop did not produce a gradeable trajectory.
- `retrieved` — number of cheatsheet entries the retriever injected at task start. Retrieval is dual-axis top-5 cosine (top-5 on the entry's `content_embedding` axis, top-5 on its `source_problem_embedding` axis, deduplicated — so at most 10 candidates) followed by a `retrieval_similarity_threshold = 0.40` floor that drops any candidate whose best-axis cosine to the task prompt is below 0.40. The final count is therefore `min(10, # active entries scoring ≥ 0.40 against this task)` — it varies because the cosine distribution against each task is different, not because retrieval is arbitrary.
- `ops` — what the curator wrote *after* this task: `<N>C/<N>U/<N>D` for CREATE/UPDATE/DELETE counts, or `—` when the curator emitted nothing (the task did not surface a transferable lesson).

| ord | task | baseline | dynamic ledger | Δ | retrieved | ops |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `task_ad52b80…` | 0.000 | 0.000 | +0.000 | 0 | 1C/0U/0D |
| 2 | `task_8985fd7…` | N/A | N/A | N/A | N/A | 1C/0U/0D |
| 3 | `task_eb001a3…` | 0.000 | 0.000 | +0.000 | 1 | 1C/0U/0D |
| 4 | `task_34254f3…` | N/A | N/A | N/A | N/A | — |
| 5 | `task_00b30f5…` | 0.000 | 0.800 | **+0.800** | 0 | 1C/0U/0D |
| 6 | `task_9627f5b…` | N/A | N/A | N/A | N/A | — |
| 7 | `task_95a1101…` | 0.000 | 0.000 | +0.000 | 0 | — |
| 8 | `task_0de09be…` | 0.000 | 0.286 | +0.286 | 0 | 1C/0U/0D |
| 9 | `task_6ed4c4d…` | 0.000 | 0.000 | +0.000 | 2 | 1C/0U/0D |
| 10 | `task_7190412…` | 0.000 | 0.000 | +0.000 | 2 | 1C/0U/0D |
| 11 | `task_bb48b8b…` | 0.000 | **1.000** | **+1.000** | 3 | — |
| 12 | `task_583c24a…` | 0.000 | 0.000 | +0.000 | 1 | 1C/0U/0D |
| 13 | `task_1da4eac…` | 0.000 | 0.000 | +0.000 | 3 | 1C/0U/0D |
| 14 | `task_de8a536…` | 0.000 | 0.000 | +0.000 | 5 | — |
| | **mean over 11 graded** | **0.000** | **0.190** | **+0.190** | | |
| | **Pass@1 over 11 graded** | **0 / 11** | **1 / 11** | | | |

`N/A` rows are tasks where the underlying agent loop did not produce a gradeable trajectory on either run (the failure happens before the rubric is reached); they are excluded from both means and Pass@1. The curator still runs on these trajectories, and any ops it emits *are* preserved in the ledger and inherited by subsequent tasks.

Raw CSVs: [`runs/grok43high-baseline/results.csv`](runs/grok43high-baseline/results.csv) (baseline), [`runs/grok43high-dl/results.csv`](runs/grok43high-dl/results.csv) (Dynamic Ledger). Per-task trajectories: `runs/<run>/task_<id>/trajectory.json`. Per-domain ledger snapshots: `runs/grok43high-dl/dynamic_ledger/<Domain>/snapshot_NNNN.json`.

## Caveats

- **Single seed.** All numbers are from one run per (task, agent profile). grok-4.3-high uses `temperature = 0.8`; same-task variance is non-trivial. A multi-seed sweep is planned.
- **Single world.** Only world `1e4d4288…` (14 IB tasks) is reported. The remaining 9 IB worlds and the other two APEX-Agents domains (Law, Management Consulting) are pre-registered in [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md) and pending.
- **One model.** Only grok-4.3-high is reported. Multi-model ablations are planned.
- **TRACE** is not reported here. TRACE uses the per-task correctness bit, which makes it not directly comparable to the no-GT Dynamic Ledger; it is included in the repo for future ablation.

## Reproduce these numbers

```bash
# Baseline
apex-agents-bench run --model grok-4.3-high --world 219 \
    --output runs/grok43high-baseline/results.csv

# Dynamic Ledger
apex-agents-bench run --model grok-4.3-high --world 219 \
    --dynamic-ledger \
    --output runs/grok43high-dl/results.csv
```

Continuing the rollout to the next IB world (and on through Law / Management Consulting) uses the same `--output` paths so the per-domain ledger persists; see [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md) for the exact `--world` values and the order they are run in. Full setup and Docker prerequisites: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
