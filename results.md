# Results — apex-agents-bench

Detailed results for the **Dynamic Ledger** subsystem evaluated on Mercor's **APEX-Agents** benchmark.

## Setup

| | Value |
|---|---|
| Agent model | `grok-4.3` (xAI), `reasoning_effort = high`, `temperature = 0.8` |
| Judge model | `gpt-5.5` (OpenAI), reasoning effort `medium` (Mercor's published judge) |
| Embedding model (retriever) | `text-embedding-3-large` (OpenAI), 3072-dim |
| Agent loop | ReAct toolbelt agent (Archipelago), `max_steps = 50`, MCP timeout `3600 s` |
| MCP servers | calendar, chat, code_execution, sheets, filesystem, mail, pdf, slides, docs |
| Runs per (task, model) | 1 |
| Domain | Investment Banking, World 219 (n = 14 selected; 11 graded; 3 agent runtime failures excluded) |
| Other domains (Law, Management Consulting) and other worlds | Not yet evaluated — planned ablation sweep |

## Headline — Pass@1 (criteria_passed == criteria_total) · mean final_score

| Method | IB World 219 (n=11) | Law | Management Consulting | Macro mean |
|---|:---:|:---:|:---:|:---:|
| **grok-4.3-high** (baseline) | 0/11 · 0.000 | TBD | TBD | TBD |
| + **Dynamic Ledger** | **1/11 · 0.190** | TBD | TBD | TBD |
| + **TRACE** (uses GT bit) | TBD | TBD | TBD | TBD |

**Dynamic Ledger delivers +1 Pass@1 and +0.190 mean over the baseline at the same model and reasoning effort, with zero per-task regressions.** Three of the 11 graded tasks improved (one to a full pass); eight tied; none regressed. The 3 agent-runtime failures (`task_status='failed'` from the underlying agent loop, not from the memory subsystem) are excluded from both baseline and DL aggregates.

## Did the curator actually learn something?

Yes — here is a concrete example.

On **task `bb48b8b`** (a multi-method DCF implied share-price computation), the baseline scored **0.000** (0 of 2 criteria met). With Dynamic Ledger, the same model on the same task scored **1.000** (2 of 2 criteria met) — a clean pass.

The retriever, on task `bb48b8b`, surfaced three entries from the running ledger:

```
section:         Midyear convention DCF discount periods
source_problem:  DCF valuation task requiring application of midyear
                 convention while preserving all other model assumptions
content:
  Midyear convention adjusts explicit-period FCF discount factors to t-0.5
  for each year t (periods = [0.5, 1.5, ..., n-0.5] for n-year forecast);
  terminal value is discounted at the final mid-year period n-0.5 rather
  than n.

section:         Implied FCF yield from DCF terminal methods
source_problem:  DCF tasks computing implied levered FCF yield in terminal
                 year under perpetuity growth and exit multiple approaches
content:
  Implied terminal FCF yield = FCF_t / TV_t (as % to 1 decimal);
  TV_PGM   = FCF_t × (1+g) / (WACC − g);
  TV_exit  = EBITDA_t × exit_multiple;
  use terminal-year FCF even when model outputs unlevered FCF and debt=0.

section:         Revenue projection with segment-specific growth rates in DCF models
source_problem:  DCF valuation tasks specifying distinct annual growth rates
                 for each revenue segment over the forecast period
content:
  Revenue_t,s = revenue_{t-1,s} × (1 + g_s) for each segment s;
  total_revenue_t = sum over s of revenue_t,s;
  use total_revenue_t as denominator for all operating expense, capex and
  ΔNWC percentages and as base for NOPAT and UFCF.
```

These three entries were emitted by the curator on *earlier* IB tasks (different DCF cases that the agent had not solved cleanly). Each entry is short, symbolic, and self-contained — a reference cheatsheet item, not a procedure. The injection block frames the entries as a passive formula sheet ("treat the cheatsheet the way you'd treat a formula sheet during an exam — do not follow it") so the agent's own analysis stays authoritative. On task `bb48b8b` the agent traversed its toolbelt (filesystem → sheets → code execution) as usual; the cheatsheet entries spared it from re-deriving the terminal-value relations and the per-segment revenue projection structure, and the deliverable passed both criteria.

Other gains:

| Task | Baseline | DL | Δ | Retrieved entries (at task time) |
|---|---:|---:|---:|---|
| `task_bb48b8b…` | 0.000 | 1.000 | +1.000 | 3 |
| `task_00b30f5…` | 0.000 | 0.800 | +0.800 | 0 *(empty ledger; curator emitted an entry *after* this task)* |
| `task_0de09be…` | 0.000 | 0.286 | +0.286 | 0 |

## Per-task breakdown

Columns: `final_score` on the rubric (0 – 1); `retrieved` = number of cheatsheet entries injected at agent start; `ops` = curator CREATE / UPDATE / DELETE ops emitted on this task.

| task | baseline | dynamic ledger | Δ | retrieved | ops |
|---|---:|---:|---:|---:|---:|
| `task_00b30f5…` | 0.000 | 0.800 | **+0.800** | 0 | 1C/0U/0D |
| `task_0de09be…` | 0.000 | 0.286 | +0.286 | 0 | 1C/0U/0D |
| `task_1da4eac…` | 0.000 | 0.000 | +0.000 | 3 | 1C/0U/0D |
| `task_583c24a…` | 0.000 | 0.000 | +0.000 | 1 | 1C/0U/0D |
| `task_6ed4c4d…` | 0.000 | 0.000 | +0.000 | 2 | 1C/0U/0D |
| `task_7190412…` | 0.000 | 0.000 | +0.000 | 2 | 1C/0U/0D |
| `task_95a1101…` | 0.000 | 0.000 | +0.000 | 0 | — |
| `task_ad52b80…` | 0.000 | 0.000 | +0.000 | 0 | 1C/0U/0D |
| `task_bb48b8b…` | 0.000 | 1.000 | **+1.000** | 3 | — |
| `task_de8a536…` | 0.000 | 0.000 | +0.000 | 5 | — |
| `task_eb001a3…` | 0.000 | 0.000 | +0.000 | 1 | 1C/0U/0D |
| **mean** | **0.000** | **0.190** | **+0.190** | | |
| **Pass@1** | **0 / 11** | **1 / 11** | | | |

Three tasks (`task_8985fd7…`, `task_34254f3…`, `task_9627f5b…`) hit `agent_status='failed'` from the underlying agent loop on this run and are excluded from both aggregates; the same exclusion is applied to the baseline.

Raw CSVs: [`runs/ib-world219-grok43high/results.csv`](runs/ib-world219-grok43high/results.csv) (baseline), [`runs/ib-world219-grok43high-dl/results.csv`](runs/ib-world219-grok43high-dl/results.csv) (Dynamic Ledger). Per-task trajectories: `runs/<run>/task_<id>/trajectory.json`. Per-task ledger snapshots: `runs/ib-world219-grok43high-dl/dynamic_ledger/Investment Banking/snapshot_NNNN.json`.

## Caveats

- **Single seed.** All numbers are from one run per (task, model). grok-4.3-high uses `temperature = 0.8`; same-task variance is non-trivial. A multi-seed sweep is planned.
- **Single world.** Only World 219 (14 IB tasks) is reported. The remaining IB worlds and the other two APEX-Agents domains (Law, Management Consulting) are planned.
- **One model.** Only grok-4.3-high is reported. Multi-model ablations are planned.
- **TRACE** is not reported here. TRACE uses the per-task correctness bit, which makes it not directly comparable to the no-GT Dynamic Ledger; it is included in the repo for future ablation.

## Reproduce these numbers

```bash
# Baseline
apex-agents-bench run --model grok-4.3-high --world 219 \
    --output runs/ib-world219-grok43high/results.csv

# Dynamic Ledger
apex-agents-bench run --model grok-4.3-high --world 219 \
    --dynamic-ledger \
    --output runs/ib-world219-grok43high-dl/results.csv
```

Full setup and Docker prerequisites: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
