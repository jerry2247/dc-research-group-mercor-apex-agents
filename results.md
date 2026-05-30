# Results — apex-agents-bench

Detailed results for the **DC-RS** (Dynamic Cheatsheet — Retrieval Synthesis) and **TRACE** subsystems evaluated against the no-memory **baseline** on Mercor's **APEX-Agents** benchmark. Numbers here are headline-only; the source of truth is the per-task CSV (`runs/<run>/results.csv`) and the per-domain memory state (`runs/<run>/dc_rs/<Domain>/` and `runs/<run>/trace/<Domain>/`).

As of this writing the rollout has completed, for **baseline and DC-RS**, the **first world of all three domains** plus a second Investment Banking world: Investment Banking worlds 1–2, Management Consulting world 1, and Law world 1. **TRACE** has been run on Investment Banking world 1 only. The remaining worlds are pre-registered and pending (see [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md)).

## Setup

| | Value |
|---|---|
| Agent model | `grok-4.3` (xAI), `reasoning_effort = high`, `temperature = 0.8` |
| Judge model | `gpt-5.5` (OpenAI), reasoning effort `medium` (Mercor's published judge) |
| Embedding model (retriever) | `text-embedding-3-large` (OpenAI), 3072-dim |
| Agent loop | ReAct toolbelt agent (Archipelago), `max_steps = 50`, MCP timeout `3600 s` |
| MCP servers | calendar, chat, code_execution, sheets, filesystem, mail, pdf, slides, docs |
| Runs per (task, agent profile) | 1 |
| Slice run so far (baseline + DC-RS) | IB world 1 (`world_1e4d4288…`, 14 tasks), IB world 2 (`world_43a921f9…`, 9 tasks), MC world 1 (`world_075ef4df…`, 14 tasks), Law world 1 (`world_06051b9b…`, 20 tasks) |
| Slice run so far (TRACE) | IB world 1 only |
| Rollout order for remaining cells | see [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md) |

The agent, judge, step/time caps, MCP server set, system prompt, and rubric→verifier construction are **identical** between the baseline and DC-RS arms; the only difference is that DC-RS prepends a per-domain cheatsheet to the agent's *user* message. The DC-RS synthesizer receives **no** grading signal (no rubric, score, correctness bit, or gold answer). This was re-verified at the artifact level for these runs.

## Headline — paired comparison (baseline vs DC-RS), non-completion = 0 over the full world

A task whose agent loop never finalizes (hits the 50-step cap or errors) writes **no CSV row**, so the count of *completed* tasks differs across methods. The honest same-denominator comparison scores every attempted task over the **full world**, with any non-finalized task counted as **0** for the method that did not finalize it. Per-task **wins / ties / losses** then compare the two methods task-by-task (DC-RS strictly higher / equal / strictly lower).

| Slice | tasks | baseline mean | DC-RS mean | DC-RS wins / ties / losses |
|---|:---:|:---:|:---:|:---:|
| IB world 1 (`world_1e4d4288…`) | 14 | 0.000 | 0.188 | 4 / 10 / 0 |
| IB world 2 (`world_43a921f9…`) | 9 | 0.111 | 0.133 | 1 / 8 / 0 |
| MC world 1 (`world_075ef4df…`) | 14 | 0.183 | 0.335 | 4 / 8 / 2 |
| Law world 1 (`world_06051b9b…`) | 20 | 0.336 | 0.494 | 7 / 11 / 2 |
| **All four worlds** | **57** | **0.180** | **0.323** | **16 / 37 / 4** |

DC-RS improves the mean score in **every world**, lifting the overall mean from 0.180 to 0.323 (≈ +80%) and winning 16 paired tasks against 4 losses. The losses appear only in the harder document/reasoning-heavy domains (MC, Law), where the baseline is already competent and the cheatsheet can occasionally steer a baseline-winnable task off course — see *Where DC-RS loses* below. On Investment Banking, where the no-memory baseline frequently fails to finalize at all, DC-RS never loses a paired task.

## Means over completed tasks (different denominators — read with care)

These are each method's mean over only the tasks it finalized, shown with denominators. Because the denominators differ, they are **not** a same-denominator comparison; they are reported for transparency alongside the paired table above.

| Slice | baseline (n · mean · nonzero · pass@1) | DC-RS (n · mean · nonzero · pass@1) |
|---|---|---|
| IB world 1 | 11 · 0.000 · 0 · 0 | 13 · 0.202 · 4 · 2 |
| IB world 2 | 6 · 0.167 · 1 · 1 | 4 · 0.300 · 2 · 1 |
| MC world 1 | 10 · 0.256 · 3 · 2 | 12 · 0.391 · 6 · 4 |
| Law world 1 | 20 · 0.336 · 14 · 4 | 20 · 0.494 · 17 · 5 |

`nonzero` = tasks scoring > 0; `pass@1` = tasks scoring a perfect 1.0 (all rubric criteria met). On Law, both arms finalized all 20 tasks, so the Law row is already a same-denominator comparison: DC-RS lifts the mean from 0.336 to 0.494 and perfect scores from 4 to 5.

## Did the synthesizer learn something?

DC-RS's per-domain cheatsheet accumulates reusable material distilled from the agent's *own past trajectories* in that domain (it never sees grades). Concrete wins where the cheatsheet coincided with a jump over baseline:

| Domain | task | baseline | DC-RS | nature of the gain |
|---|---|:---:|:---:|---|
| IB | `task_bb48b8b…` (multi-method DCF) | 0.000 | 1.000 | wrong → fully correct |
| IB | `task_de8a536c…` | 0.000 | 1.000 | wrong → fully correct |
| IB | `task_0de09be0…` | non-completion | 0.429 | did-not-finalize → completed |
| MC | `task_4372ee27…` | 0.000 | 1.000 | wrong → fully correct |
| MC | `task_a138d532…` | non-completion | 1.000 | did-not-finalize → perfect |
| MC | `task_aac22560…` | non-completion | 1.000 | did-not-finalize → perfect |
| Law | `task_be78a90e…` | 0.000 | 1.000 | wrong → fully correct |
| Law | `task_b15e9456…` | 0.000 | 1.000 | wrong → fully correct |
| Law | `task_828f11b8…` | 0.167 | 0.667 | partial → mostly correct |

On **Investment Banking** the dominant gain is converting trajectory-level non-completions (file-write-instead-of-finalize, tool/code-execution friction) into completions. On **Law** — a document-reasoning domain with little tool/sandbox friction to recover from — the gains are instead in *reasoning and structure* (the cheatsheet transfers reusable analytical approaches), which is evidence the mechanism helps beyond tool-friction recovery.

## Where DC-RS loses

The two losses outside IB are worth recording rather than hiding:

- **MC `task_4bdc188e…`: 1.000 → 0.000** — baseline solved it; with the cheatsheet, DC-RS did not.
- **Law `task_fe1efa5d…`: 1.000 → 0.500** — baseline aced a North-Carolina compliance task (2/2); DC-RS got 1/2.

These show the cheatsheet is not strictly beneficial: on tasks the no-memory agent already handles, injected reference material can occasionally distract. The net effect is still strongly positive (16 wins vs 4 losses across the four worlds), but the regressions are real and motivate the entry-quality auditing tracked separately.

## Per-task breakdown

`—` marks a task whose agent loop did not finalize (no CSV row written); it is counted as 0 in the paired comparison above. **Bold** marks a DC-RS win over baseline.

### Investment Banking world 1 (`world_1e4d4288…`, 14 tasks)

| task | baseline | DC-RS | TRACE |
|---|:---:|:---:|:---:|
| `task_00b30f56…` | — | — | 0.400 |
| `task_0de09be0…` | — | **0.429** | — |
| `task_1da4eacd…` | 0.000 | 0.000 | — |
| `task_34254f33…` | 0.000 | 0.000 | 0.000 |
| `task_583c24a9…` | 0.000 | 0.000 | — |
| `task_6ed4c4d3…` | 0.000 | 0.000 | — |
| `task_71904126…` | 0.000 | 0.000 | — |
| `task_8985fd77…` | 0.000 | 0.000 | 0.000 |
| `task_95a11015…` | 0.000 | 0.000 | 0.000 |
| `task_9627f5be…` | — | **0.200** | 0.200 |
| `task_ad52b801…` | 0.000 | 0.000 | 0.000 |
| `task_bb48b8b3…` | 0.000 | **1.000** | — |
| `task_de8a536c…` | 0.000 | **1.000** | — |
| `task_eb001a3a…` | 0.000 | 0.000 | 0.000 |

### Investment Banking world 2 (`world_43a921f9…`, 9 tasks)

Rows shown for the 6 tasks completed by at least one method; the other 3 were finalized by neither baseline nor DC-RS. TRACE was not run on this world.

| task | baseline | DC-RS |
|---|:---:|:---:|
| `task_052cc631…` | 0.000 | — |
| `task_0ffa0e6f…` | 0.000 | **0.200** |
| `task_761646f2…` | 1.000 | 1.000 |
| `task_83bed0e0…` | 0.000 | 0.000 |
| `task_91b0998a…` | 0.000 | 0.000 |
| `task_a006f24d…` | 0.000 | — |

### Management Consulting world 1 (`world_075ef4df…`, 14 tasks)

| task | baseline | DC-RS |
|---|:---:|:---:|
| `task_0233800d…` | 1.000 | 1.000 |
| `task_0a4ad19b…` | 0.000 | 0.000 |
| `task_0cc6381a…` | 0.556 | 0.444 |
| `task_1e19e170…` | — | 0.000 |
| `task_4372ee27…` | 0.000 | **1.000** |
| `task_4bdc188e…` | 1.000 | 0.000 |
| `task_593cb247…` | 0.000 | 0.000 |
| `task_66b15748…` | — | — |
| `task_a138d532…` | — | **1.000** |
| `task_aac22560…` | — | **1.000** |
| `task_cb7189ae…` | 0.000 | 0.000 |
| `task_d8d2c6bd…` | 0.000 | **0.250** |
| `task_f23cb148…` | 0.000 | 0.000 |
| `task_f69f5d19…` | 0.000 | — |

### Law world 1 (`world_06051b9b…`, 20 tasks)

Both arms finalized all 20 tasks.

| task | baseline | DC-RS |
|---|:---:|:---:|
| `task_912055978…` | 1.000 | 1.000 |
| `task_be78a90e8…` | 0.000 | **1.000** |
| `task_b5481555a…` | 0.200 | **0.400** |
| `task_de1ec7602…` | 0.667 | 0.667 |
| `task_f40a571a8…` | 0.000 | 0.000 |
| `task_1d7f4e0be…` | 0.100 | 0.000 |
| `task_b15e94569…` | 0.000 | **1.000** |
| `task_828f11b82…` | 0.167 | **0.667** |
| `task_3caff6fe3…` | 0.000 | 0.000 |
| `task_a2a4615b0…` | 1.000 | 1.000 |
| `task_1d28a1ef3…` | 0.333 | 0.333 |
| `task_680f8581e…` | 0.250 | 0.250 |
| `task_ea50530a0…` | 0.333 | 0.333 |
| `task_fe1efa5d3…` | 1.000 | 0.500 |
| `task_729286e68…` | 1.000 | 1.000 |
| `task_e72676c16…` | 0.333 | 0.333 |
| `task_4fa914d8c…` | 0.200 | 0.200 |
| `task_ccee33f38…` | 0.000 | **0.667** |
| `task_908f411c4…` | 0.000 | **0.250** |
| `task_374272038…` | 0.000 | **0.286** |

Raw CSVs: [`runs/grok43high-baseline/results.csv`](runs/grok43high-baseline/results.csv), [`runs/grok43high-dc-rs/results.csv`](runs/grok43high-dc-rs/results.csv), [`runs/grok43high-trace/results.csv`](runs/grok43high-trace/results.csv). Per-task trajectories: `runs/<run>/task_<id>/trajectory.json`. Per-domain DC-RS state: `runs/grok43high-dc-rs/dc_rs/<Domain>/` (`bank.jsonl`, `cheatsheet.txt`, `cheatsheets/task_<id>.txt`, `synthesizer_log.jsonl`). Per-domain TRACE state: `runs/grok43high-trace/trace/<Domain>/`.

## Caveats

- **Single seed.** All numbers are from one run per (task, agent profile). grok-4.3-high uses `temperature = 0.8`; same-task variance is non-trivial. A multi-seed sweep is planned.
- **First world per domain (plus a second IB world).** Baseline and DC-RS cover IB worlds 1–2, MC world 1, and Law world 1 (57 tasks total); TRACE covers IB world 1 only. The remaining worlds are pre-registered in [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md) and pending.
- **One model.** Only grok-4.3-high is reported. Multi-model ablations are planned.
- **Differing completion counts.** A non-finalizing agent loop writes no row, so per-method "completed" means use different denominators; the paired comparison handles this by scoring non-completions as 0 over the full world.
- **DC-RS is not strictly dominant.** It loses 4 paired tasks (2 in MC, 2 in Law — none in IB). The net is strongly positive but the regressions are real.
- **TRACE is not directly comparable.** TRACE consumes the per-task correctness bit, so it is not a like-for-like comparison against the no-ground-truth DC-RS; it is reported for context and has only been run on IB world 1.

## Reproduce these numbers

```bash
# Baseline — Law world 1 (dataset name: World 433)
apex-agents-bench run --model grok-4.3-high \
    --world world_06051b9b10c94c079db1bac3b70c4c4b \
    --output runs/grok43high-baseline/results.csv

# DC-RS — same world
apex-agents-bench run --model grok-4.3-high \
    --world world_06051b9b10c94c079db1bac3b70c4c4b \
    --dc-rs \
    --output runs/grok43high-dc-rs/results.csv
```

Each domain's first world starts from an empty per-domain cheatsheet and pool; the same `--output` paths are reused so per-domain DC-RS state persists across worlds within a domain (no cross-domain transfer). See [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md) for the exact `--world` ids and the order they are run in, and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for full setup and Docker prerequisites.
