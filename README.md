# apex-agents-bench

Evaluation harness for **Dynamic Ledger** and **TRACE** — two test-time-learning subsystems — on Mercor's **APEX-Agents** benchmark (multi-turn agent on an MCP toolbelt: filesystem, sheets, code execution, mail, slides, docs, pdf, calendar, chat).

## Project context

This repository sits inside a collaboration between two student teams at Stanford CS 224N, jointly mentored by Mirac Suzgun (Stanford SAIL NLP):

- **Dynamic Ledger** (*Retrieval-Augmented Structured Memory for Test-Time Learning*) — a no-ground-truth memory mechanism, developed by Jerry Gu, Sabrina Yen-Ko, and Shurui Liu.
- **TRACE** (*Tool-augmented Reasoning via Atomic Cheatsheet Editing*) — a memory mechanism that uses a per-task correctness bit, developed by Kyleen Liao, Roshen Nair, and Arnold Yang.

The two mechanisms are evaluated head-to-head on Mercor's APEX-Agents benchmark in this repository, and on the single-shot prose surface (Mercor's APEX-v1-extended) in the [`apex-bench`](https://github.com/jerry2247/dc-research-group-mercor-apex) sister repository. Both repositories share the same subsystem implementations and audit policy.

## What this repository is

A thin policy and orchestration layer over Mercor's Archipelago harness (`vendor/archipelago/`, vendored at commit `3f4a8234`). The vendored harness — system prompt, MCP server set, `react_toolbelt_agent` config, per-task fresh container, verifier construction, scoring config — is preserved exactly; this repository contributes:

- **Two memory subsystems** that can be toggled via CLI flags; both are off by default. With either flag off the pipeline is byte-identical to the no-memory baseline (pinned by a fidelity test).
- **Project policies** — judge model fixed to `gpt-5.5` (medium reasoning effort, Mercor's published judge); one run per (task, agent profile); a typed agent profile registry (`gpt-5.5-{low,medium,high,xhigh}`, `grok-4.3-{low,medium,high}`).
- **Two documented vendor patches** (`docs/HARNESS_NOTES.md`) that make the published Archipelago example runnable end-to-end.

| Subsystem | CLI flag | Ground-truth signal | Spec |
|---|---|---|---|
| **Dynamic Ledger** | `--dynamic-ledger` | None | [`docs/DYNAMIC_LEDGER_PRD.md`](docs/DYNAMIC_LEDGER_PRD.md) |
| **TRACE** | `--trace` | boolean per-task correctness bit | [`docs/TRACE_PRD.md`](docs/TRACE_PRD.md) |

The benchmark dataset is Mercor's and is **not** redistributed; it is fetched at setup time from `mercor/APEX-Agents` on Hugging Face. Docker Desktop is required for the per-task MCP toolbelt containers; see [`docs/DOCKER.md`](docs/DOCKER.md).

## Results

On the only configuration that has been run end-to-end at the time of writing — `grok-4.3-high`, gpt-5.5 judge, Investment Banking world 219 (n = 11 graded; 3 tasks excluded for agent-runtime failures in both runs):

| Method | Pass@1 | Mean `final_score` |
|---|:---:|:---:|
| Baseline (no memory) | 0 / 11 | 0.000 |
| + Dynamic Ledger | **1 / 11** | **0.190** |

Full per-domain × per-method table (with placeholders for Law, Management Consulting, and the TRACE row across all three domains), per-task breakdown, and a worked example of a Dynamic Ledger entry that converted a 0 → 1.00 score on a downstream task: [`results.md`](results.md). Pre-registered rollout order for the remaining (domain, world, method) cells: [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md).

## Reproduce

```bash
git clone https://github.com/jerry2247/dc-research-group-mercor-apex-agents.git apex-agents-bench
cd apex-agents-bench && pip install -e . && cp .env.example .env  # add API keys
# Docker Desktop must be running.

# Baseline (no memory)
apex-agents-bench run --model grok-4.3-high --world 219 \
    --output runs/ib-world219-grok43high/results.csv

# Dynamic Ledger
apex-agents-bench run --model grok-4.3-high --world 219 --dynamic-ledger \
    --output runs/ib-world219-grok43high-dl/results.csv
```

Full setup (venv, API keys, Docker pre-warm, dataset hashes) and the documented divergences from Mercor's published harness: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), [`docs/AUDIT.md`](docs/AUDIT.md), [`docs/DOCKER.md`](docs/DOCKER.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/apex_agents_bench/` | The harness — policy, runner, audit, CSV schema |
| `src/apex_agents_bench/dynamic_ledger/` | Dynamic Ledger subsystem: config, retriever, curator, injector, prompts, trajectory rendering |
| `src/apex_agents_bench/trace/` | TRACE subsystem: reflector, curator, citations parser, prompts |
| `vendor/archipelago/` | Mercor's agentic harness, vendored at `3f4a8234`, with two documented patches |
| `data/apex-agents/` | Mercor's benchmark dataset, fetched from `mercor/APEX-Agents` |
| `runs/ib-world219-grok43high/` | Baseline run, grok-4.3-high, Investment Banking world 219 |
| `runs/ib-world219-grok43high-dl/` | Dynamic Ledger run, same profile, same subset |
| `docs/` | Architecture, PRDs, reproducibility, audit, Docker. Index: [`docs/INDEX.md`](docs/INDEX.md) |

Behavioral fidelity to Mercor's published evaluation surface is enforced by pytest assertions and a code-level audit; see [`docs/AUDIT.md`](docs/AUDIT.md).

## License

Code: see [`LICENSE`](LICENSE). Mercor's vendored harness and benchmark dataset are governed by their respective upstream licenses.

## Citation

```bibtex
@misc{gu_yenko_liu_2026_dynamic_ledger,
  title  = {Dynamic Ledger: Retrieval-Augmented Structured Memory for
            Test-Time Learning},
  author = {Gu, Jerry and Yen-Ko, Sabrina and Liu, Shurui},
  note   = {Mentor: Mirac Suzgun},
  year   = {2026}
}

@misc{liao_nair_yang_2026_trace,
  title  = {TRACE: Tool-augmented Reasoning via Atomic Cheatsheet Editing},
  author = {Liao, Kyleen and Nair, Roshen and Yang, Arnold},
  year   = {2026}
}

@misc{suzgun_yuksekgonul_bianchi_jurafsky_zou_2025_dynamic_cheatsheet,
  title  = {Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory},
  author = {Suzgun, Mirac and Yuksekgonul, Mert and Bianchi, Federico and
            Jurafsky, Dan and Zou, James},
  year   = {2025},
  eprint = {2504.07952},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2504.07952}
}

@misc{suzgun_kalai_2024_meta_prompting,
  title  = {Meta-Prompting: Enhancing Language Models with Task-Agnostic
            Scaffolding},
  author = {Suzgun, Mirac and Kalai, Adam Tauman},
  year   = {2024},
  eprint = {2401.12954},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2401.12954}
}
```
