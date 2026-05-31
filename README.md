# apex-agents-bench

Evaluation harness for research on **test-time learning in tool-using agents**, extending Mirac Suzgun et al.'s *Dynamic Cheatsheet* (2025) from the single-shot prose setting to multi-step agentic work on Mercor's **APEX-Agents** benchmark (a multi-turn agent on an MCP toolbelt: filesystem, sheets, code execution, mail, slides, docs, pdf, calendar, chat).

## Project context

This repository studies whether a language-model agent can learn at test time, accumulating a self-curated memory from its own past attempts and reusing it on later tasks. We extend **Dynamic Cheatsheet** (Suzgun et al., 2025) and its retrieval-synthesis variant **DC-RS** from the single-shot prose setting to the agentic one, evaluating it against a no-memory baseline on Mercor's APEX-Agents benchmark.

Work by Jerry Gu, Kyleen Liao, Shurui Liu, Roshen Nair, Arnold Yang, and Sabrina Yen-Ko, mentored by Mirac Suzgun (Stanford SAIL NLP).

## What this repository is

A thin policy and orchestration layer over Mercor's Archipelago harness (`vendor/archipelago/`, vendored at commit `3f4a8234`). The vendored harness — system prompt, MCP server set, `react_toolbelt_agent` config, per-task fresh container, verifier construction, scoring config — is preserved exactly; this repository contributes:

- **A test-time memory subsystem** toggled via a CLI flag and off by default. With the flag off the pipeline is byte-identical to the no-memory baseline (pinned by a fidelity test).
- **Project policies** — judge model fixed to `gpt-5.5` (medium reasoning effort, Mercor's published judge); one run per (task, agent profile); a typed agent profile registry (`gpt-5.5-{low,medium,high,xhigh}`, `grok-4.3-{low,medium,high}`, `deepseek-v4-pro-max`).
- **One documented vendor patch** (`vendor/archipelago/PATCHES.md`) — a `Dockerfile` build step that compiles the code-execution sandbox library (`sandbox_fs.so`) — needed to make the published Archipelago example runnable end-to-end.

| Subsystem | CLI flag | Ground-truth signal | Spec |
|---|---|---|---|
| **DC-RS** | `--dc-rs` | None | [`docs/DC_RS_PRD.md`](docs/DC_RS_PRD.md) |
| **TRACE** | `--trace` | boolean per-task correctness bit | [`docs/TRACE_PRD.md`](docs/TRACE_PRD.md) |

The benchmark dataset is Mercor's and is **not** redistributed; it is fetched at setup time from `mercor/APEX-Agents` on Hugging Face. Docker Desktop is required for the per-task MCP toolbelt containers; see [`docs/DOCKER.md`](docs/DOCKER.md).

## Results

So far the rollout has completed, for the baseline and DC-RS, the **first world of all three domains** plus a second Investment Banking world (`grok-4.3-high` agent, gpt-5.5 judge). A non-finalizing agent loop writes no CSV row, so per-method completion counts differ; the figures below are means over the **full world** with each non-completion scored 0, which keeps the comparison same-denominator:

| Domain | worlds | tasks | Baseline | + DC-RS | DC-RS W / T / L |
|---|:---:|:---:|:---:|:---:|:---:|
| Investment Banking | 1–2 | 23 | 0.043 | 0.166 | 5 / 18 / 0 |
| Management Consulting | 1 | 14 | 0.183 | 0.335 | 4 / 8 / 2 |
| Law | 1 | 20 | 0.336 | 0.494 | 7 / 11 / 2 |
| **All three domains** | **4** | **57** | **0.180** | **0.323** | **16 / 37 / 4** |

DC-RS improves the mean in every domain (overall 0.323 vs 0.180, ≈ +80%) and wins 16 paired tasks against 4 losses. The four losses are confined to the document/reasoning-heavy domains (MC, Law) where the no-memory baseline is already strong; on Investment Banking, where the baseline frequently fails to finalize, DC-RS never loses a paired task. These are **full-world** means (non-completion scored 0); `results.md` breaks each domain out by world and also reports means over only the completed tasks (a different, non-same-denominator basis — e.g. DC-RS IB world 1 is 0.202 over its 13 completed). TRACE consumes the per-task correctness bit, so it is not a like-for-like comparison and is reported separately. Full per-domain × per-world tables, the per-task breakdown, and the completed-task means: [`results.md`](results.md). Pre-registered rollout order for the remaining cells: [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md).

## Reproduce

```bash
git clone https://github.com/jerry2247/dc-research-group-mercor-apex-agents.git apex-agents-bench
cd apex-agents-bench && pip install -e . && cp .env.example .env  # add API keys
# Docker Desktop must be running.

# Baseline (no memory) — Investment Banking world 1 (dataset name: World 219)
apex-agents-bench run --model grok-4.3-high \
    --world world_1e4d4288e63f4a08851a3cc441eb3ccb \
    --output runs/grok43high-baseline/results.csv

# DC-RS — same world
apex-agents-bench run --model grok-4.3-high \
    --world world_1e4d4288e63f4a08851a3cc441eb3ccb --dc-rs \
    --output runs/grok43high-dc-rs/results.csv
```

Full setup (venv, API keys, Docker pre-warm, dataset hashes) and the documented divergences from Mercor's published harness: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), [`docs/AUDIT.md`](docs/AUDIT.md), [`docs/DOCKER.md`](docs/DOCKER.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/apex_agents_bench/` | The harness — policy, runner, audit, CSV schema |
| `src/apex_agents_bench/dc_rs/` | DC-RS subsystem: config, retriever, synthesizer, injector, prompts, trajectory rendering, per-domain pool + cheatsheet store |
| `src/apex_agents_bench/trace/` | TRACE subsystem: reflector, curator, citations parser, prompts |
| `vendor/archipelago/` | Mercor's agentic harness, vendored at `3f4a8234`, with one documented patch |
| `data/apex-agents/` | Mercor's benchmark dataset, fetched from `mercor/APEX-Agents` |
| `runs/grok43high-baseline/` | Baseline runs, grok-4.3-high (IB worlds 1–2, MC world 1, Law world 1) |
| `runs/grok43high-dc-rs/` | DC-RS runs (+ per-domain pool & cheatsheet), grok-4.3-high (same slice) |
| `runs/grok43high-trace/` | TRACE runs (+ per-domain ledger), grok-4.3-high, IB world 1 |
| `docs/` | Architecture, PRDs, reproducibility, audit, Docker. Index: [`docs/INDEX.md`](docs/INDEX.md) |

Behavioral fidelity to Mercor's published evaluation surface is enforced by pytest assertions and a code-level audit; see [`docs/AUDIT.md`](docs/AUDIT.md).

## License

Code: see [`LICENSE`](LICENSE). Mercor's vendored harness and benchmark dataset are governed by their respective upstream licenses.

## Citation

```bibtex
@misc{gu_liu_yenko_2026_dynamic_ledger,
  title  = {Dynamic Ledger: Retrieval-Augmented Structured Memory for
            Test-Time Learning},
  author = {Gu, Jerry and Liu, Shurui and Yen-Ko, Sabrina},
  year   = {2026},
  url    = {https://drive.google.com/file/d/1I6vW9WQnt6tM2iVQPQpEAxS68Z0-IhGK/view}
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
