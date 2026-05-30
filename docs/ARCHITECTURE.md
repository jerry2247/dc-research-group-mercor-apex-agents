# Architecture

> **Read this if you're asking:** "Why is X in `src/apex_agents_bench/` and
> Y in `vendor/archipelago/`? Why is Docker required? Where does our
> wrapper end and the vendor begin?"

## The wrapper-vs-vendor split

```
apex-agents-bench/
├── src/apex_agents_bench/        <- harness (policy, runner, audit, CSV schema)
│   ├── cli.py                       Typer CLI
│   ├── config.py                    Policy constants + Settings
│   ├── paths.py                     Repo + vendor path resolution
│   ├── dataset.py                   tasks_and_rubrics.json + worlds loader
│   ├── world.py                     HF fetch + cache + materialize
│   ├── docker_env.py                Environment container lifecycle
│   ├── agent_profile.py             8 agent profiles (gpt-5.5-*, grok-4.3-*, deepseek-v4-pro-max)
│   ├── trajectory.py                Parse vendor's trajectory + grades JSON
│   ├── judge.py                     Build grading_settings + verifiers + scoring
│   ├── smoke.py                     Single-task E2E
│   ├── runner.py                    Multi-task driver with CSV resume
│   ├── catalog.py                   Dataset characterization
│   ├── task_index.py                Browseable task index
│   ├── dc_rs/                       DC-RS subsystem: pool, retriever, synthesizer, injector, prompts
│   └── trace/                       TRACE subsystem: reflector, curator, citations parser, prompts
│
└── vendor/archipelago/           <- VENDOR. Pristine Mercor source.
    ├── agents/                      Agent runner (react_toolbelt_agent, ...)
    ├── grading/                     Grading runner + scoring methods
    ├── environment/                 Docker container + MCP gateway
    ├── mcp_servers/                 9 MCP servers (calendar/code/files/...)
    ├── examples/hugging_face_task/  Mercor's reference runner -- the
    │                                template our runner.py replicates
    ├── UPSTREAM.md                  Pinned commit + diff policy
    ├── PATCHES.md                   Patch log (one active: Dockerfile sandbox_fs.so build)
    └── LICENSE_UPSTREAM             Apache-2.0 verbatim
```

**Rule of thumb:** Anything that decides *what* to run lives in `src/`.
Anything that *executes* the agent or grades it lives in `vendor/`.

## Why vendor at all (and pin a commit)?

1. **Reproducibility.** A pinned commit means "the harness behavior on
   day N is the same as day 0." No surprise upstream changes can
   silently shift our numbers.
2. **Auditability.** A single `diff -rq` between our `vendor/archipelago/`
   and a fresh upstream clone tells you exactly what we changed: the
   three added provenance files and one build patch (`environment/Dockerfile`
   compiles `sandbox_fs.so`; Patch 001 in `PATCHES.md`).
3. **Modification headroom.** When future patches become unavoidable,
   the diff policy in `vendor/archipelago/UPSTREAM.md` requires a
   marker comment + an entry in `PATCHES.md` + a regression test. This
   prevents silent drift.

## Why Docker is required

Archipelago's environment is **a real headless container** with a
Python 3.13 + Node + Chromium + LibreOffice toolchain. The 9 MCP
servers run inside it as subprocesses talking to the FastMCP gateway.
The agent runs on the host and talks to the gateway over HTTP. There
is no in-process simulation -- we use the same artifact Mercor uses.

The cost of this fidelity:

- Setup needs Docker Desktop or Docker Engine.
- A fresh container per task adds ~5-10 s of overhead.
- Disk: ~2 GB for the env image, plus world snapshots (~18.7 GB if
  you fetch them all; the runner fetches one at a time and caches).

The benefit: zero divergence from Mercor's published numbers. The agent
runs in exactly the environment Mercor validated against.

## The per-task lifecycle

`runner.run_single_task` is a near-mechanical port of Mercor's
`examples/hugging_face_task/main.py`. Step by step:

```
                         ┌───────────────────────────┐
                         │   docker compose down -v  │  fresh start
                         └──────────────┬────────────┘
                                        │
                         ┌──────────────▼────────────┐
                         │   docker compose up -d    │  ~10 s
                         │   wait /health = 200      │
                         └──────────────┬────────────┘
                                        │
   HF cache  ──────► fetch world.zip ──►│ POST /data/populate (filesystem + .apps_data)
                     fetch task_files ─►│ POST /data/populate (overlay)
                                        │
                                        │  POST /apps  (9 MCP servers)
                                        │
                  ┌─────────────────────▼────────────────────┐
                  │  uv run python -m runner.main  (agents)  │
                  │  --orchestrator-model openai/gpt-5.5     │
                  │  --orchestrator-extra-args ...           │
                  │  --agent-config {max_steps: 50, ...}     │
                  │  --output trajectory.json                │
                  └─────────────────────┬────────────────────┘
                                        │
                                        │  POST /data/snapshot  ► final_snapshot.tar.gz
                                        │  tar.gz → zip
                                        │
                  ┌─────────────────────▼────────────────────┐
                  │  uv run python -m runner.main  (grading) │
                  │  --grading-settings  {judge: gpt-5.5}    │
                  │  --verifiers         <from rubric>       │
                  │  --eval-configs      ec_output_llm       │
                  │  --scoring-config    template            │
                  │  --output grades.json                    │
                  └─────────────────────┬────────────────────┘
                                        │
                         ┌──────────────▼────────────┐
                         │   docker compose down -v  │  clean up
                         └──────────────┬────────────┘
                                        │
                  ┌─────────────────────▼────────────────────┐
                  │  append row to results.csv               │
                  │  task_id, domain, world_id, final_score, │
                  │  criteria_passed/total, agent_profile,   │
                  │  agent_model, judge_model, ...           │
                  └──────────────────────────────────────────┘
```

The two `uv run python -m runner.main` calls hit two different
runners: the agents package and the grading package, each in its own
uv-managed venv. We match the published example's working-directory
trick (`cwd=AGENTS_DIR` / `cwd=GRADING_DIR`) so each runner finds its
own `runner.main` module.

## Why two venvs (uv-managed vendor venvs + our pip wrapper venv)?

`vendor/archipelago/agents/` and `vendor/archipelago/grading/` both
ship a top-level Python package named `runner`. They cannot coexist in
a single venv as editable installs. Mercor's published path is `uv
sync` in each subdir, which gives each its own isolated venv. We do
the same thing, exactly. Our wrapper venv is pip-based, light, and
imports neither -- it only orchestrates them via `subprocess`.

## Settings as the single source of truth

A run is fully determined by a `Settings` instance plus a chosen
`AgentProfile` and CSV path. `Settings` is frozen; the runner reads it
once at the top and never consults `os.environ` for policy (only for
credentials read by LiteLLM / boto3 / HF). This makes two runs with
the same `Settings` and same dataset SHA comparable modulo provider
stochasticity.

## Diff policy (re-stated)

`vendor/archipelago/` is treated as third-party source. Any modification
requires:

1. An inline `# vendored-patch: <reason>` marker on every changed line.
2. An entry in `vendor/archipelago/PATCHES.md`.
3. A regression test under `tests/test_fidelity.py` asserting the
   marker still exists.

Pre-commit's `exclude: ^vendor/` rules block auto-formatting from
silently rewriting any vendor file.
