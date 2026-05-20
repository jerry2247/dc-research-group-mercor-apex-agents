# Harness notes

> **Read this if you're asking:** "How does Archipelago actually work
> under the hood? What's the ReAct loop? Where do tool calls go? What
> is ReSum?"

This document collects the bits of Archipelago internals that affect
our wrapper. It assumes you've read `ARCHITECTURE.md`.

## The three Archipelago processes

A single task spans three Docker / Python boundaries:

1. **Environment container** (Docker, `vendor/archipelago/environment/`).
   FastAPI service on port 8080. Hosts the MCP gateway and the 9 MCP
   servers as subprocesses. The agent's tool calls land here.

2. **Agent runner** (host-side Python via `uv run`, in
   `vendor/archipelago/agents/`). Runs the ReAct loop. Talks to the
   environment over HTTP MCP. Calls the orchestrator LLM via LiteLLM.

3. **Grading runner** (host-side Python via `uv run`, in
   `vendor/archipelago/grading/`). Loads the trajectory + before /
   after snapshots. Per criterion: extracts the relevant artifact,
   calls the judge LLM, records a `VerifierResult`. Aggregates via
   the scoring method to a `final_score`.

Our wrapper's `runner.py` orchestrates all three.

## The ReAct loop (`react_toolbelt_agent`)

`vendor/archipelago/agents/runner/agents/react_toolbelt_agent/main.py`
implements a "ReAct + Toolbelt" agent:

- The agent has a **toolbelt** -- a working set of tools selected from
  the gateway's full inventory. Starts empty.
- Each step, the agent sees: system prompt + conversation history + a
  small set of *meta-tools* (`todo_write`, `toolbelt_list_tools`,
  `toolbelt_inspect_tool`, `toolbelt_add_tool`, `toolbelt_remove_tool`)
  + whatever's currently in the toolbelt + `final_answer`.
- The agent must discover tools via `toolbelt_list_tools`, add them
  via `toolbelt_add_tool`, and only then use them. This forces an
  exploration phase and prevents the LLM from being overwhelmed by 9
  servers' worth of tools at once.
- The loop terminates when the agent calls `final_answer` *and* all
  declared todos are complete/cancelled. If it hits `max_steps=50`
  first, the run ends with `agent_status="failed"`.

## ReSum context manager

`vendor/archipelago/agents/runner/agents/react_toolbelt_agent/resum.py`

When the conversation grows past 70% of the model's context window,
ReSum summarizes the oldest messages into a compact "story so far"
turn, and continues. This lets agents that need long exploration
phases (e.g. spreadsheet-heavy Investment Banking tasks) keep going past the
naive context cap.

We do not configure ReSum -- it runs with the agent's defaults. The
70% threshold is hardcoded in the upstream and we inherit it.

## Tool execution

When the agent calls an MCP tool, the agent runner forwards the call
to the MCP gateway over HTTP. The gateway routes it to the right
server (e.g. `code_execution_server` for Python eval, `sheets_server`
for spreadsheet edits). The server runs **inside the environment
container** with read/write access to `/filesystem` and `/.apps_data`.

**Code execution caveat**: `code_execution_server` runs Python code
inside the env container. Not a fully-sandboxed runtime -- it's a
plain Python process with the same filesystem mounts the agent has.
This matches Mercor's published behavior; we do not add additional
isolation.

## Snapshot diffing (grading side)

After the agent terminates, our runner streams the env's filesystem
state out via `POST /data/snapshot` (a tar.gz). The grading runner
takes both the *initial* snapshot (the world zip we populated) and
the *final* snapshot, computes a diff, and selects the changed /
created artifacts as inputs to each verifier.

The grading runner then, for each verifier:

1. Loads the rubric criterion text (verbatim from `verifiers.json`).
2. Selects relevant artifacts via the diff and optional file-type
   extractors (Reducto for visual artifacts; pure-Python for text).
3. Calls the judge LLM with: criterion + artifacts → `{"grade":
   "pass" | "fail", "rationale": "..."}`.
4. Maps `pass` → score 1.0, `fail` → score 0.0 (the `output_llm`
   eval defn).

The scoring method (`template`) then averages verifier scores into a
single `final_score` in [0.0, 1.0]. There are richer scoring methods
in the vendor (e.g. `task_score_unweighted_and_universal_penalty`)
but the published example uses `template`; we match it.

## CWD trick (uv run)

Both `agents/runner` and `grading/runner` ship a top-level Python
package called `runner`. We can't pip-install both editable into the
same venv -- they'd collide on the module name.

Mercor's published workaround: each subdir has its own uv-managed
venv (uv.lock + .venv/). When invoking a runner, `cd` into that subdir
so its uv venv (and its `runner/` module) are on sys.path. We do the
same:

```python
subprocess.run(["uv", "run", "python", "-m", "runner.main", ...],
               cwd=archipelago_agents_dir())     # for agents
subprocess.run(["uv", "run", "python", "-m", "runner.main", ...],
               cwd=archipelago_grading_dir())    # for grading
```

This is the *only* place we run subprocesses with non-host-default
working directories.

## MCP gateway hot-swap

`POST /apps` reconfigures the MCP gateway. We post the published
`mcp_config_all_oss_servers.json` once per container. The vendor
supports hot-swapping mid-task (e.g. add a new server while the agent
is running), but we don't use that feature.

## Health probe quirks

The vendor's `/health` endpoint takes 5-30 seconds to come up after
`docker compose up -d`. We poll with a 1-second cadence and a 180-
second ceiling. The poll loop tolerates connection refused (container
still starting), HTTP non-200 (container started but FastAPI not ready
yet), and httpx errors generally.

## Cost telemetry

`vendor/archipelago/agents/runner/utils/usage.py::UsageTracker` records
per-call token counts on the agent side. The grading runner records
its own token counts on the verifier side. Both flow into trajectory
/ grades JSON respectively.

We roll up **agent-side** usage into the CSV because it is directly tied
to the model being evaluated. The CSV fields are copied from
`trajectory.json.usage`: prompt, completion, total, final-step
completion, availability, source, and a total-consistency check.

We do NOT roll judge usage into the CSV. Judge token counts remain in
`grades.json`, where the vendor writes verifier-level details. This keeps
the result table focused on useful model-output information and avoids
mixing shared grading overhead into model comparisons.

## Known gotchas

- **Port 8080 collision**: another local service on 8080 will make
  `docker compose up` fail with a port-bind error. Set
  `APEX_AGENTS_HOST_PORT=8090` (or another free port) in `.env`. The
  runner respects it.
- **World snapshot leak**: if a task gets stuck and the container
  isn't torn down, the next task sees the previous one's filesystem.
  Our runner registers an `atexit` and a SIGINT handler that calls
  `docker compose down -v` on every active container. If the Python
  process is SIGKILLed, manual cleanup: `docker ps -aq | xargs docker
  rm -f`.
- **Python 3.13 strict**: the vendor pins `>=3.13,<3.14`. On macOS,
  `brew install python@3.13` or `pyenv install 3.13`.
- **uv must be installed**: `pip install uv` or
  `curl -LsSf https://astral.sh/uv/install.sh | sh`. Our `setup.sh`
  refuses to proceed without it.
- **HuggingFace dataset is gated**: even with `HF_TOKEN`, the first
  download fails until you visit the dataset page and accept the
  eval-only terms.
- **ReSum can fail at boundaries**: if a single tool result is larger
  than the model's context, ReSum can't help; the loop just retries
  and eventually returns. We surface this via `agent_status="failed"`.

## Why we don't use the vendor's `agents/runner/save/main.py`

It writes to a webhook + S3, which is for Mercor's leaderboard
infrastructure. Local runs don't need it. We write trajectories and
grades to the local filesystem instead, matching the published
example's `--output` flag pattern.
