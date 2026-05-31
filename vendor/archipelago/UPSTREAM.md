# Archipelago vendor provenance

This directory is a verbatim copy of the upstream Mercor Archipelago
repository, pinned to a specific commit. Treat the contents as third-party
source: read it, install it, but do not edit it casually.

## Upstream

| Field | Value |
|---|---|
| Repository | https://github.com/Mercor-Intelligence/archipelago |
| Pinned commit | `3f4a8234a27d71a17e0aa19e60f116440bcd1481` |
| Commit date | 2026-04-20 |
| Vendored on | 2026-05-19 |
| License | Apache-2.0 (see `LICENSE_UPSTREAM` and `LICENSE`) |

The pin is the head of `main` as of the vendoring date. We do not track
upstream `main` automatically; bumping the pin is a deliberate operation
that requires re-running `make check` against the new commit and updating
`PATCHES.md` with any required deltas.

## Diff policy

The vendored source MUST match upstream byte-for-byte at the pinned commit
except for changes recorded in `PATCHES.md`. The only active patch is a
build-time change to `environment/Dockerfile` (Patch 001), which compiles
`sandbox_fs.so` for the code-execution server; there are **zero patches to
the Archipelago Python source**. The Archipelago call path through LiteLLM
(`agents/runner/utils/llm.py` and `grading/runner/utils/llm.py`) passes the
model name string through verbatim, so we do not need any vendor edit to
use our chosen test or judge models.

If a future patch is unavoidable, each must:

1. Carry an inline comment marker exactly of the form `# vendored-patch: <reason>`.
2. Be recorded in `PATCHES.md` with the diff and the rationale.
3. Be covered by a regression test under `tests/test_fidelity.py` proving
   the marker is still present.

Pre-commit's `exclude: ^vendor/` rules guarantee that auto-formatting cannot
accidentally rewrite this directory.

## Resync recipe

```bash
# From repo root, with the venv active:
rm -rf vendor/archipelago.tmp
git clone --no-checkout https://github.com/Mercor-Intelligence/archipelago.git vendor/archipelago.tmp
cd vendor/archipelago.tmp
git -c advice.detachedHead=false checkout <new-sha>
cd ../..

# Diff first; if PATCHES.md still applies cleanly, swap the directory in:
diff -rq vendor/archipelago vendor/archipelago.tmp | grep -v '\.git$'

# Strip .git, replace, re-run checks:
rm -rf vendor/archipelago.tmp/.git
rm -rf vendor/archipelago
mv vendor/archipelago.tmp vendor/archipelago
make install check
```

## Component map (just the pieces our wrapper invokes)

| Path | Role |
|---|---|
| `agents/runner/main.py` | Agent entry point (invoked as `python -m runner.main ...`). |
| `agents/runner/agents/registry.py` | `AGENT_REGISTRY` -- the vendor's agent registry. DC-RS and TRACE operate via runner hooks and prompt injection, so they do not register a custom `agent_config_id`. |
| `agents/runner/agents/react_toolbelt_agent/main.py` | The published agent implementation we use by default. |
| `agents/runner/utils/llm.py` | LiteLLM call site -- passes `orchestrator_model` string verbatim. No model allow-list. |
| `grading/runner/main.py` | Grading entry point (invoked separately as `python -m runner.main ...`). |
| `grading/runner/utils/llm.py` | LiteLLM call site for the judge. Reads `grading_settings.llm_judge_model`. |
| `environment/Dockerfile`, `environment/docker-compose.yaml` | Headless environment container. |
| `mcp_servers/` | The 9 MCP servers (calendar, chat, code, spreadsheets, filesystem, mail, pdfs, presentations, documents). |
| `examples/hugging_face_task/` | The canonical reference runner we faithfully re-implement in `src/apex_agents_bench/runner.py`. |
