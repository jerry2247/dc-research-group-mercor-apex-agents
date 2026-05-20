# Implementation plan

> **Read this if you're asking:** "What's the phased plan? Where does
> Dynamic Ledger fit in? What's deferred?"

## Phase 0 — Repository bootstrap ✅ (this commit)

- Vendor Archipelago at pinned SHA `3f4a8234` with `UPSTREAM.md`,
  `LICENSE_UPSTREAM`, `PATCHES.md` provenance.
- Wrapper package `src/apex_agents_bench/` with the 14 modules listed
  in `ARCHITECTURE.md`.
- 11 test files covering imports, fidelity, dataset, agent profiles,
  judge, trajectory, catalog, task index, runner, docker_env (mocked),
  smoke selection, world materialization.
- Documentation: README + 9 docs (this set).
- Tooling: pyproject.toml (Python 3.13), Makefile (12 targets),
  pre-commit (9 hooks), setup.sh + docker_check.sh + fetch_dataset.sh
  + smoke_test.sh.
- Verification: `make check` is green on the bootstrap commit.

## Phase 1 — First end-to-end run

Goal: run one task end-to-end against both `gpt-5.5-medium` and
`grok-4.3-medium` and record numerical results in `runs/`.

Steps:

1. `make setup` on a clean machine.
2. Set keys in `.env` (`OPENAI_API_KEY`, `XAI_API_KEY`, `HF_TOKEN`).
3. `make docker-check` -- daemon up, env image buildable.
4. `make fetch-dataset` -- task + world index downloaded.
5. `apex-agents-bench smoke --model gpt-5.5-medium` -- one task E2E.
6. `apex-agents-bench smoke --model grok-4.3-medium` -- same task, different agent.
7. Open `runs/smoke/<profile>__<task_id>/trajectory.json` and inspect
   the tool-call shape. Sanity-check `grades.json`.

**Gate to Phase 2**: both smokes complete with `agent_status="completed"`
and a non-zero score on at least 1/N criteria. Wall time per smoke
< 60 minutes.

## Phase 2 — Per-domain pilot runs

Goal: produce a usable cross-method comparison table.

For each of the 7 agent profiles, run:

```bash
apex-agents-bench run --model <profile> --domain "Investment Banking"   --limit 10
apex-agents-bench run --model <profile> --domain "Management Consulting" --limit 10
apex-agents-bench run --model <profile> --domain Law                     --limit 10
```

That's 7 profiles × 30 tasks = 210 task runs. Estimated cost:
~$50-100 total at GPT-5.5 medium pricing (see `docs/COST.md`).
Estimated wall time: 210 × ~3 min / max-parallel-1 ≈ 11 hours (we run
sequentially by default).

**Gate to Phase 3**: per-domain mean scores are non-trivially different
across profiles (i.e. the benchmark is discriminative for the agent
families we're comparing).

## Phase 3 — Dynamic Ledger integration

Goal: register Dynamic Ledger as a custom agent against the vendor's
`AGENT_REGISTRY`, then re-run the Phase 2 cells with the DL-enabled
agent and compare.

Plan:

1. Read `vendor/archipelago/agents/runner/agents/registry.py` and the
   `react_toolbelt_agent` implementation. Identify the seams where DL
   inserts ledger reads/writes.
2. Create `src/apex_agents_bench/agents/dynamic_ledger_agent/` (in our
   wrapper, NOT in the vendor). Register it as a new
   `agent_config_id="dynamic_ledger_react_agent"` by extending the
   vendor's registry *non-destructively* -- ideally via a Python entry
   point or a side-channel module the runner can import.
3. If the vendor's registry is not extensible from outside, this is
   the FIRST time we'd justify a vendor patch. Document it in
   `PATCHES.md`, mark the registry entries with `# vendored-patch:`,
   add a regression test.
4. Add CLI flag `--agent-impl <id>` defaulting to `react_toolbelt_agent`.
5. Re-run Phase 2 with `--agent-impl dynamic_ledger_react_agent` for
   each profile.

**Open questions for Phase 3**:
- Where does the ledger live across tasks? (Per-world? Per-domain?
  Per-profile?) The published agent is stateless across tasks; DL is
  about *persistence across tasks*. Sensible default: ledger keyed by
  `(profile, domain)` -- 21 ledgers total -- so the agent can
  accumulate domain-specific knowledge but not leak cross-profile.
- Does DL read/write happen inside the agent loop (via a new MCP
  server we author) or outside (the runner reads/writes the ledger
  around the agent call)? Inside-loop is more faithful to DC's
  design; outside-loop is simpler and easier to ablate. We'll
  prototype both and pick whichever produces cleaner results.
- The DL contribution is the user's own (from
  "Dynamic Cheatsheet 2.0 Codebase"). Adapting Suzgun's TRACE / CRUD
  framework is out of scope for this repo; if needed, that lives in
  a separate fork.

**Gate to publication**: per-domain mean scores with DL show a
statistically meaningful lift over no-DL on at least one (profile,
domain) cell, with the lift visible across multiple seeds (we'll
re-introduce N>1 for the DL ablation specifically).

## Phase 4 — Bedrock plumbing (joint with apex-bench)

Goal: enable Claude profiles on Bedrock in both apex-bench (for v1
extended) and here (for agents), at the same time, on the same
inference-profile prefixes.

The apex-evals sister harness has no Bedrock plumbing on its own
`call_llm` path (apex-bench documents this). Archipelago's LiteLLM
call passes the model string through verbatim, so we *could* enable
Claude here today by setting up AWS credentials and using the
`bedrock/us.anthropic.claude-opus-4-6-v1:0` form. We hold off so the
two repos' active model surface stays in lockstep -- enabling Claude
here without enabling it in apex-bench would create a comparison
asymmetry we'd then have to caveat.

When Phase 4 lands:
1. apex-bench gets the necessary `apex-evals/src/call_llm` patch.
2. We register `claude-opus-4.6-*`, `claude-sonnet-4.6-*`,
   `claude-haiku-4.5-*` profiles here (sketched in
   `src/apex_agents_bench/agent_profile.py::_DEFERRED_CLAUDE_PROFILES_NOTE`).
3. `tests/test_fidelity.py::test_no_claude_profile_registered` flips to
   an explicit registration check.

## Out of scope (for now)

- **APEX-v1 (the older 200-task v1 benchmark)**: superseded by
  v1-extended (which apex-bench targets).
- **Custom MCP servers**: we use the published 9-server set verbatim.
- **Browser / web-search tool**: not in the published example;
  enabling it would change task semantics.
- **Multi-seed agent runs**: project policy is N=1; we'll reintroduce
  N>1 specifically for the DL ablation but not for the headline numbers.
