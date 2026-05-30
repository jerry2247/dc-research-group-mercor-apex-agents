# Implementation plan

> **Read this if you're asking:** "What's the phased plan? Where does
> DC-RS fit in? What's deferred?"

## Status (current)

The phases below are the original roadmap; this is where execution
actually stands:

- **Phase 0 (bootstrap)** and **Phase 1 (first end-to-end run)** — complete.
- **DC-RS and TRACE are both implemented** (the Phase 3 integration
  work below shipped; TRACE was added alongside it). Both are off by
  default and gated behind `--dc-rs` / `--trace`.
- **The active rollout follows the pre-registered per-(domain, world,
  method) order in [`EVALUATION_PLAN.md`](EVALUATION_PLAN.md) using the
  `grok-4.3-high` profile** — not the 7-profile × `--limit 10` pilot
  sketched in Phase 2. As of this writing, the baseline and DC-RS have
  completed **Investment Banking worlds 1–2** and TRACE has completed
  **world 1**; numbers are in [`../results.md`](../results.md).
- **Phase 4 (Bedrock / Claude profiles)** — still deferred.

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

## Phase 3 — DC-RS integration

Goal: wire DC-RS (Dynamic Cheatsheet — Retrieval Synthesis) into the
runner as two task-level hooks, then re-run the Phase 2 cells with
`--dc-rs` on and compare.

DC-RS is *not* a custom vendor agent — it leaves the published
`react_toolbelt_agent` untouched and instead brackets the agent call
with runner hooks, so no vendor patch is needed.

Plan:

1. Add `src/apex_agents_bench/dc_rs/` (in our wrapper, NOT in the
   vendor): per-domain pool (`bank.py`), replace-slot cheatsheet store
   (`store.py`), single-axis cosine retriever, trajectory renderer, the
   single synthesizer call, the `<cheatsheet>` extractor, the anti-wipe
   guard, and the injector.
2. Hook A (before the agent): embed the task prompt, retrieve the
   top-k=3 most-similar past `(task, trajectory)` pairs from the
   per-domain pool, run ONE synthesizer LLM call to produce the new
   cheatsheet (carrying the previous one forward), apply the anti-wipe
   guard, and prepend the cheatsheet block to the USER message of
   `initial_messages.json` (SYSTEM untouched).
3. Hook B (after the agent): render the agent's trajectory and append
   the `(task_prompt, rendered_trajectory, embedding)` triple to the
   per-domain pool. No second LLM call.
4. Add CLI flags `--dc-rs / --no-dc-rs` and `--dc-rs-top-k` (default 3);
   `--dc-rs` is mutually exclusive with `--trace`. With `--dc-rs` off
   the runner takes the baseline path and the CSV is byte-identical.
5. Re-run Phase 2 with `--dc-rs` for each profile.

**Design facts settled for Phase 3** (see `docs/DC_RS_PRD.md`):
- DC-RS state is keyed **per domain** (one pool + one cheatsheet slot
  per Investment Banking / Law / Management Consulting), persisted under
  `runs/<run>/dc_rs/<Domain>/`. Retrieval never crosses domains.
- The synthesizer runs **outside** the agent loop (the runner
  synthesizes before the agent and appends after it); this is simpler,
  cheaper (exactly one LLM call per task), and easy to ablate, while
  still faithful to Suzgun et al.'s reference.
- The synthesizer consumes **no** ground-truth signal — no rubric, no
  score, no correctness bit. That is the load-bearing fidelity
  invariant; it is what distinguishes DC-RS from TRACE.

**Gate to publication**: per-domain mean scores with DC-RS show a
statistically meaningful lift over baseline on at least one (profile,
domain) cell, with the lift visible across multiple seeds (we'll
re-introduce N>1 for the DC-RS ablation specifically).

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
  N>1 specifically for the DC-RS ablation but not for the headline numbers.
