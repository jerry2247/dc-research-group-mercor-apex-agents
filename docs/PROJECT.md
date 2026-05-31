# Project: test-time learning methods on Mercor's APEX benchmarks

This document explains the broader research project and how this
repository fits into it. It is intentionally short — see the per-method
PRDs and the README for details.

## Goal

The project asks whether a language-model agent can learn at test time,
building a self-curated memory from its own past attempts and reusing it
on later tasks. The central method is **DC-RS**, an adaptation of
Suzgun et al.'s *Dynamic Cheatsheet* to the agentic setting; the harness
evaluates it against a no-memory baseline on Mercor's APEX benchmarks at
constant agent-side compute, and includes **TRACE** as a secondary
memory mechanism for comparison.

1. **Baseline** — vendor harness, no memory subsystem.
2. **DC-RS** — Dynamic Cheatsheet — Retrieval Synthesis: a
   no-ground-truth, single-synthesizer memory mechanism that maintains
   a per-domain pool of past trajectories plus a replace-slot
   cheatsheet, adapted from Suzgun et al., *Dynamic Cheatsheet:
   Test-Time Learning with Adaptive Memory* (2025, arXiv:2504.07952).
   See [`DC_RS_PRD.md`](DC_RS_PRD.md).
3. **TRACE** — a secondary comparison: a reflector + curator with
   GT-bit feedback over an atomic-bullet cheatsheet, *Tool-augmented
   Reasoning via Atomic Cheatsheet Editing*. See
   [`TRACE_PRD.md`](TRACE_PRD.md).

For both subsystems the synthesizer (DC-RS) or the reflector + curator
(TRACE) run on **the same model as the agent profile under test**. Only
the **judge** model is fixed (gpt-5.5 medium). Embeddings always use
OpenAI `text-embedding-3-large`.

## What's in this repo

- `src/apex_agents_bench/` — the baseline harness wrapper around
  Mercor's Archipelago, plus the `dc_rs/` and `trace/` subpackages.
- `docs/DC_RS_PRD.md` — the DC-RS specification.
- `docs/TRACE_PRD.md` — the TRACE specification.
- `tests/` — unit + fidelity tests; the baseline schema is byte-
  identical when neither subsystem is enabled.
- `vendor/archipelago/` — vendored Archipelago at the pinned commit
  (3f4a8234) with one build-time patch (the `environment/Dockerfile`
  `sandbox_fs.so` compile, Patch 001); see `vendor/archipelago/PATCHES.md`.

## CLI surface, at a glance

```
apex-agents-bench run --model <profile> --task-ids <id> [--output PATH]
                       [--dc-rs | --trace]
                       [--azure]
```

`--dc-rs` and `--trace` are mutually exclusive. `--azure` routes any
GPT-5.5 chat completion (judge + agent + DC-RS synthesizer + TRACE
reflector/curator) through Azure-OpenAI; embeddings always use OpenAI.

## Reading the results

The baseline CSV columns are pinned by the
`test_baseline_csv_has_no_dc_rs_columns` fidelity test. The two memory
subsystems each append a known set of columns when enabled.
The README's *Results* table reports, per domain, the mean rubric score
for the baseline and DC-RS and the per-task win/tie/loss of DC-RS against
the baseline.
