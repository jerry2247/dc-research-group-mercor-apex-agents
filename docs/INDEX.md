# Documentation index

This is a map of the project documentation. Skim the column on the right
to find the doc that answers your question.

| Doc | Length | Read if you want to answer… |
|---|---|---|
| [`README.md`](../README.md) | long | "how do I run the benchmark?" |
| [`EVALUATION_PLAN.md`](EVALUATION_PLAN.md) | medium | "which world should I run next? what's the rollout order?" |
| [`../results.md`](../results.md) | medium | "what numbers do we have so far?" |
| [`DC_RS_PRD.md`](DC_RS_PRD.md) | long | "how does `--dc-rs` actually work?" |
| [`TRACE_PRD.md`](TRACE_PRD.md) | long | "how does `--trace` actually work?" |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | medium | "why is this in src/ and that in vendor/?" |
| [`AUDIT.md`](AUDIT.md) | long | "are we really running Mercor's harness?" |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | medium | "why one run per task? why gpt-5.5?" |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | medium | "what's the phased roadmap?" |
| [`HARNESS_NOTES.md`](HARNESS_NOTES.md) | medium | "how does Archipelago actually work under the hood?" |
| [`BENCHMARK_STRUCTURE.md`](BENCHMARK_STRUCTURE.md) | medium | "what's actually in the 480 tasks?" |
| [`COST.md`](COST.md) | short | "what will this cost?" |
| [`DOCKER.md`](DOCKER.md) | medium | "Docker is acting up" |
| [`vendor/archipelago/UPSTREAM.md`](../vendor/archipelago/UPSTREAM.md) | short | "what commit did we vendor? how to resync?" |
| [`vendor/archipelago/PATCHES.md`](../vendor/archipelago/PATCHES.md) | short | "what did we change in the harness?" |

If you're a first-time reader, read in this order:

1. `README.md` -- five-minute overview and the running-it-in-one-command shape.
2. `ARCHITECTURE.md` -- where things live and why.
3. `AUDIT.md` -- evidence we match Mercor.
4. `HARNESS_NOTES.md` -- internals.
5. `IMPLEMENTATION_PLAN.md` -- where we're going next.
