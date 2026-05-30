# Cost

> **Read this if you're asking:** "How much will a full run cost?
> What if I only do one domain?"

## Headline numbers

These are rough estimates -- model prices and per-task token usage
move month-over-month. Treat them as order-of-magnitude.

| Scope | Tasks | Approx cost | When to use |
|---|---|---|---|
| One smoke task | 1 | $0.10-$1 | Verifying setup; cheapest E2E sanity check. |
| One domain pilot (10 tasks per profile) | 10 | $1-$10 | First-look comparisons; tractable in an evening. |
| One full domain (160 tasks per profile) | 160 | $20-$150 | Headline per-domain number, 7-15 hours wall time. |
| Full benchmark (480 tasks per profile) | 480 | $60-$450 | The "official-style" run. ~1 day wall time per profile. |

Multiplied by **7 profiles** for an all-profiles sweep, the full
benchmark sits in the $500-$3,000 range per pass. Most research
iterations should stay in the per-domain pilot tier.

## What costs what

Per task, the cost is split across:

| Cost source | Typical share | Notes |
|---|---|---|
| Agent LLM calls (orchestrator) | ~80% | ReAct steps × prompt+completion tokens. The high-reasoning profiles spend more here. |
| Judge LLM calls | ~15% | 1 call per criterion. Avg 4 criteria per task → 4 judge calls. |
| Reducto extractions (if enabled) | <5% | Only fires when the agent produces chart-heavy artifacts (PDFs / spreadsheets with embedded images). Most tasks grade fine without it. |
| Disk + Docker | 0% | Local. |
| Network (HF downloads) | 0% | Once cached, free; first task per world fetches ~500 MB. |

The user said "we will not be using the full task, so the cost is not
fully relevant" -- the design here matches that: per-domain pilots
and individual world runs are the expected use case, and they sit
well under $50.

## Profile-by-profile expected cost per task

Very rough, based on per-profile prompt/completion mix at OpenAI's and
xAI's published prices as of 2026-05:

| Profile | Per task | Notes |
|---|---|---|
| `gpt-5.5-low` | $0.10-$0.30 | Cheap; less reasoning per step. |
| `gpt-5.5-medium` | $0.20-$0.50 | Default-ish balance. |
| `gpt-5.5-high` | $0.40-$1.00 | More reasoning tokens; same model. |
| `gpt-5.5-xhigh` | $0.80-$2.00 | Significantly more tokens; reaches the 50-step cap less often. |
| `grok-4.3-low` | $0.15-$0.40 | Cheap; xAI's tier. |
| `grok-4.3-medium` | $0.30-$0.70 | |
| `grok-4.3-high` | $0.60-$1.40 | |

Judge (`gpt-5.5` medium) adds ~$0.05-$0.20 per task on top, regardless
of agent profile.

## Resume saves money

A partial run is re-runnable. The runner reads `results.csv` and skips
any `status="completed"` row. If a 160-task run is interrupted at task
80, re-running the same command picks up at task 81 -- you don't pay
for the first 80 again. (Resume is "smart" only on a *successful*
task; failed tasks aren't recorded as completed, so they will be
retried.)

## What you'll see in the CSV

The CSV includes **agent-side token telemetry only**:
`agent_prompt_tokens`, `agent_completion_tokens`, `agent_total_tokens`,
`agent_final_step_completion_tokens`, plus availability/consistency
fields. These values come from Archipelago's `trajectory.json` top-level
`usage` block, which the vendor fills from LiteLLM `response.usage`.

Judge token usage is intentionally not rolled into the CSV. It remains in
the per-task `grades.json` artifact because it is shared evaluation
overhead, not a model-output metric for cross-method comparisons.

Cost columns are NOT in the CSV. LiteLLM pricing-map estimates are not
provider-billed charges, so exact billing analysis should be done after a
run from provider invoices or a separate explicit pricing script over the
saved agent token counts.

## Cost-saving tactics that are safe

- `--limit 10` for first-look pilots.
- `--domain "Investment Banking"` for the cheapest single-domain (IB tasks
  average ~1.36 expert-hours, shortest of the three).
- `gpt-5.5-low` or `grok-4.3-low` for cheap baselines.
- `--task-ids <id>,<id>` to re-run only specific failed tasks.

## Cost-saving tactics that change the benchmark (do not use silently)

- Lowering `--max-steps` below 50.
- Lowering `--timeout-seconds` below 3600.
- Subsetting MCP servers (the wrapper refuses this).
- Changing the system prompt (the wrapper refuses this -- the prompt
  is a literal copy of the published example).
