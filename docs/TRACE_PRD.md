# TRACE — apex-agents-bench

**TRACE** (Tool-augmented Reasoning via Atomic Cheatsheet Editing;
Liao, Nair, Yang, Stanford CS224N) is a test-time-learning subsystem
layered on the Mercor / Archipelago ReAct-toolbelt harness. Unlike
the Dynamic Ledger, TRACE *uses* the ground-truth correctness bit —
intentionally, per the paper.

We follow the paper's pipeline faithfully with these scoped
adaptations to the agentic, code-executing benchmark:

1. **OpenAI embeddings** (`text-embedding-3-large`) rather than the
   paper's task-specific embedding choice.
2. **No bullet length cap** — the paper caps atomic bullets at ~600
   characters; we let bullets grow to fit the tool-call workflows the
   benchmark surfaces.
3. **No SFT step** — the paper's optional supervised-fine-tuning stage
   is omitted.
4. **One TRACE framework** — we ship the GT-using reflector + curator
   pair the paper describes; no ablation variants are wired into the
   runtime.
5. **Same model for reflector + curator + agent** — the reflector and
   curator both run on the active `AgentProfile`'s `orchestrator_model`
   with the same `orchestrator_extra_args` (reasoning effort, etc.).
   Only the **judge** model is fixed (gpt-5.5 medium).

TRACE is **off by default**. With `--trace` off the runner takes the
baseline code path; CSV schema is byte-identical to the no-TRACE
shape. `--trace` and `--dynamic-ledger` are mutually exclusive.

## Pipeline

```
   ┌──────────┐    ┌────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
   │ RETRIEVE │───▶│ INJECT │───▶│ GENERATE │───▶│  CITE   │───▶│  REFLECT │───▶│  CURATE  │
   │ dual k=5 │    │ user   │    │ vendor   │    │ parse + │    │ same     │    │ same     │
   │ cosine   │    │ msg    │    │ agent    │    │ strip   │    │ model    │    │ model    │
   └──────────┘    └────────┘    └──────────┘    └─────────┘    └──────────┘    └─────┬────┘
        ▲                                              │             ▲                 │
        │                                              ▼             │                 │
        │                                          ┌────────┐   gt_bit                 │
        │                                          │ GRADE  │   (criteria_passed       │
        │                                          │ vendor │    == criteria_total)    │
        │                                          │ judge  │ ────────┘                │
        │                                          └────────┘                          │
        │                                                                              │
        └─────────────────────────  L_{i+1}  ◀─────────────────────────────────────────┘
```

Per task: **two** LLM calls into the TRACE pipeline (reflector, then
curator). Both calls receive the boolean `gt_correct`. Neither sees
the rubric text, per-criterion verdicts, expected answer, or judge
rationale.

## Bullet shape

```python
class Bullet(BaseModel):
    bullet_id: str                              # "bullet-N"
    section: str                               # short categorical label
    content: str                               # free-form, no length cap
    source_problem: str                        # curator's paraphrase; second retrieval key
    active: bool = True
    helpful: int = 0                           # cited on a case judged correct (gt_correct=True)
    harmful: int = 0                           # cited on a case judged incorrect (gt_correct=False)
    usage: int = 0                             # total cites
    created: int                               # 0-indexed per-domain ordinal at create-time
    updated: int                               # same, for the most recent edit
    content_embedding: list[float]             # text-embedding-3-large; 3072d
    source_problem_embedding: list[float]      # text-embedding-3-large; 3072d
```

Counters condition the reflector and curator's edit decisions — a
bullet with high `harmful` and low `helpful` is a deletion candidate;
a bullet with high `helpful` is preserved or sharpened.

## Hooks into `runner.run_single_task`

| Hook | When | Effect |
|------|------|--------|
| **A. retrieve + inject** | before agent subprocess writes `initial_messages.json` | dual top-k retrieval (k=5 per axis) on `(content_embedding, source_problem_embedding)`; render cheatsheet block + citation instruction; prepend to USER message; vendor SYSTEM prompt untouched. |
| **B. citations + shadow** | after agent subprocess returns | parse `<citations>[bullet-...]</citations>` from the last `final_answer.reasoning`; write `trajectory_graded.json` with the tag stripped from the reasoning; record cited bullet ids. |
| **C. counters + reflect + curate + apply + persist** | after grading completes | bump cited bullets' `usage` and `helpful`/`harmful` per `gt_correct`; call reflector with `(cheatsheet, problem, trajectory, cited_bullets, gt_correct)` → emits `<reflector_proposals>`; call curator with the above plus the reflector's proposals → emits `<cheatsheet_updates>`; apply ops in `DELETE → CONSOLIDATE → UPDATE → CREATE` order; persist per-domain snapshot. |

All three hooks are guarded by `if trace_runtime is not None`. With
TRACE off they do not run; the CSV schema is the baseline shape.

## Op contract

Both the reflector and curator emit a JSON array of operations,
identical schema:

| Op           | Args                                                  | Effect                                       |
|--------------|-------------------------------------------------------|----------------------------------------------|
| `CREATE`     | `section, content, source_problem`                    | New bullet; subject to create-time dedup     |
| `UPDATE`     | `bullet_id, content`                                  | Replace + re-embed; bump `updated`           |
| `DELETE`     | `bullet_id`                                           | Soft-delete                                  |
| `CONSOLIDATE`| `bullet_ids[], section, content, source_problem`      | Soft-delete sources; mint merged bullet with summed counters |
| `NO_OP`      | `reason` (optional)                                   | Explicit "nothing to do this turn"          |

The reflector wraps its output in `<reflector_proposals>...`; the
curator wraps its output in `<cheatsheet_updates>...`. Hallucinated
`bullet_id`s are dropped silently.

CONSOLIDATE preserves counters across sources: `helpful_new =
sum(helpful)`, `harmful_new = sum(harmful)`, `usage_new = sum(usage)`.

## Configuration

```python
@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    top_k_per_axis: int = 5

    # Filled in from the active AgentProfile by the runner
    reflector_model: str | None = None
    curator_model: str | None = None
    model_extra_args: dict | None = None

    reflector_temperature: float = 1.0
    curator_temperature: float = 1.0
    reflector_max_tokens: int = 16000
    curator_max_tokens: int = 16000
    reflector_timeout_seconds: int = 1800
    curator_timeout_seconds: int = 1800

    create_time_similarity_threshold: float = 0.85
    trajectory_max_chars_per_tool_result: int = 4000
```

CLI flags: `--trace / --no-trace`, `--trace-top-k`. Mutually exclusive
with `--dynamic-ledger`.

## Per-domain isolation

Each domain has its own TRACE ledger and snapshot history under
`runs/<run>/trace/<Domain>/snapshot_NNNN.json`. Retrieval at task `i`
in domain D sees the active bullets of D's ledger only; never
cross-domain.

## CSV columns added when TRACE is on

```
trace_enabled
trace_snapshot_index_before
retrieved_bullet_count
retrieved_bullet_ids                  JSON list
citations_present                     bool
citations_count                       int
citations_malformed_count             int
gt_correct_bit                        bool  (criteria_passed == criteria_total)
reflector_proposal_count              int
curator_create_count                  (committed)
curator_create_blocked_count          (rejected by create-time dedup)
curator_update_count
curator_delete_count
curator_consolidate_count
curator_no_op                         bool
trace_active_bullet_count_after
trace_total_bullet_count_after
trace_total_active_chars_after
reflector_prompt_tokens
reflector_completion_tokens
reflector_wall_seconds
curator_prompt_tokens
curator_completion_tokens
curator_wall_seconds
trajectory_chars_seen_by_curator
```

## Tests

```
tests/test_trace_bullet.py                 Bullet + TraceLedger + counters
tests/test_trace_curator_reflector.py      parsers + apply_ops (incl. CONSOLIDATE counter sum)
tests/test_trace_injector_citations.py     render + augment + citations extract / strip
tests/test_trace_fidelity.py               load-bearing signature, CSV, prompt invariants
```

The fidelity tests cover:
- the reflector signature includes `gt_correct` (intentional);
- the curator signature includes `gt_correct` AND `reflector_proposals`;
- the baseline CSV is unchanged when TRACE is off;
- the TRACE-on CSV preserves the baseline as a prefix;
- the reflector / curator system prompts reference the GT bit;
- the injection block specifies the citation format;
- the curator user template threads every required input.

## Run it

```bash
apex-agents-bench run \
    --model grok-4.3-high \
    --task-ids task_XXX \
    --trace \
    --output runs/trace-on/results.csv
```

The first task in a domain starts from an empty cheatsheet. Subsequent
tasks see the reflector + curator's accumulated edits.

## Citation

This subsystem implements the **TRACE** method (Tool-augmented
Reasoning via Atomic Cheatsheet Editing) by Liao, Nair, and Yang,
published as a Stanford CS224N final project. We follow the paper's
reflector + curator pipeline with the adaptations enumerated above
(OpenAI embeddings, no bullet length cap, no SFT step, same model for
the reflector and curator and agent, GT bit = boolean
`criteria_passed == criteria_total`).

```bibtex
@misc{liao_nair_yang_2026_trace,
  title  = {TRACE: Tool-augmented Reasoning via Atomic Cheatsheet Editing},
  author = {Liao, Kyleen and Nair, Roshen and Yang, Arnold},
  year   = {2026}
}
```

If a canonical citation entry becomes available (e.g., arXiv or
workshop submission), this block should be updated to match.

