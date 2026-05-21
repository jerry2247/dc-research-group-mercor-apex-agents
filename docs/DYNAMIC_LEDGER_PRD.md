# Dynamic Ledger — apex-agents-bench

Test-time learning subsystem layered on the Mercor / Archipelago
ReAct-toolbelt harness. Implements the **Dynamic Ledger** approach
from the **Dynamic Cheatsheet 2.0** codebase (Jerry Gu, Shurui Liu,
Sabrina Yen-Ko; mentor: Mirac Suzgun) — adapted to a multi-turn
tool-using agent on a rubric-graded benchmark. The Ledger has **no
ground-truth signal** at any point — the curator never sees the
criterion text, never sees the judge's per-criterion verdict, never
sees the expected answer.

## Design at a glance

```
   ┌──────────┐     ┌────────┐     ┌──────────────┐     ┌───────────┐
   │ RETRIEVE │────▶│ INJECT │────▶│   GENERATE   │────▶│  CURATE   │
   │ dual k=5 │     │ user   │     │ vendor agent │     │  gpt-5.5  │
   │ cosine   │     │ msg    │     │ unmodified   │     │  1 call   │
   └──────────┘     └────────┘     └──────────────┘     └─────┬─────┘
        ▲                                                     │
        │                                                     │
        └─────────────  L_{i+1}  ◀──────────────────────────  ┘
```

Per task, exactly one curator LLM call. No grader-in-the-loop. No
outcome bit threading. No citations. No shadow trajectory.

The Ledger is **off by default**. With `--dynamic-ledger` off, the
pipeline is byte-identical to the baseline runner — pinned by
`test_dynamic_ledger_off_csv_schema_unchanged`.

## Entry shape

```python
class Entry(BaseModel):
    entry_id: str                              # "entry-N" — monotonic per domain
    section: str                               # short categorical label
    content: str                               # free-form playbook text, no length cap
    source_problem: str                        # curator's paraphrase — second retrieval key
    active: bool = True                        # soft-delete flag
    created: int                               # 0-indexed per-domain task ordinal
    updated: int                               # last edit ordinal
    content_embedding: list[float]             # text-embedding-3-large; 3072d
    source_problem_embedding: list[float]      # text-embedding-3-large; 3072d
```

No counters (no helpful / harmful / usage). The Dynamic Ledger has no
GT, so quality signals from grading cannot reach the curator. Per-entry
diagnostic counters would invite a back-channel; we keep the schema
clean.

## Pipeline

### Hook A · retrieve + inject

Active entries in the task's domain are dual-retrieved against the
task prompt:

- `top_c = top-k(content_embedding cosine, k=5)`
- `top_p = top-k(source_problem_embedding cosine, k=5)`
- `B_i   = dedup-by-entry_id(top_p + top_c)`   (source-problem axis first)

The unioned subset is rendered into a `## Reference notes from prior
cases in this area` block and prepended to the USER message in
`initial_messages.json`. The vendor SYSTEM prompt is left untouched
(fidelity test pins this).

### Generate

The vendor's ReAct-toolbelt agent runs as a Dockerized subprocess.
**No changes** to vendor code, no changes to `final_answer` format —
the strategies block is just additional context in the user message.

### Hook B · curate

After grading, the curator runs **once**. Inputs:

- the per-domain Dynamic Ledger (its active entries, serialized as JSON for
  the curator's `<playbook>` block),
- the verbatim task prompt (WITHOUT the strategies-block injection),
- the agent's rendered trajectory (truncated per
  `cfg.trajectory_max_chars_per_tool_result`).

**Forbidden in the curator signature:** `criteria`, `score`, `scores`,
`gt_bit`, `gt_correct_bit`, `expected_answer`, `gold_response`,
`judge_rationale`, `verifier_result`, `final_score`. Pinned by the
load-bearing fidelity test `test_curator_signature_has_no_outcome`.

The curator emits a single `<memory_updates>` XML block holding a JSON
array of ops:

| Op       | Args                                | Effect                                  |
|----------|-------------------------------------|-----------------------------------------|
| `CREATE` | `section, content, source_problem`  | New entry; subject to create-time dedup |
| `UPDATE` | `entry_id, content`                 | Replace + re-embed; bump `updated`      |
| `DELETE` | `entry_id`                          | Soft-delete                             |

The three operations match the Dynamic Ledger approach in the
**Dynamic Cheatsheet 2.0** codebase. No `CONSOLIDATE`, no `NO_OP`.

Op application order: `DELETE → UPDATE → CREATE`. Hallucinated
`entry_id`s are dropped silently. Any op outside `{CREATE, UPDATE,
DELETE}` is parsed and dropped — the curator's wider prompt cannot
introduce ops not in this set.

### Dedup

Create-time only, against the **retrieved subset** `B_i` (not the
whole ledger). Candidate content is embedded, then compared to each
retrieved entry's `content_embedding`. If max cosine > 0.85 the CREATE
is rejected with a `skipped_similar` counter bump.

### Curate-always policy

The curator fires whenever the Dynamic Ledger is on, regardless of
agent status (`completed` / `failed` / `error` / `cancelled`). The
agent's status is itself a coarse outcome proxy we deliberately do
not condition on. Failed trajectories often contain instructive
"what-to-avoid" patterns; the curator decides whether anything is
worth capturing.

## Configuration

```python
@dataclass(frozen=True)
class DynamicLedgerConfig:
    enabled: bool = False
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    top_k_per_axis: int = 5
    curator_model: str | None = None        # filled from AgentProfile at runtime
    curator_extra_args: dict | None = None  # filled from AgentProfile at runtime
    curator_temperature: float = 1.0
    curator_max_tokens: int = 16000
    curator_timeout_seconds: float = 1800
    create_time_similarity_threshold: float = 0.85
    trajectory_max_chars_per_tool_result: int = 4000
```

CLI flag: `--dynamic-ledger / --no-dynamic-ledger`. Default OFF.

### Curator model policy

The curator runs on **the same model as the agent profile under test**,
with the same `orchestrator_extra_args` (reasoning effort, etc.). If
the agent is `grok-4.3-high`, the curator is also `grok-4.3-high`; if
the agent is `gpt-5.5-medium`, the curator is also `gpt-5.5-medium`.
The runner fills `cfg.curator_model` and `cfg.curator_extra_args` from
the active `AgentProfile` before the first curator call. Only the
**judge** model is fixed (gpt-5.5 medium).

Setting `cfg.curator_model` explicitly is allowed for experiments that
need to hold the curator model constant across agent profiles, but the
CLI does not surface that knob.

## Per-domain isolation

Each domain (Investment Banking, Law, Management Consulting) has its
own Ledger and its own snapshot history under
`runs/<run>/dynamic_ledger/<Domain>/snapshot_NNNN.json`. Retrieval at
task `i` in domain D sees the active entries of D's ledger only;
never cross-domain.

## Snapshots & resume

After each task in domain D, the runner saves
`snapshot_<ordinal>.json` and appends one line to `curator_log.jsonl`
with op counts, token usage, and wall time. On resume, the runtime
loads `snapshot_<max_completed>.json` so the in-flight ledger is
exactly what it would have been at that point.

## CSV columns added when the Ledger is on

```
dynamic_ledger_enabled
dynamic_ledger_snapshot_index_before
retrieved_entry_count
retrieved_entry_ids                  JSON list
curator_create_count                 (committed)
curator_create_blocked_count         (rejected by create-time dedup)
curator_update_count
curator_delete_count
dynamic_ledger_active_entry_count_after
dynamic_ledger_total_entry_count_after
dynamic_ledger_total_active_chars_after
curator_prompt_tokens
curator_completion_tokens
curator_wall_seconds
trajectory_chars_seen_by_curator
```

**No** GT-related columns. **No** criteria-related columns. **No**
citation-related columns.

## Curator-side principles

The curator system prompt asks for a *senior reviewer*: critically
diagnose what the colleague did well vs. thinly, prescribe domain
standard practice when it is clear, hedge with conditions when it
genuinely isn't. Five hard rules:

1. **R1** — Workflows + critical diagnosis, not outcomes. Embed
   literal code / tool-call shapes; show failing AND succeeding
   argument forms.
2. **R2** — Ground in concrete examples from the trajectory. Values
   in entries are illustrative; the injection block explicitly tells
   the next agent not to copy specific numbers, paths, or names.
3. **R3** — Prescribe standard practice when it is clear; hedge with
   conditions when it isn't.
4. **R4** — Extract substantive domain insights, not just process.
5. **R5** — Capture the micro-details that break practitioners.

Two entry shapes: elaborate playbook entries (multi-paragraph
workflows, no length cap) and focused action notes (one tight
paragraph for a narrow lesson). The default behavior is two-to-four
ops per session; zero ops are allowed when the
existing playbook already covers everything observed.

## Tests

```
tests/test_dynamic_ledger_entry.py            Entry + DynamicLedger
tests/test_dynamic_ledger_store.py            SnapshotStore + resume
tests/test_dynamic_ledger_retriever_dedup.py  dual retrieval + dedup
tests/test_dynamic_ledger_curator.py          parser + apply_ops
tests/test_dynamic_ledger_injector.py         render + initial_messages augmentation
tests/test_dynamic_ledger_trajectory_render.py rendering for the curator
tests/test_dynamic_ledger_fidelity.py         load-bearing invariants
```

The fidelity tests cover:
- the curator signature contains no GT-leaking parameter,
- the baseline CSV schema is byte-identical with the ledger off,
- the ledger-on CSV preserves the baseline as a prefix,
- the curator prompt mentions "will not be told",
- the injection block tells the agent not to copy values,
- the vendor SYSTEM prompt is preserved.

## Run it

```bash
apex-agents-bench run \
    --model grok-4.3-high \
    --task-ids task_XXX \
    --dynamic-ledger \
    --output runs/dl-on/results.csv
```

The first task in a domain starts from an empty ledger. Each subsequent
task in that domain receives retrieval results from prior tasks in the
same domain. The pipeline is fully deterministic given the trajectory,
the curator's seeded sampling, and the embedding service.

## Attribution

The **Dynamic Ledger** method itself — itemised strategy memory with
typed `CREATE` / `UPDATE` / `DELETE` operations, dual-axis
(strategy + source-problem) embedding retrieval, and a create-time
similarity filter — is introduced in the **Dynamic Cheatsheet 2.0**
codebase by **Jerry Gu, Shurui Liu, and Sabrina Yen-Ko** (Stanford
SAIL; mentor: Mirac Suzgun). It is one of three memory architectures
studied in DC2 alongside the Dynamic Cheatsheet variants (Suzgun et
al., 2025) and ACE (Zhang et al., 2025).

The reference implementation lives at
`src/dc2/methods/dl.py` and `prompts/dl/{generator,curator}.md` in
the DC2 codebase.

```bibtex
@misc{gu_liu_yang_2025_dynamic_ledger,
  title   = {Dynamic Ledger: itemised, dual-indexed strategy memory},
  author  = {Gu, Jerry and Liu, Shurui and Yen-Ko, Sabrina},
  note    = {Stanford SAIL; introduced in the Dynamic Cheatsheet 2.0
             codebase. Mentor: Mirac Suzgun},
  year    = {2025}
}
```

### What this repository changes (adaptations, not the core method)

The DL **flow and framework** here match the DC2 reference:
single-call curator emitting typed `CREATE` / `UPDATE` / `DELETE`
ops inside `<memory_updates>`, no GT signal, dual top-k retrieval
on content and source-problem embeddings, create-time cosine
similarity gate (0.85) against the retrieved subset. The
adaptations to this benchmark are scoped:

- **Curator prompt content** — critical-diagnosis / 5-rule (R1-R5)
  framing replaces the DC2 prompt body. The output contract is
  unchanged (same `<memory_updates>` block, same three ops; any op
  outside `{CREATE, UPDATE, DELETE}` the curator's wider prompt
  describes is silently dropped by the parser).
- **Per-domain ledger** — one ledger per APEX-Agents domain
  (Investment Banking / Law / Management Consulting); DC2 uses one
  global ledger.
- **Free-form strategy text** — DC2 mandates a `Description /
  Applicability / Example / Anti-pattern` 4-field body; ours is
  free-form to fit longer tool-call workflows.
- **k = 5 per axis** (DC2 default is 3).
- **Trajectory-aware curator input** — the curator sees the agent's
  full tool-call trajectory (DC2's solver transcript is a single
  prose deliverable; ours is a multi-turn ReAct trace).

