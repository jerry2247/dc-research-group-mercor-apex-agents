# DC-RS — apex-agents-bench

**DC-RS** (Dynamic Cheatsheet — Retrieval Synthesis) is a
no-ground-truth test-time-learning subsystem layered on the Mercor /
Archipelago ReAct-toolbelt harness. It is a faithful port of Suzgun et
al.'s *Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory*
(2025, arXiv:2504.07952) to the agentic, tool-using, code-executing
benchmark.

Unlike TRACE, DC-RS *never* sees the ground-truth correctness bit, the
rubric text, per-criterion verdicts, the expected answer, or the judge
rationale. Its only inputs are the previous cheatsheet, the retrieved
past trajectories, and the current task prompt. This no-GT property is
the load-bearing fidelity invariant of the method, asserted by the
narrow synthesizer signature (see *Tests*).

We follow the paper's pipeline faithfully with these scoped adaptations
to the agentic, code-executing benchmark:

1. **OpenAI embeddings** (`text-embedding-3-large`, 3072-dim) for the
   single-axis cosine retrieval over the per-domain pool.
2. **Trajectory in place of deliverable** — in the prose setting the
   "answer" half of a retrieved pair is a written deliverable; here it
   is a rendered transcript of the agent's tool calls, tool results,
   and reasoning (see *Trajectory rendering*).
3. **One synthesizer model** — the synthesizer runs on the active
   `AgentProfile`'s `synthesizer_model` with the same
   `synthesizer_extra_args` (reasoning effort, etc.). Only the
   **judge** model is fixed (gpt-5.5 medium).
4. **Anti-wipe guard** — a small backstop that rescues a single-turn
   wholesale wipe of the cheatsheet (see *Anti-wipe guard*); the
   paper's "the cheatsheet grows and is sharpened, never wiped"
   discipline is otherwise carried by the prompt.

DC-RS is **off by default**. With `--dc-rs` off the runner takes the
baseline code path; the CSV schema is byte-identical to the no-DC-RS
shape (the agent sees byte-identical `initial_messages.json`).
`--dc-rs` and `--trace` are mutually exclusive.

## Pipeline

```
   before agent                                              after agent
   ┌──────────┐   ┌───────────┐   ┌────────┐   ┌──────────┐   ┌──────────┐
   │ EMBED +  │──▶│ SYNTHESIZE │──▶│ INJECT │──▶│ GENERATE │──▶│  APPEND  │
   │ RETRIEVE │   │ 1 LLM call │   │ user   │   │ vendor   │   │ to pool  │
   │ top-k=3  │   │ <cheatsheet│   │ msg    │   │ agent    │   │ (no LLM) │
   │ cosine   │   │  > +copy-  │   │        │   │          │   │          │
   │ per-dom. │   │  forward   │   │        │   │          │   │          │
   └──────────┘   └───────────┘   └────────┘   └──────────┘   └────┬─────┘
        ▲              │                                            │
        │              ▼                                            │
        │        anti-wipe guard                                    │
        │        write cheatsheet.txt                               │
        │        (replace slot, copy-forward)                       │
        │                                                           │
        └──────────────  pool[D]  ◀────────────────────────────────┘
                      (task, trajectory, embedding) appended
```

Per task: **exactly one** LLM call into the DC-RS pipeline — the
synthesizer — and it runs **before** the agent. After the agent runs
there is **no** second LLM call; the just-completed
`(task_prompt, rendered_trajectory, prompt_embedding)` triple is simply
appended to the per-domain pool. The synthesizer receives **no**
grading signal of any kind.

**Contrast with TRACE.** TRACE uses a post-task reflector + curator
pair that *does* see the boolean correctness bit and emits CRUD ops on
an atomic-bullet ledger. DC-RS has no curator, no ops, and no
ground-truth: the per-domain cheatsheet is a single text slot replaced
whole each task (copy-forward), and the per-domain pool is append-only.

## State per domain

DC-RS holds two pieces of state per benchmark domain (Investment
Banking, Law, Management Consulting), fully isolated — a Law task's
retrieval never sees an Investment Banking pair, and the Law cheatsheet
never bleeds into Management Consulting:

1. **POOL** — an append-only list of
   `(task_prompt, rendered_trajectory, prompt_embedding)` triples,
   persisted as `bank.jsonl`. No usage counters, no helpful/harmful
   flags, no soft-delete. The pool grows by exactly one entry per
   completed task in that domain.
2. **CHEATSHEET** — a single replace-slot of free-form text
   (`cheatsheet.txt`) that persists across tasks and is replaced whole
   each task. The synthesizer carries the previous cheatsheet forward
   (copy-forward) and emits the new one; the first task in a domain
   starts from the literal `(empty)`.

```python
class BankEntry(BaseModel):
    bank_id: str                      # "bank-NNNNN", per-domain ordinal
    task_id: str
    domain: str = ""                  # diagnostic provenance only
    task_prompt: str                  # the "problem" half of the pair
    rendered_trajectory: str          # the "answer" half: agent transcript
    prompt_embedding: list[float]     # text-embedding-3-large; 3072d
    added: int                        # 0-indexed per-domain ordinal at append
```

## Hooks into `runner.run_single_task`

| Hook | When | Effect |
|------|------|--------|
| **A. embed + retrieve + synthesize + guard + inject** | before agent subprocess writes / reads `initial_messages.json` | embed the task prompt; single-axis top-k=3 cosine retrieval over THIS domain's pool; render the retrieved cases block; ONE synthesizer LLM call returns a fresh `<cheatsheet>` (carrying the previous cheatsheet forward); apply the anti-wipe guard; write `cheatsheet.txt` + archive under `cheatsheets/task_<id>.txt`; prepend the injection block (wrapping the cheatsheet) to the USER message; vendor SYSTEM prompt untouched. |
| **B. append to pool** | after agent subprocess returns | render the agent's trajectory; append the `(task_prompt, rendered_trajectory, prompt_embedding)` triple to the per-domain pool (`bank.jsonl`). **No LLM call.** |

Both hooks are guarded by `if dc_rs_runtime is not None`. With DC-RS
off they do not run; the agent's `initial_messages.json` is
byte-identical to the baseline and the CSV schema is the baseline
shape.

If the cheatsheet is empty or the literal `(empty)`, the injector
leaves `initial_messages.json` untouched and returns the empty prefix,
so the first task in each domain sees byte-identical baseline content.

## The single synthesizer call

The synthesizer is the only LLM call DC-RS makes per task. Faithful to
the reference, the whole prompt (instructions + previous cheatsheet +
retrieved cases + current task) is rendered into ONE user message and
sent with no system message:

```python
litellm.completion(
    model=cfg.synthesizer_model,
    messages=[{"role": "user", "content": user_msg}],
    temperature=cfg.synthesizer_temperature,
    max_tokens=cfg.synthesizer_max_tokens,
    timeout=cfg.synthesizer_timeout_seconds,
)  # plus the profile's synthesizer_extra_args (reasoning_effort, etc.)
```

The synthesizer signature is narrow by design:

```python
synthesize(*, current_cheatsheet, retrieved_cases_block, task_prompt, cfg)
```

There is no `criteria`, `score`, `gt_correct`, `expected_answer`,
`judge_rationale`, `rubric`, or `task_id` parameter. This is the
no-GT invariant: the synthesizer never sees grading data, and never
sees identifiers it could use to tag entries with the current task —
entries must read as accumulated general knowledge.

The model wraps its output in `<cheatsheet>...</cheatsheet>`; the inner
body becomes the new cheatsheet. If the wrapper is missing or empty,
the runtime falls back to the verbatim retrieved-cases block
(degradation, not failure), and the CSV records
`synthesizer_used_fallback = True`.

### What the cheatsheet should hold

The cheatsheet's highest-value content is **reusable** material
distilled from the retrieved past trajectories — not a replay of any
one case:

- reusable code / tool-call snippets (with placeholders for case data);
- solution strategies for recurring task shapes;
- formulas, relations, and definitions in symbolic form;
- conventions (orderings, calculation bases, the channel/shape a result
  must be delivered in);
- environment and sandbox facts (a service's required argument form, a
  property of the execution surface, a filesystem convention);
- pitfalls, edge cases, and verification checks.

The single most reliable raw material is **error→fix transitions**: a
trajectory that shows a tool or code call returning an ERROR followed
by a corrected call that SUCCEEDED is self-evidently better at the
correction, *independent of the grade the synthesizer cannot see*. The
prompt directs the model to mine these first and turn each into a
reusable fact or corrected snippet.

## Anti-wipe guard

The synthesizer prompt instructs the model to default to re-emitting
every prior `<memory_item>` verbatim before any edits. In practice the
model occasionally violates that default and emits a cheatsheet with
zero `<memory_item>` blocks — a wholesale wipe. The guard is a small
backstop:

```python
apply_wipe_guard(previous_cheatsheet, new_cheatsheet) -> (cheatsheet, wipe_rescued)
```

If the new cheatsheet has **zero** `<memory_item>` blocks AND the
previous one had **at least one**, the previous cheatsheet is kept for
the persistent slot and `wipe_rescued = True`. Refinements (fewer items
than before, but not zero) are accepted as written — only a full
wipe-to-zero is rescued.

## Trajectory rendering

The "answer" half of each pool pair is a compact transcript of the
agent's run, produced by `render_trajectory_for_synthesizer`:

- assistant reasoning text and tool-call **arguments** are rendered in
  full — these are the high-signal portions;
- each tool **result** is truncated to
  `trajectory_max_chars_per_tool_result` (default 4000) to bound the
  prompt size that long tool outputs would otherwise blow up;
- the upstream Archipelago system prompt is not re-rendered.

The same renderer builds the `{retrieved_cases}` block the synthesizer
reads (past trajectories) and the trajectory stored in the pool.

## Configuration

```python
@dataclass(frozen=True)
class DCRSConfig:
    enabled: bool = False
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    top_k: int = 3

    # Filled in from the active AgentProfile by the runner.
    synthesizer_model: str | None = None
    synthesizer_extra_args: dict | None = None

    synthesizer_temperature: float = 1.0
    synthesizer_max_tokens: int = 24000
    synthesizer_timeout_seconds: int = 1800

    trajectory_max_chars_per_tool_result: int = 4000
```

CLI flags: `--dc-rs / --no-dc-rs`, `--dc-rs-top-k` (default 3, Suzgun's
published value). Mutually exclusive with `--trace`. `--azure` routes
the synthesizer's GPT-5.5 chat completion through Azure-OpenAI when the
selected profile is a GPT-5.5 profile; embeddings always use OpenAI.

## On-disk layout

```
runs/<run>/dc_rs/
  <Domain>/                          # Investment Banking, Law, Management Consulting
    bank.jsonl                       # per-domain pool — source of truth for resume
    cheatsheet.txt                   # persistent replace-slot — source of truth for resume
    cheatsheets/task_<id>.txt        # per-task cheatsheet archive (diagnostic only)
    synthesizer_log.jsonl            # per-task synth call diagnostics
```

`bank.jsonl` and `cheatsheet.txt` are load-bearing for resume; the
`cheatsheets/` archive and `synthesizer_log.jsonl` are diagnostic. On
resume the runtime pre-loads every domain that already has on-disk
state; the results CSV is the source of truth only for which tasks are
already completed.

## Per-domain isolation

Each domain has its own pool and its own cheatsheet slot. Retrieval at
task `i` in domain D scores against D's pool only; never cross-domain.
The depth-first rollout (see `EVALUATION_PLAN.md`) lets each domain's
cheatsheet reach steady state across that domain's worlds before the
rollout moves on.

## CSV columns added when DC-RS is on

```
dc_rs_enabled
dc_rs_bank_size_before                # pool size at task start
dc_rs_bank_size_after                 # pool size after this task's append
dc_rs_retrieved_count
dc_rs_retrieved_bank_ids              # JSON list
dc_rs_appended_bank_id
synthesizer_prompt_tokens
synthesizer_completion_tokens
synthesizer_wall_seconds
synthesizer_cheatsheet_chars
synthesizer_used_fallback             # bool — wrapper tag missing / empty
synthesizer_wipe_rescued              # bool — anti-wipe guard fired
trajectory_chars_appended
```

These columns are appended **after** the baseline columns; the baseline
header order is preserved as a prefix. With DC-RS off, the CSV is
byte-identical to the baseline shape.

## Prompt files

Two prompt files, both under `src/apex_agents_bench/dc_rs/prompts/`:

| File | Role |
|------|------|
| `synthesizer_prompt.txt` | the single synthesizer prompt — instructions, the no-grading-signal framing, the error→fix mining directive, the grow-and-sharpen-never-wipe update protocol, the entry-quality bars, and the `<cheatsheet>` output format; rendered with `{current_cheatsheet}`, `{retrieved_cases}`, `{task_prompt}`. |
| `generator_injection_block.txt` | the block prepended to the agent's USER message; frames the cheatsheet as a passive *formula sheet* ("you do not follow it; consult it when you need a specific fact"), so the agent's own reading of the task stays authoritative; rendered with `{cheatsheet}`. |

## Tests

```
tests/test_dc_rs_bank.py          BankEntry roundtrip / extra-field tolerance / per-domain sequencing
tests/test_dc_rs_curation.py      anti-wipe guard (rescue, pass-through, zero-to-zero, tag counting)
tests/test_dc_rs_extract.py       <cheatsheet> extraction, fallback, case-insensitivity, multiline
```

CSV-schema fidelity is pinned by `test_baseline_csv_has_no_dc_rs_columns`
(baseline header when off) and `test_dc_rs_on_csv_extends_baseline_at_end`
(DC-RS-on CSV preserves the baseline as a prefix), both in
`tests/test_dc_rs_fidelity.py`.

## Run it

```bash
apex-agents-bench run \
    --model grok-4.3-high \
    --task-ids task_XXX \
    --dc-rs \
    --output runs/dc-rs-on/results.csv
```

The first task in a domain starts from an empty cheatsheet and an empty
pool. Subsequent tasks in the same domain see the carried-forward
cheatsheet and retrieve from the accumulated pool.

## Citation

This subsystem implements the **Dynamic Cheatsheet — Retrieval
Synthesis (DC-RS)** method of Suzgun et al., adapted to the agentic
benchmark with the scoped adaptations enumerated above (OpenAI
embeddings, trajectory-in-place-of-deliverable, one synthesizer model,
anti-wipe guard).

```bibtex
@misc{suzgun_yuksekgonul_bianchi_jurafsky_zou_2025_dynamic_cheatsheet,
  title  = {Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory},
  author = {Suzgun, Mirac and Yuksekgonul, Mert and Bianchi, Federico and
            Jurafsky, Dan and Zou, James},
  year   = {2025},
  eprint = {2504.07952},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2504.07952}
}
```
