# DL — Dynamic Ledger — apex-agents-bench

Dynamic Ledger (DL) is the third memory subsystem in apex-agents-bench,
alongside **DC-RS** (no ground truth, one monolithic cheatsheet, single
synthesizer call before the agent) and **TRACE** (uses ground truth,
itemised bullets with helpful/harmful counters, reflector + curator after
grading, citations).

DL is an **adaptation** of the original Dynamic Ledger (Dynamic Cheatsheet
2.0). It keeps DL's defining mechanics — *individual typed entries indexed
for dual retrieval, edited by a curator through typed CRUD operations* —
and re-expresses the entry quality and the curator's voice in the style of
this repo's DC-RS subsystem. It is **not** a copy of the original prompts.

## Where DL sits among the three methods

| Axis | DC-RS | TRACE | **DL** |
|---|---|---|---|
| Ground-truth signal | no | yes | **no** |
| Memory unit | one cheatsheet blob / domain | itemised bullets | **itemised typed entries** |
| Entry has a required `type` | no (5 sections, soft) | no (free `section`) | **yes (one of 5 categories)** |
| Retrieval | single-axis cosine | dual-axis (content + source) | **dual-axis (content + source)** |
| `top_k` | 3 | 8 / axis | **3 / axis** |
| LLM calls / task | 1 (synthesizer, **before** agent) | 2 (reflector + curator, after grade) | **1 (curator, after agent)** |
| Memory write | whole cheatsheet replaced | CRUD + consolidate + counters | **CRUD (create/update/delete)** |
| Citations | no | yes | **no** |
| Create-time dedup | no | yes | **no** |
| Persistence | bank.jsonl + cheatsheet.txt | per-domain snapshots (soft-delete) | **per-domain snapshots (soft-delete)** |

The contrast DL adds to the paper: an itemised, dual-retrieval, CRUD memory
that — unlike TRACE — consumes **no grading signal**. It learns purely from
the trajectory the agent just produced, exactly as the original DL learns
from the solver transcript.

**Design stance: DC-RS is the model, not TRACE.** The entry-quality bars,
the curator's voice, the five categories, the errors→fixes spine, and the
`<memory_item>` rendering are all taken from this repo's DC-RS subsystem.
DL deliberately does **not** borrow TRACE's machinery: no reflector, no
second LLM call, no ground-truth bit, no citations, no helpful/harmful/usage
counters, no consolidate op, no create-time dedup. What makes DL its own
method is the *ledger* nature — entries are individually stored and
individually editable — which changes the curator from a wholesale rewriter
(DC-RS) into a CRUD editor, and adds the required `type` and the
`source_problem` second retrieval axis.

## Pipeline (per task, per domain)

1. **Hook A — retrieve + inject (before the agent, no LLM).**
   Embed the task prompt once. **Dual-axis retrieval**: top-k=3 by entry
   `content_embedding` AND top-k=3 by `source_problem_embedding`; union by
   `entry_id` (source-problem axis first). Render the retrieved active
   entries — grouped under their five category headers, in DC-RS
   `<memory_item>` shape — into the generator injection block, and prepend
   it to the USER message of `initial_messages.json`. Stash the query
   embedding and the retrieved entries for Hook B.

2. **Agent runs** unmodified (no citation instruction — DL needs none).

3. **Hook B — curate (after the agent, no GT).**
   Render this task's trajectory. ONE curator LLM call sees the **retrieved
   entries** (rendered as DC-RS `<memory_item>` blocks tagged with their
   `entry_id` + `type` — the only window it may UPDATE/DELETE), the
   **current task**, and the **trajectory**. It emits a `<ledger_updates>`
   JSON batch of CREATE / UPDATE / DELETE operations. Apply
   deterministically: DELETE (soft) → UPDATE (re-embed content) → CREATE
   (embed content + source_problem). **No dedup.** Persist a per-domain
   snapshot. The grade is never read.

DL makes exactly one LLM call per task (the curator), after the agent —
mirroring the original DL's `observe`.

## Entry shape (`dl/entry.py`)

`DLEntry` (pydantic): `entry_id` (`entry-N`), `type` (one of the five
canonical tokens), `content` (the `<description>`/`<example>` body),
`source_problem` (retrieval-focused paraphrase — the second embedding
axis), `active` (soft-delete flag), `created`/`updated` (ordinals),
`content_embedding`, `source_problem_embedding`. No counters (DL has no
citations and no GT, so nothing to count).

`DLLedger` (per domain): `domain`, `next_entry_ord`, `entries: dict`. Ops:
`add`, `update_content`, `soft_delete`, `active_entries`, `get`,
`serialize_for_llm`.

### The five types (required on CREATE)

The five DC-RS categories, as compact canonical tokens mapped to display
headers:

| token | display section |
|---|---|
| `snippet` | Reusable Code and Tool-Call Snippets |
| `strategy` | Solution Strategies for Recurring Task Shapes |
| `formula` | Formulas, Definitions, and Conventions |
| `environment` | Environment and Sandbox Facts |
| `pitfall` | Pitfalls, Edge Cases, and Verification Checks |

The parser drops any op whose `type` is not one of the five.

## Op contract (`<ledger_updates>` JSON array)

```
{"op": "CREATE", "type": "<one of five>", "content": "<description>...</description>\n<example>...</example>", "source_problem": "<retrieval-focused paraphrase>"}
{"op": "UPDATE", "entry_id": "entry-N", "content": "<full replacement content>", "type": "<optional re-file>"}
{"op": "DELETE", "entry_id": "entry-N"}
```

`entry_id` on UPDATE/DELETE must come from the retrieved window; ops with
unknown/inactive ids are counted skipped, not applied. Apply order is
DELETE → UPDATE → CREATE (deletes first so the window is settled; creates
last). UPDATE re-embeds the content axis and preserves `source_problem`.
The parser also accepts the short alias `id` for `entry_id` on UPDATE/DELETE.

**UPDATE is enrichment, not just correction.** The curator is told to UPDATE
whenever a run can make an entry *better* — a more efficient tool/code path,
an added trigger or case, a missing workflow step, a cleaner snippet, a
named alternative, or tighter wording — not only when the old entry was
wrong. UPDATE replaces the entry's content wholesale (the curator must
re-emit everything that should remain and integrate the improvement), and
may re-file the entry's `type`.

## The curator prompt (`dl/prompts/curator_prompt.txt`)

A faithful **adaptation of DC-RS's synthesizer prompt**, keeping its tone,
framing, and quality bars. What changes — and why — relative to DC-RS:

- **Wholesale replace → CRUD batch.** DC-RS re-emits the entire cheatsheet
  every turn ("re-emit every `<memory_item>` verbatim, never wipe"). DL's
  entries persist in the store; the curator emits only deltas. The entire
  copy-forward / anti-wipe / wipe-rescue apparatus is removed (it has no
  meaning when entries are individually stored).
- **Past cases → this case's trajectory.** DC-RS runs *before* the agent and
  reads *retrieved past* trajectories. DL runs *after* the agent and reads
  *this task's* trajectory. The "errors-and-their-corrections is your most
  reliable signal" spine is retained but pointed at the just-finished run.
- **`entry_id` references.** Sharpen/retire become explicit UPDATE/DELETE by
  id, scoped to the retrieved window.
- **Required `type`.** Every CREATE classifies the entry into one of five.
- **New `source_problem` field.** A retrieval key DC-RS has no concept of;
  the prompt explains it describes the *situation*, not the entry.
- **Output is `<ledger_updates>` JSON, not `<cheatsheet>` XML.**
- **No ground-truth signal** (preserved DC-RS invariant; enforced by a
  fidelity test on the `curate` signature).
- **Mix of operations** is explicitly encouraged (create the new, update the
  improved, delete the disproven) — an empty array only when nothing clears
  the bar.

Retained near-verbatim from DC-RS: the agent-operation description, the
errors→fixes signal, the five categories, and the entry quality bars
(specific-and-concrete, genuinely reusable, precisely correct, does not
silently narrow, self-contained, carries real content), and the entry
content shape (`<description>` + `<example>`). The quality bars keep DC-RS's
demand for **specific, domain-vocabulary entries** — the curator is told to
name the actual tools, operations, argument/field names, and domain terms,
and to strip only this case's *data values* (never the shared tool/domain
vocabulary). Each of the five types carries a type-specific description+example
shape (a `snippet`'s example is a code/tool-call form; a `formula`'s is
symbolic; etc.) so the entry format matches the entry type.

Placeholders: `{retrieved_entries}`, `{task_prompt}`, `{rendered_trajectory}`.

## The generator injection (`dl/prompts/generator_injection_block.txt`)

Almost identical to DC-RS's generator injection block (reference-material,
consult-don't-obey, do-not-cite framing). The only change is the body: a
list of typed entries grouped under the five category headers instead of
the monolithic synthesized cheatsheet. No citation instruction.
Placeholder: `{entries_block}`.

## Configuration (`dl/config.py`)

`DLConfig`: `enabled`, `embedding_model="text-embedding-3-large"`,
`embedding_dim=3072`, `top_k=3`, `curator_model`, `curator_extra_args`,
`curator_temperature=1.0`, `curator_max_tokens=24000`,
`curator_timeout_seconds=1800`, `trajectory_max_chars_per_tool_result=8000`.
No GT knobs; no dedup threshold. The runner fills `curator_model` from the
active AgentProfile (only the judge is fixed) and applies Azure routing /
in-process Azure call kwargs exactly as for DC-RS/TRACE.

## On-disk layout

```
runs/<run>/dl/
  <Domain>/
    snapshot_<NNNN>.json     # full per-domain ledger, soft-delete; source of truth for resume
    curator_log.jsonl        # one line per curator call (diagnostic)
```

## Per-domain isolation

One ledger per benchmark domain. A Finance task never retrieves a Legal
entry; the Legal ledger never bleeds into Finance. Each domain's ledger
starts empty when that domain's first task runs.

## CSV columns added when DL is on

`dl_enabled`, `dl_snapshot_index_before`, `dl_retrieved_count`,
`dl_retrieved_entry_ids`, `dl_curator_create_count`,
`dl_curator_update_count`, `dl_curator_delete_count`,
`dl_curator_skipped_invalid_id_count`, `dl_curator_parse_error`,
`dl_active_entry_count_after`, `dl_total_entry_count_after`,
`dl_total_active_chars_after`, `dl_curator_prompt_tokens`,
`dl_curator_completion_tokens`, `dl_curator_wall_seconds`,
`dl_trajectory_chars_seen_by_curator`. Appended after the baseline headers;
disjoint from them. `--dc-rs`, `--trace`, `--dl` are mutually exclusive
(three-way).

## Tests

`tests/test_dl_*.py` mirror the DC-RS / TRACE suites: entry/ledger,
retriever (dual-axis order + union), store/resume, curator (parse + apply
order + re-embed + invalid-id handling + mix of ops), injector, runtime
isolation, trajectory render, and **fidelity** (curator signature has no
GT/score/criteria/task_id; prompt is domain-agnostic; no hardcoded
tools/lessons; output format `<ledger_updates>`; all five types named;
required `type`/`source_problem`; generator block is consult-don't-obey and
has no citation instruction; CSV extends baseline at the end; three-way
mutex).

## Run it

```
apex-agents-bench run --model grok-4.3-high --world <world_id> --dl \
    --output runs/<name>/results.csv
```

## Citation

Adapts the Dynamic Ledger method (Dynamic Cheatsheet 2.0) to the
apex-agents-bench agentic harness; entry quality and curator voice follow
this repo's DC-RS subsystem, itself a faithful port of Suzgun et al.,
*Dynamic Cheatsheet* (arXiv:2504.07952).
