# Benchmark structure

> **Read this if you're asking:** "What's actually in the 480 tasks?
> What does an Investment Banking task look like vs a Law task?"

## Overall (verified against the live `mercor/apex-agents` index, 2026-05-19)

| Dimension | Value |
|---|---|
| Total tasks | **480** |
| Total worlds | **33** |
| Domains | 3 -- exact strings: `"Investment Banking"`, `"Law"`, `"Management Consulting"` (case + space-sensitive) |
| Tasks per domain | 160 each |
| Worlds per domain | **10 Investment Banking / 11 Management Consulting / 12 Law** |
| Tasks per world | 8 / 15 / 20 (min / median / max) |
| Criteria per task | 1 / 3 / 10 (min / median / max); **mean 4.06** |
| Prompt chars | 127 / 571 / 2053 (min / median / max); mean 628 |
| Tasks with extra input files | **175 of 480 = 36.5%** |

Source: `data/apex-agents/tasks_and_rubrics.json` + `world_descriptions.json`
on disk after `make fetch-dataset`. Reproduce via `apex-agents-bench catalog`.

## A world, concretely

A *world* is a starter filesystem + `.apps_data` directory representing
a simulated professional work environment. Each world ships as a
single tar.gz / zip in `world_files_zipped/<world_id>.zip` on
HuggingFace. Average ~166 files per world (a mix of XLSX, PDF, DOCX,
PPTX, PNG, HTML, plus per-app state files like calendar events and
chat history).

Worlds are SHARED across tasks. The 160 Investment Banking tasks live
across 10 IB worlds (~16 tasks per world on average; range 8-20). The
agent starts each task from a clean copy of the world (per-task fresh
container), so no prior task's edits leak in.

## Domains

### Investment Banking (10 worlds, 160 tasks)

Examples drawn from the dataset card / the public default task:

> Calculate the accretion / dilution of both BBDC and TVPG
> shareholders, sensitized for different Cash consideration and Bid
> Premium. Edit the existing merger model and add two sensitivity
> analyses ...

The world ships an existing merger model XLSX, supporting PDFs, and
the task expects the agent to extend the spreadsheet and produce a
sensitized output. Code execution (Python in the
`code_execution_server`) and spreadsheet editing
(`sheets_server`) are both load-bearing.

Avg task time per Mercor's reporting: ~1.36 hours of expert time.

### Management Consulting (11 worlds, 160 tasks)

Long-horizon analyses across multiple data sources, with deliverables
in slides or docs. Spreadsheets, presentations, documents, mail, and
chat all in play.

Avg task time: ~1.69 hours.

### Law (12 worlds, 160 tasks)

Memos, contract analyses, regulatory correspondence. Heavy PDF and
DOCX content; agents must read, summarize, and emit polished text
products.

Avg task time: ~2.40 hours.

## Browsing the index locally

After `make fetch-dataset`:

```bash
apex-agents-bench worlds                                       # all 33
apex-agents-bench worlds --domain "Investment Banking"         # 10 worlds
apex-agents-bench tasks --world <world_id>                     # tasks in one world
apex-agents-bench tasks --domain Law -n 5                      # first 5 Law tasks
apex-agents-bench show <task_id>                               # full prompt + rubric
apex-agents-bench catalog                                      # JSON summary
```

These all read from `data/apex-agents/tasks_and_rubrics.json` and
`data/apex-agents/world_descriptions.json` -- no network calls.

## Rubric shape

Each task has a `rubric` field: a list of binary verifier criteria.
Example shape (from the dataset card):

```json
[
  {
    "verifier_id": "ver_<uuid>",
    "criteria": "The agent created report.pdf with the sensitivity table for BBDC accretion/dilution at the specified bid premiums and cash considerations"
  },
  {
    "verifier_id": "ver_<uuid>",
    "criteria": "The values in the sensitivity table are computed correctly given the stated assumptions"
  },
  ...
]
```

The first criterion is graded with `is_primary_objective=true`; all
criteria are binary pass/fail. The judge LLM reads the criterion text
+ the relevant artifact (extracted from the snapshot diff) and emits
`{"grade": "pass" | "fail", "rationale": "..."}`.

Mean per task: **4.06** criteria; median 3; max 10. 480 tasks
× 4.06 criteria ≈ **1,949 binary judgments per full run**.

## Tasks with extra input files

**175 of 480 tasks (36.5%)** ship per-task input files in addition to
the world snapshot. (The HF dataset card cited ~12% historically; the
live `tasks_and_rubrics.json` says 175 -- we trust the live value.)
These extras are downloaded via
`huggingface_hub.snapshot_download(allow_patterns=["task_files/<task_id>/**"])`
on demand and overlay the world filesystem at task start.

`apex-agents-bench tasks` shows a `*` in the `inp` column for tasks
that have extra inputs.

## Reference metadata fields (NOT used by grading)

Each task also ships three reference fields:

- `expected_output` -- a short string describing the expected deliverable.
- `gold_response` -- an example reference response (may be long).
- `gold_response_type` -- the type of the gold response (e.g. text vs file).

These are **not** consumed by the published `output_llm` grading path
(verifiers grade agent artifacts against rubric criteria, not against
the gold strings), so the runner deliberately does NOT thread them
into `verifiers.json`. They are surfaced by `apex-agents-bench show
<task_id>` for analysis purposes only. Passing them to the grading
runner would be a fidelity break.
