"""Memory pool — the persistent state DC-RS reads from and appends to.

One ``Bank`` per benchmark domain. Each domain's pool is isolated:
retrieval for a Finance task only sees prior Finance pairs, never
Legal. Per-domain isolation is enforced at the runtime layer
(``DCRSRuntime.banks: dict[str, Bank]``); the ``Bank`` data structure
itself is the per-domain unit.

Each ``BankEntry`` records one past ``(task_prompt, rendered_trajectory)``
pair along with the prompt embedding used for cosine retrieval and the
domain it was produced in (for diagnostic provenance). In the agentic
setting the "answer" half of the pair is the agent's TRAJECTORY (a
truncated text transcript of its tool calls, results, and reasoning),
not a prose deliverable. The pool is append-only; no usage counters, no
helpful/harmful flags, no soft-delete.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict


class BankEntry(BaseModel):
    """One past pair in a per-domain pool, plus its prompt embedding.

    ``domain`` defaults to ``""`` so entries written by an earlier
    flat-layout codebase can still be loaded. New code always sets it
    explicitly to the domain the pair was produced in; this field is
    diagnostic provenance only — it is not used to filter retrieval
    (filtering is by file location, since each domain has its own
    ``bank.jsonl``).

    ``rendered_trajectory`` is the agentic analogue of the prose repo's
    ``deliverable``: a compact text transcript of what the agent did on
    that task."""

    model_config = ConfigDict(extra="ignore")

    bank_id: str
    task_id: str
    domain: str = ""
    task_prompt: str
    rendered_trajectory: str
    prompt_embedding: list[float]
    added: int


@dataclass
class Bank:
    """A single domain's pool. Per-run there is one Bank per domain;
    isolation is at the runtime layer (``DCRSRuntime.banks``)."""

    entries: list[BankEntry] = field(default_factory=list)

    def append(self, entry: BankEntry) -> None:
        self.entries.append(entry)

    def next_bank_id(self) -> str:
        return f"bank-{len(self.entries) + 1:05d}"

    def next_added_ordinal(self) -> int:
        return len(self.entries)
