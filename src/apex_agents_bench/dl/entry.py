"""DL entry data model + ``DLLedger`` container.

A DL *entry* is one itemised, typed memory with TWO retrieval keys (its
own content and the source problem that produced it). Distinct from:

  * the DC-RS ``BankEntry`` — which stores a whole ``(task, trajectory)``
    pair, never an itemised lesson, and has a single embedding axis; and
  * the TRACE ``Bullet`` — which carries helpful/harmful/usage counters
    fed by the ground-truth bit and citations.

DL has no counters: it consumes no ground-truth signal and has no
citation mechanism, so there is nothing to count. What it adds over both
is a REQUIRED ``type`` — one of the five DC-RS categories — so every
entry is classified at creation.

The ledger is the deterministic primitive: an LLM curator *proposes*
CREATE / UPDATE / DELETE operations, but applying them is pure code, so
two runs that emit the same operation stream produce the same ledger.
Deletes are soft (``active=False``) so a snapshot can be replayed and so
an entry's id is never reused.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# The five canonical entry types. These are the DC-RS section categories
# expressed as compact tokens; ``TYPE_TO_SECTION`` maps each to the
# display header used when rendering entries for the generator. A CREATE
# whose ``type`` is not one of these five is dropped by the parser.
ENTRY_TYPES: tuple[str, ...] = (
    "snippet",
    "strategy",
    "formula",
    "environment",
    "pitfall",
)

TYPE_TO_SECTION: dict[str, str] = {
    "snippet": "Reusable Code and Tool-Call Snippets",
    "strategy": "Solution Strategies for Recurring Task Shapes",
    "formula": "Formulas, Definitions, and Conventions",
    "environment": "Environment and Sandbox Facts",
    "pitfall": "Pitfalls, Edge Cases, and Verification Checks",
}


class DLEntry(BaseModel):
    """One Dynamic Ledger entry with dual-retrieval keys and a required type."""

    model_config = ConfigDict(extra="ignore")

    entry_id: str
    type: str
    """One of :data:`ENTRY_TYPES`. Required at creation; carried through
    updates (an update may re-file the entry into a different type)."""

    content: str
    """The reusable memory body, in the DC-RS ``<description>`` /
    ``<example>`` shape. Embedded as one of the two retrieval keys."""

    source_problem: str
    """A retrieval-focused paraphrase of the SITUATION that produced this
    entry (not a description of the entry itself). Embedded as the second
    retrieval key so a future task that *resembles* the original situation
    can find this entry even when the query does not closely match the
    entry's own content text."""

    active: bool = True
    created: int
    updated: int
    content_embedding: list[float] = Field(default_factory=list)
    source_problem_embedding: list[float] = Field(default_factory=list)


def format_entry_id(n: int) -> str:
    if n < 0:
        raise ValueError(f"entry_id ordinal out of range: {n}")
    return f"entry-{n}"


def parse_entry_id(entry_id: str) -> int:
    if not entry_id.startswith("entry-") or len(entry_id) <= len("entry-"):
        raise ValueError(f"malformed entry_id: {entry_id!r}")
    try:
        return int(entry_id[len("entry-") :])
    except ValueError as exc:
        raise ValueError(f"malformed entry_id: {entry_id!r}") from exc


class DLLedger(BaseModel):
    """The per-domain DL ledger — entries plus the per-domain id counter."""

    model_config = ConfigDict(extra="ignore")

    domain: str
    next_entry_ord: int = 1
    entries: dict[str, DLEntry] = Field(default_factory=dict)

    def active_entries(self) -> list[DLEntry]:
        return [e for e in self.entries.values() if e.active]

    def get(self, entry_id: str) -> DLEntry | None:
        return self.entries.get(entry_id)

    def add(
        self,
        *,
        type: str,
        content: str,
        source_problem: str,
        content_embedding: list[float],
        source_problem_embedding: list[float],
        created: int,
    ) -> DLEntry:
        entry_id = format_entry_id(self.next_entry_ord)
        self.next_entry_ord += 1
        e = DLEntry(
            entry_id=entry_id,
            type=type,
            content=content,
            source_problem=source_problem,
            created=created,
            updated=created,
            content_embedding=list(content_embedding),
            source_problem_embedding=list(source_problem_embedding),
        )
        self.entries[entry_id] = e
        return e

    def update_content(
        self,
        entry_id: str,
        *,
        content: str,
        content_embedding: list[float],
        updated: int,
        type: str | None = None,
    ) -> DLEntry | None:
        e = self.entries.get(entry_id)
        if e is None or not e.active:
            return None
        updates: dict[str, Any] = {
            "content": content,
            "content_embedding": list(content_embedding),
            "updated": updated,
        }
        if type is not None:
            updates["type"] = type
        new = e.model_copy(update=updates)
        self.entries[entry_id] = new
        return new

    def soft_delete(self, entry_id: str, *, updated: int) -> DLEntry | None:
        e = self.entries.get(entry_id)
        if e is None or not e.active:
            return None
        new = e.model_copy(update={"active": False, "updated": updated})
        self.entries[entry_id] = new
        return new

    def serialize_for_llm(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in self.active_entries():
            out.append(
                {
                    "entry_id": e.entry_id,
                    "type": e.type,
                    "content": e.content,
                    "source_problem": e.source_problem,
                    "created": e.created,
                    "updated": e.updated,
                }
            )
        return out
