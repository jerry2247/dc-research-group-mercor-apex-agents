"""Memory entry data model + :class:`DynamicLedger` container.

The Dynamic Ledger is a per-domain collection of free-form text entries
that the agent retrieves from at prompt time and that the curator
appends to / refines after each task. The data model has no
ground-truth-related fields (no helpful/harmful counters, no
correctness flag, no GT-derived score). The curator decides what to add
or revise from the work itself.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Entry(BaseModel):
    """One Dynamic Ledger entry."""

    model_config = ConfigDict(extra="ignore")

    entry_id: str
    section: str
    content: str
    source_problem: str
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
        return int(entry_id[len("entry-"):])
    except ValueError as exc:
        raise ValueError(f"malformed entry_id: {entry_id!r}") from exc


class DynamicLedger(BaseModel):
    """The per-domain Dynamic Ledger — entries plus the per-domain entry_id
    counter."""

    model_config = ConfigDict(extra="ignore")

    domain: str
    next_entry_ord: int = 1
    entries: dict[str, Entry] = Field(default_factory=dict)

    def active_entries(self) -> list[Entry]:
        return [e for e in self.entries.values() if e.active]

    def get(self, entry_id: str) -> Entry | None:
        return self.entries.get(entry_id)

    def add(
        self,
        *,
        section: str,
        content: str,
        source_problem: str,
        content_embedding: list[float],
        source_problem_embedding: list[float],
        created: int,
    ) -> Entry:
        entry_id = format_entry_id(self.next_entry_ord)
        self.next_entry_ord += 1
        entry = Entry(
            entry_id=entry_id,
            section=section,
            content=content,
            source_problem=source_problem,
            created=created,
            updated=created,
            content_embedding=list(content_embedding),
            source_problem_embedding=list(source_problem_embedding),
        )
        self.entries[entry_id] = entry
        return entry

    def update_content(
        self, entry_id: str, *, content: str, content_embedding: list[float], updated: int
    ) -> Entry | None:
        e = self.entries.get(entry_id)
        if e is None or not e.active:
            return None
        new = e.model_copy(
            update={
                "content": content,
                "content_embedding": list(content_embedding),
                "updated": updated,
            }
        )
        self.entries[entry_id] = new
        return new

    def soft_delete(self, entry_id: str, *, updated: int) -> Entry | None:
        e = self.entries.get(entry_id)
        if e is None or not e.active:
            return None
        new = e.model_copy(update={"active": False, "updated": updated})
        self.entries[entry_id] = new
        return new

    def serialize_for_curator(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in self.active_entries():
            out.append(
                {
                    "entry_id": e.entry_id,
                    "section": e.section,
                    "content": e.content,
                    "source_problem": e.source_problem,
                    "created": e.created,
                    "updated": e.updated,
                }
            )
        return out
