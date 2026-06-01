"""Unit tests for the DL entry/ledger model + id helpers."""

from __future__ import annotations

import pytest

from apex_agents_bench.dl.entry import (
    ENTRY_TYPES,
    TYPE_TO_SECTION,
    DLEntry,
    DLLedger,
    format_entry_id,
    parse_entry_id,
)


def _emb(x: float) -> list[float]:
    return [x, 1.0]


def test_entry_types_and_sections_align() -> None:
    assert set(ENTRY_TYPES) == set(TYPE_TO_SECTION)
    assert len(ENTRY_TYPES) == 5
    for t in ENTRY_TYPES:
        assert TYPE_TO_SECTION[t]


def test_format_and_parse_entry_id_roundtrip() -> None:
    for n in (0, 1, 7, 1234):
        assert parse_entry_id(format_entry_id(n)) == n
    with pytest.raises(ValueError):
        format_entry_id(-1)
    for bad in ("entry", "entry-", "entry-x", "bullet-1", "1"):
        with pytest.raises(ValueError):
            parse_entry_id(bad)


def test_add_assigns_monotonic_ids_and_sets_created_updated() -> None:
    led = DLLedger(domain="D")
    e1 = led.add(
        type="snippet",
        content="c1",
        source_problem="s1",
        content_embedding=_emb(1.0),
        source_problem_embedding=_emb(2.0),
        created=1,
    )
    e2 = led.add(
        type="pitfall",
        content="c2",
        source_problem="s2",
        content_embedding=_emb(3.0),
        source_problem_embedding=_emb(4.0),
        created=2,
    )
    assert e1.entry_id == "entry-1"
    assert e2.entry_id == "entry-2"
    assert e1.created == e1.updated == 1
    assert led.next_entry_ord == 3
    assert len(led.active_entries()) == 2


def test_update_content_reembeds_and_can_refile_type() -> None:
    led = DLLedger(domain="D")
    e = led.add(
        type="snippet",
        content="old",
        source_problem="s",
        content_embedding=_emb(1.0),
        source_problem_embedding=_emb(2.0),
        created=1,
    )
    updated = led.update_content(
        e.entry_id, content="new", content_embedding=_emb(9.0), updated=5, type="strategy"
    )
    assert updated is not None
    assert updated.content == "new"
    assert updated.content_embedding == _emb(9.0)
    assert updated.type == "strategy"
    assert updated.updated == 5
    # source_problem + its embedding preserved across a content update
    assert updated.source_problem == "s"
    assert updated.source_problem_embedding == _emb(2.0)


def test_soft_delete_keeps_entry_but_drops_from_active() -> None:
    led = DLLedger(domain="D")
    e = led.add(
        type="formula",
        content="c",
        source_problem="s",
        content_embedding=_emb(1.0),
        source_problem_embedding=_emb(2.0),
        created=1,
    )
    assert led.soft_delete(e.entry_id, updated=3) is not None
    assert led.get(e.entry_id) is not None  # still present
    assert led.get(e.entry_id).active is False
    assert led.active_entries() == []
    # second delete is a no-op (already inactive)
    assert led.soft_delete(e.entry_id, updated=4) is None
    # update on an inactive entry is refused
    assert (
        led.update_content(e.entry_id, content="x", content_embedding=_emb(1.0), updated=5) is None
    )


def test_update_and_delete_unknown_id_return_none() -> None:
    led = DLLedger(domain="D")
    assert (
        led.update_content("entry-99", content="x", content_embedding=_emb(1.0), updated=1) is None
    )
    assert led.soft_delete("entry-99", updated=1) is None


def test_serialize_for_llm_only_active_and_no_embeddings() -> None:
    led = DLLedger(domain="D")
    a = led.add(
        type="snippet",
        content="ca",
        source_problem="sa",
        content_embedding=_emb(1.0),
        source_problem_embedding=_emb(2.0),
        created=1,
    )
    b = led.add(
        type="pitfall",
        content="cb",
        source_problem="sb",
        content_embedding=_emb(3.0),
        source_problem_embedding=_emb(4.0),
        created=2,
    )
    led.soft_delete(b.entry_id, updated=3)
    rows = led.serialize_for_llm()
    assert len(rows) == 1
    row = rows[0]
    assert row["entry_id"] == a.entry_id
    assert row["type"] == "snippet"
    assert row["content"] == "ca"
    assert row["source_problem"] == "sa"
    assert "content_embedding" not in row
    assert "source_problem_embedding" not in row


def test_extra_fields_ignored_on_load() -> None:
    # ConfigDict(extra="ignore") tolerates fields a future schema might add.
    e = DLEntry.model_validate(
        {
            "entry_id": "entry-1",
            "type": "snippet",
            "content": "c",
            "source_problem": "s",
            "created": 1,
            "updated": 1,
            "some_future_field": 123,
        }
    )
    assert e.entry_id == "entry-1"
