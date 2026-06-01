"""Unit tests for DL entry rendering (generator block + curator window)."""

from __future__ import annotations

from apex_agents_bench.dl.entry import TYPE_TO_SECTION, DLLedger
from apex_agents_bench.dl.formatting import (
    render_entries_for_curator,
    render_entries_for_generator,
)


def _led_with(types_contents: list[tuple[str, str]]) -> DLLedger:
    led = DLLedger(domain="D")
    for t, c in types_contents:
        led.add(
            type=t,
            content=c,
            source_problem=f"sp-{c}",
            content_embedding=[1.0, 0.0],
            source_problem_embedding=[0.0, 1.0],
            created=led.next_entry_ord,
        )
    return led


def test_generator_empty_is_literal_empty() -> None:
    assert render_entries_for_generator([]) == "(empty)"


def test_generator_groups_under_section_headers() -> None:
    led = _led_with([("snippet", "SNIP"), ("pitfall", "PIT")])
    entries = list(led.active_entries())
    out = render_entries_for_generator(entries)
    assert f"## {TYPE_TO_SECTION['snippet']}" in out
    assert f"## {TYPE_TO_SECTION['pitfall']}" in out
    assert "<memory_item>" in out
    assert "SNIP" in out
    assert "PIT" in out
    # the generator block must NOT expose entry ids
    assert "entry-1" not in out
    assert "entry_id" not in out


def test_generator_sections_in_canonical_order() -> None:
    # add pitfall first, snippet second; output must still lead with snippet
    led = _led_with([("pitfall", "PIT"), ("snippet", "SNIP")])
    out = render_entries_for_generator(list(led.active_entries()))
    assert out.index(TYPE_TO_SECTION["snippet"]) < out.index(TYPE_TO_SECTION["pitfall"])


def test_curator_window_is_tagged_memory_items_with_ids_and_types() -> None:
    led = _led_with([("snippet", "A"), ("formula", "B")])
    entries = list(led.active_entries())
    out = render_entries_for_curator(entries)
    # DC-RS <memory_item> shape, tagged with entry_id + type (NOT JSON).
    assert '<memory_item entry_id="entry-1" type="snippet">' in out
    assert '<memory_item entry_id="entry-2" type="formula">' in out
    assert out.count("<memory_item") == 2
    assert "A" in out and "B" in out


def test_curator_window_empty_is_a_sentinel_phrase() -> None:
    out = render_entries_for_curator([])
    assert out != "[]"
    assert "no entries" in out.lower()
