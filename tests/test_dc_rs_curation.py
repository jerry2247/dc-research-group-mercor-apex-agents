"""Anti-wipe guard tests for the post-synthesis curation layer."""

from __future__ import annotations

from apex_agents_bench.dc_rs.curation import apply_wipe_guard


def _ch(item_count: int) -> str:
    """Build a cheatsheet body with N memory_item blocks for testing."""
    blocks = "\n".join(
        f"<memory_item><description>desc {i}</description><example>ex {i}</example></memory_item>"
        for i in range(item_count)
    )
    return f"## section\n{blocks}\n"


def test_guard_rescues_wipe_when_previous_had_items() -> None:
    """The headline case: previous cheatsheet had entries, the synthesizer
    output has zero — keep the previous."""
    prev = _ch(3)
    new = "## section\n## another section\n"  # no items
    effective, rescued = apply_wipe_guard(prev, new)
    assert rescued is True
    assert effective is prev  # same object, byte-identical


def test_guard_passes_through_when_previous_was_empty() -> None:
    """Turn 1: previous is empty, new is empty — no rescue needed."""
    prev = "(empty)"
    new = "## section\n## another section\n"  # no items
    effective, rescued = apply_wipe_guard(prev, new)
    assert rescued is False
    assert effective is new


def test_guard_passes_through_when_new_has_any_items() -> None:
    """Refinement: previous had 3, new has 1 — the synthesizer chose to
    consolidate; not a wipe, accept it."""
    prev = _ch(3)
    new = _ch(1)
    effective, rescued = apply_wipe_guard(prev, new)
    assert rescued is False
    assert effective is new


def test_guard_passes_through_when_both_have_items() -> None:
    """Default growth: previous had 3, new has 4 — accept."""
    prev = _ch(3)
    new = _ch(4)
    effective, rescued = apply_wipe_guard(prev, new)
    assert rescued is False
    assert effective is new


def test_guard_does_not_rescue_zero_to_zero() -> None:
    """Both empty: nothing to rescue."""
    prev = "## section\n"  # no items
    new = "## section\n## another\n"  # no items
    effective, rescued = apply_wipe_guard(prev, new)
    assert rescued is False
    assert effective is new


def test_guard_counts_memory_item_tag_only() -> None:
    """Sanity: the guard counts ``<memory_item>`` open tags. Prose
    mentioning ``memory_item`` without a real tag must not be miscounted
    as an entry."""
    prev = "this body mentions the word memory_item but has no real tags"
    new = "## section\n## another\n"
    _, rescued = apply_wipe_guard(prev, new)
    assert rescued is False  # prev_items = 0, not >0, so no rescue


def test_guard_preserves_previous_text_exactly_on_rescue() -> None:
    """The rescue preserves the previous cheatsheet text exactly — no
    transformation."""
    prev = "## A\n<memory_item>x</memory_item>\n## B\n<memory_item>y</memory_item>\n"
    new = ""
    effective, rescued = apply_wipe_guard(prev, new)
    assert rescued is True
    assert effective == prev
