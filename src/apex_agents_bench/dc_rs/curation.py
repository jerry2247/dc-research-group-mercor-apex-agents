"""Post-synthesis safety nets for the DC-RS synthesizer.

The synthesizer prompt instructs the model to default to re-emitting
every prior ``<memory_item>`` verbatim. In practice the LLM occasionally
violates that default and emits a cheatsheet with zero entries on a
single turn — a "wipe". On the next turn retrieval re-surfaces the
prior trajectories and the synthesizer typically re-derives the lost
entries from those examples, but not always: an entry that lived for
just one or two turns can be lost permanently.

``apply_wipe_guard`` is a small backstop: if the synthesizer's output
on this turn has zero ``<memory_item>`` blocks but the previous
cheatsheet had at least one, keep the previous cheatsheet for the
persistent slot. Refinements (fewer items than before but not zero)
are accepted as written — only a full wipe-to-zero is rescued.
"""

from __future__ import annotations

_MEMORY_ITEM_TAG = "<memory_item>"


def apply_wipe_guard(previous_cheatsheet: str, new_cheatsheet: str) -> tuple[str, bool]:
    """Rescue a single-turn wipe.

    If ``new_cheatsheet`` contains zero ``<memory_item>`` blocks AND
    ``previous_cheatsheet`` contains at least one, return
    ``(previous_cheatsheet, True)``. Otherwise return
    ``(new_cheatsheet, False)``.

    The second element signals "a wipe was rescued" for logging and
    diagnostic CSV reporting.
    """
    new_items = new_cheatsheet.count(_MEMORY_ITEM_TAG)
    prev_items = previous_cheatsheet.count(_MEMORY_ITEM_TAG)
    if new_items == 0 and prev_items > 0:
        return previous_cheatsheet, True
    return new_cheatsheet, False
