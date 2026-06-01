"""Render retrieved DL entries for the two consumers: the generator
(injection block) and the curator (the editable window).

Both renderings follow DC-RS's ``<memory_item>`` shape rather than a JSON
serialization:

  * the generator sees entries grouped under their five category headers,
    as plain ``<memory_item>`` blocks with no ids — reference material it
    consults but does not edit;
  * the curator sees the same entries as ``<memory_item>`` blocks tagged
    with ``entry_id`` and ``type`` attributes — the only window it may
    UPDATE or DELETE, and the ids it must reference.
"""

from __future__ import annotations

from apex_agents_bench.dl.entry import ENTRY_TYPES, TYPE_TO_SECTION, DLEntry


def render_entries_for_generator(retrieved: list[DLEntry]) -> str:
    """Group retrieved entries under their five category headers, as DC-RS
    ``<memory_item>`` blocks (no ids — the generator does not edit). Returns
    the literal ``"(empty)"`` when nothing was retrieved (the first tasks in
    a domain)."""
    if not retrieved:
        return "(empty)"

    by_type: dict[str, list[DLEntry]] = {t: [] for t in ENTRY_TYPES}
    for e in retrieved:
        by_type.setdefault(e.type, []).append(e)

    chunks: list[str] = []
    for t in ENTRY_TYPES:
        items = by_type.get(t) or []
        if not items:
            continue
        chunks.append(f"## {TYPE_TO_SECTION[t]}")
        for e in items:
            chunks.append(f"<memory_item>\n{e.content.strip()}\n</memory_item>")
    return "\n\n".join(chunks).strip() if chunks else "(empty)"


def render_entries_for_curator(retrieved: list[DLEntry]) -> str:
    """Render the retrieved window as DC-RS ``<memory_item>`` blocks tagged
    with ``entry_id`` and ``type`` — the only entries the curator may UPDATE
    or DELETE, and the ids it must reference. Returns a short sentinel when
    empty so the curator knows the ledger has nothing relevant yet."""
    if not retrieved:
        return "(no entries retrieved yet — the ledger has nothing relevant to this case)"
    chunks: list[str] = []
    for e in retrieved:
        chunks.append(
            f'<memory_item entry_id="{e.entry_id}" type="{e.type}">\n'
            f"{e.content.strip()}\n"
            f"</memory_item>"
        )
    return "\n\n".join(chunks)
