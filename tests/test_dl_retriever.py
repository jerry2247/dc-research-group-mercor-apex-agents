"""Unit tests for the DL dual-axis retriever (content + source-problem)."""

from __future__ import annotations

from apex_agents_bench.dl.entry import DLLedger
from apex_agents_bench.dl.retriever import retrieve


def _add(
    led: DLLedger, *, content_vec: list[float], source_vec: list[float], type: str = "snippet"
):
    return led.add(
        type=type,
        content=f"content for {content_vec}",
        source_problem=f"source for {source_vec}",
        content_embedding=content_vec,
        source_problem_embedding=source_vec,
        created=led.next_entry_ord,
    )


def test_retrieve_empty_ledger_returns_empty() -> None:
    led = DLLedger(domain="D")
    assert retrieve(led, query_embedding=[1.0, 0.0], k=3) == []


def test_retrieve_k_zero_returns_empty() -> None:
    led = DLLedger(domain="D")
    _add(led, content_vec=[1.0, 0.0], source_vec=[1.0, 0.0])
    assert retrieve(led, query_embedding=[1.0, 0.0], k=0) == []


def test_retrieve_unions_both_axes() -> None:
    led = DLLedger(domain="D")
    # e1 matches strongly on CONTENT axis only
    e1 = _add(led, content_vec=[1.0, 0.0], source_vec=[-1.0, 0.0])
    # e2 matches strongly on SOURCE axis only
    e2 = _add(led, content_vec=[-1.0, 0.0], source_vec=[1.0, 0.0])
    # e3 matches neither
    _add(led, content_vec=[0.0, -1.0], source_vec=[0.0, -1.0])
    out = retrieve(led, query_embedding=[1.0, 0.0], k=1)
    ids = {e.entry_id for e in out}
    # k=1 per axis → e2 from source axis, e1 from content axis; union = both
    assert ids == {e1.entry_id, e2.entry_id}


def test_retrieve_source_axis_first_in_order() -> None:
    led = DLLedger(domain="D")
    e_src = _add(led, content_vec=[0.0, 1.0], source_vec=[1.0, 0.0])  # source match
    e_con = _add(led, content_vec=[1.0, 0.0], source_vec=[0.0, 1.0])  # content match
    out = retrieve(led, query_embedding=[1.0, 0.0], k=1)
    # source-problem axis is taken first, so the source match leads.
    assert out[0].entry_id == e_src.entry_id
    assert out[1].entry_id == e_con.entry_id


def test_retrieve_dedups_entry_appearing_on_both_axes() -> None:
    led = DLLedger(domain="D")
    # one entry strong on BOTH axes — must appear exactly once
    e_both = _add(led, content_vec=[1.0, 0.0], source_vec=[1.0, 0.0])
    _add(led, content_vec=[0.0, -1.0], source_vec=[0.0, -1.0])
    out = retrieve(led, query_embedding=[1.0, 0.0], k=1)
    ids = [e.entry_id for e in out]
    assert ids.count(e_both.entry_id) == 1


def test_retrieve_skips_inactive_entries() -> None:
    led = DLLedger(domain="D")
    e = _add(led, content_vec=[1.0, 0.0], source_vec=[1.0, 0.0])
    led.soft_delete(e.entry_id, updated=99)
    assert retrieve(led, query_embedding=[1.0, 0.0], k=3) == []
