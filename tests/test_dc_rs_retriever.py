"""Unit tests for the single-axis cosine retriever over a per-domain pool.

The retriever operates on whatever ``Bank`` it is handed; per-domain
isolation lives one level up at the runtime layer.
"""

from __future__ import annotations

from apex_agents_bench.dc_rs.bank import Bank, BankEntry
from apex_agents_bench.dc_rs.retriever import retrieve


def _entry(idx: int, vec: list[float]) -> BankEntry:
    return BankEntry(
        bank_id=f"bank-{idx:05d}",
        task_id=f"t-{idx}",
        task_prompt=f"p{idx}",
        rendered_trajectory=f"tr{idx}",
        prompt_embedding=vec,
        added=idx - 1,
    )


def test_retrieve_empty_pool_returns_empty_list() -> None:
    bank = Bank()
    assert retrieve(bank, query_embedding=[1.0, 0.0], k=3) == []


def test_retrieve_returns_at_most_k() -> None:
    bank = Bank()
    bank.append(_entry(1, [1.0, 0.0]))
    bank.append(_entry(2, [0.9, 0.1]))
    bank.append(_entry(3, [0.0, 1.0]))
    bank.append(_entry(4, [0.8, 0.2]))
    bank.append(_entry(5, [-1.0, 0.0]))
    out = retrieve(bank, query_embedding=[1.0, 0.0], k=3)
    assert len(out) == 3
    assert [r.entry.task_id for r in out] == ["t-1", "t-2", "t-4"]
    assert out[0].similarity >= out[1].similarity >= out[2].similarity


def test_retrieve_pool_smaller_than_k_returns_all() -> None:
    bank = Bank()
    bank.append(_entry(1, [1.0, 0.0]))
    bank.append(_entry(2, [0.0, 1.0]))
    out = retrieve(bank, query_embedding=[1.0, 0.0], k=3)
    assert len(out) == 2
    assert out[0].entry.task_id == "t-1"


def test_retrieve_default_k_is_three() -> None:
    bank = Bank()
    for i in range(1, 6):
        bank.append(_entry(i, [1.0 - i * 0.1, i * 0.1]))
    out = retrieve(bank, query_embedding=[1.0, 0.0])
    assert len(out) == 3


def test_retrieve_k_zero_returns_empty() -> None:
    bank = Bank()
    bank.append(_entry(1, [1.0, 0.0]))
    assert retrieve(bank, query_embedding=[1.0, 0.0], k=0) == []
