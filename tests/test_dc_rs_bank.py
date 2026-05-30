"""Unit tests for DC-RS bank entry shape and Bank sequencing (agentic port).

The agentic ``BankEntry`` stores a ``rendered_trajectory`` transcript in
place of the prose repo's ``deliverable``.
"""

from __future__ import annotations

import json

from apex_agents_bench.dc_rs.bank import Bank, BankEntry


def _make_entry(idx: int, *, domain: str = "Finance") -> BankEntry:
    return BankEntry(
        bank_id=f"bank-{idx:05d}",
        task_id=f"t-{idx}",
        domain=domain,
        task_prompt=f"prompt {idx}",
        rendered_trajectory=f"trajectory {idx}",
        prompt_embedding=[float(idx), float(idx + 1), float(idx + 2)],
        added=idx - 1,
    )


def test_bank_entry_roundtrip_via_json() -> None:
    e = _make_entry(7)
    raw = e.model_dump_json()
    parsed = BankEntry.model_validate_json(raw)
    assert parsed.bank_id == "bank-00007"
    assert parsed.task_id == "t-7"
    assert parsed.domain == "Finance"
    assert parsed.task_prompt == "prompt 7"
    assert parsed.rendered_trajectory == "trajectory 7"
    assert parsed.prompt_embedding == [7.0, 8.0, 9.0]
    assert parsed.added == 6


def test_bank_entry_ignores_extra_fields() -> None:
    payload = {
        "bank_id": "bank-00001",
        "task_id": "t-1",
        "domain": "Legal",
        "task_prompt": "p",
        "rendered_trajectory": "tr",
        "prompt_embedding": [1.0],
        "added": 0,
        "future_field": "value the schema does not know about",
    }
    e = BankEntry.model_validate_json(json.dumps(payload))
    assert e.bank_id == "bank-00001"
    assert e.domain == "Legal"


def test_bank_entry_domain_defaults_to_empty_for_backward_compat() -> None:
    """A pre-per-domain bank.jsonl line (no ``domain`` key) must still
    load — the field defaults to ``""``."""
    payload = {
        "bank_id": "bank-00001",
        "task_id": "t-1",
        "task_prompt": "p",
        "rendered_trajectory": "tr",
        "prompt_embedding": [1.0],
        "added": 0,
    }
    e = BankEntry.model_validate_json(json.dumps(payload))
    assert e.domain == ""


def test_bank_sequencing_is_per_domain_unit() -> None:
    """``Bank`` is the per-domain unit. Each domain holds its own Bank in
    ``DCRSRuntime.banks``; the Bank class itself does not take a domain
    parameter because the per-domain dispatch happens at the runtime
    layer."""
    bank = Bank()
    assert bank.entries == []
    assert bank.next_bank_id() == "bank-00001"
    assert bank.next_added_ordinal() == 0
    bank.append(_make_entry(1))
    assert bank.next_bank_id() == "bank-00002"
    assert bank.next_added_ordinal() == 1
    bank.append(_make_entry(2))
    assert bank.next_bank_id() == "bank-00003"
    assert bank.next_added_ordinal() == 2
    assert [e.task_id for e in bank.entries] == ["t-1", "t-2"]
