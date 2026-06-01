"""Unit tests for the DL curator: parse_ledger_updates + apply_ops + curate."""

from __future__ import annotations

import sys
import types

import pytest

from apex_agents_bench.dl.config import DLConfig
from apex_agents_bench.dl.curator import (
    CuratedOp,
    apply_ops,
    curate,
    parse_ledger_updates,
)
from apex_agents_bench.dl.entry import DLLedger


# --- fake embedding client (returns one 2-vector per input text) -----------
class _Embed:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(i + 1), 1.0] for i in range(len(texts))]


# --- fake litellm (records kwargs) -----------------------------------------
class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMsg(content)


class _FakeUsage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _FakeResp:
    def __init__(self, content: str, p: int, c: int) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(p, c)


def _install_fake_litellm(monkeypatch: pytest.MonkeyPatch, content: str, capture: dict) -> None:
    def fake_completion(**kwargs):
        capture.update(kwargs)
        return _FakeResp(content, p=11, c=22)

    fake_litellm = types.SimpleNamespace(completion=fake_completion)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)


def _seed(led: DLLedger, n: int) -> None:
    for i in range(1, n + 1):
        led.add(
            type="snippet",
            content=f"c{i}",
            source_problem=f"s{i}",
            content_embedding=[float(i), 0.0],
            source_problem_embedding=[0.0, float(i)],
            created=i,
        )


# ---- parse ----------------------------------------------------------------


def test_parse_no_block_returns_error() -> None:
    ops, err = parse_ledger_updates("no block here")
    assert ops == []
    assert err is not None


def test_parse_mixed_crud_batch() -> None:
    text = """
    <ledger_updates>
    [
      {"op": "CREATE", "type": "pitfall", "content": "x", "source_problem": "sp"},
      {"op": "UPDATE", "entry_id": "entry-2", "content": "y"},
      {"op": "DELETE", "entry_id": "entry-3"}
    ]
    </ledger_updates>
    """
    ops, err = parse_ledger_updates(text)
    assert err is None
    assert [o.op for o in ops] == ["CREATE", "UPDATE", "DELETE"]
    assert ops[0].type == "pitfall"
    assert ops[0].source_problem == "sp"
    assert ops[1].entry_id == "entry-2"
    assert ops[2].entry_id == "entry-3"


def test_parse_drops_create_with_invalid_type() -> None:
    text = (
        "<ledger_updates>"
        '[{"op":"CREATE","type":"banana","content":"x","source_problem":"s"}]'
        "</ledger_updates>"
    )
    ops, err = parse_ledger_updates(text)
    assert err is None
    assert ops == []


def test_parse_drops_create_missing_required_fields() -> None:
    text = (
        "<ledger_updates>"
        '[{"op":"CREATE","type":"snippet","content":"x"}]'  # no source_problem
        "</ledger_updates>"
    )
    ops, _ = parse_ledger_updates(text)
    assert ops == []


def test_parse_accepts_empty_array() -> None:
    ops, err = parse_ledger_updates("<ledger_updates>[]</ledger_updates>")
    assert err is None
    assert ops == []


def test_parse_update_with_bad_type_drops_the_type_only() -> None:
    text = (
        "<ledger_updates>"
        '[{"op":"UPDATE","entry_id":"entry-1","content":"y","type":"nope"}]'
        "</ledger_updates>"
    )
    ops, _ = parse_ledger_updates(text)
    assert len(ops) == 1
    assert ops[0].type == ""  # invalid type is dropped, update still stands


# ---- apply ----------------------------------------------------------------


def test_apply_create_embeds_both_axes_no_dedup() -> None:
    led = DLLedger(domain="D")
    emb = _Embed()
    ops = [
        CuratedOp(op="CREATE", type="snippet", content="a", source_problem="sa"),
        CuratedOp(op="CREATE", type="snippet", content="a", source_problem="sa"),  # identical
    ]
    stats = apply_ops(ledger=led, ops=ops, embed=emb, current_ordinal=1)
    # No dedup: BOTH creates land.
    assert stats.create == 2
    assert len(led.active_entries()) == 2
    # each CREATE embeds [content, source_problem] → 2 texts per call
    assert all(len(c) == 2 for c in emb.calls)


def test_apply_update_reembeds_content_only() -> None:
    led = DLLedger(domain="D")
    _seed(led, 1)
    emb = _Embed()
    ops = [CuratedOp(op="UPDATE", entry_id="entry-1", content="new body")]
    stats = apply_ops(ledger=led, ops=ops, embed=emb, current_ordinal=5)
    assert stats.update == 1
    assert led.get("entry-1").content == "new body"
    assert led.get("entry-1").updated == 5
    # UPDATE embeds just [content]
    assert emb.calls == [["new body"]]


def test_apply_delete_then_create_ordering() -> None:
    led = DLLedger(domain="D")
    _seed(led, 2)
    emb = _Embed()
    ops = [
        CuratedOp(op="CREATE", type="formula", content="new", source_problem="s"),
        CuratedOp(op="DELETE", entry_id="entry-1"),
    ]
    stats = apply_ops(ledger=led, ops=ops, embed=emb, current_ordinal=3)
    assert stats.delete == 1
    assert stats.create == 1
    ids = {e.entry_id for e in led.active_entries()}
    assert "entry-1" not in ids  # deleted
    assert "entry-2" in ids
    assert "entry-3" in ids  # the create (next ordinal after 2 seeded)


def test_apply_invalid_ids_counted_skipped() -> None:
    led = DLLedger(domain="D")
    _seed(led, 1)
    emb = _Embed()
    ops = [
        CuratedOp(op="UPDATE", entry_id="entry-99", content="x"),
        CuratedOp(op="DELETE", entry_id="entry-77"),
    ]
    stats = apply_ops(ledger=led, ops=ops, embed=emb, current_ordinal=2)
    assert stats.skipped_invalid_entry_id == 2
    assert stats.update == 0
    assert stats.delete == 0


def test_apply_update_on_deleted_entry_is_skipped() -> None:
    led = DLLedger(domain="D")
    _seed(led, 1)
    emb = _Embed()
    ops = [
        CuratedOp(op="DELETE", entry_id="entry-1"),
        CuratedOp(op="UPDATE", entry_id="entry-1", content="x"),
    ]
    stats = apply_ops(ledger=led, ops=ops, embed=emb, current_ordinal=2)
    # delete first → update then finds it inactive → skipped
    assert stats.delete == 1
    assert stats.update == 0
    assert stats.skipped_invalid_entry_id == 1


def test_apply_create_carries_type_into_entry() -> None:
    led = DLLedger(domain="D")
    emb = _Embed()
    apply_ops(
        ledger=led,
        ops=[CuratedOp(op="CREATE", type="environment", content="c", source_problem="s")],
        embed=emb,
        current_ordinal=1,
    )
    assert led.active_entries()[0].type == "environment"


# ---- curate (LLM call wiring) ---------------------------------------------


def test_curate_requires_model() -> None:
    led = DLLedger(domain="D")
    with pytest.raises(RuntimeError):
        curate(led, [], "task", "traj", cfg=DLConfig(enabled=True, curator_model=None))


def test_curate_single_user_message_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict = {}
    _install_fake_litellm(
        monkeypatch,
        content='<ledger_updates>[{"op":"DELETE","entry_id":"entry-1"}]</ledger_updates>',
        capture=capture,
    )
    led = DLLedger(domain="D")
    cfg = DLConfig(enabled=True, curator_model="openai/gpt-5.5", curator_extra_args={})
    res = curate(led, [], "the task prompt", "the trajectory", cfg=cfg)
    # single user message, system None (faithful to DC-RS synthesizer shape)
    msgs = capture["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "the task prompt" in msgs[0]["content"]
    assert "the trajectory" in msgs[0]["content"]
    assert res.prompt_tokens == 11
    assert res.completion_tokens == 22
    assert [o.op for o in res.ops] == ["DELETE"]


def test_curate_profile_extra_args_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict = {}
    _install_fake_litellm(
        monkeypatch, content="<ledger_updates>[]</ledger_updates>", capture=capture
    )
    cfg = DLConfig(
        enabled=True,
        curator_model="openai/gpt-5.5",
        curator_extra_args={"reasoning_effort": "high", "temperature": 0.3},
    )
    curate(DLLedger(domain="D"), [], "t", "tr", cfg=cfg)
    assert capture["reasoning_effort"] == "high"
    # extra_args override the default temperature without a kwarg collision
    assert capture["temperature"] == 0.3
