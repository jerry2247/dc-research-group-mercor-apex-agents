"""Per-domain isolation invariants on ``DCRSRuntime`` (agentic port).

The runtime is the layer that enforces per-domain separation. These
tests assert that two domains in the same runtime never share bank
entries or cheatsheet state — neither in memory nor on disk — and that
resume pre-loads existing on-disk state. All IO is under ``tmp_path``;
the embedder is a local stub, so nothing hits the network.
"""

from __future__ import annotations

from pathlib import Path

from apex_agents_bench.dc_rs.config import DCRSConfig
from apex_agents_bench.dc_rs.runtime import DCRSRuntime
from apex_agents_bench.dc_rs.store import EMPTY_CHEATSHEET


class _StubEmbed:
    """No-op embedder (satisfies the ``EmbeddingClient`` Protocol). The
    runtime tests don't exercise embedding — they only exercise
    per-domain dispatch on banks + cheatsheets."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def _cfg() -> DCRSConfig:
    return DCRSConfig(enabled=True, synthesizer_model="stub")


def test_runtime_starts_with_empty_per_domain_state(tmp_path: Path) -> None:
    rt = DCRSRuntime.create(cfg=_cfg(), run_dir=tmp_path, embed=_StubEmbed())
    assert rt.banks == {}
    assert rt.cheatsheets == {}
    # Asking for a domain that has nothing on disk yields an empty Bank.
    bank = rt.bank_for("Finance")
    assert bank.entries == []
    assert rt.cheatsheet_for("Finance") == EMPTY_CHEATSHEET


def test_runtime_append_entry_isolates_two_domains(tmp_path: Path) -> None:
    """Appending to Finance must not put anything in Legal — neither in
    memory nor in the on-disk per-domain dir."""
    rt = DCRSRuntime.create(cfg=_cfg(), run_dir=tmp_path, embed=_StubEmbed())
    rt.append_entry(
        domain="Finance",
        task_id="t-1",
        task_prompt="finance prompt",
        rendered_trajectory="finance trajectory",
        prompt_embedding=[1.0, 0.0],
    )
    rt.append_entry(
        domain="Finance",
        task_id="t-2",
        task_prompt="another finance prompt",
        rendered_trajectory="another finance trajectory",
        prompt_embedding=[0.9, 0.1],
    )

    fin = rt.bank_for("Finance")
    leg = rt.bank_for("Legal")
    assert [e.task_id for e in fin.entries] == ["t-1", "t-2"]
    assert leg.entries == []

    # Per-domain bank_ids restart at bank-00001 — they are NOT a global
    # counter shared across domains.
    rt.append_entry(
        domain="Legal",
        task_id="t-3",
        task_prompt="legal prompt",
        rendered_trajectory="legal trajectory",
        prompt_embedding=[0.0, 1.0],
    )
    leg = rt.bank_for("Legal")
    assert [e.bank_id for e in leg.entries] == ["bank-00001"]
    assert [e.task_id for e in leg.entries] == ["t-3"]
    # Finance still has its original two entries with their original bank_ids.
    fin = rt.bank_for("Finance")
    assert [e.bank_id for e in fin.entries] == ["bank-00001", "bank-00002"]


def test_runtime_cheatsheet_slot_isolates_two_domains(tmp_path: Path) -> None:
    rt = DCRSRuntime.create(cfg=_cfg(), run_dir=tmp_path, embed=_StubEmbed())
    rt.write_cheatsheet("Finance", "FINANCE BODY")
    rt.write_cheatsheet("Legal", "LEGAL BODY")
    assert rt.cheatsheet_for("Finance") == "FINANCE BODY"
    assert rt.cheatsheet_for("Legal") == "LEGAL BODY"

    # Overwriting Finance does not touch Legal.
    rt.write_cheatsheet("Finance", "FINANCE V2")
    assert rt.cheatsheet_for("Finance") == "FINANCE V2"
    assert rt.cheatsheet_for("Legal") == "LEGAL BODY"


def test_runtime_resume_preloads_existing_per_domain_state(tmp_path: Path) -> None:
    """A fresh runtime built over a run_dir that already has per-domain
    state must pre-load every domain's bank + cheatsheet so the first
    task of the resumed run sees them."""
    rt1 = DCRSRuntime.create(cfg=_cfg(), run_dir=tmp_path, embed=_StubEmbed())
    rt1.append_entry(
        domain="Finance",
        task_id="t-1",
        task_prompt="p",
        rendered_trajectory="tr",
        prompt_embedding=[1.0, 0.0],
    )
    rt1.write_cheatsheet("Finance", "FIN CHEAT")
    rt1.append_entry(
        domain="Legal",
        task_id="t-2",
        task_prompt="p",
        rendered_trajectory="tr",
        prompt_embedding=[0.0, 1.0],
    )
    rt1.write_cheatsheet("Legal", "LEG CHEAT")

    rt2 = DCRSRuntime.create(cfg=_cfg(), run_dir=tmp_path, embed=_StubEmbed())
    assert sorted(rt2.banks.keys()) == ["Finance", "Legal"]
    assert sorted(rt2.cheatsheets.keys()) == ["Finance", "Legal"]
    assert [e.task_id for e in rt2.bank_for("Finance").entries] == ["t-1"]
    assert [e.task_id for e in rt2.bank_for("Legal").entries] == ["t-2"]
    assert rt2.cheatsheet_for("Finance") == "FIN CHEAT"
    assert rt2.cheatsheet_for("Legal") == "LEG CHEAT"


def test_runtime_append_entry_stamps_domain_on_bank_entry(tmp_path: Path) -> None:
    """New bank entries written by the runtime carry the domain they
    were produced under — diagnostic provenance for downstream
    analysis even if files are later merged."""
    rt = DCRSRuntime.create(cfg=_cfg(), run_dir=tmp_path, embed=_StubEmbed())
    rt.append_entry(
        domain="Medicine",
        task_id="t-1",
        task_prompt="p",
        rendered_trajectory="tr",
        prompt_embedding=[1.0],
    )
    e = rt.bank_for("Medicine").entries[0]
    assert e.domain == "Medicine"
    assert e.rendered_trajectory == "tr"
