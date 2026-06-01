"""Per-domain isolation + ordinal/persist invariants on DLRuntime."""

from __future__ import annotations

from pathlib import Path

from apex_agents_bench.dl.config import DLConfig
from apex_agents_bench.dl.runtime import DLRuntime, dl_csv_fragment_empty


class _Embed:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _rt(tmp_path: Path) -> DLRuntime:
    return DLRuntime.create(cfg=DLConfig(enabled=True), run_dir=tmp_path, embed=_Embed())


def test_ledgers_are_per_domain_isolated(tmp_path: Path) -> None:
    rt = _rt(tmp_path)
    fin = rt.ledger_for("Finance")
    law = rt.ledger_for("Law")
    assert fin is not law
    fin.add(
        type="snippet",
        content="c",
        source_problem="s",
        content_embedding=[1.0, 0.0],
        source_problem_embedding=[0.0, 1.0],
        created=1,
    )
    # Law ledger is untouched by a Finance write
    assert law.active_entries() == []
    assert fin.domain == "Finance"
    assert law.domain == "Law"


def test_current_ordinal_increments_per_domain(tmp_path: Path) -> None:
    rt = _rt(tmp_path)
    assert rt.current_ordinal_for("Finance") == 1
    assert rt.current_ordinal_for("Finance") == 2
    # independent counter per domain
    assert rt.current_ordinal_for("Law") == 1


def test_persist_then_resume_reloads_state(tmp_path: Path) -> None:
    rt = _rt(tmp_path)
    led = rt.ledger_for("Finance")
    led.add(
        type="pitfall",
        content="keepme",
        source_problem="s",
        content_embedding=[1.0, 0.0],
        source_problem_embedding=[0.0, 1.0],
        created=1,
    )
    rt.current_ordinal_for("Finance")  # → 1
    rt.persist("Finance")

    # a fresh runtime over the same run_dir pre-loads the snapshot
    rt2 = DLRuntime.create(cfg=DLConfig(enabled=True), run_dir=tmp_path, embed=_Embed())
    led2 = rt2.ledger_for("Finance")
    assert len(led2.active_entries()) == 1
    assert led2.active_entries()[0].content == "keepme"
    # next ordinal continues past the loaded snapshot index
    assert rt2.next_ordinal["Finance"] == 1


def test_csv_fragment_empty_has_all_columns() -> None:
    frag = dl_csv_fragment_empty()
    from apex_agents_bench.runner import _DL_CSV_COLUMNS

    assert set(frag.keys()) == set(_DL_CSV_COLUMNS)
