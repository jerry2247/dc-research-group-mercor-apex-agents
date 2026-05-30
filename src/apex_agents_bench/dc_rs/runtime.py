"""Per-run in-flight DC-RS state (apex-agents-bench).

Holds one pool and one cheatsheet slot per benchmark domain, the
embedding client, and the run directory where state is persisted. Each
domain is fully isolated: a Finance task's retrieval never sees Legal
pairs, and the Legal cheatsheet does not carry over to Finance.

Constructed once by the runner at the start of a run; mutated as tasks
complete. This is the agentic port of the sibling apex-bench runtime;
the only data-shape change is that each appended pool entry stores a
``rendered_trajectory`` (a transcript of the agent's tool-use run)
rather than a prose ``deliverable``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from apex_agents_bench.dc_rs.bank import Bank, BankEntry
from apex_agents_bench.dc_rs.config import DCRSConfig
from apex_agents_bench.dc_rs.embeddings import EmbeddingClient, LiteLLMEmbeddingClient
from apex_agents_bench.dc_rs.store import Store

log = logging.getLogger(__name__)


@dataclass
class DCRSRuntime:
    """Per-run DC-RS state, keyed by benchmark domain.

    ``banks`` and ``cheatsheets`` are lazy-populated on first access to
    a domain. Resume pre-populates them from disk in ``create()`` so
    every domain that already has on-disk state is loaded before the
    first task fires.
    """

    cfg: DCRSConfig
    run_dir: Path
    embed: EmbeddingClient
    store: Store = field(init=False)
    banks: dict[str, Bank] = field(init=False, default_factory=dict)
    cheatsheets: dict[str, str] = field(init=False, default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        cfg: DCRSConfig,
        run_dir: Path,
        embed: EmbeddingClient | None = None,
    ) -> DCRSRuntime:
        """Build a runtime and pre-load every domain that already has
        state on disk (resume).

        For each domain subdirectory under ``runs/<run>/dc_rs/``, the
        on-disk ``bank.jsonl`` is the source of truth for that domain's
        pool and ``cheatsheet.txt`` is the source of truth for the
        persistent cheatsheet slot. The results CSV is the source of
        truth only for which tasks have been completed.
        """
        if embed is None:
            embed = LiteLLMEmbeddingClient(model=cfg.embedding_model)
        store = Store.for_run(run_dir)
        rt = cls(cfg=cfg, run_dir=run_dir, embed=embed)
        rt.store = store
        for domain in store.discover_domains():
            rt.bank_for(domain)
            rt.cheatsheet_for(domain)
        return rt

    # ---- per-domain accessors ----------------------------------------

    def bank_for(self, domain: str) -> Bank:
        """Return (loading on first call) the Bank for ``domain``."""
        if domain not in self.banks:
            self.banks[domain] = self.store.load_bank(domain)
        return self.banks[domain]

    def cheatsheet_for(self, domain: str) -> str:
        """Return (loading on first call) the persistent cheatsheet slot
        for ``domain``. Returns the literal ``"(empty)"`` on the first
        task in a domain that has no prior cheatsheet on disk."""
        if domain not in self.cheatsheets:
            self.cheatsheets[domain] = self.store.read_cheatsheet(domain)
        return self.cheatsheets[domain]

    def write_cheatsheet(self, domain: str, cheatsheet: str) -> None:
        """Replace the persistent cheatsheet slot for ``domain`` in
        memory and on disk."""
        self.cheatsheets[domain] = cheatsheet
        self.store.write_cheatsheet(domain, cheatsheet)

    def archive_cheatsheet(self, domain: str, task_id: str, cheatsheet: str) -> Path:
        return self.store.archive_cheatsheet(domain, task_id, cheatsheet)

    def append_synth_log(self, domain: str, record: dict) -> None:
        self.store.append_synth_log(domain, record)

    def append_entry(
        self,
        *,
        domain: str,
        task_id: str,
        task_prompt: str,
        rendered_trajectory: str,
        prompt_embedding: list[float],
    ) -> str:
        """Mint a new BankEntry under ``domain``, persist it, and return
        its bank_id. The "answer" half of the pair is the agent's
        ``rendered_trajectory`` transcript."""
        bank = self.bank_for(domain)
        bank_id = bank.next_bank_id()
        added = bank.next_added_ordinal()
        entry = BankEntry(
            bank_id=bank_id,
            task_id=task_id,
            domain=domain,
            task_prompt=task_prompt,
            rendered_trajectory=rendered_trajectory,
            prompt_embedding=prompt_embedding,
            added=added,
        )
        bank.append(entry)
        self.store.append_bank_entry(domain, entry)
        return bank_id


def dc_rs_csv_fragment_empty() -> dict:
    """Default fragment for the DC-RS CSV columns when the run is on but
    a per-task error prevents filling individual fields."""
    return {
        "dc_rs_enabled": True,
        "dc_rs_bank_size_before": 0,
        "dc_rs_bank_size_after": 0,
        "dc_rs_retrieved_count": 0,
        "dc_rs_retrieved_bank_ids": "[]",
        "dc_rs_appended_bank_id": "",
        "synthesizer_prompt_tokens": 0,
        "synthesizer_completion_tokens": 0,
        "synthesizer_wall_seconds": 0.0,
        "synthesizer_cheatsheet_chars": 0,
        "synthesizer_used_fallback": False,
        "synthesizer_wipe_rescued": False,
        "trajectory_chars_appended": 0,
    }
