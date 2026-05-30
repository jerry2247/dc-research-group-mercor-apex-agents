"""On-disk persistence for the DC-RS subsystem (per-domain layout).

One pool and one cheatsheet slot per benchmark domain. The on-disk
layout mirrors that:

    runs/<run>/dc_rs/
      <Domain>/                            # e.g. Finance, Legal, Medicine, Consulting
        bank.jsonl                         # source of truth for that domain's pool
        cheatsheet.txt                     # most recent synthesized cheatsheet for the domain
        cheatsheets/task_<id>.txt          # per-task archive (diagnostic only)
        synthesizer_log.jsonl              # per-task synth call diagnostics

``bank.jsonl`` and ``cheatsheet.txt`` are load-bearing for resume; the
``cheatsheets/`` archive and ``synthesizer_log.jsonl`` are diagnostic.
Each domain's directory is fully independent: no shared file, no
cross-domain reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from apex_agents_bench.dc_rs.bank import Bank, BankEntry

EMPTY_CHEATSHEET = "(empty)"


@dataclass
class Store:
    """File-system view of a run's per-domain DC-RS state.

    All methods take ``domain`` as their first argument; nothing in this
    store crosses domains.
    """

    root: Path

    @classmethod
    def for_run(cls, run_dir: Path) -> Store:
        root = run_dir / "dc_rs"
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    # ---- per-domain directory ----------------------------------------

    def domain_dir(self, domain: str) -> Path:
        d = self.root / domain
        d.mkdir(parents=True, exist_ok=True)
        (d / "cheatsheets").mkdir(parents=True, exist_ok=True)
        return d

    def discover_domains(self) -> list[str]:
        """Return the sorted list of domain subdirectories on disk.

        Used by the runtime on resume to pre-load any domain that
        already has state. A directory counts as a domain iff it is a
        direct child of ``self.root`` (i.e. it sits next to other
        ``<Domain>/`` subdirs); the ``cheatsheets/`` per-domain archive
        lives INSIDE a domain dir, not next to it, so this method does
        not need to filter it out.
        """
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    # ---- pool --------------------------------------------------------

    def bank_path(self, domain: str) -> Path:
        return self.domain_dir(domain) / "bank.jsonl"

    def load_bank(self, domain: str) -> Bank:
        """Read every line of the domain's bank.jsonl into a fresh Bank."""
        bank = Bank()
        path = self.bank_path(domain)
        if not path.is_file():
            return bank
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                bank.append(BankEntry.model_validate_json(line))
        return bank

    def append_bank_entry(self, domain: str, entry: BankEntry) -> None:
        with self.bank_path(domain).open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    # ---- cheatsheet slot ---------------------------------------------

    def cheatsheet_path(self, domain: str) -> Path:
        return self.domain_dir(domain) / "cheatsheet.txt"

    def read_cheatsheet(self, domain: str) -> str:
        path = self.cheatsheet_path(domain)
        if not path.is_file():
            return EMPTY_CHEATSHEET
        text = path.read_text(encoding="utf-8")
        return text if text.strip() else EMPTY_CHEATSHEET

    def write_cheatsheet(self, domain: str, cheatsheet: str) -> None:
        self.cheatsheet_path(domain).write_text(cheatsheet, encoding="utf-8")

    def archive_cheatsheet(self, domain: str, task_id: str, cheatsheet: str) -> Path:
        path = self.domain_dir(domain) / "cheatsheets" / f"task_{task_id}.txt"
        path.write_text(cheatsheet, encoding="utf-8")
        return path

    # ---- synthesizer log ---------------------------------------------

    def synth_log_path(self, domain: str) -> Path:
        return self.domain_dir(domain) / "synthesizer_log.jsonl"

    def append_synth_log(self, domain: str, record: dict) -> None:
        with self.synth_log_path(domain).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
