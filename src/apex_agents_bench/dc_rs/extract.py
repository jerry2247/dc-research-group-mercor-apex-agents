"""Extract the inner cheatsheet body from the synthesizer's raw response.

Faithful to Suzgun et al.'s DC-RS reference (``dc_rs.py:_CHEATSHEET_RE``
and ``_extract_cheatsheet``):

    _CHEATSHEET_RE = re.compile(
        r"<cheatsheet>\\s*(.*?)\\s*</cheatsheet>",
        re.DOTALL | re.IGNORECASE,
    )

If the wrapper is present, the inner text becomes the cheatsheet.
If it is missing, the runtime falls back to the verbatim
retrieved-cases block — degradation, not failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CHEATSHEET_RE = re.compile(
    r"<cheatsheet>\s*(.*?)\s*</cheatsheet>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractResult:
    cheatsheet: str
    used_fallback: bool


def extract_cheatsheet(raw_response: str, *, fallback: str) -> ExtractResult:
    """Return the inner cheatsheet body, or the fallback string if the
    synthesizer omitted the wrapper tag."""
    if raw_response:
        match = _CHEATSHEET_RE.search(raw_response)
        if match:
            inner = match.group(1).strip()
            if inner:
                return ExtractResult(cheatsheet=inner, used_fallback=False)
    return ExtractResult(cheatsheet=fallback, used_fallback=True)
