"""Unit tests for the <cheatsheet>...</cheatsheet> extractor + fallback path.

Mirrors Suzgun's ``_CHEATSHEET_RE`` which uses ``re.DOTALL | re.IGNORECASE``.
"""

from __future__ import annotations

from apex_agents_bench.dc_rs.extract import extract_cheatsheet


def test_extract_happy_path() -> None:
    raw = "preamble noise\n<cheatsheet>\nthe body here\n</cheatsheet>\ntrailing"
    out = extract_cheatsheet(raw, fallback="FALLBACK")
    assert out.used_fallback is False
    assert out.cheatsheet == "the body here"


def test_extract_missing_tags_falls_back() -> None:
    raw = "the model forgot the wrapper"
    out = extract_cheatsheet(raw, fallback="FALLBACK BODY")
    assert out.used_fallback is True
    assert out.cheatsheet == "FALLBACK BODY"


def test_extract_empty_body_falls_back() -> None:
    raw = "<cheatsheet>   </cheatsheet>"
    out = extract_cheatsheet(raw, fallback="FALLBACK")
    assert out.used_fallback is True
    assert out.cheatsheet == "FALLBACK"


def test_extract_empty_response_falls_back() -> None:
    out = extract_cheatsheet("", fallback="FALLBACK")
    assert out.used_fallback is True
    assert out.cheatsheet == "FALLBACK"


def test_extract_multiline_body_preserved() -> None:
    body = "- first note\n- second note\n  with a continuation"
    raw = f"<cheatsheet>\n{body}\n</cheatsheet>"
    out = extract_cheatsheet(raw, fallback="FALLBACK")
    assert out.used_fallback is False
    assert out.cheatsheet == body


def test_extract_case_insensitive_wrapper() -> None:
    """Suzgun's regex uses ``re.IGNORECASE`` so a model emitting
    ``<Cheatsheet>`` (or any case) still matches."""
    raw = "<Cheatsheet>body</CHEATSHEET>"
    out = extract_cheatsheet(raw, fallback="FALLBACK")
    assert out.used_fallback is False
    assert out.cheatsheet == "body"
