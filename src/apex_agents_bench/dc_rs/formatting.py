"""Render retrieved pool entries as the synthesizer input block.

Faithful to Suzgun et al.'s DC-RS reference ``_format_entries``, adapted
to the agentic setting: each prior pair is a ``(task_prompt,
rendered_trajectory)`` case rather than a ``(task_prompt, deliverable)``
pair, so each rendered block shows the prior task AND a transcript of
what the agent did on it.

  * empty pool → the literal string ``"(empty)"``;
  * non-empty → a preamble note framing the prior cases as evidence
    (not authoritative answers) followed by per-case blocks in REVERSED
    order (most-similar last so it sits closest to the current case the
    synthesizer's prompt template appends below), and a
    ``#### PRIOR CASES (END)`` footer.
"""

from __future__ import annotations

from apex_agents_bench.dc_rs.retriever import Retrieved


def format_retrieved_cases(retrieved: list[Retrieved]) -> str:
    """Return the markdown block that fills ``{retrieved_cases}``.

    When the pool is empty (the first task in a domain), returns the
    literal string ``"(empty)"`` — matching the reference behaviour so
    the synthesizer prompt template substitutes cleanly without any
    extra wrapper headers.
    """
    if not retrieved:
        return "(empty)"

    chunks: list[str] = [
        "### PRIOR CASES (START)\n\n"
        "Note: The task/trajectory pairs listed below are taken from previous "
        "test cases and are meant to assist you in understanding potential "
        "solution strategies, tool-use patterns, and pitfalls. While they can "
        "offer insight and inspiration, they should not be blindly copied, as "
        "they may contain errors or may not fit the current case. Approach them "
        "with a critical mindset — analyse their logic, verify their "
        "correctness, and adapt them as needed."
    ]
    # Reversed so the most-similar retrieved case sits closest to the
    # current case appended below by the synthesizer template.
    for idx, r in enumerate(reversed(retrieved), start=1):
        entry = r.entry
        chunks.append(
            f"### PRIOR CASE #{idx} (similarity {r.similarity:.2f})\n\n"
            f"#### Task:\n{entry.task_prompt}\n\n"
            f"#### What the agent did (trajectory):\n{entry.rendered_trajectory}\n---"
        )
    chunks.append("#### PRIOR CASES (END)")
    return "\n\n".join(chunks).strip()
