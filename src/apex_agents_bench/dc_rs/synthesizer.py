"""Single synthesizer LLM call.

Faithful to Suzgun et al.'s DC-RS reference, which renders the whole
curator prompt (instructions + previous cheatsheet + retrieved cases +
current task) into ONE user message and sends it with ``system=None``.
We do the same: the single ``synthesizer_prompt.txt`` template contains
the entire prompt, and the LiteLLM call uses
``messages=[{"role":"user", ...}]`` only.

This is the ONLY LLM call DC-RS makes per task, and it runs BEFORE the
agent. No ground-truth signal is consumed anywhere in this module.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from apex_agents_bench.dc_rs.config import DCRSConfig
from apex_agents_bench.dc_rs.extract import extract_cheatsheet

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


@dataclass(frozen=True)
class SynthesizerResult:
    """Outcome of one synthesizer call."""

    cheatsheet: str
    raw_response: str
    used_fallback: bool
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float


def synthesize(
    *,
    current_cheatsheet: str,
    retrieved_cases_block: str,
    task_prompt: str,
    cfg: DCRSConfig,
) -> SynthesizerResult:
    """Run the synthesizer LLM call and return the parsed cheatsheet.

    Four keyword-only arguments — the previous cheatsheet, the formatted
    retrieved-cases block, the current task prompt, and the config.
    There is no ``criteria``, ``score``, ``gt_correct``,
    ``expected_answer``, ``judge_rationale``, ``rubric``, or ``task_id``
    parameter. This narrow signature is the load-bearing fidelity
    invariant of DC-RS: the synthesizer never sees grading data and
    never sees identifiers it could use to tag entries with the current
    task — entries must read as accumulated general knowledge.

    The synthesizer prompt template (a single file) is rendered with
    the three placeholders, and the result is sent as a single user
    message.
    """
    if cfg.synthesizer_model is None:
        raise RuntimeError(
            "DCRSConfig.synthesizer_model is None — the runner must fill it "
            "from the active AgentProfile before calling synthesize()."
        )

    template = _load_prompt("synthesizer_prompt.txt")
    user_msg = (
        template.replace("{current_cheatsheet}", current_cheatsheet)
        .replace("{retrieved_cases}", retrieved_cases_block)
        .replace("{task_prompt}", task_prompt)
    )

    import litellm

    started = time.time()
    kwargs: dict = {
        "model": cfg.synthesizer_model,
        "messages": [
            {"role": "user", "content": user_msg},
        ],
        "temperature": cfg.synthesizer_temperature,
        "max_tokens": cfg.synthesizer_max_tokens,
        "timeout": cfg.synthesizer_timeout_seconds,
    }
    # The profile's extra args may carry their own temperature / timeout /
    # reasoning_effort etc. Using ``dict.update`` lets the profile override
    # the defaults without raising a kwarg-collision TypeError, which would
    # silently abort the LLM call before LiteLLM is even reached.
    kwargs.update(cfg.synthesizer_extra_args or {})
    resp = litellm.completion(**kwargs)
    elapsed = time.time() - started

    choices = getattr(resp, "choices", None) or resp["choices"]
    msg = choices[0].message
    content = getattr(msg, "content", None) or msg["content"]
    if not isinstance(content, str):
        content = str(content)

    usage_obj = getattr(resp, "usage", None) or resp.get("usage", {}) or {}
    prompt_tokens = int(
        getattr(usage_obj, "prompt_tokens", None)
        if getattr(usage_obj, "prompt_tokens", None) is not None
        else (usage_obj.get("prompt_tokens") if isinstance(usage_obj, dict) else 0) or 0
    )
    completion_tokens = int(
        getattr(usage_obj, "completion_tokens", None)
        if getattr(usage_obj, "completion_tokens", None) is not None
        else (usage_obj.get("completion_tokens") if isinstance(usage_obj, dict) else 0) or 0
    )

    extracted = extract_cheatsheet(content, fallback=retrieved_cases_block)
    return SynthesizerResult(
        cheatsheet=extracted.cheatsheet,
        raw_response=content,
        used_fallback=extracted.used_fallback,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_seconds=round(elapsed, 2),
    )
