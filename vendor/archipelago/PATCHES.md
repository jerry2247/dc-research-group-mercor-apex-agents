# Vendor patch log

Every modification to files under `vendor/archipelago/` MUST be recorded
here. As of the vendoring date there are **zero active patches** -- the
Archipelago agent and grading runners accept arbitrary LiteLLM model
strings without an allow-list check, so the test-model surface
(`gpt-5.5-*`, `grok-4.3-*`) works on the unmodified vendor source.

## Active patches

_(none)_

## Why no patch is needed for `gpt-5.5` / `grok-4.3`

`apex-bench`'s sister vendor (`vendor/apex_evals/`) required a 2-line patch
to `src/call_llm/litellm_client.py` because Mercor's `apex-evals` harness
gates model ids through a `MODEL_MAPPINGS` exact-match dict before calling
LiteLLM. Archipelago is different: both call sites pass the model string
straight to `litellm.acompletion(model=<string>)`:

```python
# vendor/archipelago/agents/runner/utils/llm.py:134-164
kwargs: dict[str, Any] = {
    "model": model,
    "messages": messages,
    "timeout": llm_response_timeout,
    **extra_args,
}
...
response = await acompletion(**kwargs)

# vendor/archipelago/grading/runner/utils/llm.py:175-194
kwargs: dict[str, Any] = {
    "model": model,
    "messages": messages,
    "timeout": timeout,
    **(extra_args or {}),
}
response = await litellm.acompletion(**kwargs)
```

`tests/test_fidelity.py::test_archipelago_passes_model_string_verbatim`
greps both call sites to guard this invariant.

## Diff policy (re-stated)

If a future patch becomes unavoidable:

1. Add the diff with an inline `# vendored-patch: <reason>` comment.
2. Record it here with: patch number, file, status (active|retired), one-line
   rationale, exact diff in a fenced block, and a resync note explaining
   what to check when bumping the upstream pin.
3. Add a regression test in `tests/test_fidelity.py` asserting the marker
   is still present and asserting the surrounding code shape.
4. Update `vendor/archipelago/UPSTREAM.md` if the patch affects the resync
   recipe.

## Retired patches

_(none)_
