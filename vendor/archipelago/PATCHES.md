# Vendor patch log

Every modification to files under `vendor/archipelago/` MUST be recorded
here. There are **zero active patches** to the Archipelago **source
code** -- the agent and grading runners accept arbitrary LiteLLM model
strings without an allow-list check, so the test-model surface
(`gpt-5.5-*`, `grok-4.3-*`) works on the unmodified runner source. The
one active patch is a **build-time** change to `environment/Dockerfile`
(Patch 001 below), which compiles the `sandbox_fs.so` library the
`code_execution_server` requires at startup.

## Active patches

### Patch 002 — `agents/runner/utils/llm.py` routes gpt-5.5 through the Responses API

**File**: `vendor/archipelago/agents/runner/utils/llm.py`
**Status**: active
**Rationale**: OpenAI/Azure `gpt-5.5` is a reasoning model. Over the
chat-completions API it returns reasoning text but **drops structured
`tool_calls`** when a `reasoning_effort` is set together with the agent's
"think before acting" system prompt — proven empirically: chat completions
return `finish_reason=stop` with no `tool_calls`, while the Responses API
returns the `function_call` correctly. The react_toolbelt agent then never
receives a tool call, loops to the step cap, and every task fails with
`status=failed` and no gradeable output. LiteLLM 1.83.0 contains the bridge
rule ("OpenAI/Azure gpt-5.4+ chat-completions calls with both tools +
reasoning_effort must be bridged to the Responses API") in
`responses_api_bridge_check`, but `litellm.completion()`'s call site does
not forward `tools`/`reasoning_effort` to that check, so the auto-bridge
never fires. We bridge explicitly for the **gpt-5.5 family only**: in
`generate_response`, when `_is_reasoning_bridge_model(model)` and tools are
present, call `litellm.aresponses(...)` and translate its `output` items
back into a chat-shaped `ModelResponse` (with `choices[0].message.tool_calls`
and chat-shaped `usage`). Every other model (grok-4.3, deepseek, gpt-4o,
...) is gated out and takes the unmodified chat-completions path
byte-for-byte.

**Shape** (one inserted block of helpers near the top of the module, and
one gated branch inside `generate_response` immediately before the existing
`if stream:` block; **no existing line is modified or removed** — the diff
is 118 insertions, 0 deletions):

```python
# near module top
def _is_reasoning_bridge_model(model: str) -> bool:
    m = model.lower()
    return "gpt-5.5" in m or "gpt-5-5" in m
# ... _chat_tools_to_responses_tools(...), _responses_output_to_model_response(...)

# inside generate_response, before `if stream:`
if _is_reasoning_bridge_model(model) and tools:
    resp_kwargs = {"model": model, "input": messages,
                   "tools": _chat_tools_to_responses_tools(tools),
                   "timeout": llm_response_timeout, **extra_args}
    raw = await aresponses(**resp_kwargs)
    return _responses_output_to_model_response(model, raw)
```

**Resync note**: When bumping the upstream pin, check whether (a) the
vendored agent gains native reasoning-model / Responses-API tool-call
handling, or (b) the installed LiteLLM version fixes the
`responses_api_bridge_check` call site to forward `tools`/`reasoning_effort`
from `completion()` (making the manual bridge unnecessary). If either is
true, drop this patch and move it to "Retired patches". The
chat-completions path for grok/others must remain unchanged.

**Regression test**:
`tests/test_fidelity.py::test_vendor_gpt55_responses_bridge_patch_present`
asserts the bridge helpers + the `aresponses(` branch are present in the
vendored `llm.py` and that grok is gated out.

### Patch 001 — `environment/Dockerfile` adds gcc compile step for `sandbox_fs.so`

**File**: `vendor/archipelago/environment/Dockerfile`
**Status**: active
**Rationale**: At the pinned vendor commit (3f4a8234), the production
`environment/Dockerfile` ships `sandbox_fs.c` (the LD_PRELOAD library used
by `code_execution_server`) but does not compile it. The server's
`verify_sandbox_available()` then raises `RuntimeError` at startup, no
tools are published, and the MCP gateway's `/apps` readiness check times
out after 300s with a 503. The vendor's own `mcp_servers/code/Dockerfile.gvisor-test`
contains the missing build step — we transplant it into the production
Dockerfile.

**Diff** (one inserted block, immediately after the `pdfs` install):

```dockerfile
# vendored-patch: compile sandbox_fs.so for the code_execution_server.
RUN mkdir -p /app/lib && gcc -shared -fPIC -O2 \
    -o /app/lib/sandbox_fs.so \
    /app/mcp_servers/code/mcp_servers/code_execution_server/sandbox_fs.c \
    -ldl -lpthread
```

**Resync note**: When bumping the upstream pin, check whether
`environment/Dockerfile` upstream has gained a sandbox_fs.so build step. If
yes, drop this patch and move it to "Retired patches" with the upstream
commit that fixed it. If no, keep this patch and verify the file path
`/app/mcp_servers/code/mcp_servers/code_execution_server/sandbox_fs.c`
still exists in the new vendor source.

**Regression test**: TBD — a fidelity test should assert the
`# vendored-patch: compile sandbox_fs.so` marker is present in the Dockerfile.

## Why no patch is needed for the test/judge models

Archipelago does not gate model ids through an allow-list. Both call
sites pass the model string straight to
`litellm.acompletion(model=<string>)`, so `gpt-5.5`, `grok-4.3`, and
`deepseek-v4-pro` route by prefix on the unmodified runner source:

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
