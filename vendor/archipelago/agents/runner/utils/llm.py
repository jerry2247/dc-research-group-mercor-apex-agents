"""LLM utilities for agents using LiteLLM."""

from typing import Any

import litellm
from litellm import acompletion, aresponses
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.files.main import ModelResponse
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from runner.agents.models import LitellmAnyMessage
from runner.utils.decorators import with_retry
from runner.utils.settings import get_settings

settings = get_settings()

# Configure LiteLLM proxy routing if configured
if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
    litellm.use_litellm_proxy = True


# --- PATCH (apex-agents-bench): GPT-5.5 reasoning-model tool-calling --------
# OpenAI/Azure gpt-5.5 is a reasoning model. Over the chat-completions API it
# emits reasoning text but DROPS structured tool_calls when a reasoning effort
# is set together with the agent's "think before acting" system prompt (proven
# empirically: chat returns finish_reason=stop with no tool_calls; the
# Responses API returns the function_call correctly). LiteLLM 1.83.0 contains
# the bridge rule ("gpt-5.4+ chat calls with tools + reasoning_effort must be
# bridged to the Responses API") but its completion() call site never forwards
# tools/reasoning_effort to the bridge check, so the auto-bridge never fires.
# We bridge explicitly for the gpt-5.5 family ONLY: call the Responses API and
# translate its output back into a chat-shaped ModelResponse so the agent loop
# (history threading, tool dispatch, usage tracking) is byte-identical to the
# chat path. Every other model (grok, etc.) is untouched.
def _is_reasoning_bridge_model(model: str) -> bool:
    """True only for the gpt-5.5 family (any provider prefix). All other
    models take the unmodified chat-completions path."""
    m = model.lower()
    return "gpt-5.5" in m or "gpt-5-5" in m


def _chat_tools_to_responses_tools(
    tools: list[ChatCompletionToolParam],
) -> list[dict[str, Any]]:
    """Chat tool param -> Responses API flat tool schema."""
    out: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", {})  # type: ignore[union-attr]
        out.append(
            {
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return out


def _responses_output_to_model_response(
    model: str, raw_response: Any
) -> ModelResponse:
    """Translate a Responses API result into a chat-shaped ModelResponse with
    choices[0].message.tool_calls and chat-shaped usage, so all downstream
    chat-path consumers work unchanged."""
    rd = (
        raw_response.model_dump()
        if hasattr(raw_response, "model_dump")
        else dict(raw_response)
    )
    tool_calls: list[dict[str, Any]] = []
    content_parts: list[str] = []
    for item in rd.get("output", []) or []:
        itype = item.get("type")
        if itype == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )
        elif itype == "message":
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    content_parts.append(c.get("text", ""))
    usage = rd.get("usage", {}) or {}
    prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
    completion_tokens = (
        usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0
    )
    return ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": tool_calls or None,
                },
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens", 0)
            or (prompt_tokens + completion_tokens),
        },
        model=model,
    )
# --- END PATCH -------------------------------------------------------------


def _is_context_window_error(e: Exception) -> bool:
    """
    Detect context window exceeded errors that LiteLLM doesn't properly classify.

    Some providers (notably Gemini) return context window errors as BadRequestError
    instead of ContextWindowExceededError. This predicate catches those cases
    by checking the error message content.

    Known error patterns:
    - Gemini: "input token count exceeds the maximum number of tokens allowed"
    - OpenAI: "context_length_exceeded" (usually caught as ContextWindowExceededError)
    - Anthropic: "prompt is too long" (usually caught as ContextWindowExceededError)
    """
    error_str = str(e).lower()

    # Common patterns indicating context/token limit exceeded
    context_patterns = [
        "token count exceeds",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "maximum number of tokens",
        "prompt is too long",
        "input too long",
        "exceeds the model's maximum context",
    ]

    return any(pattern in error_str for pattern in context_patterns)


def _is_non_retriable_bad_request(e: Exception) -> bool:
    """
    Detect BadRequestErrors that are deterministic and should NOT be retried.

    These are configuration/validation errors that will always fail regardless
    of retry attempts. Retrying wastes time and resources.

    Note: Patterns must be specific enough to avoid matching transient errors
    like rate limits (e.g., "maximum of 100 requests" should NOT match).
    """
    error_str = str(e).lower()

    non_retriable_patterns = [
        # Tool count errors - be specific to avoid matching rate limits
        "tools are supported",  # "Maximum of 128 tools are supported"
        "too many tools",
        # Model/auth errors
        "model not found",
        "does not exist",
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "unsupported parameter",
        "unsupported value",
    ]

    return any(pattern in error_str for pattern in non_retriable_patterns)


def _should_skip_retry(e: Exception) -> bool:
    """Combined check for all non-retriable errors."""
    return _is_context_window_error(e) or _is_non_retriable_bad_request(e)


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_should_skip_retry,
)
async def generate_response(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[ChatCompletionToolParam],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> ModelResponse:
    """
    Generate a response from the LLM with retry logic.

    Args:
        model: The model identifier to use
        messages: The conversation messages (input AllMessageValues or output Message)
        tools: Available tools for the model to call
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments to pass to the completion call
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The model response
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": llm_response_timeout,
        **extra_args,
    }

    if tools:
        kwargs["tools"] = tools

    # If LiteLLM proxy is configured, add tracking tags
    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        tags = ["service:trajectory"]
        if trajectory_id:
            tags.append(f"trajectory_id:{trajectory_id}")
        kwargs["extra_body"] = {"tags": tags}

    # PATCH (apex-agents-bench): gpt-5.5 reasoning models drop tool_calls over
    # chat-completions; route them through the Responses API and translate the
    # output back to a chat-shaped ModelResponse. Only gpt-5.5 is affected;
    # every other model falls through to the unchanged chat path below.
    if _is_reasoning_bridge_model(model) and tools:
        resp_kwargs: dict[str, Any] = {
            "model": model,
            "input": messages,
            "tools": _chat_tools_to_responses_tools(tools),
            "timeout": llm_response_timeout,
            **extra_args,
        }
        if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
            resp_kwargs["api_base"] = settings.LITELLM_PROXY_API_BASE
            resp_kwargs["api_key"] = settings.LITELLM_PROXY_API_KEY
            tags = ["service:trajectory"]
            if trajectory_id:
                tags.append(f"trajectory_id:{trajectory_id}")
            resp_kwargs["extra_body"] = {"tags": tags}
        raw = await aresponses(**resp_kwargs)
        return _responses_output_to_model_response(model, raw)

    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        stream_iter: Any = await acompletion(**kwargs)
        chunks: list[ModelResponse] = []
        async for chunk in stream_iter:
            chunks.append(chunk)
        rebuilt = litellm.stream_chunk_builder(chunks, messages=messages)
        if rebuilt is None:
            raise RuntimeError("stream_chunk_builder returned None — empty stream")
        return ModelResponse.model_validate(rebuilt)

    response = await acompletion(**kwargs)
    return ModelResponse.model_validate(response)


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_should_skip_retry,
)
async def call_responses_api(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[dict[str, Any]],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> Any:
    """
    Generate a response using a provider's Responses API (e.g., web search) with retry logic.

    Uses litellm.aresponses() which is the native async version.

    Args:
        model: The model identifier to use (e.g., 'openai/gpt-4o')
        messages: The conversation messages
        tools: Tools for web search (e.g., [{"type": "web_search"}])
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments (reasoning, etc.)
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The OpenAI responses API response object
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "input": messages,
        "tools": tools,
        "timeout": llm_response_timeout,
        **extra_args,
    }

    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        kwargs["api_base"] = settings.LITELLM_PROXY_API_BASE
        kwargs["api_key"] = settings.LITELLM_PROXY_API_KEY
        tags = ["service:trajectory"]
        if trajectory_id:
            tags.append(f"trajectory_id:{trajectory_id}")
        kwargs["extra_body"] = {"tags": tags}

    if stream:
        kwargs["stream"] = True
        stream_iter: Any = await aresponses(**kwargs)
        completed_response = None
        async for event in stream_iter:
            if getattr(event, "type", None) == "response.completed":
                completed_response = getattr(event, "response", None)
        if completed_response is None:
            raise RuntimeError(
                "No response.completed event received from Responses API stream"
            )
        return completed_response

    response = await aresponses(**kwargs)
    return response
