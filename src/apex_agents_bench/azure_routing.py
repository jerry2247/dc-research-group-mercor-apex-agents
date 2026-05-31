"""Azure-routing helper for GPT-5.5 chat completions.

When ``AzureConfig.enabled`` is True, any ``gpt-5.5*`` model identifier that
would otherwise hit OpenAI is routed to the Azure-OpenAI deployment instead.

This project's Azure resource is an Azure AI Foundry resource that exposes the
OpenAI-compatible ``/openai/v1`` surface. LiteLLM reaches that surface through
its ``openai/`` provider (the standard OpenAI request shape), NOT the classic
``azure/`` provider: the ``azure/`` provider builds the
``/openai/deployments/<name>/chat/completions`` path, which returns HTTP 404 on
this resource. Empirically verified: ``openai/<deployment>`` against
``<resource>.services.ai.azure.com/openai/v1`` with the Azure key returns 200.

Routing therefore changes ONLY the destination (endpoint URL + key), never the
request itself: the model id stays ``openai/<deployment>`` and every sampling
parameter (reasoning_effort, verbosity, temperature, timeout, ...) is identical
to the non-Azure path. The endpoint and key are supplied out-of-band:

  - the agent and judge run in subprocesses -> :func:`azure_subprocess_env`
    injects ``OPENAI_BASE_URL`` + ``OPENAI_API_KEY`` (Azure values) into the
    child env. Those subprocesses make no other ``openai/`` call (no
    embeddings), so the override is total and safe.
  - the DC-RS synthesizer and TRACE reflector/curator run in THIS process,
    where the env still holds the real ``OPENAI_API_KEY`` for embeddings ->
    :func:`azure_call_kwargs` returns ``api_base`` + ``api_key`` to pass
    explicitly into that one ``litellm.completion`` call, leaving the process
    env (and embeddings) untouched.

The runner supplies these credentials per request:

  - ``AZURE_API_KEY``       (the Azure resource key)
  - ``AZURE_API_BASE``      (the OpenAI-compatible base, e.g.
                            ``https://<resource>.services.ai.azure.com/openai/v1``)
  - ``AZURE_API_VERSION``   (a recent preview, e.g. ``2025-07-01-preview``;
                            kept for reference only -- this /openai/v1 route
                            does not read it)

The embedding model (``text-embedding-3-large``) is left untouched -- embeddings
always go through OpenAI (``OPENAI_API_KEY``) regardless of ``AzureConfig``.

Affected call sites (when enabled):
  - the agent profile's orchestrator model in the agent subprocess invocation
  - the judge model in :func:`apex_agents_bench.judge.write_grading_settings`
  - the DC-RS synthesizer's ``cfg.synthesizer_model``
  - the TRACE reflector/curator's ``cfg.reflector_model`` /
    ``cfg.curator_model``

Affected NOT (by design):
  - ``text-embedding-3-large`` (always OpenAI)
  - non-OpenAI provider models (``xai/grok-*``, ``bedrock/...``, etc.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AzureConfig:
    """Azure-OpenAI routing configuration.

    When ``enabled=False`` (default), no routing happens -- the project's
    OpenAI-based default paths are preserved byte-for-byte.

    ``deployment_name`` is the name of the Azure deployment that hosts
    GPT-5.5. The value defaults to the ``AZURE_GPT55_DEPLOYMENT_NAME``
    env var, then to ``"gpt-5.5"`` if that env var is unset. Users
    whose Azure deployment is named differently (e.g. ``gpt-5-5``)
    should set the env var.
    """

    enabled: bool = False
    deployment_name: str = ""

    def resolved_deployment_name(self) -> str:
        if self.deployment_name:
            return self.deployment_name
        return os.environ.get("AZURE_GPT55_DEPLOYMENT_NAME", "gpt-5.5")


def _is_gpt55(model_id: str) -> bool:
    bare = model_id.split("/")[-1] if "/" in model_id else model_id
    return bare.startswith("gpt-5.5")


def route_model_id(model_id: str, *, cfg: AzureConfig) -> str:
    """Return the model id LiteLLM should be called with.

    Identity when Azure is off or the model is not gpt-5.5. When Azure is on
    and the model IS gpt-5.5, returns ``openai/<deployment_name>`` -- the
    ``openai/`` provider aimed at the Azure ``/openai/v1`` surface (see module
    docstring). The credentials/endpoint are supplied separately via
    :func:`azure_subprocess_env` or :func:`azure_call_kwargs`; the model id and
    all sampling parameters are unchanged from the non-Azure path.
    """
    if not cfg.enabled:
        return model_id
    if not _is_gpt55(model_id):
        return model_id
    return f"openai/{cfg.resolved_deployment_name()}"


def azure_api_base() -> str | None:
    """The Azure OpenAI-compatible base URL from the environment, or None."""
    return os.environ.get("AZURE_API_BASE") or None


def azure_api_key() -> str | None:
    """The Azure resource key from the environment, or None."""
    return os.environ.get("AZURE_API_KEY") or None


def azure_subprocess_env(cfg: AzureConfig) -> dict[str, str]:
    """Env overrides to inject into a child subprocess (agent / grading) so its
    ``openai/`` calls reach Azure instead of OpenAI.

    Returns an empty dict when Azure is disabled (no override -> the subprocess
    inherits the parent env unchanged). When enabled, points ``OPENAI_BASE_URL``
    and ``OPENAI_API_KEY`` at the Azure resource. Safe because the agent and
    grading subprocesses issue no other ``openai/`` request (no embeddings); the
    only LLM they call is the gpt-5.5 orchestrator / judge, which is exactly what
    we want on Azure.
    """
    if not cfg.enabled:
        return {}
    out: dict[str, str] = {}
    base = azure_api_base()
    key = azure_api_key()
    if base:
        out["OPENAI_BASE_URL"] = base
    if key:
        out["OPENAI_API_KEY"] = key
    return out


def azure_call_kwargs(cfg: AzureConfig) -> dict[str, str]:
    """Kwargs to splat into an in-process ``litellm.completion`` call (DC-RS
    synthesizer / TRACE reflector+curator) so that single call reaches Azure
    WITHOUT disturbing the process environment.

    Returns an empty dict when Azure is disabled. Used instead of env overrides
    here because this process keeps the real ``OPENAI_API_KEY`` in its env for
    ``text-embedding-3-large`` embeddings, which must stay on OpenAI.
    """
    if not cfg.enabled:
        return {}
    out: dict[str, str] = {}
    base = azure_api_base()
    key = azure_api_key()
    if base:
        out["api_base"] = base
    if key:
        out["api_key"] = key
    return out


def route_provider_for_credentials(model_id: str) -> str | None:
    """Return the credentials-provider hint for ``_required_api_key``.

    When the model id is ``azure/...`` the preflight check should look
    for ``AZURE_API_KEY`` instead of ``OPENAI_API_KEY``. (Retained for the
    classic ``azure/`` form; this project routes gpt-5.5 via ``openai/`` and
    enforces Azure credentials in the runner preflight when ``--azure`` is on.)
    """
    if model_id.startswith("azure/"):
        return "azure"
    return None
