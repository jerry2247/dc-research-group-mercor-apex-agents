"""Azure-routing helper for GPT-5.5 chat completions.

When ``AzureConfig.enabled`` is True, any ``gpt-5.5*`` model identifier
that would otherwise hit OpenAI is rewritten to the Azure-OpenAI
deployment instead (``azure/<deployment_name>``). LiteLLM resolves
Azure credentials from the standard environment variables:

  - ``AZURE_API_KEY``       (or ``AZURE_OPENAI_API_KEY``)
  - ``AZURE_API_BASE``      (the endpoint URL)
  - ``AZURE_API_VERSION``   (e.g. ``2024-08-01-preview``)

These are not surfaced through this module; they are read by LiteLLM
directly per-request.

The embedding model (``text-embedding-3-large``) is left untouched —
embeddings always go through OpenAI (``OPENAI_API_KEY``) regardless of
``AzureConfig.enabled``. This mirrors the project goal: "The embedding
model should still be through the openAI key."

Affected call sites (when enabled):
  - the judge model in :func:`apex_agents_bench.judge.write_grading_settings`
  - the agent profile's orchestrator model in the agent subprocess invocation
  - the Dynamic Ledger curator's ``cfg.curator_model``
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

    When ``enabled=False`` (default), no routing happens — the project's
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

    Identity when Azure is off or the model is not gpt-5.5. When Azure
    is on and the model IS gpt-5.5, returns ``azure/<deployment_name>``.
    """
    if not cfg.enabled:
        return model_id
    if not _is_gpt55(model_id):
        return model_id
    return f"azure/{cfg.resolved_deployment_name()}"


def route_provider_for_credentials(model_id: str) -> str | None:
    """Return the credentials-provider hint for ``_required_api_key``.

    When the model id is ``azure/...`` the preflight check should look
    for ``AZURE_API_KEY`` instead of ``OPENAI_API_KEY``.
    """
    if model_id.startswith("azure/"):
        return "azure"
    return None
