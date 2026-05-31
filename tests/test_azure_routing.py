"""Unit tests for the Azure routing helper."""

from __future__ import annotations

import ast
import inspect
import os

from apex_agents_bench.azure_routing import (
    AzureConfig,
    azure_call_kwargs,
    azure_subprocess_env,
    route_model_id,
    route_provider_for_credentials,
)


def test_route_off_is_identity() -> None:
    cfg = AzureConfig(enabled=False)
    assert route_model_id("openai/gpt-5.5", cfg=cfg) == "openai/gpt-5.5"
    assert route_model_id("xai/grok-4.3-high", cfg=cfg) == "xai/grok-4.3-high"


def test_route_on_gpt55_with_default_deployment() -> None:
    cfg = AzureConfig(enabled=True)
    # AZURE_GPT55_DEPLOYMENT_NAME may be set in the env; the default is "gpt-5.5".
    # gpt-5.5 routes to the openai/ provider aimed at the Azure /openai/v1
    # surface (NOT the classic azure/ provider, which 404s on this resource).
    deployment = os.environ.get("AZURE_GPT55_DEPLOYMENT_NAME", "gpt-5.5")
    assert route_model_id("openai/gpt-5.5", cfg=cfg) == f"openai/{deployment}"
    assert route_model_id("gpt-5.5", cfg=cfg) == f"openai/{deployment}"
    assert route_model_id("gpt-5.5-medium", cfg=cfg) == f"openai/{deployment}"


def test_route_on_with_explicit_deployment() -> None:
    cfg = AzureConfig(enabled=True, deployment_name="my-gpt55")
    assert route_model_id("openai/gpt-5.5", cfg=cfg) == "openai/my-gpt55"


def test_route_on_non_gpt55_is_identity() -> None:
    cfg = AzureConfig(enabled=True)
    assert route_model_id("xai/grok-4.3-high", cfg=cfg) == "xai/grok-4.3-high"
    assert route_model_id("deepseek/deepseek-v4-pro", cfg=cfg) == "deepseek/deepseek-v4-pro"
    assert route_model_id("anthropic/claude-3", cfg=cfg) == "anthropic/claude-3"
    assert route_model_id("text-embedding-3-large", cfg=cfg) == "text-embedding-3-large"


def test_azure_env_helpers_are_subprocess_or_call_scoped(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_API_BASE", "https://unit-test.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_API_KEY", "azure-secret")

    cfg = AzureConfig(enabled=True)
    assert azure_subprocess_env(cfg) == {
        "OPENAI_BASE_URL": "https://unit-test.services.ai.azure.com/openai/v1",
        "OPENAI_API_KEY": "azure-secret",
    }
    assert azure_call_kwargs(cfg) == {
        "api_base": "https://unit-test.services.ai.azure.com/openai/v1",
        "api_key": "azure-secret",
    }
    assert azure_subprocess_env(AzureConfig(enabled=False)) == {}
    assert azure_call_kwargs(AzureConfig(enabled=False)) == {}


def test_agent_and_grader_calls_both_receive_azure_subprocess_env() -> None:
    from apex_agents_bench import runner

    tree = ast.parse(inspect.getsource(runner.run_single_task))
    azure_call_sites = {
        node.func.id: {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg is not None}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_run_agent_subprocess", "_run_grading_subprocess"}
    }

    assert azure_call_sites["_run_agent_subprocess"]["extra_env"] == (
        "azure_subprocess_env(azure_eff)"
    )
    assert azure_call_sites["_run_grading_subprocess"]["extra_env"] == (
        "azure_subprocess_env(azure_eff)"
    )


def test_run_loads_dotenv_before_in_process_azure_kwargs() -> None:
    from apex_agents_bench import runner

    source = inspect.getsource(runner.run)
    assert source.index("_load_dotenv_for_run()") < source.index("azure_call_kwargs")


def test_route_provider_for_credentials_azure() -> None:
    assert route_provider_for_credentials("azure/gpt-5.5") == "azure"
    assert route_provider_for_credentials("openai/gpt-5.5") is None
    assert route_provider_for_credentials("xai/grok-4.3-high") is None


def test_required_api_key_recognizes_azure() -> None:
    from apex_agents_bench.runner import _required_api_key

    assert _required_api_key("azure/gpt-5.5") == "AZURE_API_KEY"
    assert _required_api_key("openai/gpt-5.5") == "OPENAI_API_KEY"
    assert _required_api_key("deepseek/deepseek-v4-pro", "deepseek") == "DEEPSEEK_API_KEY"


def test_run_options_default_azure_is_off() -> None:
    import dataclasses

    from apex_agents_bench.runner import RunOptions

    fields = {f.name: f for f in dataclasses.fields(RunOptions)}
    assert "azure" in fields
    factory = fields["azure"].default_factory  # type: ignore[union-attr]
    assert factory().enabled is False
