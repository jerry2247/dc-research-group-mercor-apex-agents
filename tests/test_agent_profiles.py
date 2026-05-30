"""Agent profile registry tests."""

from __future__ import annotations

import pytest

from apex_agents_bench.agent_profile import (
    AgentProfile,
    all_profiles,
    get_profile,
    profile_names,
    profiles_by_family,
)


def test_registry_has_expected_families() -> None:
    families = {p.family for p in all_profiles()}
    assert families == {"gpt-5.5", "grok-4.3"}


def test_profile_names_are_unique() -> None:
    names = profile_names()
    assert len(names) == len(set(names))


def test_gpt55_has_four_efforts() -> None:
    names = {p.name for p in all_profiles() if p.family == "gpt-5.5"}
    assert names == {"gpt-5.5-low", "gpt-5.5-medium", "gpt-5.5-high", "gpt-5.5-xhigh"}


def test_grok43_has_three_efforts() -> None:
    names = {p.name for p in all_profiles() if p.family == "grok-4.3"}
    assert names == {"grok-4.3-low", "grok-4.3-medium", "grok-4.3-high"}


def test_get_profile_unknown_name_helpful_error() -> None:
    with pytest.raises(KeyError, match="Unknown agent profile"):
        get_profile("does-not-exist")


def test_get_profile_suggests_family_matches() -> None:
    with pytest.raises(KeyError, match="Did you mean"):
        get_profile("gpt-5.5-pro")


def test_profiles_by_family_groups_correctly() -> None:
    g = profiles_by_family()
    assert set(g.keys()) == {"gpt-5.5", "grok-4.3"}
    assert len(g["gpt-5.5"]) == 4
    assert len(g["grok-4.3"]) == 3


def test_to_extra_args_json_returns_copy() -> None:
    p = get_profile("gpt-5.5-high")
    d = p.to_extra_args_json()
    d["mutated"] = True
    # Mutation must not leak back into the registry.
    assert "mutated" not in p.orchestrator_extra_args


def test_typeof_returned_object() -> None:
    p = get_profile("grok-4.3-medium")
    assert isinstance(p, AgentProfile)
