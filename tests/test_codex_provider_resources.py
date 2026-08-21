from __future__ import annotations

from mana_agent.integrations.codex.provider import (
    CodexCredential, CodexCredentialStore, CodexExecutionMode, CodexUsage,
    CodexUsageStore, CredentialKind, CodexIdentityError, choose_codex_mode, choose_codex_resource,
)


def test_credential_store_persists_reference_only(tmp_path):
    store = CodexCredentialStore(tmp_path)
    store.save(CodexCredential(CredentialKind.API, "OPENAI_API_KEY", authenticated=True))
    assert "OPENAI_API_KEY" in (tmp_path / "codex.json").read_text()
    assert "secret" not in (tmp_path / "codex.json").read_text()


def test_subscription_usage_never_has_dollar_cost_and_cache_round_trips(tmp_path):
    usage = CodexUsage(CodexExecutionMode.SUBSCRIPTION, True, quota_remaining=3, quota_capacity=10, quota_consumed=7)
    store = CodexUsageStore(tmp_path)
    store.save(usage)
    assert store.load(CodexExecutionMode.SUBSCRIPTION) == usage


def test_subscription_quota_wins_then_api_is_fallback_resource():
    api = CodexUsage(CodexExecutionMode.API, True, tokens_consumed=10, estimated_cost_usd=0.02)
    subscription = CodexUsage(CodexExecutionMode.SUBSCRIPTION, True, quota_remaining=2)
    assert choose_codex_mode({CodexExecutionMode.API: api, CodexExecutionMode.SUBSCRIPTION: subscription}, "coding") is CodexExecutionMode.SUBSCRIPTION
    exhausted = CodexUsage(CodexExecutionMode.SUBSCRIPTION, True, quota_remaining=0)
    assert choose_codex_mode({CodexExecutionMode.API: api, CodexExecutionMode.SUBSCRIPTION: exhausted}, "ci") is CodexExecutionMode.API


def test_resource_decision_retains_explicit_api_fallback_reason():
    api = CodexUsage(CodexExecutionMode.API, True, estimated_cost_usd=0.02)
    exhausted = CodexUsage(CodexExecutionMode.SUBSCRIPTION, True, quota_remaining=0, quota_capacity=10)
    decision = choose_codex_resource({CodexExecutionMode.SUBSCRIPTION: exhausted, CodexExecutionMode.API: api}, "coding")
    assert decision.selected_mode is CodexExecutionMode.API
    assert "unavailable" in decision.reason


def test_healthy_subscription_is_preferred_for_coding_even_when_api_is_available():
    api = CodexUsage(CodexExecutionMode.API, True, estimated_cost_usd=0.02)
    subscription = CodexUsage(CodexExecutionMode.SUBSCRIPTION, True, quota_remaining=80, quota_capacity=100)
    assert choose_codex_resource({CodexExecutionMode.API: api, CodexExecutionMode.SUBSCRIPTION: subscription}, "coding").selected_mode is CodexExecutionMode.SUBSCRIPTION


def test_expired_subscription_requires_authentication_instead_of_silent_api_switch():
    api = CodexUsage(CodexExecutionMode.API, True, estimated_cost_usd=0.02)
    expired = CodexUsage(CodexExecutionMode.SUBSCRIPTION, False, capacity_status="unknown")
    try:
        choose_codex_resource({CodexExecutionMode.SUBSCRIPTION: expired, CodexExecutionMode.API: api}, "coding")
    except CodexIdentityError as exc:
        assert "authentication" in str(exc)
    else:
        raise AssertionError("expired subscription must not silently switch to API")
