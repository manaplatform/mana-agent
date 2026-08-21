from __future__ import annotations

from mana_agent.integrations.codex.provider import (
    CodexCredential, CodexCredentialStore, CodexExecutionMode, CodexUsage,
    CodexUsageStore, CredentialKind, choose_codex_mode,
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
