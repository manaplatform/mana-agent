from __future__ import annotations

from mana_agent.integrations.codex.provider import (
    CodexAccountingRecord,
    CodexAccountingStore,
    CodexAuthenticationService,
    CodexAuthenticationStatus,
    CodexCredential,
    CodexCredentialStore,
    CodexExecutionMode,
    CodexExecutionError,
    CodexExecutionMetadata,
    CodexExecutionState,
    CodexUsage,
    CodexUsageProvider,
    CodexUsageStore,
    CodexProvider,
    CodexRoutingDecisionStore,
    CredentialKind,
    choose_codex_mode,
)


class Identity:
    def login(self, reference, *, account_identity=""):
        return CodexCredential(CredentialKind.SUBSCRIPTION, reference, account_identity or "acct-1", expires_at=1000, authenticated=True)

    def validate(self, credential):
        return credential

    def refresh(self, credential):
        return CodexCredential(credential.kind, credential.reference, credential.account_identity, expires_at=2000, authenticated=True, refresh_state="refreshed")

    def revoke(self, credential):
        return None


class UsageClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def fetch_usage(self, credential, mode):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_subscription_authentication_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr("mana_agent.integrations.codex.provider.time.time", lambda: 10)
    service = CodexAuthenticationService(CodexCredentialStore(tmp_path), Identity())
    assert service.login("mana-secret://codex", account_identity="acct-1").authenticated
    assert service.status().status is CodexAuthenticationStatus.AUTHENTICATED
    assert service.logout().status is CodexAuthenticationStatus.REVOKED


def test_expired_subscription_refreshes_and_invalid_session_is_not_available(tmp_path, monkeypatch):
    monkeypatch.setattr("mana_agent.integrations.codex.provider.time.time", lambda: 1500)
    store = CodexCredentialStore(tmp_path)
    store.save(CodexCredential(CredentialKind.SUBSCRIPTION, "ref", "acct", expires_at=1000, authenticated=True))
    service = CodexAuthenticationService(store, Identity())
    assert service.refresh_if_needed().status is CodexAuthenticationStatus.AUTHENTICATED
    invalid = CodexUsage(CodexExecutionMode.SUBSCRIPTION, True, quota_remaining=4, capacity_status="unknown")
    assert not invalid.available


def test_refresh_failure_preserves_expired_auth_required_state(tmp_path, monkeypatch):
    monkeypatch.setattr("mana_agent.integrations.codex.provider.time.time", lambda: 1500)
    store = CodexCredentialStore(tmp_path)
    store.save(CodexCredential(CredentialKind.SUBSCRIPTION, "ref", "acct", expires_at=1000, authenticated=True))

    class FailingIdentity(Identity):
        def refresh(self, credential):
            raise RuntimeError("session revoked by Luna")

    status = CodexAuthenticationService(store, FailingIdentity()).recover_session()
    assert status.status is CodexAuthenticationStatus.EXPIRED
    assert "authentication" in status.reason


def test_usage_provider_normalizes_api_and_subscription_without_mixing_billing(tmp_path):
    credentials = CodexCredential(CredentialKind.API, "ref", authenticated=True)
    api = CodexUsageProvider(UsageClient({"total_tokens": 12, "estimated_cost_usd": 0.03}), store=CodexUsageStore(tmp_path), clock=lambda: 10)
    api_usage = api.current(credentials, CodexExecutionMode.API)
    assert api_usage.tokens_consumed == 12 and api_usage.estimated_cost_usd == 0.03
    subscription = CodexUsageProvider(UsageClient({"quota_capacity": 10, "quota_consumed": 3, "reset_at": 50}), store=CodexUsageStore(tmp_path / "sub"), clock=lambda: 10)
    sub_usage = subscription.current(CodexCredential(CredentialKind.SUBSCRIPTION, "ref", authenticated=True), CodexExecutionMode.SUBSCRIPTION)
    assert sub_usage.quota_remaining == 7 and sub_usage.estimated_cost_usd is None


def test_unavailable_usage_is_unknown_and_subscription_falls_back_to_api(tmp_path):
    client = UsageClient(RuntimeError("endpoint unavailable"))
    provider = CodexUsageProvider(client, store=CodexUsageStore(tmp_path))
    unknown = provider.current(CodexCredential(CredentialKind.SUBSCRIPTION, "ref", authenticated=True), CodexExecutionMode.SUBSCRIPTION)
    api = CodexUsage(CodexExecutionMode.API, True, tokens_consumed=1, estimated_cost_usd=0.01)
    assert unknown.capacity_status == "unknown"
    assert choose_codex_mode({CodexExecutionMode.SUBSCRIPTION: unknown, CodexExecutionMode.API: api}, "coding") is CodexExecutionMode.API


def test_accounting_keeps_api_cost_and_subscription_quota_separate(tmp_path):
    store = CodexAccountingStore(tmp_path)
    store.record(CodexAccountingRecord("codex", CodexExecutionMode.API, "api-1", input_tokens=2, output_tokens=3, usd_cost=0.02))
    store.record(CodexAccountingRecord("codex", CodexExecutionMode.SUBSCRIPTION, "sub-1", quota_consumed=5, account_identity="acct"))
    rows = (tmp_path / "codex-accounting.jsonl").read_text().splitlines()
    assert '"usd_cost": 0.02' in rows[0] and '"quota_consumed": 5' in rows[1]


def test_execution_persists_routing_reason_and_accounting_record(tmp_path):
    class Backend:
        async def execute(self, task, workspace):
            return type("Result", (), {"token_usage": {"input_tokens": 4, "output_tokens": 6, "usd_cost": 0.03}})()

    provider = CodexProvider(
        Backend(),
        mode=CodexExecutionMode.API,
        credentials=CodexCredentialStore(tmp_path / "credentials"),
        usage=CodexUsageStore(tmp_path / "usage"),
        accounting=CodexAccountingStore(tmp_path / "accounting"),
        routing=CodexRoutingDecisionStore(tmp_path / "routing"),
    )

    import asyncio
    task = type("Task", (), {"task_id": "execution-1"})()
    result = asyncio.run(provider.execute(task, object()))

    assert result.token_usage["output_tokens"] == 6
    accounting = (tmp_path / "accounting" / "codex-accounting.jsonl").read_text()
    routing = (tmp_path / "routing" / "codex-routing.jsonl").read_text()
    assert '"execution_id": "execution-1"' in accounting
    assert '"usd_cost": 0.03' in accounting
    assert '"routing_reason": ["API resource selected"]' in accounting
    assert '"reason": ["API resource selected"]' in routing


def test_codex_metadata_serializes_stable_lifecycle_values():
    metadata = CodexExecutionMetadata(
        "execution-1",
        "codex",
        CodexExecutionMode.SUBSCRIPTION,
        CodexExecutionState.FALLBACK_SELECTED,
        "codex/subscription",
        fallback_path=({"from": "subscription", "to": "api", "reason": "quota exhausted"},),
        routing_reason=("quota exhausted", "API resource selected"),
        decision_id="decision-1",
    )

    assert metadata.as_dict() == {
        "execution_id": "execution-1",
        "provider": "codex",
        "mode": "subscription",
        "state": "FALLBACK_SELECTED",
        "selected_resource": "codex/subscription",
        "accounting_reference": "",
        "fallback_path": [{"from": "subscription", "to": "api", "reason": "quota exhausted"}],
        "failure_reason": "",
        "fallback_failure_reason": "",
        "routing_reason": ["quota exhausted", "API resource selected"],
        "decision_id": "decision-1",
    }
def test_subscription_execution_records_identity_quota_and_reset_cycle(tmp_path):
    class Backend:
        async def execute(self, task, workspace):
            return type("Result", (), {"token_usage": {"quota_consumed": 2}})()

    credentials = CodexCredentialStore(tmp_path / "credentials")
    credentials.save(
        CodexCredential(
            CredentialKind.SUBSCRIPTION,
            "subscription-ref",
            account_identity="acct-1",
            authenticated=True,
        )
    )
    usage = CodexUsageStore(tmp_path / "usage")
    usage.save(
        CodexUsage(
            CodexExecutionMode.SUBSCRIPTION,
            True,
            account_identity="acct-1",
            quota_consumed=3,
            quota_remaining=7,
            quota_capacity=10,
            reset_at=1234,
        )
    )
    provider = CodexProvider(
        Backend(),
        mode=CodexExecutionMode.SUBSCRIPTION,
        credentials=credentials,
        usage=usage,
        accounting=CodexAccountingStore(tmp_path / "accounting"),
        routing=CodexRoutingDecisionStore(tmp_path / "routing"),
    )

    import asyncio
    asyncio.run(provider.execute(type("Task", (), {"task_id": "subscription-execution"})(), object()))

    accounting = (tmp_path / "accounting" / "codex-accounting.jsonl").read_text()
    assert '"account_identity": "acct-1"' in accounting
    assert '"quota_consumed": 2.0' in accounting
    assert '"reset_at": 1234' in accounting
    assert '"routing_reason": ["coding task", "subscription authenticated", "quota healthy"]' in accounting


def test_authenticated_subscription_with_healthy_quota_completes_with_resource_metadata(tmp_path):
    class Backend:
        async def execute(self, task, workspace):
            return type("Result", (), {"token_usage": {"quota_consumed": 2}})()

    credentials = CodexCredentialStore(tmp_path / "credentials")
    credentials.save(CodexCredential(CredentialKind.SUBSCRIPTION, "ref", "acct", authenticated=True))
    usage = CodexUsageStore(tmp_path / "usage")
    usage.save(CodexUsage(CodexExecutionMode.SUBSCRIPTION, True, "acct", quota_consumed=2, quota_remaining=8, quota_capacity=10))
    provider = CodexProvider(Backend(), mode=CodexExecutionMode.SUBSCRIPTION, credentials=credentials, usage=usage, accounting=CodexAccountingStore(tmp_path / "accounting"))

    import asyncio
    result = asyncio.run(provider.execute(type("Task", (), {"task_id": "healthy-subscription"})(), object()))

    assert result.codex_metadata.state is CodexExecutionState.COMPLETED
    assert result.codex_metadata.selected_resource == "codex/subscription"
    assert result.codex_metadata.accounting_reference == "codex-accounting:healthy-subscription"


def test_exhausted_subscription_falls_back_to_api_once_with_original_reason(tmp_path):
    class Backend:
        async def execute(self, task, workspace):
            return type("Result", (), {"token_usage": {"input_tokens": 1, "output_tokens": 1}})()

    credentials = CodexCredentialStore(tmp_path / "credentials")
    credentials.save(CodexCredential(CredentialKind.SUBSCRIPTION, "ref", "acct", authenticated=True))
    usage = CodexUsageStore(tmp_path / "usage")
    usage.save(CodexUsage(CodexExecutionMode.SUBSCRIPTION, True, "acct", quota_remaining=0, quota_capacity=10))
    api = CodexProvider(Backend(), mode=CodexExecutionMode.API, credentials=CodexCredentialStore(tmp_path / "api-credentials"), accounting=CodexAccountingStore(tmp_path / "api-accounting"))
    provider = CodexProvider(Backend(), mode=CodexExecutionMode.SUBSCRIPTION, credentials=credentials, usage=usage, routing=CodexRoutingDecisionStore(tmp_path / "routing"), fallback_provider=api)

    import asyncio
    result = asyncio.run(provider.execute(type("Task", (), {"task_id": "quota-fallback"})(), object()))

    assert result.codex_metadata.state is CodexExecutionState.FALLBACK_SELECTED
    assert result.codex_metadata.mode is CodexExecutionMode.API
    assert result.codex_metadata.fallback_path[0]["reason"] == "subscription quota exhausted"


def test_expired_subscription_stops_with_auth_required(tmp_path, monkeypatch):
    monkeypatch.setattr("mana_agent.integrations.codex.provider.time.time", lambda: 2000)
    credentials = CodexCredentialStore(tmp_path / "credentials")
    credentials.save(CodexCredential(CredentialKind.SUBSCRIPTION, "ref", "acct", expires_at=1000, authenticated=True))
    provider = CodexProvider(object(), mode=CodexExecutionMode.SUBSCRIPTION, credentials=credentials)

    import asyncio
    try:
        asyncio.run(provider.execute(type("Task", (), {"task_id": "expired-subscription"})(), object()))
    except CodexExecutionError as exc:
        assert exc.state is CodexExecutionState.AUTH_REQUIRED
    else:
        raise AssertionError("expired subscription must require authentication")


def test_usage_provider_unavailability_is_unknown_capacity_and_does_not_claim_unlimited(tmp_path):
    provider = CodexUsageProvider(UsageClient(RuntimeError("usage endpoint unavailable")), store=CodexUsageStore(tmp_path))
    usage = provider.current(CodexCredential(CredentialKind.SUBSCRIPTION, "ref", "acct", authenticated=True), CodexExecutionMode.SUBSCRIPTION)

    assert usage.capacity_status == "unknown"
    assert not usage.available
    assert usage.quota_health == 0
