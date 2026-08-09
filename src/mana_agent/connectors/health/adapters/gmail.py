"""Gmail connector health adapter using real provider semantics."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable

from mana_agent.connectors.health.models import (
    CapabilitySignal,
    ConnectorHealthCapabilities,
    DeliveryReceipt,
    DeliveryState,
    HealthReasonCode,
    PathSignals,
    ProbeCategory,
    ProbeOutcome,
    ProbeResult,
    RecoveryActionKind,
    SyntheticProbeMode,
    utc_now,
)
from mana_agent.connectors.health.probes import failed, passed, rate_limited

logger = logging.getLogger(__name__)


class GmailHealthAdapter:
    """Health probes for a single Gmail account.

    Process/gateway aliveness is independent. A configured account with a live
    Mana process is NOT healthy until auth + connectivity probes succeed.
    """

    def __init__(
        self,
        *,
        account_id: str,
        account_address: str = "",
        enabled: bool = True,
        provider_factory: Callable[[], Any] | None = None,
        runtime_alive: bool = True,
        synthetic_mode: SyntheticProbeMode = SyntheticProbeMode.SAFE_ENDPOINT,
        delivery_receipts: list[DeliveryReceipt] | None = None,
        last_ingress_at: datetime | None = None,
        token_refresh: Callable[[], bool] | None = None,
    ) -> None:
        self.account_id = account_id
        self.account_address = account_address
        self._enabled = enabled
        self._provider_factory = provider_factory
        self._runtime_alive = runtime_alive
        self._synthetic_mode = synthetic_mode
        self._delivery_receipts = list(delivery_receipts or [])
        self._last_ingress_at = last_ingress_at
        self._token_refresh = token_refresh
        self._last_auth_ok = False
        self._last_transport_ok = False
        self._client: Any = None

    @property
    def connector_id(self) -> str:
        return f"gmail:{self.account_id}"

    @property
    def connector_type(self) -> str:
        return "gmail"

    def health_capabilities(self) -> ConnectorHealthCapabilities:
        return ConnectorHealthCapabilities(
            auth=True,
            connectivity=True,
            ingress=True,
            egress=True,
            subscriptions=False,
            acknowledgements=True,
        )

    def supported_probe_categories(self) -> list[ProbeCategory]:
        return [
            ProbeCategory.AUTH,
            ProbeCategory.CONNECTIVITY,
            ProbeCategory.INGRESS,
            ProbeCategory.EGRESS,
            ProbeCategory.ACKNOWLEDGEMENT,
        ]

    def synthetic_probe_mode(self) -> SyntheticProbeMode:
        return self._synthetic_mode

    def is_enabled(self) -> bool:
        return self._enabled

    def describe(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "connector_type": self.connector_type,
            "account_id": self.account_id,
            "address": self.account_address,
            "enabled": self._enabled,
        }

    def collect_signals(self) -> PathSignals:
        return PathSignals(
            runtime_alive=self._runtime_alive,
            transport_connected=self._last_transport_ok,
            authenticated=CapabilitySignal.OK if self._last_auth_ok else CapabilitySignal.UNKNOWN,
            ingress_operational=CapabilitySignal.UNKNOWN,
            egress_operational=CapabilitySignal.UNKNOWN,
            subscription_operational=CapabilitySignal.NOT_APPLICABLE,
            acknowledgements_operational=CapabilitySignal.UNKNOWN,
        )

    def list_recovery_actions(self, reason_codes: list[str]) -> list[RecoveryActionKind]:
        actions = [
            RecoveryActionKind.TOKEN_REFRESH,
            RecoveryActionKind.CLIENT_RECREATE,
            RecoveryActionKind.CONNECTION_POOL_RESET,
        ]
        return actions

    def recent_delivery_receipts(self, *, limit: int = 20) -> list[DeliveryReceipt]:
        return self._delivery_receipts[-limit:]

    def record_delivery(self, receipt: DeliveryReceipt) -> None:
        self._delivery_receipts.append(receipt)

    def set_runtime_alive(self, alive: bool) -> None:
        self._runtime_alive = alive

    def note_ingress(self, when: datetime | None = None) -> None:
        self._last_ingress_at = when or utc_now()

    async def run_probe(self, category: ProbeCategory, *, mode: SyntheticProbeMode) -> ProbeResult:
        if category is ProbeCategory.AUTH:
            return await self._probe_auth()
        if category is ProbeCategory.CONNECTIVITY:
            return await self._probe_connectivity()
        if category is ProbeCategory.INGRESS:
            return await self._probe_ingress(mode)
        if category is ProbeCategory.EGRESS:
            return self._probe_egress(mode)
        if category is ProbeCategory.ACKNOWLEDGEMENT:
            return self._probe_ack()
        return ProbeResult(
            category=category,
            outcome=ProbeOutcome.UNSUPPORTED,
            message=f"Gmail does not support probe {category.value}",
        )

    async def execute_recovery(self, action: RecoveryActionKind) -> ProbeResult:
        if action is RecoveryActionKind.TOKEN_REFRESH:
            if self._token_refresh is None:
                return failed(
                    ProbeCategory.AUTH,
                    HealthReasonCode.TOKEN_REFRESH_FAILED,
                    "No token refresh handler configured",
                )
            try:
                ok = await asyncio.to_thread(self._token_refresh)
            except Exception as exc:
                return failed(
                    ProbeCategory.AUTH,
                    HealthReasonCode.TOKEN_REFRESH_FAILED,
                    f"Token refresh failed: {type(exc).__name__}",
                )
            if ok:
                self._client = None
                return passed(ProbeCategory.AUTH, message="Token refresh succeeded")
            return failed(
                ProbeCategory.AUTH,
                HealthReasonCode.TOKEN_REFRESH_FAILED,
                "Token refresh returned failure",
            )
        if action in {
            RecoveryActionKind.CLIENT_RECREATE,
            RecoveryActionKind.CONNECTION_POOL_RESET,
        }:
            self._client = None
            probe = await self._probe_auth()
            if probe.outcome is ProbeOutcome.PASSED:
                return passed(ProbeCategory.CONNECTIVITY, message=f"Recovery {action.value} succeeded")
            return failed(
                ProbeCategory.CONNECTIVITY,
                HealthReasonCode.RECONNECT_FAILED,
                probe.message or f"Recovery {action.value} failed",
            )
        return failed(
            ProbeCategory.CONNECTIVITY,
            HealthReasonCode.RECONNECT_FAILED,
            f"Unsupported recovery action {action.value}",
        )

    async def _probe_auth(self) -> ProbeResult:
        started = time.perf_counter()
        try:
            provider = self._get_provider()
            if hasattr(provider, "health_check"):
                health = await provider.health_check()
                latency = (time.perf_counter() - started) * 1000
                if getattr(health, "healthy", False):
                    self._last_auth_ok = True
                    self._last_transport_ok = True
                    return passed(ProbeCategory.AUTH, latency_ms=latency, message="Gmail auth OK")
                self._last_auth_ok = False
                return failed(
                    ProbeCategory.AUTH,
                    HealthReasonCode.AUTH_EXPIRED,
                    getattr(health, "message", None) or "Gmail auth failed",
                    latency_ms=latency,
                )
            # Fallback: connect/profile
            await provider.connect()
            latency = (time.perf_counter() - started) * 1000
            self._last_auth_ok = True
            self._last_transport_ok = True
            return passed(ProbeCategory.AUTH, latency_ms=latency, message="Gmail profile reachable")
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            self._last_auth_ok = False
            self._last_transport_ok = False
            name = type(exc).__name__
            message = str(exc)
            if "AuthenticationRequired" in name or "401" in message or "credentials" in message.lower():
                return failed(
                    ProbeCategory.AUTH,
                    HealthReasonCode.AUTH_EXPIRED,
                    "Gmail credentials rejected or expired",
                    latency_ms=latency,
                    details={"exception_type": name},
                )
            if "403" in message or "Authorization" in name:
                return failed(
                    ProbeCategory.AUTH,
                    HealthReasonCode.AUTH_REVOKED,
                    "Gmail authorization revoked or insufficient",
                    latency_ms=latency,
                    details={"exception_type": name},
                )
            if "429" in message or "rate" in message.lower():
                return rate_limited(ProbeCategory.AUTH, "Gmail rate limited", retry_after=60.0)
            return failed(
                ProbeCategory.AUTH,
                HealthReasonCode.PROBE_FAILED,
                f"Gmail auth probe failed: {name}",
                latency_ms=latency,
                details={"exception_type": name},
            )

    async def _probe_connectivity(self) -> ProbeResult:
        # Connectivity is verified via the same safe profile endpoint.
        result = await self._probe_auth()
        return result.model_copy(update={"category": ProbeCategory.CONNECTIVITY})

    async def _probe_ingress(self, mode: SyntheticProbeMode) -> ProbeResult:
        """Verify the ingress *mechanism* without treating quiet inboxes as failure.

        Uses a metadata-only list with maxResults=1 as a safe endpoint probe.
        Success means the API path works, not that new mail exists.
        """
        started = time.perf_counter()
        try:
            provider = self._get_provider()
            if hasattr(provider, "search_messages"):
                from mana_agent.connectors.email.models import EmailQuery

                await provider.search_messages(EmailQuery(folders=["INBOX"], limit=1))
            elif hasattr(provider, "list_folders"):
                await provider.list_folders()
            else:
                return ProbeResult(
                    category=ProbeCategory.INGRESS,
                    outcome=ProbeOutcome.SKIPPED,
                    message="No safe ingress probe available",
                )
            latency = (time.perf_counter() - started) * 1000
            self.note_ingress()
            return passed(
                ProbeCategory.INGRESS,
                latency_ms=latency,
                message="Gmail ingress mechanism operational",
            )
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            name = type(exc).__name__
            if "AuthenticationRequired" in name or "401" in str(exc):
                return failed(
                    ProbeCategory.INGRESS,
                    HealthReasonCode.AUTH_EXPIRED,
                    "Ingress probe hit auth failure",
                    latency_ms=latency,
                )
            if "429" in str(exc):
                return rate_limited(ProbeCategory.INGRESS, retry_after=60.0)
            return failed(
                ProbeCategory.INGRESS,
                HealthReasonCode.INGRESS_STALLED,
                f"Gmail ingress probe failed: {name}",
                latency_ms=latency,
                details={"exception_type": name},
            )

    def _probe_egress(self, mode: SyntheticProbeMode) -> ProbeResult:
        """Egress health from recent delivery receipts — never send real mail by default."""
        if mode in {SyntheticProbeMode.ACTIVE, SyntheticProbeMode.TEST_CHANNEL}:
            return ProbeResult(
                category=ProbeCategory.EGRESS,
                outcome=ProbeOutcome.SKIPPED,
                message="Active Gmail egress probes are not enabled by default",
            )
        receipts = self.recent_delivery_receipts(limit=10)
        if not receipts:
            # No evidence of failure either — mark unknown/skip rather than inventing success.
            return ProbeResult(
                category=ProbeCategory.EGRESS,
                outcome=ProbeOutcome.SKIPPED,
                message="No recent delivery receipts; egress not actively verified",
            )
        latest = receipts[-1]
        if latest.state is DeliveryState.FAILED:
            return failed(
                ProbeCategory.EGRESS,
                HealthReasonCode.EGRESS_FAILED,
                latest.failure_reason or "Recent Gmail delivery failed",
            )
        if latest.state in {
            DeliveryState.PROVIDER_ACCEPTED,
            DeliveryState.DELIVERED,
            DeliveryState.ACKNOWLEDGED,
            DeliveryState.SUBMITTED,
        }:
            return passed(ProbeCategory.EGRESS, message=f"Recent delivery state={latest.state.value}")
        return ProbeResult(
            category=ProbeCategory.EGRESS,
            outcome=ProbeOutcome.SKIPPED,
            message=f"Latest delivery state is {latest.state.value}",
        )

    def _probe_ack(self) -> ProbeResult:
        receipts = [
            r
            for r in self.recent_delivery_receipts(limit=20)
            if r.state in {DeliveryState.ACKNOWLEDGED, DeliveryState.FAILED, DeliveryState.SUBMITTED, DeliveryState.PROVIDER_ACCEPTED}
        ]
        if not receipts:
            return ProbeResult(
                category=ProbeCategory.ACKNOWLEDGEMENT,
                outcome=ProbeOutcome.SKIPPED,
                message="No acknowledgement samples available",
            )
        # Gmail send API acceptance is provider_accepted, not end-user delivery.
        timed_out = [
            r
            for r in receipts
            if r.state is DeliveryState.SUBMITTED
            and r.submitted_at is not None
            and (utc_now() - r.submitted_at).total_seconds() > 300
        ]
        if timed_out:
            return failed(
                ProbeCategory.ACKNOWLEDGEMENT,
                HealthReasonCode.ACK_TIMEOUT,
                "Gmail delivery acknowledgement timed out",
            )
        return passed(ProbeCategory.ACKNOWLEDGEMENT, message="Recent Gmail provider acceptances present")

    def _get_provider(self) -> Any:
        if self._client is not None:
            return self._client
        if self._provider_factory is None:
            raise RuntimeError("Gmail provider factory is not configured")
        self._client = self._provider_factory()
        return self._client


def discover_gmail_adapters(
    *,
    runtime_alive: bool = True,
    provider_builder: Callable[[Any], Any] | None = None,
) -> list[GmailHealthAdapter]:
    """Discover configured Gmail accounts and build health adapters."""
    try:
        from mana_agent.connectors.email.config import load_accounts
    except Exception:
        return []
    adapters: list[GmailHealthAdapter] = []
    for account in load_accounts():
        if account.provider != "gmail":
            continue

        def factory(acc=account):
            if provider_builder is not None:
                return provider_builder(acc)
            return _default_gmail_provider(acc)

        def token_refresh(acc=account) -> bool:
            return _refresh_gmail_token(acc)

        adapters.append(
            GmailHealthAdapter(
                account_id=account.id,
                account_address=account.address.address if account.address else "",
                enabled=bool(account.enabled),
                provider_factory=factory,
                runtime_alive=runtime_alive,
                token_refresh=token_refresh,
            )
        )
    return adapters


def _default_gmail_provider(account: Any) -> Any:
    from mana_agent.connectors.email.auth.credential_store import CredentialStore
    from mana_agent.connectors.email.providers.gmail import GmailProvider

    if not account.secret_ref:
        raise RuntimeError("Gmail account has no secret reference")
    credentials_data = CredentialStore().get(account.secret_ref)
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Gmail API client is missing: install mana-agent[email]") from exc
    credentials = Credentials(**credentials_data)
    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
    return GmailProvider(account=account, service=service)


def _refresh_gmail_token(account: Any) -> bool:
    if not account.secret_ref:
        return False
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        from mana_agent.connectors.email.auth.credential_store import CredentialStore

        store = CredentialStore()
        data = store.get(account.secret_ref)
        credentials = Credentials(**data)
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            store.put(
                {
                    "token": credentials.token,
                    "refresh_token": credentials.refresh_token,
                    "token_uri": credentials.token_uri,
                    "client_id": credentials.client_id,
                    "client_secret": credentials.client_secret,
                    "scopes": list(credentials.scopes or []),
                },
                reference=account.secret_ref,
            )
            return True
        return bool(credentials.valid)
    except Exception:
        logger.exception("gmail token refresh failed for %s", account.id)
        return False
