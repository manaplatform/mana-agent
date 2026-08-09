"""Telegram connector health adapter (polling/webhook path awareness)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
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


class TelegramHealthAdapter:
    """Health for Telegram is independent of the background process state.

    A running gateway/process with a broken subscription or stalled poller is
    DEGRADED / OFFLINE, never HEALTHY.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        transport: str = "polling",
        runtime_alive: bool = False,
        client_factory: Callable[[], Any] | None = None,
        status_provider: Callable[[], dict[str, Any]] | None = None,
        webhook_info_provider: Callable[[], Any] | None = None,
        restart_transport: Callable[[], Any] | None = None,
        reregister_webhook: Callable[[], Any] | None = None,
        synthetic_mode: SyntheticProbeMode = SyntheticProbeMode.SAFE_ENDPOINT,
        ingress_stall_seconds: float = 600.0,
        delivery_receipts: list[DeliveryReceipt] | None = None,
    ) -> None:
        self._enabled = enabled
        self._transport = transport
        self._runtime_alive = runtime_alive
        self._client_factory = client_factory
        self._status_provider = status_provider
        self._webhook_info_provider = webhook_info_provider
        self._restart_transport = restart_transport
        self._reregister_webhook = reregister_webhook
        self._synthetic_mode = synthetic_mode
        self._ingress_stall_seconds = ingress_stall_seconds
        self._delivery_receipts = list(delivery_receipts or [])
        self._last_auth_ok = False
        self._last_transport_ok = False
        self._client: Any = None

    @property
    def connector_id(self) -> str:
        return "telegram"

    @property
    def connector_type(self) -> str:
        return "telegram"

    def health_capabilities(self) -> ConnectorHealthCapabilities:
        return ConnectorHealthCapabilities(
            auth=True,
            connectivity=True,
            ingress=True,
            egress=True,
            subscriptions=self._transport == "webhook",
            acknowledgements=True,
        )

    def supported_probe_categories(self) -> list[ProbeCategory]:
        categories = [
            ProbeCategory.AUTH,
            ProbeCategory.CONNECTIVITY,
            ProbeCategory.INGRESS,
            ProbeCategory.EGRESS,
            ProbeCategory.ACKNOWLEDGEMENT,
        ]
        if self._transport == "webhook":
            categories.append(ProbeCategory.SUBSCRIPTION)
        return categories

    def synthetic_probe_mode(self) -> SyntheticProbeMode:
        return self._synthetic_mode

    def is_enabled(self) -> bool:
        return self._enabled

    def describe(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "connector_type": self.connector_type,
            "transport": self._transport,
            "runtime_alive": self._runtime_alive,
            "enabled": self._enabled,
        }

    def set_runtime_alive(self, alive: bool) -> None:
        self._runtime_alive = alive

    def collect_signals(self) -> PathSignals:
        status = self._status()
        last_error = status.get("last_error")
        return PathSignals(
            runtime_alive=self._runtime_alive or bool(status.get("running")),
            transport_connected=self._last_transport_ok and not last_error,
            authenticated=CapabilitySignal.OK if self._last_auth_ok else CapabilitySignal.UNKNOWN,
            ingress_operational=CapabilitySignal.UNKNOWN,
            egress_operational=CapabilitySignal.UNKNOWN,
            subscription_operational=(
                CapabilitySignal.UNKNOWN
                if self._transport == "webhook"
                else CapabilitySignal.NOT_APPLICABLE
            ),
            acknowledgements_operational=CapabilitySignal.UNKNOWN,
        )

    def list_recovery_actions(self, reason_codes: list[str]) -> list[RecoveryActionKind]:
        actions = [
            RecoveryActionKind.CLIENT_RECREATE,
            RecoveryActionKind.TRANSPORT_RECONNECT,
            RecoveryActionKind.POLLER_RESTART,
            RecoveryActionKind.CONSUMER_RESTART,
        ]
        if self._transport == "webhook":
            actions.append(RecoveryActionKind.WEBHOOK_REREGISTER)
            actions.append(RecoveryActionKind.SUBSCRIPTION_RENEW)
        return actions

    def recent_delivery_receipts(self, *, limit: int = 20) -> list[DeliveryReceipt]:
        return self._delivery_receipts[-limit:]

    def record_delivery(self, receipt: DeliveryReceipt) -> None:
        self._delivery_receipts.append(receipt)

    async def run_probe(self, category: ProbeCategory, *, mode: SyntheticProbeMode) -> ProbeResult:
        if category is ProbeCategory.AUTH:
            return await self._probe_auth()
        if category is ProbeCategory.CONNECTIVITY:
            return await self._probe_auth()  # getMe is connectivity+auth for Telegram
        if category is ProbeCategory.INGRESS:
            return self._probe_ingress()
        if category is ProbeCategory.EGRESS:
            return self._probe_egress(mode)
        if category is ProbeCategory.SUBSCRIPTION:
            return await self._probe_subscription()
        if category is ProbeCategory.ACKNOWLEDGEMENT:
            return self._probe_ack()
        return ProbeResult(
            category=category,
            outcome=ProbeOutcome.UNSUPPORTED,
            message=f"Telegram does not support probe {category.value}",
        )

    async def execute_recovery(self, action: RecoveryActionKind) -> ProbeResult:
        if action in {
            RecoveryActionKind.CLIENT_RECREATE,
            RecoveryActionKind.TRANSPORT_RECONNECT,
            RecoveryActionKind.POLLER_RESTART,
            RecoveryActionKind.CONSUMER_RESTART,
        }:
            self._client = None
            if self._restart_transport is not None:
                try:
                    await _maybe_await(self._restart_transport())
                except Exception as exc:
                    return failed(
                        ProbeCategory.CONNECTIVITY,
                        HealthReasonCode.RECONNECT_FAILED,
                        f"Transport restart failed: {type(exc).__name__}",
                    )
            probe = await self._probe_auth()
            if probe.outcome is ProbeOutcome.PASSED:
                return passed(ProbeCategory.CONNECTIVITY, message=f"Recovery {action.value} succeeded")
            return failed(
                ProbeCategory.CONNECTIVITY,
                HealthReasonCode.RECONNECT_FAILED,
                probe.message,
            )
        if action in {
            RecoveryActionKind.WEBHOOK_REREGISTER,
            RecoveryActionKind.SUBSCRIPTION_RENEW,
        }:
            if self._reregister_webhook is None:
                return failed(
                    ProbeCategory.SUBSCRIPTION,
                    HealthReasonCode.SUBSCRIPTION_MISSING,
                    "Webhook re-registration handler not configured",
                )
            try:
                ok = await _maybe_await(self._reregister_webhook())
            except Exception as exc:
                return failed(
                    ProbeCategory.SUBSCRIPTION,
                    HealthReasonCode.WEBHOOK_UNREACHABLE,
                    f"Webhook re-register failed: {type(exc).__name__}",
                )
            if ok:
                return passed(ProbeCategory.SUBSCRIPTION, message="Webhook re-registered")
            return failed(
                ProbeCategory.SUBSCRIPTION,
                HealthReasonCode.WEBHOOK_UNREACHABLE,
                "Webhook re-registration returned failure",
            )
        return failed(
            ProbeCategory.CONNECTIVITY,
            HealthReasonCode.RECONNECT_FAILED,
            f"Unsupported recovery {action.value}",
        )

    async def _probe_auth(self) -> ProbeResult:
        started = time.perf_counter()
        try:
            client = self._get_client()
            identity = await client.get_me()
            latency = (time.perf_counter() - started) * 1000
            self._last_auth_ok = True
            self._last_transport_ok = True
            username = getattr(identity, "username", None) or getattr(identity, "id", "?")
            return passed(
                ProbeCategory.AUTH,
                latency_ms=latency,
                message=f"Telegram auth OK (@{username})",
            )
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            self._last_auth_ok = False
            self._last_transport_ok = False
            name = type(exc).__name__
            text = str(exc)
            if "401" in text or "Unauthorized" in text or "auth" in name.lower():
                return failed(
                    ProbeCategory.AUTH,
                    HealthReasonCode.AUTH_REVOKED,
                    "Telegram bot token rejected",
                    latency_ms=latency,
                    details={"exception_type": name},
                )
            if "429" in text or "rate" in name.lower():
                retry_after = getattr(exc, "retry_after", 30.0)
                return rate_limited(ProbeCategory.AUTH, "Telegram rate limited", retry_after=float(retry_after or 30))
            if "timeout" in text.lower() or "Timeout" in name:
                return failed(
                    ProbeCategory.CONNECTIVITY,
                    HealthReasonCode.CONNECTION_TIMEOUT,
                    "Telegram connectivity timeout",
                    latency_ms=latency,
                )
            return failed(
                ProbeCategory.AUTH,
                HealthReasonCode.PROBE_FAILED,
                f"Telegram auth probe failed: {name}",
                latency_ms=latency,
                details={"exception_type": name},
            )

    def _probe_ingress(self) -> ProbeResult:
        """Detect silent ingress failure without treating quiet chats as outages.

        For polling: successful poll loop and offset progression capability.
        For webhook: subscription validity checked separately; here we inspect
        queue consumer / last error / last completed update freshness relative
        to a *running* expectation only when the transport claims to be running.
        """
        status = self._status()
        last_error = status.get("last_error")
        if last_error:
            return failed(
                ProbeCategory.INGRESS,
                HealthReasonCode.INGRESS_STALLED,
                f"Telegram ingress error: {str(last_error)[:200]}",
            )
        running = bool(status.get("running"))
        # Process running alone is not enough — need transport evidence.
        if self._runtime_alive and not running and not self._last_transport_ok:
            return failed(
                ProbeCategory.INGRESS,
                HealthReasonCode.PROCESS_ONLY_ALIVE,
                "Telegram process/runtime is alive but transport is not operational",
            )
        queue = status.get("queue") or {}
        if isinstance(queue, dict):
            failed_count = int(queue.get("failed") or queue.get("failed_count") or 0)
            if failed_count > 0 and not status.get("last_completed_update"):
                return failed(
                    ProbeCategory.INGRESS,
                    HealthReasonCode.INGRESS_STALLED,
                    "Telegram queue has failures and no completed updates",
                )
        last_completed = status.get("last_completed_update")
        # Absence of messages is OK. Only fail when transport is claimed running
        # but the poller/webhook path has a known structural problem.
        if running or self._last_transport_ok:
            return passed(
                ProbeCategory.INGRESS,
                message="Telegram ingress mechanism operational"
                + (f" (last_update={last_completed})" if last_completed else " (quiet is healthy)"),
            )
        if self._runtime_alive:
            return failed(
                ProbeCategory.INGRESS,
                HealthReasonCode.INGRESS_STALLED,
                "Runtime alive but Telegram ingress is not running",
            )
        return ProbeResult(
            category=ProbeCategory.INGRESS,
            outcome=ProbeOutcome.SKIPPED,
            message="Telegram connector is not running; ingress not expected",
        )

    def _probe_egress(self, mode: SyntheticProbeMode) -> ProbeResult:
        if mode in {SyntheticProbeMode.ACTIVE, SyntheticProbeMode.TEST_CHANNEL}:
            return ProbeResult(
                category=ProbeCategory.EGRESS,
                outcome=ProbeOutcome.SKIPPED,
                message="Active Telegram message probes require explicit test_channel configuration",
            )
        receipts = self.recent_delivery_receipts(limit=10)
        if not receipts:
            return ProbeResult(
                category=ProbeCategory.EGRESS,
                outcome=ProbeOutcome.SKIPPED,
                message="No recent Telegram delivery receipts",
            )
        latest = receipts[-1]
        if latest.state is DeliveryState.FAILED:
            return failed(
                ProbeCategory.EGRESS,
                HealthReasonCode.EGRESS_FAILED,
                latest.failure_reason or "Telegram send failed",
            )
        return passed(ProbeCategory.EGRESS, message=f"Recent delivery state={latest.state.value}")

    async def _probe_subscription(self) -> ProbeResult:
        if self._transport != "webhook":
            return ProbeResult(
                category=ProbeCategory.SUBSCRIPTION,
                outcome=ProbeOutcome.UNSUPPORTED,
                message="Subscription probe applies to webhook transport",
            )
        try:
            if self._webhook_info_provider is not None:
                info = await _maybe_await(self._webhook_info_provider())
            else:
                client = self._get_client()
                if not hasattr(client, "get_webhook_info"):
                    return ProbeResult(
                        category=ProbeCategory.SUBSCRIPTION,
                        outcome=ProbeOutcome.SKIPPED,
                        message="Webhook info API unavailable",
                    )
                info = await client.get_webhook_info()
            url = ""
            if isinstance(info, dict):
                url = str(info.get("url") or "")
                last_error = info.get("last_error_message") or info.get("last_error")
            else:
                url = str(getattr(info, "url", "") or "")
                last_error = getattr(info, "last_error_message", None)
            if not url:
                return failed(
                    ProbeCategory.SUBSCRIPTION,
                    HealthReasonCode.SUBSCRIPTION_MISSING,
                    "Telegram webhook subscription is missing",
                )
            if last_error:
                return failed(
                    ProbeCategory.SUBSCRIPTION,
                    HealthReasonCode.WEBHOOK_UNREACHABLE,
                    f"Webhook last error: {str(last_error)[:200]}",
                )
            return passed(ProbeCategory.SUBSCRIPTION, message="Webhook subscription present")
        except Exception as exc:
            return failed(
                ProbeCategory.SUBSCRIPTION,
                HealthReasonCode.SUBSCRIPTION_EXPIRED,
                f"Webhook probe failed: {type(exc).__name__}",
            )

    def _probe_ack(self) -> ProbeResult:
        receipts = self.recent_delivery_receipts(limit=20)
        if not receipts:
            return ProbeResult(
                category=ProbeCategory.ACKNOWLEDGEMENT,
                outcome=ProbeOutcome.SKIPPED,
                message="No acknowledgement samples",
            )
        now = utc_now()
        for receipt in receipts:
            if (
                receipt.state is DeliveryState.SUBMITTED
                and receipt.submitted_at is not None
                and (now - receipt.submitted_at) > timedelta(seconds=120)
            ):
                return failed(
                    ProbeCategory.ACKNOWLEDGEMENT,
                    HealthReasonCode.ACK_TIMEOUT,
                    "Telegram delivery ack timeout",
                )
        return passed(ProbeCategory.ACKNOWLEDGEMENT, message="Recent Telegram acks present")

    def _status(self) -> dict[str, Any]:
        if self._status_provider is None:
            return {}
        try:
            return dict(self._status_provider() or {})
        except Exception:
            return {}

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is None:
            raise RuntimeError("Telegram client factory is not configured")
        self._client = self._client_factory()
        return self._client


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def discover_telegram_adapter(
    *,
    process_running: bool | None = None,
) -> TelegramHealthAdapter | None:
    try:
        from mana_agent.connectors.telegram.config import load_telegram_config
    except Exception:
        return None
    try:
        config = load_telegram_config()
    except Exception:
        return None
    if not config.enabled and process_running is not True:
        # Still register as disabled so status is accurate.
        return TelegramHealthAdapter(enabled=False, transport=config.effective_transport)

    runtime_alive = bool(process_running)
    if process_running is None:
        try:
            from mana_agent.background.manager import BackgroundProcessManager

            manager = BackgroundProcessManager()
            manager.recover_stale()
            runtime_alive = any(
                row.singleton_key == "connector.telegram" and row.state in {"starting", "running"}
                for row in manager.list()
            )
        except Exception:
            runtime_alive = False

    def client_factory():
        from mana_agent.connectors.telegram.client import TelegramBotClient

        return TelegramBotClient(config.bot_token, timeout_seconds=config.request_timeout_seconds)

    def status_provider() -> dict[str, Any]:
        return {
            "enabled": config.enabled,
            "running": runtime_alive,
            "effective_transport": config.effective_transport,
        }

    async def reregister_webhook():
        client = client_factory()
        try:
            from urllib.parse import urljoin

            base = config.webhook.public_url.rstrip("/") + "/"
            url = urljoin(base, config.webhook.path.lstrip("/"))
            return await client.set_webhook(url, config.webhook_secret, drop_pending_updates=False)
        finally:
            await client.close()

    return TelegramHealthAdapter(
        enabled=config.enabled,
        transport=config.effective_transport,
        runtime_alive=runtime_alive,
        client_factory=client_factory,
        status_provider=status_provider,
        reregister_webhook=reregister_webhook if config.effective_transport == "webhook" else None,
    )
