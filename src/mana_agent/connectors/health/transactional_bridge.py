"""Policy-gated recoveries that mutate external connector state."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .models import RecoveryActionKind
from .recovery import TRANSACTIONAL_ACTIONS

logger = logging.getLogger(__name__)


class ConnectorTransactionalBridge:
    """Route risky recoveries through the policy-gated action lifecycle.

    Simple local reconnects never enter this path. Webhook re-registration and
    subscription mutations must follow propose → preview → policy → approval →
    execute → verify when a gateway is bound.
    """

    def __init__(
        self,
        *,
        action_gateway: Any | None = None,
        auto_allow_local: bool = True,
        decision_fn: Callable[[str, RecoveryActionKind, str], bool] | None = None,
    ) -> None:
        self.action_gateway = action_gateway
        self.auto_allow_local = auto_allow_local
        self.decision_fn = decision_fn
        self._pending: dict[str, dict[str, Any]] = {}

    def authorize(
        self,
        connector_id: str,
        action: RecoveryActionKind,
        reason: str,
    ) -> bool:
        if action not in TRANSACTIONAL_ACTIONS:
            return True
        if self.decision_fn is not None:
            return bool(self.decision_fn(connector_id, action, reason))
        if self.action_gateway is None:
            # Fail closed for external mutations without a policy gateway.
            logger.warning(
                "connector.recovery.policy_denied connector_id=%s action=%s reason=no_gateway",
                connector_id,
                action.value,
            )
            return False
        try:
            proposal = {
                "tool": "connector.health.recovery",
                "connector_id": connector_id,
                "action": action.value,
                "reason": reason,
                "externally_visible": True,
                "reversibility": "compensatable",
            }
            if hasattr(self.action_gateway, "propose_and_decide"):
                decision = self.action_gateway.propose_and_decide(proposal)
                allowed = bool(getattr(decision, "allowed", False) or getattr(decision, "outcome", "") in {"allow", "ALLOW"})
            elif hasattr(self.action_gateway, "evaluate"):
                decision = self.action_gateway.evaluate(proposal)
                allowed = bool(decision)
            else:
                allowed = False
            self._pending[f"{connector_id}:{action.value}"] = {
                "proposal": proposal,
                "allowed": allowed,
            }
            return allowed
        except Exception:
            logger.exception(
                "connector.recovery.policy_error connector_id=%s action=%s",
                connector_id,
                action.value,
            )
            return False
