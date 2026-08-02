from __future__ import annotations

from dataclasses import dataclass, field

from .adapters import ActionAdapter
from .gateway import ActionGateway, ActionOutcome
from .models import ActionPreview, ActionState, TransactionIntent, TransactionStrategy, utc_now
from .store import ActionStore


@dataclass
class TransactionOutcome:
    transaction: TransactionIntent
    actions: list[ActionOutcome] = field(default_factory=list)
    failed_action_ids: list[str] = field(default_factory=list)
    compensated_action_ids: list[str] = field(default_factory=list)


class TransactionCoordinator:
    """Coordinate ordered actions without claiming cross-adapter atomicity."""

    def __init__(self, gateway: ActionGateway, store: ActionStore) -> None:
        self.gateway, self.store = gateway, store

    def create(self, transaction: TransactionIntent) -> TransactionIntent:
        if not transaction.coordinated_not_atomic:
            raise ValueError("cross-adapter transactions must be labeled coordinated_not_atomic")
        actions = []
        for action_id in transaction.action_ids:
            action = self.store.get_action(action_id)
            if action is None or action.transaction_id != transaction.transaction_id:
                raise ValueError("transaction actions must be proposed and bound to this transaction")
            if action.preview is None or action.policy_decision is None:
                raise ValueError("transaction actions require previews and policy decisions")
            actions.append(action)
        transaction.per_action_reversibility = {
            action.action_id: action.reversibility for action in actions
        }
        transaction.transaction_preview = ActionPreview(
            summary=f"coordinate {len(actions)} policy-gated actions (not atomic)",
            resources=[
                {
                    "action_id": action.action_id,
                    "tool": action.tool_name,
                    "operation": action.operation_name,
                    "targets": action.target_resources,
                    "preview_digest": action.preview_digest,
                    "policy_outcome": action.policy_decision.outcome.value,
                    "reversibility": action.reversibility.value,
                }
                for action in actions
            ],
            expected_side_effects=[
                effect for action in actions for effect in action.expected_side_effects
            ],
            risks=[
                "coordination spans actions that cannot provide cross-adapter atomicity",
                *[
                    f"{action.action_id}: {risk}"
                    for action in actions
                    for risk in (action.preview.risks if action.preview else [])
                    if risk
                ],
            ],
        )
        self.store.save_transaction(transaction)
        return transaction

    def execute(self, transaction_id: str, adapters: dict[str, ActionAdapter], approvals: dict[str, str] | None = None) -> TransactionOutcome:
        transaction = self.store.get_transaction(transaction_id)
        if transaction is None:
            raise LookupError("unknown transaction")
        if set(adapters) != set(transaction.action_ids):
            raise ValueError("transaction adapters must exactly match transaction actions")
        outcome = TransactionOutcome(transaction=transaction)
        completed: set[str] = set()
        for action_id in transaction.action_ids:
            unmet = set(transaction.dependencies.get(action_id, [])) - completed
            if unmet:
                outcome.failed_action_ids.append(action_id)
                transaction.manual_recovery_required = True
                if transaction.strategy is not TransactionStrategy.CONTINUE_SAFE_ACTIONS:
                    break
                continue
            try:
                action_outcome = self.gateway.execute(adapters[action_id], approval_id=(approvals or {}).get(action_id, ""))
            except Exception:
                outcome.failed_action_ids.append(action_id)
                if transaction.strategy in {TransactionStrategy.STOP_ON_FAILURE, TransactionStrategy.COMPENSATE_COMPLETED_ACTIONS, TransactionStrategy.MANUAL_RECOVERY_REQUIRED}:
                    break
                continue
            outcome.actions.append(action_outcome)
            if action_outcome.action.state is ActionState.COMMITTED:
                completed.add(action_id)
            else:
                outcome.failed_action_ids.append(action_id)
                if transaction.strategy is not TransactionStrategy.CONTINUE_SAFE_ACTIONS:
                    break
        if outcome.failed_action_ids:
            if transaction.strategy is TransactionStrategy.COMPENSATE_COMPLETED_ACTIONS:
                for completed_outcome in reversed(outcome.actions):
                    if completed_outcome.action.state is not ActionState.COMMITTED:
                        continue
                    try:
                        compensation_action = self.gateway.compensate(
                            completed_outcome.action.action_id,
                            adapters[completed_outcome.action.action_id],
                            approval_id=(approvals or {}).get(
                                f"compensate:{completed_outcome.action.action_id}", ""
                            ),
                        )
                    except Exception:
                        transaction.manual_recovery_required = True
                        continue
                    if compensation_action.state is ActionState.COMMITTED:
                        outcome.compensated_action_ids.append(completed_outcome.action.action_id)
                    else:
                        transaction.manual_recovery_required = True
            else:
                transaction.manual_recovery_required = True
        uncompensated = completed - set(outcome.compensated_action_ids)
        transaction.final_verification_summary = (
            f"{len(uncompensated)} of {len(transaction.action_ids)} actions remain committed with complete verification; "
            f"{len(outcome.compensated_action_ids)} were compensated by separately gated actions; "
            f"{len(outcome.failed_action_ids)} failed. This transaction was coordinated, not atomic."
        )
        transaction.updated_at = utc_now()
        self.store.save_transaction(transaction)
        return outcome
