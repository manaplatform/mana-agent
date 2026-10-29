"""Conservative equivalence judgment for a user-approved batch of requests.

This is not an authorization decision: each matching request must still be
individually authorized and granted by the normal inbox/action machinery.
"""
from __future__ import annotations

from typing import Any

from mana_agent.human_inbox.models import InboxRequestType, ResponseOperation
from mana_agent.transactional_actions.models import ActionState, PolicyOutcome


def same_approval_request(reference: Any, candidate: Any, *, action_store: Any = None) -> bool:
    """Return true only when the available authoritative material proves equality.

    Missing data, redacted-only general requests, different scopes and lookup
    failures are deliberately treated as non-matches.
    """
    try:
        if reference.inbox_item_id == candidate.inbox_item_id:
            return False
        if reference.request_type is not InboxRequestType.APPROVAL or candidate.request_type is not InboxRequestType.APPROVAL:
            return False
        if ResponseOperation.APPROVE not in candidate.allowed_responses:
            return False
        for field in ("tenant_id", "project_id", "task_id", "root_task_id", "branch_id",
                      "parent_task_id", "execution_id", "attempt_id", "execution_attempt_id",
                      "checkpoint_id", "integration_stage", "recovery_reason", "intervention_id",
                      "recovery_intervention_id", "requested_by_agent_id", "assigned_reviewer_type",
                      "assigned_reviewer_id", "risk_level"):
            if getattr(reference, field) != getattr(candidate, field):
                return False
        if not reference.task_id:
            return False
        if bool(reference.action_intent_id) != bool(candidate.action_intent_id):
            return False
        if not reference.action_intent_id:
            # Do not infer equivalence from generic titles or redacted context.
            return bool(
                reference.minimal_context
                and not reference.protected_context_ref and not candidate.protected_context_ref
                and reference.title == candidate.title and reference.summary == candidate.summary
                and reference.minimal_context == candidate.minimal_context
                and reference.reversibility == candidate.reversibility
                and reference.action_id == candidate.action_id
                and reference.policy_decision_id == candidate.policy_decision_id
                and reference.disclosed_fields == candidate.disclosed_fields
            )
        if action_store is None:
            return False
        a = action_store.get_action(reference.action_intent_id)
        b = action_store.get_action(candidate.action_intent_id)
        if a is None or b is None or not a.policy_decision or not b.policy_decision:
            return False
        # These adapters bind even redacted invocation data to a fingerprint.
        # Other adapters may persist only redacted arguments, which is not
        # sufficient evidence that two invocations are identical.
        required_fingerprints = {
            "shell": ("argv_fingerprint", "environment_fingerprint"),
            "mcp": ("arguments_sha256",),
        }
        required = required_fingerprints.get(a.tool_name)
        if not required or any(not a.normalized_arguments.get(key) or not b.normalized_arguments.get(key) for key in required):
            return False
        if b.state is not ActionState.AWAITING_APPROVAL:
            return False
        if any(x.policy_decision.outcome is not PolicyOutcome.REQUIRE_APPROVAL for x in (a, b)):
            return False
        if not a.preview or not b.preview or not a.preview_digest or not b.preview_digest:
            return False
        if any(x.protected_context_ref and not x.protected_context_digest for x in (a, b)):
            return False
        if reference.action_digest != a.approval_digest() or candidate.action_digest != b.approval_digest():
            return False
        fields = ("parent_task_id", "transaction_id", "actor", "originating_agent", "tool_name",
                  "operation_name", "target_resources", "normalized_arguments", "requested_capabilities",
                  "expected_side_effects", "data_disclosure", "blast_radius", "reversibility",
                  "preview_digest", "protected_context_digest", "verification_plan", "compensation_strategy")
        return all(getattr(a, field) == getattr(b, field) for field in fields) and (
            a.policy_decision.policy_fingerprint == b.policy_decision.policy_fingerprint
            and a.policy_decision.required_approval_scope == b.policy_decision.required_approval_scope
            and a.policy_decision.reason_codes == b.policy_decision.reason_codes
            and a.policy_decision.matched_rules == b.policy_decision.matched_rules
        )
    except Exception:
        return False
