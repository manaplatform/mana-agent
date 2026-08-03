from __future__ import annotations

import hashlib
import json
import uuid
import getpass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mana_agent.utils.redaction import redact_secrets


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ActionState(str, Enum):
    PROPOSED = "proposed"
    PREVIEWING = "previewing"
    AWAITING_POLICY = "awaiting_policy"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMMITTED = "committed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_ACTION_STATES = {
    ActionState.COMMITTED,
    ActionState.COMPENSATED,
    ActionState.FAILED,
    ActionState.CANCELLED,
    ActionState.EXPIRED,
}

VALID_TRANSITIONS: dict[ActionState, frozenset[ActionState]] = {
    ActionState.PROPOSED: frozenset({ActionState.PREVIEWING, ActionState.CANCELLED, ActionState.EXPIRED}),
    ActionState.PREVIEWING: frozenset({ActionState.AWAITING_POLICY, ActionState.FAILED, ActionState.CANCELLED, ActionState.EXPIRED}),
    ActionState.AWAITING_POLICY: frozenset({ActionState.AWAITING_APPROVAL, ActionState.APPROVED, ActionState.FAILED, ActionState.CANCELLED, ActionState.EXPIRED}),
    ActionState.AWAITING_APPROVAL: frozenset({ActionState.APPROVED, ActionState.CANCELLED, ActionState.EXPIRED}),
    ActionState.APPROVED: frozenset({ActionState.EXECUTING, ActionState.CANCELLED, ActionState.EXPIRED}),
    ActionState.EXECUTING: frozenset({ActionState.VERIFYING, ActionState.FAILED}),
    ActionState.VERIFYING: frozenset({ActionState.COMMITTED, ActionState.COMPENSATING, ActionState.FAILED}),
    ActionState.COMMITTED: frozenset({ActionState.COMPENSATING}),
    ActionState.COMPENSATING: frozenset({ActionState.COMPENSATED, ActionState.FAILED}),
    ActionState.COMPENSATED: frozenset(),
    ActionState.FAILED: frozenset({ActionState.COMPENSATING}),
    ActionState.CANCELLED: frozenset(),
    ActionState.EXPIRED: frozenset(),
}


class Reversibility(str, Enum):
    FULLY_REVERSIBLE = "fully_reversible"
    COMPENSATABLE = "compensatable"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class DataDisclosure(str, Enum):
    NONE = "none"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    EXTERNAL_PUBLIC = "external_public"
    EXTERNAL_PRIVATE = "external_private"
    UNKNOWN = "unknown"


class BlastRadius(str, Enum):
    SINGLE_RESOURCE = "single_resource"
    MULTIPLE_RESOURCES = "multiple_resources"
    WORKSPACE = "workspace"
    EXTERNAL_ACCOUNT = "external_account"
    ORGANISATION = "organisation"
    PHYSICAL = "physical"
    UNKNOWN = "unknown"


class PolicyOutcome(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class ApprovalScope(str, Enum):
    ACTION_ONCE = "action_once"
    TRANSACTION = "transaction"


class TransactionStrategy(str, Enum):
    STOP_ON_FAILURE = "stop_on_failure"
    CONTINUE_SAFE_ACTIONS = "continue_safe_actions"
    COMPENSATE_COMPLETED_ACTIONS = "compensate_completed_actions"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"


class ActionPreview(StrictModel):
    summary: str
    resources: list[dict[str, Any]] = Field(default_factory=list)
    diff: str = ""
    exact_invocation: dict[str, Any] = Field(default_factory=dict)
    expected_side_effects: list[str] = Field(default_factory=list)
    disclosed_data: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    externally_visible: bool | None = None
    potentially_billable: bool | None = None
    supports_native_idempotency: bool = False
    supports_dry_run: bool = False

    def redacted(self) -> dict[str, Any]:
        # Omit unknown tri-state labels so existing stored preview digests remain
        # stable across the schema addition; explicitly declared labels are bound.
        return redact_secrets(self.model_dump(mode="json", exclude_none=True))

    def digest(self) -> str:
        payload = json.dumps(self.redacted(), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PolicyDecision(StrictModel):
    decision_id: str = ""
    outcome: PolicyOutcome
    reason_codes: list[str] = Field(min_length=1)
    explanation: str
    matched_rules: list[str] = Field(default_factory=list)
    required_approval_scope: ApprovalScope | None = None
    decided_at: datetime = Field(default_factory=utc_now)
    policy_fingerprint: str
    expires_at: datetime
    assigned_reviewer_type: str = "person"
    assigned_reviewer_id: str = Field(default_factory=getpass.getuser)

    @model_validator(mode="before")
    @classmethod
    def derive_compatible_decision_id(cls, value: Any) -> Any:
        if isinstance(value, dict) and not str(value.get("decision_id") or "").strip():
            value = dict(value)
            material = {key: item for key, item in value.items() if key != "decision_id"}
            encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
            value["decision_id"] = "policy_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
        return value

    @model_validator(mode="after")
    def approval_scope_matches_outcome(self) -> "PolicyDecision":
        if self.outcome is PolicyOutcome.REQUIRE_APPROVAL and self.required_approval_scope is None:
            raise ValueError("approval decisions require an approval scope")
        if self.outcome is not PolicyOutcome.REQUIRE_APPROVAL and self.required_approval_scope is not None:
            raise ValueError("approval scope is valid only for require_approval")
        if self.outcome is PolicyOutcome.REQUIRE_APPROVAL and (
            self.assigned_reviewer_type not in {"person", "group", "role"}
            or not self.assigned_reviewer_id.strip()
        ):
            raise ValueError("approval policy decisions require an explicit reviewer assignment")
        return self


class VerificationEvidence(StrictModel):
    complete: bool
    summary: str
    checks: list[dict[str, Any]] = Field(default_factory=list)
    immutable_remote_id: str = ""
    observed_at: datetime = Field(default_factory=utc_now)


class CompensationEvidence(StrictModel):
    complete: bool
    summary: str
    checks: list[dict[str, Any]] = Field(default_factory=list)
    observed_at: datetime = Field(default_factory=utc_now)


class ActionIntent(StrictModel):
    action_id: str = Field(default_factory=lambda: f"act_{uuid.uuid4().hex}", min_length=8)
    parent_task_id: str
    transaction_id: str = ""
    actor: str
    originating_agent: str
    tool_name: str
    operation_name: str
    target_resources: list[str] = Field(min_length=1)
    normalized_arguments: dict[str, Any]
    requested_capabilities: list[str] = Field(min_length=1)
    expected_side_effects: list[str] = Field(min_length=1)
    data_disclosure: DataDisclosure = DataDisclosure.NONE
    blast_radius: BlastRadius = BlastRadius.SINGLE_RESOURCE
    reversibility: Reversibility = Reversibility.UNKNOWN
    idempotency_key: str = Field(min_length=8, max_length=256)
    verification_plan: list[str] = Field(min_length=1)
    compensation_strategy: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(default_factory=lambda: utc_now() + timedelta(minutes=15))
    state: ActionState = ActionState.PROPOSED
    state_version: int = Field(default=0, ge=0)
    preview: ActionPreview | None = None
    preview_digest: str = ""
    policy_decision: PolicyDecision | None = None
    execution_attempts: int = Field(default=0, ge=0)
    execution_result: dict[str, Any] = Field(default_factory=dict)
    verification: VerificationEvidence | None = None
    compensation: CompensationEvidence | None = None
    error: str = ""
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("normalized_arguments")
    @classmethod
    def arguments_must_not_contain_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        redacted = redact_secrets(value)
        if redacted != value:
            raise ValueError("normalized_arguments must be redacted before persistence")
        return value

    @model_validator(mode="after")
    def timestamps_and_preview_match(self) -> "ActionIntent":
        if self.expires_at <= self.created_at:
            raise ValueError("action expiration must be after creation")
        if self.preview is not None and self.preview_digest != self.preview.digest():
            raise ValueError("preview_digest does not match preview")
        return self

    def binding_digest(self) -> str:
        decision = self.policy_decision.model_dump(mode="json") if self.policy_decision else None
        material = {
            "action_id": self.action_id,
            "transaction_id": self.transaction_id,
            "tool_name": self.tool_name,
            "operation_name": self.operation_name,
            "target_resources": self.target_resources,
            "normalized_arguments": self.normalized_arguments,
            "requested_capabilities": self.requested_capabilities,
            "preview_digest": self.preview_digest,
            "policy_decision": decision,
            "expires_at": self.expires_at.isoformat(),
        }
        encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def intent_digest(self) -> str:
        """Stable material identity before preview and policy enrich the action."""
        material = {
            "transaction_id": self.transaction_id,
            "parent_task_id": self.parent_task_id,
            "actor": self.actor,
            "originating_agent": self.originating_agent,
            "tool_name": self.tool_name,
            "operation_name": self.operation_name,
            "target_resources": self.target_resources,
            "normalized_arguments": self.normalized_arguments,
            "requested_capabilities": self.requested_capabilities,
            "expected_side_effects": self.expected_side_effects,
            "data_disclosure": self.data_disclosure.value,
            "blast_radius": self.blast_radius.value,
            "reversibility": self.reversibility.value,
            "idempotency_key": self.idempotency_key,
            "verification_plan": self.verification_plan,
            "compensation_strategy": self.compensation_strategy,
        }
        encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def approval_digest(self) -> str:
        """Canonical human authorization boundary for the exact action."""
        material = {
            "action_type": {"tool": self.tool_name, "operation": self.operation_name},
            "target_resources": self.target_resources,
            "parameters": self.normalized_arguments,
            "disclosed_side_effects": self.expected_side_effects,
            "risk_classification": {
                "reversibility": self.reversibility.value,
                "data_disclosure": self.data_disclosure.value,
                "blast_radius": self.blast_radius.value,
                "effect_labels": self.approval_effect_labels(),
            },
            "preview_digest": self.preview_digest,
            "policy_decision_id": self.policy_decision.decision_id if self.policy_decision else "",
            "policy_fingerprint": self.policy_decision.policy_fingerprint if self.policy_decision else "",
            "expires_at": self.expires_at.isoformat(),
        }
        encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def approval_effect_labels(self) -> dict[str, bool | None]:
        """Return explicit human-facing effect labels without guessing unknowns."""
        reversibility_unknown = self.reversibility is Reversibility.UNKNOWN
        disclosure_unknown = self.data_disclosure is DataDisclosure.UNKNOWN
        return {
            "reversible": (
                None
                if reversibility_unknown
                else self.reversibility
                in {Reversibility.FULLY_REVERSIBLE, Reversibility.PARTIALLY_REVERSIBLE}
            ),
            "compensatable": (
                None
                if reversibility_unknown
                else self.reversibility is Reversibility.COMPENSATABLE
            ),
            "irreversible": (
                None
                if reversibility_unknown
                else self.reversibility is Reversibility.IRREVERSIBLE
            ),
            "externally_visible": (
                self.preview.externally_visible if self.preview is not None else None
            ),
            "data_disclosing": (
                None
                if disclosure_unknown
                else self.data_disclosure is not DataDisclosure.NONE
            ),
            "potentially_billable": (
                self.preview.potentially_billable if self.preview is not None else None
            ),
        }

    def transition(self, target: ActionState) -> None:
        if target not in VALID_TRANSITIONS[self.state]:
            raise ValueError(f"invalid action transition: {self.state.value} -> {target.value}")
        self.state = target
        self.state_version += 1
        self.updated_at = utc_now()


class TransactionIntent(StrictModel):
    transaction_id: str = Field(default_factory=lambda: f"txn_{uuid.uuid4().hex}")
    parent_task_id: str
    action_ids: list[str] = Field(min_length=1)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    shared_policy_context: dict[str, Any] = Field(default_factory=dict)
    transaction_preview: ActionPreview | None = None
    per_action_reversibility: dict[str, Reversibility] = Field(default_factory=dict)
    strategy: TransactionStrategy = TransactionStrategy.STOP_ON_FAILURE
    commit_conditions: list[str] = Field(min_length=1)
    compensation_plan: list[str] = Field(default_factory=list)
    coordinated_not_atomic: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    final_verification_summary: str = ""
    manual_recovery_required: bool = False

    @model_validator(mode="after")
    def dependencies_reference_members(self) -> "TransactionIntent":
        members = set(self.action_ids)
        if any(action not in members or any(dep not in members for dep in deps) for action, deps in self.dependencies.items()):
            raise ValueError("transaction dependencies must reference member actions")
        return self

    def binding_digest(self) -> str:
        """Bind approvals to the immutable coordination plan, not its runtime summary."""
        material = {
            "transaction_id": self.transaction_id,
            "parent_task_id": self.parent_task_id,
            "action_ids": self.action_ids,
            "dependencies": self.dependencies,
            "shared_policy_context": self.shared_policy_context,
            "transaction_preview": self.transaction_preview.model_dump(mode="json") if self.transaction_preview else None,
            "per_action_reversibility": {
                key: value.value for key, value in self.per_action_reversibility.items()
            },
            "strategy": self.strategy.value,
            "commit_conditions": self.commit_conditions,
            "compensation_plan": self.compensation_plan,
            "coordinated_not_atomic": self.coordinated_not_atomic,
        }
        encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
