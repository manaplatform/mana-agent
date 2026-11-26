"""Transactional execution for model-selected automation creation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from mana_agent.automations.service import AutomationService, AutomationValidationError
from mana_agent.utils.redaction import redact_secrets
from mana_agent.transactional_actions.adapters import ActionAdapter
from mana_agent.transactional_actions.models import (
    ActionIntent, ActionPreview, BlastRadius, DataDisclosure, Reversibility,
    VerificationEvidence,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class AutomationCreateAdapter(ActionAdapter):
    def __init__(self, *, root: Path, arguments: dict[str, Any], invoke: Callable[[], Any], parent_task_id: str) -> None:
        self.root = Path(root).resolve()
        self.arguments = dict(arguments)
        self.invoke = invoke
        self.parent_task_id = str(parent_task_id).strip()
        self.decision_id = str(arguments.get("source_decision_id") or "").strip()
        if not self.parent_task_id or not self.decision_id:
            raise ValueError("Automation creation requires a parent task and source decision")
        self.arguments_sha256 = _digest(self.arguments)
        self.target_runtime = str(arguments.get("target_runtime") or "local")

    def build_intent(self) -> ActionIntent:
        return ActionIntent(
            parent_task_id=self.parent_task_id, actor="model_tool", originating_agent="ask_agent",
            tool_name="automation", operation_name="create",
            target_resources=[f"automation://{_digest(str(self.root))}"],
            normalized_arguments={"source_decision_id": self.decision_id, "arguments_sha256": self.arguments_sha256, "target_runtime": self.target_runtime},
            requested_capabilities=["automation.create"],
            expected_side_effects=["Persist and deploy a scheduled automation; its job may execute later."],
            data_disclosure=DataDisclosure.UNKNOWN, blast_radius=BlastRadius.SINGLE_RESOURCE,
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            idempotency_key=f"automation:create:{_digest(self.parent_task_id)}:{self.arguments_sha256}",
            verification_plan=["Read the created automation from the durable store and compare its snapshot."],
            compensation_strategy="Use a separately authorized automation deletion.",
        )

    def preview(self, action: ActionIntent) -> ActionPreview:
        return ActionPreview(
            summary="Create and deploy a scheduled automation",
            resources=[{"workspace": str(self.root), "target_runtime": self.target_runtime}],
            exact_invocation=redact_secrets(self.arguments),
            expected_side_effects=action.expected_side_effects,
            risks=["The scheduled job may have future side effects."],
            externally_visible=None, potentially_billable=None,
        )

    def protected_action_context(self) -> dict[str, Any]:
        return {"tool_name": "automation_create", "arguments": self.arguments}

    def execute(self, action: ActionIntent) -> dict[str, Any]:
        raw = self.invoke()
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            detail = payload.get("message") or payload.get("error") if isinstance(payload, dict) else None
            raise RuntimeError(str(redact_secrets(detail or "Automation creation returned an invalid result"))[:1000])
        record = payload.get("persisted")
        if not isinstance(record, dict) or not record.get("id"):
            raise RuntimeError("Automation creation returned no persisted definition")
        # ToolCallGate returns a model-facing, redacted projection that can omit
        # job fields such as prompt. Fingerprint the authoritative store instead.
        persisted = AutomationService(self.root).get(str(record["id"]))
        return {"automation_id": persisted.id, "snapshot_sha256": _digest(persisted.to_dict())}

    def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
        try:
            record = AutomationService(self.root).get(str(result.get("automation_id") or ""))
            from mana_agent.automations.runtime_tools import _Create

            requested = _Create.model_validate(self.arguments)
            expected_description = requested.description
            if requested.idempotency_key:
                expected_description = f"{expected_description.rstrip()} [idempotency:{requested.idempotency_key}]".strip()
            matches = (
                record.description == expected_description
                and _digest(record.to_dict()) == result.get("snapshot_sha256")
                and record.name == requested.name
                and record.trigger == requested.trigger
                and record.job == requested.job
                and record.timezone == requested.timezone
                and record.target_runtime == requested.target_runtime
                and record.retry_policy == requested.retry_policy
                and record.misfire_policy == requested.misfire_policy
                and record.idempotency_key == requested.idempotency_key
                and set(record.permission_references) >= set(requested.permission_references)
            )
        except (AutomationValidationError, ValueError):
            matches = False
        return VerificationEvidence(
            complete=matches,
            summary="Persisted automation snapshot matches the creation result." if matches else "Persisted automation could not be verified against the creation result.",
            checks=[{"check": "persisted_automation_matches", "observed": matches}],
        )
