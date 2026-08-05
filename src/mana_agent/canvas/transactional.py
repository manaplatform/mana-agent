"""Transactional adapter for model-selected durable Canvas mutations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from mana_agent.transactional_actions.adapters import ActionAdapter
from mana_agent.transactional_actions.models import (
    ActionIntent,
    ActionPreview,
    BlastRadius,
    DataDisclosure,
    Reversibility,
    VerificationEvidence,
)
from mana_agent.utils.redaction import redact_secrets


_OPERATIONS = {
    "canvas_create_surface": "create_surface",
    "canvas_update_components": "update_components",
    "canvas_update_data": "update_data",
    "canvas_delete_surface": "delete_surface",
}


class CanvasActionAdapter(ActionAdapter):
    """Execute one validated Canvas mutation through the action gateway."""

    def __init__(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        invoke: Callable[[], Any],
        parent_task_id: str,
        actor: str,
        originating_agent: str,
    ) -> None:
        self.tool_name = str(tool_name).strip()
        self.operation = _OPERATIONS.get(self.tool_name, "")
        if not self.operation:
            raise ValueError(f"unsupported Canvas mutation tool: {self.tool_name}")
        self.arguments = dict(arguments)
        self.invoke = invoke
        self.parent_task_id = str(parent_task_id).strip()
        if not self.parent_task_id:
            raise ValueError("Canvas action requires a durable parent task")
        self.actor = str(actor)
        self.originating_agent = str(originating_agent)
        self.session_id = self._required_argument("session_id")
        self.conversation_id = self._required_argument("conversation_id")
        self.surface_id = self._required_argument("surface_id")
        self.source_decision_id = self._required_argument("source_decision_id")
        encoded = json.dumps(self.arguments, sort_keys=True, ensure_ascii=False, default=str)
        self.arguments_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        task_sha256 = hashlib.sha256(self.parent_task_id.encode("utf-8")).hexdigest()
        self.idempotency_key = (
            f"canvas:{task_sha256}:{self.tool_name}:{self.arguments_sha256}"
        )

    def _required_argument(self, name: str) -> str:
        value = str(self.arguments.get(name) or "").strip()
        if not value:
            raise ValueError(f"Canvas action requires {name}")
        return value

    def build_intent(self) -> ActionIntent:
        return ActionIntent(
            parent_task_id=self.parent_task_id,
            actor=self.actor,
            originating_agent=self.originating_agent,
            tool_name="canvas",
            operation_name=self.operation,
            target_resources=[f"canvas://{self.session_id}/{self.surface_id}"],
            normalized_arguments={
                "tool_name": self.tool_name,
                "session_id": self.session_id,
                "conversation_id": self.conversation_id,
                "surface_id": self.surface_id,
                "source_decision_id": self.source_decision_id,
                "arguments_sha256": self.arguments_sha256,
            },
            requested_capabilities=[f"canvas.{self.operation}"],
            expected_side_effects=[
                f"persist Canvas {self.operation} for {self.surface_id}"
            ],
            data_disclosure=DataDisclosure.NONE,
            blast_radius=BlastRadius.SINGLE_RESOURCE,
            reversibility=(
                Reversibility.FULLY_REVERSIBLE
                if self.operation == "create_surface"
                else Reversibility.PARTIALLY_REVERSIBLE
            ),
            idempotency_key=self.idempotency_key,
            verification_plan=[
                "require an explicit successful Canvas tool result",
                "verify the returned durable surface snapshot matches the selected target",
            ],
            compensation_strategy=(
                "Use a separately model-selected and policy-gated Canvas delete action."
            ),
        )

    def preview(self, action: ActionIntent) -> ActionPreview:
        return ActionPreview(
            summary=f"Canvas {self.operation.replace('_', ' ')}: {self.surface_id}",
            resources=[
                {
                    "session_id": self.session_id,
                    "conversation_id": self.conversation_id,
                    "surface_id": self.surface_id,
                    "operation": self.operation,
                }
            ],
            exact_invocation=action.normalized_arguments,
            expected_side_effects=action.expected_side_effects,
            risks=(
                ["deletes a durable Canvas surface"]
                if self.operation == "delete_surface"
                else ["updates one local durable Canvas surface"]
            ),
            externally_visible=False,
            potentially_billable=False,
        )

    def protected_action_context(self) -> dict[str, Any]:
        """Retain exact model arguments outside the redacted action/audit records."""
        return {"tool_name": self.tool_name, "arguments": self.arguments}

    def execute(self, action: ActionIntent) -> dict[str, Any]:
        raw = self.invoke()
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Canvas tool returned non-JSON output") from exc
        elif isinstance(raw, dict):
            payload = dict(raw)
        else:
            raise RuntimeError("Canvas tool returned an unsupported result type")
        if not bool(payload.get("ok")):
            detail = str(
                payload.get("message")
                or payload.get("error")
                or payload.get("error_code")
                or "Canvas tool returned ok=false without a diagnostic detail"
            )
            raise RuntimeError(redact_secrets(detail)[:1000])
        snapshot = payload.get("result")
        if not isinstance(snapshot, dict):
            raise RuntimeError("Canvas tool did not return a durable surface snapshot")
        return {
            "ok": True,
            "operation": self.operation,
            "surface": snapshot,
        }

    def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
        snapshot = result.get("surface")
        matching_target = isinstance(snapshot, dict) and all(
            str(snapshot.get(name) or "") == expected
            for name, expected in {
                "session_id": self.session_id,
                "conversation_id": self.conversation_id,
                "surface_id": self.surface_id,
            }.items()
        )
        deleted = bool(snapshot.get("deleted")) if isinstance(snapshot, dict) else False
        components = snapshot.get("components") if isinstance(snapshot, dict) else None
        complete_surface = isinstance(components, list) and any(
            isinstance(component, dict) and component.get("id") == "root"
            for component in components
        )
        operation_complete = (
            deleted
            if self.operation == "delete_surface"
            else not deleted and (complete_surface if self.operation == "create_surface" else True)
        )
        complete = bool(result.get("ok")) and matching_target and operation_complete
        return VerificationEvidence(
            complete=complete,
            summary=(
                "Canvas mutation was matched to its returned durable surface snapshot."
                if complete
                else "Canvas mutation did not return a matching durable surface snapshot."
            ),
            checks=[
                {"check": "canvas_result_ok", "observed": bool(result.get("ok"))},
                {"check": "surface_target_matches", "observed": matching_target},
                {"check": "operation_persisted", "observed": operation_complete},
            ],
        )


__all__ = ["CanvasActionAdapter"]
