"""Transactional adapter for model-selected saved API integration changes."""

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
    "api_docs_import": "import",
    "api_docs_import_semantic": "import",
    "api_integration_update": "update",
    "api_integration_delete": "delete",
}


class ApiIntegrationActionAdapter(ActionAdapter):
    """Persist one model-selected API integration change through the action gateway."""

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
            raise ValueError(f"unsupported API integration tool: {self.tool_name}")
        self.arguments = dict(arguments)
        self.invoke = invoke
        self.parent_task_id = str(parent_task_id).strip()
        if not self.parent_task_id:
            raise ValueError("API integration action requires a durable parent task")
        self.actor = str(actor)
        self.originating_agent = str(originating_agent)
        self.source_decision_id = self._required_argument("source_decision_id")
        self.session_id = self._required_argument("session_id")
        self.target = self._target()
        material = json.dumps(redact_secrets(self.arguments), sort_keys=True, ensure_ascii=False, default=str)
        self.arguments_sha256 = hashlib.sha256(material.encode("utf-8")).hexdigest()
        parent_sha256 = hashlib.sha256(self.parent_task_id.encode("utf-8")).hexdigest()
        self.idempotency_key = f"api-integration:{parent_sha256}:{self.tool_name}:{self.arguments_sha256}"

    def _required_argument(self, name: str) -> str:
        value = str(self.arguments.get(name) or "").strip()
        if not value:
            raise ValueError(f"API integration action requires {name}")
        return value

    def _target(self) -> str:
        integration_id = str(self.arguments.get("integration_id") or "").strip()
        if integration_id:
            return f"api-integration://{integration_id}"
        name = str(self.arguments.get("name") or "").strip()
        if not name:
            raise ValueError("API documentation import requires a non-empty integration name")
        return f"api-integration://name/{hashlib.sha256(name.encode()).hexdigest()[:24]}"

    def build_intent(self) -> ActionIntent:
        return ActionIntent(
            parent_task_id=self.parent_task_id,
            actor=self.actor,
            originating_agent=self.originating_agent,
            tool_name="api_integration",
            operation_name=self.operation,
            target_resources=[self.target],
            normalized_arguments={
                "tool_name": self.tool_name,
                "source_decision_id": self.source_decision_id,
                "session_id": self.session_id,
                "target": self.target,
                "arguments_sha256": self.arguments_sha256,
            },
            requested_capabilities=[f"api.integration.{self.operation}"],
            expected_side_effects=[
                f"persist API integration {self.operation} for {self.target}"
            ],
            data_disclosure=DataDisclosure.NONE,
            blast_radius=BlastRadius.SINGLE_RESOURCE,
            reversibility=(
                Reversibility.IRREVERSIBLE
                if self.operation == "delete"
                else Reversibility.PARTIALLY_REVERSIBLE
            ),
            idempotency_key=self.idempotency_key,
            verification_plan=[
                "require an explicit successful API integration tool result",
                "verify the returned result identifies the selected integration change",
            ],
            compensation_strategy=(
                "Use a separately model-selected and policy-gated API integration update or delete action."
            ),
        )

    def preview(self, action: ActionIntent) -> ActionPreview:
        return ActionPreview(
            summary=f"API integration {self.operation}: {self.target}",
            resources=[{"target": self.target, "operation": self.operation}],
            exact_invocation=action.normalized_arguments,
            expected_side_effects=action.expected_side_effects,
            risks=(
                ["deletes a saved API integration"]
                if self.operation == "delete"
                else ["changes one local saved API integration"]
            ),
            externally_visible=False,
            potentially_billable=False,
        )

    def protected_action_context(self) -> dict[str, Any]:
        return {"tool_name": self.tool_name, "arguments": self.arguments}

    def execute(self, action: ActionIntent) -> dict[str, Any]:
        raw = self.invoke()
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError("API integration tool returned non-JSON output") from exc
        elif isinstance(raw, dict):
            payload = dict(raw)
        else:
            raise RuntimeError("API integration tool returned an unsupported result type")
        if not bool(payload.get("ok")):
            detail = str(
                payload.get("message")
                or payload.get("error")
                or payload.get("error_code")
                or "API integration tool returned ok=false without a diagnostic detail"
            )
            raise RuntimeError(redact_secrets(detail)[:1000])
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("API integration tool did not return a structured result")
        return {"ok": True, "operation": self.operation, "target": self.target, "api_result": result}

    def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
        api_result = result.get("api_result")
        integration = api_result.get("integration") if isinstance(api_result, dict) else None
        imported = (
            isinstance(integration, dict)
            and bool(integration.get("integration_id"))
            and bool(api_result.get("saved"))
        )
        complete = bool(result.get("ok")) and (
            imported if self.operation == "import" else isinstance(api_result, dict)
        )
        return VerificationEvidence(
            complete=complete,
            summary=(
                "API integration change returned verified durable integration evidence."
                if complete
                else "API integration change did not return the required durable evidence."
            ),
            checks=[
                {"check": "api_result_ok", "observed": bool(result.get("ok"))},
                {"check": "integration_persisted", "observed": imported},
            ],
        )


__all__ = ["ApiIntegrationActionAdapter"]
