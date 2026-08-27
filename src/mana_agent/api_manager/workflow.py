"""API workflow completion validation, state tracking, and failure diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mana_agent.api_manager.redaction import redact_mapping

API_WORKFLOW_EVIDENCE: dict[str, str] = {
    "api_docs_inspect": "documentation_inspection",
    "browser_inspect": "documentation_inspection",
    "api_docs_import": "integration_import",
    "api_docs_import_semantic": "integration_import",
    "api_integration_update": "integration_configuration",
    "api_operations_search": "operation_search",
    "api_request_preview": "request_preview",
    "api_request_execute": "request_execution",
}

API_LIFECYCLE_ORDER: tuple[str, ...] = (
    "documentation_inspection",
    "integration_import",
    "integration_configuration",
    "operation_search",
    "request_preview",
    "request_execution",
)

DEFAULT_MAX_ATTEMPTS_PER_ACTION: int = 3
DEFAULT_MAX_CONSECUTIVE_NO_PROGRESS: int = 3
DEFAULT_MAX_DOCUMENTATION_RELOADS: int = 2
DEFAULT_MAX_IMPORT_REFRESH_ATTEMPTS: int = 2


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_trace_payload(trace: dict[str, Any]) -> dict[str, Any]:
    """Return the authoritative structured tool payload when available."""
    for key in ("result", "output_preview", "result_summary", "error"):
        value: Any = trace.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(decoded, dict):
                return decoded
    return {}


def _extract_raw_payload(trace: dict[str, Any]) -> Any:
    """Return the first available raw result representation."""
    for key in ("result", "output_preview", "result_summary"):
        value = trace.get(key)
        if value not in (None, ""):
            return value
    return None


@dataclass
class ApiWorkflowController:
    """Deterministic lifecycle state machine and controller for API workflows."""

    workflow_decision_id: str = ""
    task_intent: str = ""
    required_actions: list[str] = field(default_factory=list)
    completed_actions: set[str] = field(default_factory=set)
    waived_actions: list[str] = field(default_factory=list)
    current_action: str = ""
    attempts_by_action: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_error_by_action: dict[str, str] = field(default_factory=dict)
    last_progress_at: str = field(default_factory=_utc_now_iso)
    consecutive_no_progress: int = 0

    integration_id: str = ""
    operation_id: str = ""
    documentation_reference: str = ""
    refresh_integration_id: str = ""
    permission_request_id: str = ""
    permission_requests: list[dict[str, Any]] = field(default_factory=list)

    last_api_tool: str = ""
    last_api_error_code: str = ""
    last_successful_action: str = ""

    terminal_outcome: str = ""
    terminal_evidence: dict[str, Any] = field(default_factory=dict)
    terminal_attempted: bool = False
    terminal_valid: bool = False
    terminal_failure_reason: str = ""

    execution_evidence: dict[str, Any] = field(default_factory=dict)
    actual_tool_events: list[dict[str, Any]] = field(default_factory=list)

    inspection_progress: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_fingerprint: tuple[Any, ...] | None = None

    max_attempts_per_action: int = DEFAULT_MAX_ATTEMPTS_PER_ACTION
    max_consecutive_no_progress: int = DEFAULT_MAX_CONSECUTIVE_NO_PROGRESS
    max_documentation_reloads: int = DEFAULT_MAX_DOCUMENTATION_RELOADS
    max_import_refresh_attempts: int = DEFAULT_MAX_IMPORT_REFRESH_ATTEMPTS

    is_stalled: bool = False
    stalled_action: str = ""
    decision_attempted: bool = False
    decision_valid: bool = False
    decision_missing: bool = False
    decision_invalid_reason: str = ""

    def compute_progress_fingerprint(self) -> tuple[Any, ...]:
        """Compute an immutable tuple representing current workflow state."""
        return (
            tuple(sorted(self.completed_actions)),
            self.current_action,
            self.integration_id,
            self.operation_id,
            self.documentation_reference,
            self.refresh_integration_id,
            self.permission_request_id,
            self.last_api_error_code,
        )

    def next_required_action(self) -> str | None:
        """Return the first declared action that is not completed and not waived."""
        for action in self.required_actions:
            if action not in self.completed_actions and action not in self.waived_actions:
                return action
        return None

    def initialize_decision(self, decision: dict[str, Any], decision_id: str = "") -> None:
        """Initialize controller state from validated api_workflow_decide."""
        self.decision_attempted = True
        self.decision_valid = True
        self.decision_missing = False
        self.workflow_decision_id = str(
            decision.get("source_decision_id")
            or decision.get("decision_id")
            or decision_id
            or ""
        )
        self.task_intent = str(decision.get("task_intent") or "")
        self.required_actions = [
            str(item)
            for item in decision.get("required_actions") or []
            if str(item).strip()
        ]
        self.current_action = self.next_required_action() or ""
        self.last_fingerprint = self.compute_progress_fingerprint()

    def record_tool_trace(self, trace: dict[str, Any]) -> None:
        """Process one tool invocation trace through the lifecycle state machine."""
        tool_name = str(trace.get("tool_name") or "")
        result = _extract_trace_payload(trace)
        raw_result = _extract_raw_payload(trace)
        trace_succeeded = str(trace.get("status") or "").lower() == "ok"
        result_succeeded = result.get("ok") is True

        if not self.decision_valid:
            if tool_name == "api_workflow_decide":
                self.decision_attempted = True
                if result_succeeded and isinstance(result.get("result"), dict):
                    candidate = result["result"]
                    if candidate.get("safe_to_continue") is True:
                        self.initialize_decision(candidate)
                    else:
                        self.decision_invalid_reason = "Workflow decision was marked unsafe."
                else:
                    self.decision_invalid_reason = "Workflow decision did not succeed."
            elif not self.decision_attempted and (tool_name in API_WORKFLOW_EVIDENCE or tool_name == "api_workflow_terminal"):
                self.decision_missing = True
                self.decision_invalid_reason = "Operational tool called before workflow decision."

        action = API_WORKFLOW_EVIDENCE.get(tool_name)
        if tool_name in API_WORKFLOW_EVIDENCE or tool_name in {
            "api_workflow_decide",
            "api_workflow_terminal",
            "api_integrations_list",
            "api_integration_get",
            "api_integration_delete",
        }:
            self.last_api_tool = tool_name
            if not result_succeeded and not trace_succeeded:
                self.last_api_error_code = str(
                    result.get("error_code")
                    or result.get("error")
                    or trace.get("error")
                    or "tool_error"
                )
            else:
                self.last_api_error_code = ""

        # Extract tool event for durable recording
        self._record_safe_tool_event(tool_name, action, trace, result, trace_succeeded, result_succeeded)

        if action:
            self.current_action = action
            self.attempts_by_action[action] += 1

        clipped_success_evidence = (
            action != "request_execution"
            and trace_succeeded
            and isinstance(raw_result, str)
            and len(raw_result) >= 4000
            and not result
        )

        # 1. Documentation inspection tracking
        if (
            tool_name == "api_docs_inspect"
            and trace_succeeded
            and result_succeeded
            and isinstance(result.get("result"), dict)
        ):
            inspected = result["result"]
            doc_ref = str(inspected.get("documentation_ref") or "").strip()
            if doc_ref:
                self.documentation_reference = doc_ref
                try:
                    offset = max(0, int(inspected.get("offset") or 0))
                except (TypeError, ValueError):
                    offset = -1
                text_value = str(inspected.get("text") or "")
                observed_end = offset + len(text_value) if offset >= 0 else -1

                progress = self.inspection_progress.setdefault(
                    doc_ref,
                    {"next_offset": 0, "complete": False},
                )
                if not progress["complete"] and offset == progress["next_offset"]:
                    truncated = bool(inspected.get("truncated", False))
                    if truncated:
                        next_offset = inspected.get("next_offset")
                        if (
                            isinstance(next_offset, int)
                            and next_offset == observed_end
                            and next_offset > offset
                        ):
                            progress["next_offset"] = next_offset
                    else:
                        progress["next_offset"] = observed_end
                        progress["complete"] = True
                        self.completed_actions.add("documentation_inspection")
                        self.last_successful_action = "documentation_inspection"

        # 2. Terminal documentation declaration
        elif tool_name == "api_workflow_terminal":
            self.terminal_attempted = True
            if (
                trace_succeeded
                and result_succeeded
                and isinstance(result.get("result"), dict)
            ):
                terminal = result["result"]
                outcome = str(terminal.get("outcome") or "").strip()
                doc_ref = str(terminal.get("documentation_ref") or "").strip()
                progress = self.inspection_progress.get(doc_ref)

                if outcome != "unsupported_documentation":
                    self.terminal_failure_reason = "Unsupported API workflow terminal outcome."
                elif not progress or progress.get("complete") is not True:
                    self.terminal_failure_reason = (
                        "Terminal unsupported_documentation requires complete contiguous documentation inspection."
                    )
                elif "documentation_inspection" not in self.required_actions:
                    self.terminal_failure_reason = (
                        "Terminal unsupported_documentation requires documentation_inspection in the workflow decision."
                    )
                else:
                    self.terminal_valid = True
                    self.terminal_outcome = outcome
                    self.terminal_evidence = {
                        "outcome": outcome,
                        "documentation_ref": doc_ref,
                        "reason": str(terminal.get("reason") or ""),
                    }
            else:
                self.terminal_failure_reason = (
                    "API workflow terminal declaration did not return successful structured evidence."
                )

        # 3. Integration import handling (including duplicate collisions)
        elif action == "integration_import":
            details = result.get("details") if isinstance(result.get("details"), dict) else {}
            refresh_id = (
                result.get("refresh_integration_id")
                or details.get("refresh_integration_id")
            )
            if refresh_id:
                self.refresh_integration_id = str(refresh_id)

            if not result_succeeded and not clipped_success_evidence:
                self.last_api_error_code = str(result.get("error_code") or "import_failed")
                self.last_error_by_action[action] = self.last_api_error_code
            else:
                res_dict = result.get("result") if isinstance(result.get("result"), dict) else result
                if isinstance(res_dict, dict):
                    integration = res_dict.get("integration")
                    if isinstance(integration, dict) and integration.get("integration_id"):
                        self.integration_id = str(integration["integration_id"])
                self.completed_actions.add(action)
                self.last_successful_action = action
                self.last_error_by_action.pop(action, None)

        # 4. Operation search handling
        elif action == "operation_search":
            if result_succeeded or clipped_success_evidence:
                res_val = result.get("result")
                if isinstance(res_val, list) and res_val:
                    first_op = res_val[0]
                    if isinstance(first_op, dict) and first_op.get("operation_id"):
                        self.operation_id = str(first_op["operation_id"])
                        if first_op.get("integration_id") and not self.integration_id:
                            self.integration_id = str(first_op["integration_id"])
                self.completed_actions.add(action)
                self.last_successful_action = action
                self.last_error_by_action.pop(action, None)
            else:
                self.last_api_error_code = str(result.get("error_code") or "search_failed")
                self.last_error_by_action[action] = self.last_api_error_code

        # 5. Request preview handling
        elif action == "request_preview":
            preview_result = result.get("result")
            if result_succeeded and isinstance(preview_result, dict):
                self.completed_actions.add(action)
                self.last_successful_action = action
                self.last_error_by_action.pop(action, None)
            elif (
                result.get("error_code") in {"permission_required", "approval_required"}
                and isinstance(result.get("details"), dict)
                and str(result["details"].get("permission_scope") or "") == "api.request.execute"
                and str(result["details"].get("permission_request_id") or "").strip()
            ):
                self.permission_request_id = str(result["details"]["permission_request_id"])
                self.permission_requests.append(dict(result["details"]))
                self.completed_actions.add(action)
                self.last_successful_action = action
                self.last_error_by_action.pop(action, None)
            else:
                self.last_api_error_code = str(result.get("error_code") or "preview_failed")
                self.last_error_by_action[action] = self.last_api_error_code

        # 6. Request execution handling (strictly requires preview)
        elif action == "request_execution":
            executed = (
                result.get("result")
                if isinstance(result.get("result"), dict)
                else result
            )
            if not isinstance(executed, dict):
                self.last_api_error_code = str(result.get("error_code") or "execution_failed")
                self.last_error_by_action[action] = self.last_api_error_code
            elif (
                executed.get("executed") is not True
                or executed.get("upstream_ok") is not True
                or not isinstance(executed.get("status_code"), int)
            ):
                self.last_api_error_code = str(
                    executed.get("error_code")
                    or result.get("error_code")
                    or "upstream_failure"
                )
                self.last_error_by_action[action] = self.last_api_error_code
            elif "request_preview" not in self.completed_actions:
                self.last_api_error_code = "preview_required_before_execution"
                self.last_error_by_action[action] = self.last_api_error_code
            elif "operation_search" in self.required_actions and "operation_search" not in self.completed_actions:
                self.last_api_error_code = "operation_search_required_before_execution"
                self.last_error_by_action[action] = self.last_api_error_code
            else:
                self.completed_actions.add(action)
                self.last_successful_action = action
                self.last_error_by_action.pop(action, None)
                evidence = {
                    key: executed.get(key)
                    for key in (
                        "integration_id",
                        "operation_id",
                        "method",
                        "redacted_url",
                        "status_code",
                        "content_type",
                        "body_kind",
                        "json_body",
                        "text_body",
                        "file_reference",
                        "latency_ms",
                        "upstream_ok",
                        "executed",
                    )
                    if executed.get(key) not in (None, "")
                }
                self.execution_evidence = redact_mapping(evidence)

        # 7. Other actions
        elif action:
            if result_succeeded or clipped_success_evidence:
                self.completed_actions.add(action)
                self.last_successful_action = action
                self.last_error_by_action.pop(action, None)

        self._recompute_waived()
        self._update_stagnation()

    def _record_safe_tool_event(
        self,
        tool_name: str,
        action: str | None,
        trace: dict[str, Any],
        payload: dict[str, Any],
        trace_succeeded: bool,
        result_succeeded: bool,
    ) -> None:
        """Record safe redacted lifecycle event into actual_tool_events."""
        if not tool_name:
            return
        status = "ok" if (trace_succeeded and result_succeeded) else "error"
        error_code = ""
        if status == "error":
            error_code = str(
                payload.get("error_code")
                or payload.get("error")
                or trace.get("error")
                or "tool_error"
            )

        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        integration_id = (
            str(payload.get("integration_id") or "")
            or str(details.get("integration_id") or "")
            or self.integration_id
        )
        operation_id = (
            str(payload.get("operation_id") or "")
            or str(details.get("operation_id") or "")
            or self.operation_id
        )

        event = {
            "type": "api_tool_event",
            "tool_name": tool_name,
            "action": action or ("workflow_decision" if tool_name == "api_workflow_decide" else "terminal_declaration" if tool_name == "api_workflow_terminal" else "management"),
            "status": status,
            "error_code": error_code,
            "timestamp": _utc_now_iso(),
            "integration_id": integration_id,
            "operation_id": operation_id,
        }
        if self.refresh_integration_id:
            event["refresh_integration_id"] = self.refresh_integration_id
        self.actual_tool_events.append(event)

    def _recompute_waived(self) -> None:
        """Update waived actions when terminal unsupported_documentation is valid."""
        if not self.terminal_valid:
            self.waived_actions = []
            return

        waivable = {
            "integration_import",
            "integration_configuration",
            "operation_search",
            "request_preview",
            "request_execution",
        }
        conflicting = sorted(self.completed_actions & waivable)
        if conflicting:
            self.terminal_valid = False
            self.terminal_failure_reason = (
                "unsupported_documentation contradicts already completed downstream actions: "
                + ", ".join(conflicting)
                + "."
            )
            self.waived_actions = []
        else:
            self.waived_actions = [
                act for act in self.required_actions
                if act in waivable and act not in self.completed_actions
            ]

    def _update_stagnation(self) -> None:
        """Compute progress fingerprint and detect repetitions / stalls."""
        current_fp = self.compute_progress_fingerprint()
        if current_fp == self.last_fingerprint:
            self.consecutive_no_progress += 1
        else:
            self.consecutive_no_progress = 0
            self.last_progress_at = _utc_now_iso()
            self.last_fingerprint = current_fp

        next_act = self.next_required_action()
        current_act = next_act or self.current_action
        if (
            self.consecutive_no_progress >= self.max_consecutive_no_progress
            or (current_act and self.attempts_by_action[current_act] >= self.max_attempts_per_action)
        ):
            self.is_stalled = True
            self.stalled_action = current_act

    def format_continuation_prompt(self) -> str:
        """Generate a minimal, structured single-action continuation prompt."""
        next_act = self.next_required_action()
        if not next_act:
            return ""

        context_state = {
            "workflow_decision_id": self.workflow_decision_id,
            "required_actions": self.required_actions,
            "completed_actions": sorted(self.completed_actions),
            "next_action": next_act,
        }
        if self.integration_id:
            context_state["integration_id"] = self.integration_id
        if self.operation_id:
            context_state["operation_id"] = self.operation_id
        if self.documentation_reference:
            context_state["documentation_reference"] = self.documentation_reference
        if self.refresh_integration_id:
            context_state["refresh_integration_id"] = self.refresh_integration_id

        state_header = f"API Workflow State:\n{json.dumps(context_state, indent=2)}\n\n"

        if next_act == "documentation_inspection":
            return (
                f"{state_header}Current lifecycle action: documentation_inspection.\n"
                "Call api_docs_inspect with the target documentation source. "
                "Do not answer in prose."
            )
        elif next_act == "integration_import":
            if self.refresh_integration_id:
                return (
                    f"{state_header}Current lifecycle action: integration_import (refresh duplicate).\n"
                    f"The integration already exists. Retry api_docs_import_semantic (or api_docs_import) "
                    f"with refresh_integration_id={self.refresh_integration_id!r} to complete integration import. "
                    "Do not search operations, preview, or execute yet. Do not answer in prose."
                )
            return (
                f"{state_header}Current lifecycle action: integration_import.\n"
                "Call api_docs_import_semantic (or api_docs_import) to import the API documentation. "
                "Do not inspect documentation again. Do not answer in prose."
            )
        elif next_act == "integration_configuration":
            return (
                f"{state_header}Current lifecycle action: integration_configuration.\n"
                "Call api_integration_update to configure the integration. Do not answer in prose."
            )
        elif next_act == "operation_search":
            return (
                f"{state_header}Current lifecycle action: operation_search.\n"
                f"Integration: {self.integration_id or '<integration_id>'}\n"
                "Search enabled operations using api_operations_search for matching endpoints. "
                "Do not inspect documentation again. Do not import again. "
                "Do not execute the request yet. Do not answer in prose."
            )
        elif next_act == "request_preview":
            return (
                f"{state_header}Current lifecycle action: request_preview.\n"
                f"Integration: {self.integration_id}, Operation: {self.operation_id}\n"
                "Call api_request_preview with the selected operation. Request preview is mandatory "
                "before execution (including read-only requests). Do not execute yet. Do not answer in prose."
            )
        elif next_act == "request_execution":
            return (
                f"{state_header}Current lifecycle action: request_execution.\n"
                f"Integration: {self.integration_id}, Operation: {self.operation_id}\n"
                "Call api_request_execute with the previewed request. "
                "Do not answer in prose until execution completes."
            )
        return (
            f"{state_header}Current lifecycle action: {next_act}.\n"
            f"Call the corresponding API tool. Do not answer in prose until all required actions have completed."
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return the durable api_workflow metadata dictionary."""
        missing = [
            action for action in self.required_actions
            if action not in self.completed_actions and action not in self.waived_actions
        ]
        return {
            "decision_id": self.workflow_decision_id,
            "workflow_decision_id": self.workflow_decision_id,
            "task_intent": self.task_intent,
            "required_actions": list(self.required_actions),
            "completed_actions": sorted(self.completed_actions),
            "missing_actions": missing,
            "waived_actions": list(self.waived_actions),
            "current_action": self.current_action,
            "attempts_by_action": dict(self.attempts_by_action),
            "last_successful_action": self.last_successful_action,
            "last_tool": self.last_api_tool,
            "last_api_tool": self.last_api_tool,
            "last_error_code": self.last_api_error_code,
            "last_api_error_code": self.last_api_error_code,
            "terminal_outcome": self.terminal_outcome,
            "terminal_evidence": dict(self.terminal_evidence),
            "execution_evidence": dict(self.execution_evidence),
            "stalled": self.is_stalled,
            "stalled_action": self.stalled_action,
            "integration_id": self.integration_id,
            "operation_id": self.operation_id,
            "documentation_reference": self.documentation_reference,
            "refresh_integration_id": self.refresh_integration_id,
            "permission_request_id": self.permission_request_id,
        }

    def evaluate(self) -> dict[str, Any]:
        """Evaluate final completion status against the model decision."""
        if self.decision_missing or not self.decision_attempted:
            return {
                "valid": False,
                "error_code": "api_workflow_decision_missing",
                "message": (
                    "Model decision failed: api_workflow. The first API-route tool call "
                    "was not a validated workflow decision. No completion was recorded."
                ),
                "task_intent": "",
                "required_actions": [],
                "completed_actions": [],
                "missing_actions": [],
                "unexpected_actions": [],
                "execution_evidence": {},
                "waived_actions": [],
                "terminal_outcome": "",
                "terminal_evidence": {},
                "last_successful_action": "",
                "last_api_tool": self.last_api_tool,
                "last_api_error_code": self.last_api_error_code or "api_workflow_decision_missing",
                "workflow_decision_id": "",
                "stalled": False,
                "stalled_action": "",
                "actual_tool_events": self.actual_tool_events,
            }

        if not self.decision_valid:
            return {
                "valid": False,
                "error_code": "api_workflow_decision_invalid",
                "message": (
                    "Model decision failed: api_workflow. The workflow decision was "
                    "invalid or unsafe. No completion was recorded."
                ),
                "task_intent": self.task_intent,
                "required_actions": [],
                "completed_actions": [],
                "missing_actions": [],
                "unexpected_actions": [],
                "execution_evidence": {},
                "waived_actions": [],
                "terminal_outcome": "",
                "terminal_evidence": {},
                "last_successful_action": "",
                "last_api_tool": self.last_api_tool,
                "last_api_error_code": self.last_api_error_code or "api_workflow_decision_invalid",
                "workflow_decision_id": self.workflow_decision_id,
                "stalled": False,
                "stalled_action": "",
                "actual_tool_events": self.actual_tool_events,
            }

        missing = [
            action for action in self.required_actions
            if action not in self.completed_actions and action not in self.waived_actions
        ]
        unexpected = sorted(
            action for action in self.completed_actions
            if action not in self.required_actions
        )

        if self.terminal_attempted and not self.terminal_valid:
            error_code = "api_workflow_terminal_invalid"
            message = (
                "API workflow terminal evidence is invalid"
                + (f": {self.terminal_failure_reason}" if self.terminal_failure_reason else ".")
            )
        elif unexpected:
            error_code = "api_workflow_action_not_selected"
            message = (
                "API tools executed actions absent from the workflow decision: "
                + ", ".join(unexpected)
                + "."
            )
        elif self.is_stalled:
            error_code = "api_workflow_stalled"
            message = (
                f"API workflow stalled at action {self.stalled_action!r} after "
                f"{self.attempts_by_action[self.stalled_action]} repeated attempts with no progress."
            )
        elif missing:
            error_code = "api_workflow_incomplete"
            message = (
                "API workflow is incomplete; missing successful evidence for: "
                + ", ".join(missing)
                + "."
            )
        else:
            error_code = ""
            message = (
                "API workflow terminated with evidence-backed unsupported documentation."
                if self.terminal_valid
                else "API workflow completion evidence is valid."
            )

        valid = (
            not missing
            and not unexpected
            and not self.is_stalled
            and not (self.terminal_attempted and not self.terminal_valid)
        )

        return {
            "valid": valid,
            "error_code": error_code,
            "message": message,
            "task_intent": self.task_intent,
            "required_actions": list(self.required_actions),
            "completed_actions": sorted(self.completed_actions),
            "waived_actions": list(self.waived_actions),
            "missing_actions": missing,
            "unexpected_actions": unexpected,
            "execution_evidence": self.execution_evidence,
            "terminal_outcome": self.terminal_outcome if self.terminal_valid else "",
            "terminal_evidence": self.terminal_evidence if self.terminal_valid else {},
            "last_successful_action": self.last_successful_action,
            "last_api_tool": self.last_api_tool,
            "last_api_error_code": self.last_api_error_code,
            "workflow_decision_id": self.workflow_decision_id,
            "stalled": self.is_stalled,
            "stalled_action": self.stalled_action,
            "actual_tool_events": self.actual_tool_events,
            "continuation_prompt": self.format_continuation_prompt(),
            "api_workflow": self.to_metadata(),
        }


def evaluate_api_workflow_completion(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate exact successful tool evidence against the model workflow decision."""
    if not traces:
        return {
            "valid": False,
            "error_code": "api_workflow_decision_missing",
            "message": (
                "Model decision failed: api_workflow. The first API-route tool call "
                "was not a validated workflow decision. No completion was recorded."
            ),
            "task_intent": "",
            "required_actions": [],
            "completed_actions": [],
            "missing_actions": [],
            "unexpected_actions": [],
            "execution_evidence": {},
            "waived_actions": [],
            "terminal_outcome": "",
            "terminal_evidence": {},
            "last_successful_action": "",
            "last_api_tool": "",
            "last_api_error_code": "api_workflow_decision_missing",
            "workflow_decision_id": "",
            "stalled": False,
            "stalled_action": "",
            "actual_tool_events": [],
        }

    controller = ApiWorkflowController()
    for trace in traces:
        controller.record_tool_trace(trace)

    return controller.evaluate()

