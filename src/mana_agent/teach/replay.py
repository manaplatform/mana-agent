"""Permission-aware dry, guided, and normal replay."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .models import ManaFlow, ReplayResult, StepResult, TeachError
from .verification import BuiltinVerifier


COMMITTING_MARKERS = ("send", "delete", "purchase", "publish", "pay", "submit", "post")


class SafeReplayExecutor:
    def __init__(
        self,
        *,
        action_executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        permission_checker: Callable[[str], bool] | None = None,
        confirmation_checker: Callable[[str], bool] | None = None,
        verifier: BuiltinVerifier | None = None,
    ):
        self.action_executor = action_executor
        self.permission_checker = permission_checker or (lambda _scope: False)
        self.confirmation_checker = confirmation_checker or (lambda _step: False)
        self.verifier = verifier or BuiltinVerifier()

    def replay(
        self,
        flow: ManaFlow,
        *,
        mode: str,
        inputs: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ReplayResult:
        if mode not in {"dry_run", "guided", "normal"}:
            raise TeachError("Replay mode must be dry_run, guided, or normal.")
        resolved = self._resolve_inputs(flow, inputs or {})
        results: list[StepResult] = []
        execution_context = {**(context or {}), "inputs": resolved}
        for step in flow.steps:
            missing_permissions = [scope for scope in step.permissions if not self.permission_checker(scope)]
            committing = step.requires_confirmation or any(marker in step.action for marker in COMMITTING_MARKERS)
            if mode == "dry_run":
                message = "Would require confirmation." if committing else "Validated for dry run."
                results.append(StepResult(step_id=step.id, status="planned", message=message))
                continue
            if missing_permissions:
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="failed",
                        message=f"Missing permissions: {', '.join(missing_permissions)}",
                    )
                )
                return self._result(flow, mode, "failed", results, [])
            if committing and not self.confirmation_checker(step.id):
                results.append(
                    StepResult(step_id=step.id, status="waiting_confirmation", message="Explicit confirmation required.")
                )
                return self._result(flow, mode, "unverified", results, [])
            if mode == "guided" and (step.requires_review or step.confidence < 0.8) and not self.confirmation_checker(step.id):
                results.append(StepResult(step_id=step.id, status="waiting_confirmation", message="Guided approval required."))
                return self._result(flow, mode, "unverified", results, [])
            if self.action_executor is None:
                results.append(StepResult(step_id=step.id, status="failed", message="No replay executor is available."))
                return self._result(flow, mode, "failed", results, [])
            evidence = self.action_executor(step.action, step.with_)
            if not isinstance(evidence, dict):
                results.append(
                    StepResult(
                        step_id=step.id,
                        status="failed",
                        message="Replay executor returned invalid evidence.",
                    )
                )
                return self._result(flow, mode, "failed", results, [])
            if not evidence.get("ok", False):
                results.append(StepResult(step_id=step.id, status="failed", message="Action failed.", evidence=evidence))
                return self._result(flow, mode, "failed", results, [evidence])
            results.append(StepResult(step_id=step.id, status="completed", message="Step completed.", evidence=evidence))
            execution_context.update(evidence)
        if mode == "dry_run":
            # A dry run validates safety/readiness; it never claims the real outcome exists.
            return self._result(flow, mode, "unverified", results, [])
        evidence: list[dict[str, Any]] = []
        required = [rule for rule in flow.verify if rule.required]
        if not required:
            return self._result(flow, mode, "unverified", results, [{"error": "No final verification rule."}])
        passed = True
        for rule in required:
            rule_passed, item = self.verifier.verify(rule, execution_context)
            evidence.append({"rule_id": rule.id, "passed": rule_passed, **item})
            passed = passed and rule_passed
        return self._result(flow, mode, "verified" if passed else "failed", results, evidence)

    def _resolve_inputs(self, flow: ManaFlow, supplied: dict[str, Any]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for name, definition in flow.inputs.items():
            value = supplied.get(name, definition.default)
            if definition.required and value is None:
                raise TeachError(f"Required flow input is missing: {name}")
            if definition.secret and isinstance(value, str) and not value.startswith("secret://"):
                raise TeachError(f"Secret input {name} must use an existing secret:// reference.")
            resolved[name] = value
        unknown = set(supplied) - set(flow.inputs)
        if unknown:
            raise TeachError(f"Unknown flow inputs: {', '.join(sorted(unknown))}")
        return resolved

    @staticmethod
    def _result(
        flow: ManaFlow,
        mode: str,
        status: str,
        steps: list[StepResult],
        evidence: list[dict[str, Any]],
    ) -> ReplayResult:
        now = datetime.now(timezone.utc)
        return ReplayResult(
            flow_id=flow.id,
            flow_version=flow.version,
            mode=mode,
            verification_status=status,
            steps=steps,
            evidence=evidence,
            finished_at=now,
        )
