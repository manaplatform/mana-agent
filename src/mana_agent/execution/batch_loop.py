"""Resumable Batch turns used by the Autonomous Loop and native coding engine.

This module advances transport checkpoints. The model chooses tools and semantic
continuation; the Supervisor owns leases, action receipts and completion.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from mana_agent.batch.policy import BatchPending, batch_tool_call_context
from mana_agent.batch.transport import BatchModelTransport
from mana_agent.config.autonomous_config import AutonomousExecutionConfig
from mana_agent.execution.continuation_decision import ContinuationDecision, ContinuationAction
from mana_agent.execution.continuation_evaluator import _CONTINUATION_PROMPT, _coerce_continuation_decision
from mana_agent.execution_supervisor.models import ActionRequestState, SideEffectClassification
from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.config.model_discovery import GLOBAL_CAPABILITY_REGISTRY


def workspace_fingerprint(root: Path) -> str:
    """Hash local state without copying source or secrets into telemetry."""
    digest = hashlib.sha256()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, timeout=10)
    if head.returncode == 0:
        digest.update(head.stdout)
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, capture_output=True, timeout=10)
        files = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root, capture_output=True, timeout=10)
        if diff.returncode or files.returncode:
            raise RuntimeError("Could not establish Batch workspace revision")
        digest.update(diff.stdout)
        paths = [root / name.decode() for name in files.stdout.split(b"\0") if name]
    else:
        paths = sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Batch workspace contains an unconfined file reference")
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(65536), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def batch_output_limit(supervisor: Any, task_id: str, provider: str, model: str,
                       messages: list[dict[str, Any]], tools: list[dict[str, Any]], schema: Any) -> int:
    profile = GLOBAL_CAPABILITY_REGISTRY.get(provider, model)
    input_tokens = estimate_value_tokens(messages) + estimate_value_tokens(tools) + estimate_value_tokens(schema or {})
    capacity = profile.context_window - input_tokens if profile else 0
    if supervisor:
        task = supervisor.get_task(task_id)
        if task.token_budget is not None:
            capacity = min(capacity, task.token_budget - task.token_usage - input_tokens)
        if task.monetary_budget is not None:
            from mana_agent.context_cost.profiles import ModelIdentity, ModelTokenProfileResolver
            pricing = ModelTokenProfileResolver().resolve(ModelIdentity(provider=provider, model=model))
            if pricing.input_price_per_million is None or pricing.output_price_per_million is None:
                raise RuntimeError("Cannot enforce Batch spending limit without model pricing")
            remaining = task.monetary_budget - task.actual_cost - float(pricing.input_price_per_million) * input_tokens / 2_000_000
            if pricing.output_price_per_million:
                capacity = min(capacity, int(remaining * 2_000_000 / float(pricing.output_price_per_million)))
    output_limit = min(4096, capacity, profile.max_output_tokens if profile else 0)
    if output_limit < 1:
        raise RuntimeError("Batch request exceeds the remaining context or task budget")
    return output_limit


class BatchTaskLoop:
    def __init__(self, manager: Any, *, supervisor: Any = None, config: AutonomousExecutionConfig | None = None,
                 attempt_id: str = "", lease_token: str = "") -> None:
        self.transport = BatchModelTransport(manager)
        self.store = self.transport.operations
        self.supervisor = supervisor
        self.config = config or AutonomousExecutionConfig()
        self.attempt_id, self.lease_token = attempt_id, lease_token

    def advance(
        self, *, task_id: str, goal: str, root: Path, model: str, provider: str,
        messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        dispatch: Callable[[str, dict[str, Any], str], Any],
        resume: dict[str, Any] | None = None,
        authorize: Callable[[str], Any] | None = None,
    ) -> dict[str, Any]:
        root = root.resolve()
        run = self.store.load_run(task_id)
        if run is None:
            run = {
                "task_id": task_id, "root": str(root), "goal": goal, "model": model, "provider": provider,
                "messages": messages, "tools": tools, "phase": "work", "step": 0, "status": "running",
                "model_turns": 0, "tool_calls": 0, "no_progress": 0, "completed_steps": [],
                "changed_files": [], "commands_run": [], "verification_results": [], "trace": [], "resume": resume or {},
                "workspace_fingerprint": workspace_fingerprint(root), "accounted_operations": [],
                "limits": self.config.model_dump(mode="json"),
                "owner_root": (resume or {}).get("owner_root", str(root)),
            }
            self.store.save_run(task_id, run)
        if (run["root"], run["goal"], run["model"], run["provider"]) != (str(root), goal, model, provider):
            raise ValueError("Batch resume does not match the original task, workspace, or model")
        if run["tools"] != tools:
            raise ValueError("Registered Batch tool contracts changed; a fresh model decision is required")
        # A process restart must not silently replace the original task's
        # authorized turn/tool limits with the new process defaults.
        self.config = AutonomousExecutionConfig.model_validate(run["limits"])
        run.setdefault("executed_call_ids", [entry["call_id"] for entry in run.get("trace", []) if entry.get("call_id")])
        run.setdefault("consumed_call_ids", list(run["executed_call_ids"]))
        if run["status"] in {"completed", "failed", "cancelled", "waiting_for_user"}:
            return run
        if workspace_fingerprint(root) != run["workspace_fingerprint"]:
            raise RuntimeError("stale_revision_rejected: Batch workspace changed while awaiting a model result")
        self._validate_authority(task_id)
        run["status"] = "running"
        self.store.save_run(task_id, run)
        while True:
            self._validate_authority(task_id)
            if run["model_turns"] >= self.config.max_model_turns or run["tool_calls"] >= self.config.max_tool_calls:
                raise RuntimeError("Batch task execution budget exhausted")
            phase = run["phase"]
            if phase == "tools":
                self._execute_tools(run, dispatch, authorize)
                if run["status"] == "waiting_for_user":
                    return run
                run["phase"] = "work"
                run["step"] += 1
                run["workspace_fingerprint"] = workspace_fingerprint(root)
                self.store.save_run(task_id, run)
                continue
            request_messages = run["messages"]
            schema = None
            request_tools = tools
            if phase == "continuation":
                request_tools = []
                schema = ContinuationDecision.model_json_schema()
                request_messages = [
                    {"role": "system", "content": _CONTINUATION_PROMPT},
                    {"role": "user", "content": json.dumps({
                        "task_id": task_id, "user_goal": goal, "step_output": run.get("last_answer", ""),
                        "completed_steps": run["completed_steps"], "tool_calls": run["tool_calls"],
                        "independent_verification_status": "unverified", "remaining_work": "Evaluate actual tool evidence",
                    }, ensure_ascii=False)},
                ]
            try:
                output_limit = batch_output_limit(self.supervisor, task_id, provider, model, request_messages, request_tools, schema)
                response = self.transport.invoke(
                    task_id=task_id, phase=phase, step=run["step"], model=model, provider=provider,
                    messages=request_messages, tools=request_tools, schema=schema,
                    max_output_tokens=output_limit,
                )
            except BatchPending as pending:
                run["pending_operation"] = pending.operation.model_dump(mode="json")
                run["status"] = "waiting"
                self.store.save_run(task_id, run)
                raise
            except Exception:
                self._account_failed_result(run, phase)
                raise
            run["model_turns"] += 1
            self._account_result(run, phase)
            run["last_operation_id"] = f"batch:{task_id}:{phase}:{run['step']}"
            run.pop("pending_operation", None)
            if phase == "continuation":
                decision = _coerce_continuation_decision(response["content"], task_id=task_id)
                if decision.task_id != task_id:
                    raise ValueError("Batch continuation decision selected a different task")
                run["continuation_decision"] = decision.model_dump(mode="json")
                if decision.action is ContinuationAction.FINALIZE:
                    # The caller still owes independent Supervisor verification.
                    run["status"] = "completed"
                    run["final_delivery_pending"] = True
                    run["answer"] = decision.user_message or run["last_answer"]
                elif decision.action is ContinuationAction.WAIT_FOR_USER:
                    run["status"] = "waiting_for_user"
                    run["answer"] = decision.user_message
                elif decision.action is ContinuationAction.BLOCKED:
                    run["status"] = "failed"
                    run["answer"] = decision.user_message or decision.reason
                else:
                    run["no_progress"] = 0 if decision.progress_detected else run["no_progress"] + 1
                    if run["no_progress"] >= self.config.max_no_progress_cycles:
                        raise RuntimeError("Batch task made no progress; explicit replanning is required")
                    run["phase"] = "work"
                    run["step"] += 1
                    run["messages"].append({"role": "user", "content": decision.remaining_work})
                self.store.save_run(task_id, run)
                if run["status"] in {"completed", "failed", "waiting_for_user"}:
                    return run
                continue
            run["messages"].append(response)
            if response.get("tool_calls"):
                run["phase"] = "tools"
                run["pending_calls"] = response["tool_calls"]
                run["completed_call_ids"] = []
            else:
                run["last_answer"] = response["content"]
                run["phase"] = "continuation"
                run["step"] += 1
            self.store.save_run(task_id, run)

    def _validate_authority(self, task_id: str) -> Any:
        if self.supervisor is None:
            return None  # Submission-only embedding; tool execution requires authority below.
        task = self.supervisor.get_task(task_id)
        if self.supervisor.cancellation_requested(task_id):
            raise RuntimeError("Batch task was cancelled")
        if not self.attempt_id or not self.lease_token:
            raise RuntimeError("Batch execution requires the caller's current lease credentials")
        self.supervisor._validate_lease(task, attempt_id=self.attempt_id, lease_token=self.lease_token)
        if task.token_budget is not None and task.token_usage >= task.token_budget:
            raise RuntimeError("Batch task token budget exhausted")
        if task.monetary_budget is not None and task.actual_cost >= task.monetary_budget:
            raise RuntimeError("Batch task spending budget exhausted")
        return task

    def _account_result(self, run: dict[str, Any], phase: str) -> None:
        operation_id = f"batch:{run['task_id']}:{phase}:{run['step']}"
        if operation_id in run["accounted_operations"]:
            return
        operation = self.store.get(operation_id)
        job = self.transport.manager.store.get_job(operation["batch_id"])
        if self.supervisor is not None:
            def account(task: Any) -> None:
                ids = task.provider_metadata.setdefault("accounted_batch_operations", [])
                if operation_id in ids:
                    return
                ids.append(operation_id)
                usage = job.usage
                totals = task.provider_metadata.setdefault("batch_usage", {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "cost_known": True})
                totals["input_tokens"] += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                totals["output_tokens"] += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
                totals["cost"] += job.cost_estimate_usd
                totals["cost_known"] = totals["cost_known"] and job.metadata.get("cost_estimate_known") == "true"
                task.token_usage += int(usage.get("total_tokens") or
                    int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0) + int(usage.get("output_tokens") or usage.get("completion_tokens") or 0))
                task.actual_cost += job.cost_estimate_usd
                task.actual_cost_known = totals["cost_known"]
            self.supervisor.store.update_task(run["task_id"], account)
        run["accounted_operations"].append(operation_id)

    def _account_failed_result(self, run: dict[str, Any], phase: str, *, operation_step: int | None = None) -> None:
        """Failed responses can still consume provider tokens and money."""
        step = run["step"] if operation_step is None else operation_step
        operation_id = f"batch:{run['task_id']}:{phase}:{step}"
        operation = self.store.get(operation_id)
        job = self.transport.manager.store.get_job(operation["batch_id"]) if operation and operation["batch_id"] else None
        if job and job.is_terminal() and operation_id not in run["accounted_operations"]:
            self._account_result({**run, "step": step}, phase)
            run["model_turns"] += 1
            self.store.save_run(run["task_id"], run)

    def _execute_tools(self, run: dict[str, Any], dispatch: Callable[..., Any], authorize: Callable[[str], Any] | None) -> None:
        if self.supervisor is None:
            raise RuntimeError("Batch tool execution requires an Execution Supervisor lease")
        contracts = {tool["function"]["name"]: tool["function"] for tool in run["tools"]}
        if run.get("inflight_call_id"):
            raise RuntimeError("Interrupted Batch tool outcome requires durable action reconciliation before replay")
        for call in run["pending_calls"]:
            if call["id"] in run["completed_call_ids"]:
                continue
            if call["id"] in run["consumed_call_ids"]:
                raise ValueError("Batch response replayed a tool call from a completed wave")
            task = self._validate_authority(run["task_id"])
            name = call["function"]["name"]
            if name not in contracts:
                raise ValueError("Batch selected an unregistered or unauthorized tool")
            args = json.loads(call["function"]["arguments"])
            from mana_agent.batch.tool_arguments import validate_arguments
            validate_arguments(args, contracts[name].get("parameters", {}))
            if authorize:
                authorize(name)
            if run["tool_calls"] >= self.config.max_tool_calls:
                raise RuntimeError("Batch tool-call budget exhausted")
            fingerprint = hashlib.sha256(json.dumps({"task": task.task_id, "call_id": call["id"], "name": name, "args": args}, sort_keys=True).encode()).hexdigest()
            run["inflight_call_id"] = call["id"]
            self.store.save_run(task.task_id, run)
            action = self.supervisor.prepare_action(
                task.task_id, attempt_id=self.attempt_id, lease_token=self.lease_token,
                tool_name=name, action_fingerprint=fingerprint, classification=SideEffectClassification.UNKNOWN,
            )
            self.supervisor.update_action(action.action_id, request_state=ActionRequestState.STARTED)
            # If dispatch is interrupted, its STARTED receipt prohibits replay.
            with batch_tool_call_context(call["id"]):
                result = dispatch(name, args, call["id"])
            encoded = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            try:
                data = json.loads(encoded)
            except (ValueError, TypeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            self.supervisor.update_action(action.action_id, request_state=ActionRequestState.SUCCEEDED,
                result_reference=f"batch-run:{task.task_id}:{call['id']}")
            run["messages"].append({"role": "tool", "tool_call_id": call["id"], "content": encoded})
            run["completed_call_ids"].append(call["id"])
            run["executed_call_ids"].append(call["id"])
            run["consumed_call_ids"].append(call["id"])
            run.pop("inflight_call_id", None)
            run["tool_calls"] += 1
            run["trace"].append({"tool_name": name, "call_id": call["id"], "status": "executed"})
            run["completed_steps"].append({"tool": name, "result": encoded})
            if data.get("ok"):
                paths = data.get("changed_files") or data.get("touched_files") or data.get("files_changed") or []
                if name in {"write_file", "create_file", "edit_file", "multi_edit_file", "delete_file"}:
                    paths = [args.get("path")]
                run["changed_files"] = list(dict.fromkeys(run["changed_files"] + [str(path) for path in paths if path]))
                if name in {"run_command", "run_script_once", "shell"}:
                    run["commands_run"].append(str(args.get("cmd") or args.get("command") or args.get("script") or ""))
            if data.get("inbox_item_id") or data.get("awaiting_approval"):
                run["status"] = "waiting_for_user"
                run["pending_approval_id"] = data.get("inbox_item_id") or ""
                run["answer"] = "Execution is paused for the required approval."
            fingerprint_after = workspace_fingerprint(Path(run["root"]))
            if fingerprint_after != run["workspace_fingerprint"]:
                run["verification_results"] = []
            if name in {"verify_project", "run_command", "run_script_once", "shell"}:
                run.setdefault("verification_results", []).append({"tool": name, "arguments": args, "result": data})
            run["workspace_fingerprint"] = fingerprint_after
            self.store.save_run(task.task_id, run)
            if run["status"] == "waiting_for_user":
                # Every call in this assistant message needs a correlated
                # output before the next model wave. Sibling calls have not
                # executed; report their deferral instead of losing them or
                # sending an invalid, partially answered tool-call sequence.
                for deferred in run["pending_calls"]:
                    if deferred["id"] in run["completed_call_ids"]:
                        continue
                    run["messages"].append({"role": "tool", "tool_call_id": deferred["id"],
                        "content": json.dumps({"ok": False, "not_executed": True,
                            "reason": "Task paused for required human input; request this action again after resumption"})})
                    run["completed_call_ids"].append(deferred["id"])
                    run["consumed_call_ids"].append(deferred["id"])
                self.store.save_run(task.task_id, run)
                return
