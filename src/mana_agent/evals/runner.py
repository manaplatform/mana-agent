from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import date
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.config import ChatGatewayConfig

from .config import EvalSuite
from .environment import capture_environment
from .ids import execution_id, stable_hash
from .models import (
    CommandRecord,
    EvalRun,
    EvalTask,
    EvalVariant,
    PatchRecord,
    RouteRecord,
    RunStatus,
    TestResult,
    ToolCallRecord,
    UsageRecord,
)
from .recorder import ArtifactEvalRecorder, EvalExecutionContext, use_eval_context
from .redaction import redact_text
from .pricing import DEFAULT_PRICING, Price, PricingRegistry
from .scoring import score_run
from .storage import EvalStorage, atomic_write
from .workspace import EvalWorkspace, workspace_backend

GatewayFactory = Callable[[Path, EvalVariant], Any]
ProgressSink = Callable[[dict[str, Any]], None]


class EvalExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class RunSelection:
    task: EvalTask
    variant: EvalVariant
    trial_number: int


class EvalRunner:
    def __init__(
        self,
        *,
        storage: EvalStorage,
        gateway_factory: GatewayFactory | None = None,
        progress_sink: ProgressSink | None = None,
        retain_workspaces: bool = False,
    ) -> None:
        self.storage = storage
        self.gateway_factory = gateway_factory or self._default_gateway
        self.progress_sink = progress_sink
        self.retain_workspaces = retain_workspaces

    @staticmethod
    def _default_gateway(root: Path, variant: EvalVariant) -> AgentChatGateway:
        return AgentChatGateway(
            root,
            config=ChatGatewayConfig(
                model=variant.main_model,
                lane_overrides=variant.lane_configuration,
                session_id=execution_id("eval_session"),
                agent_timeout_seconds=max(1, int(variant.context_limits.get("timeout_seconds", 30))),
            ),
        )

    def run(self, *, suite: EvalSuite, task: EvalTask, variant: EvalVariant, trial_number: int, experiment_id: str, replayed_from_run_id: str | None = None, force_execute: bool = False) -> EvalRun:
        variant = variant.model_copy(update={"trial_number": trial_number})
        fingerprint = stable_hash(
            {"suite": suite.fingerprint, "task": task.fingerprint, "variant": variant.fingerprint, "trial": trial_number}
        )
        cached = None if force_execute else self.storage.successful_fingerprint(fingerprint)
        if cached is not None:
            self._progress("run.cached", cached.run_id, task.task_id, variant.variant_id)
            return cached
        run_id = execution_id("run")
        run = EvalRun(
            run_id=run_id,
            run_fingerprint=fingerprint,
            experiment_id=experiment_id,
            task_id=task.task_id,
            variant_id=variant.variant_id,
            trial_number=trial_number,
            replayed_from_run_id=replayed_from_run_id,
            repository_commit=task.repository_commit,
            status=RunStatus.PENDING,
        )
        backend = workspace_backend(
            variant.workspace_backend,
            self.storage.root / "workspaces",
            retain=self.retain_workspaces,
        )
        workspace: EvalWorkspace | None = None
        run_dir: Path | None = None
        recorder: ArtifactEvalRecorder | None = None
        started = time.monotonic()
        try:
            source_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(task.repository), capture_output=True, text=True, check=False,
            )
            source_dirty = bool(source_status.stdout.strip())
            workspace = backend.create(task.repository, task.repository_commit, run_id=run_id)
            environment = capture_environment(
                workspace.path,
                workspace_backend=variant.workspace_backend,
                prompt_hashes=variant.prompt_hashes,
                tool_policy_hash=stable_hash({"policy": variant.tool_policy, "allow": variant.tool_allowlist, "deny": variant.tool_denylist}),
                lane_policy_hash=stable_hash(variant.lane_configuration),
                environment_image=variant.environment_specification,
                deterministic_task=bool(task.description and task.repository_commit),
                has_test_command=bool(task.test_commands) or task.expected.no_repository_mutation,
            )
            run = run.model_copy(update={
                "status": RunStatus.RUNNING,
                "repository_commit": environment.repository_commit,
                "initial_dirty_state": source_dirty,
                "environment": environment,
            })
            run_dir = self.storage.create_run(
                run,
                config={"suite": suite.model_dump(mode="json"), "task": task.model_dump(mode="json"), "variant": variant.model_dump(mode="json")},
                environment=environment.model_dump(mode="json"),
            )
            recorder = ArtifactEvalRecorder(
                run_dir,
                run_id=run_id,
                task_id=task.task_id,
                variant_id=variant.variant_id,
                extra_redaction_patterns=tuple(suite.extra_redaction_patterns),
            )
            context = EvalExecutionContext(run_id, task.task_id, variant.variant_id, recorder)
            recorder.record_run_started({"workspace": str(workspace.path), "fingerprint": fingerprint})
            self._progress("run.started", run_id, task.task_id, variant.variant_id)
            with use_eval_context(context):
                setup_records, setup_success = self._commands(task.setup_commands, workspace.path, task.timeout_seconds, recorder, event="setup")
                if not setup_success:
                    raise EvalExecutionError("evaluation setup command failed")
                turn_started = time.monotonic()
                result = None
                attempt = 0
                for attempt in range(1, variant.retry_policy.max_attempts + 1):
                    recorder.record("run.attempt.started", {"attempt": attempt})
                    try:
                        gateway = self.gateway_factory(workspace.path, variant)
                        session_id = gateway.create_session(frontend="eval")
                        result = gateway.process_turn(session_id, task.description)
                    except Exception as attempt_error:
                        recorder.record(
                            "run.attempt.failed",
                            {"attempt": attempt, "error_type": type(attempt_error).__name__, "error": str(attempt_error)},
                        )
                        if attempt >= variant.retry_policy.max_attempts:
                            raise
                        continue
                    recorder.record(
                        "run.attempt.finished",
                        {"attempt": attempt, "error": getattr(result, "error", None)},
                    )
                    if not getattr(result, "error", None) or attempt >= variant.retry_policy.max_attempts:
                        break
                if result is None:
                    raise EvalExecutionError("evaluation produced no gateway result")
                turn_latency = time.monotonic() - turn_started
                route = self._route_record(result, variant, turn_latency, task.description)
                if route:
                    route = route.model_copy(update={"retry_attempts": max(0, attempt - 1)})
                if route:
                    recorder.record_route_decision(route.model_dump(mode="json"))
                tests, _ = self._tests(task.test_commands, workspace.path, task.timeout_seconds, recorder)
            patch = self._capture_patch(workspace.path, run_dir, recorder)
            tool_calls = self._tool_records(result)
            event_data = self._event_records(run_dir, suite)
            calculated_cost = (
                sum(item.calculated_cost for item in event_data["usage"] if item.calculated_cost is not None)
                if event_data["usage"] and all(item.calculated_cost is not None for item in event_data["usage"])
                else None
            )
            command_records = setup_records + [
                CommandRecord(
                    command=item.command,
                    working_directory=".",
                    exit_code=item.exit_code,
                    duration_seconds=item.duration_seconds,
                    stdout=item.stdout,
                    stderr=item.stderr,
                    timed_out=item.timed_out,
                )
                for item in tests
            ]
            run = run.model_copy(update={
                "routes": [route] if route else [],
                "model_calls": event_data["model_calls"],
                "tool_calls": tool_calls,
                "commands": command_records,
                "patches": [patch] if patch.changed_files else [],
                "usage": event_data["usage"],
                "calculated_cost": calculated_cost,
                "tests": tests,
                "reviewer_results": event_data["reviews"],
                "verifier_results": event_data["verifications"],
                "final_answer": str(getattr(result, "answer", "") or ""),
                "errors": [str(getattr(result, "error", ""))] if getattr(result, "error", None) else [],
                "warnings": list(getattr(result, "warnings", []) or []),
                "latency_seconds": time.monotonic() - started,
                "artifact_paths": {
                    "run": str(run_dir / "run.json"),
                    "events": str(run_dir / "events.jsonl"),
                    "config": str(run_dir / "config.yaml"),
                    "environment": str(run_dir / "environment.json"),
                    "patch": str(run_dir / "final.patch"),
                },
            })
            outcome = score_run(
                task=task,
                run=run,
                tests=tests,
                weights=suite.scoring,
                setup_success=True,
                first_attempt_success=attempt == 1 and not bool(getattr(result, "error", None)),
            )
            terminal = RunStatus.COMPLETED if outcome.task_success else RunStatus.FAILED
            run = run.model_copy(update={"outcome": outcome, "status": terminal, "completed_at": datetime.now(timezone.utc)})
            artifacts = self._final_artifacts(run, route, tests, patch, result)
            recorder.record_run_finished({"status": terminal.value, "score": outcome.normalized_score})
            self.storage.finalize_run(run, artifacts)
            self.storage.index_run(run, suite_name=suite.name, suite_version=suite.version)
            self._progress("run.finished", run_id, task.task_id, variant.variant_id, status=terminal.value)
            return run
        except BaseException as exc:
            if run_dir is not None and not (run_dir / ".complete").exists():
                if recorder is not None:
                    recorder.record_run_failed({"error_type": type(exc).__name__, "error": str(exc)})
                failed = run.model_copy(update={
                    "status": RunStatus.FAILED,
                    "completed_at": datetime.now(timezone.utc),
                    "latency_seconds": time.monotonic() - started,
                    "errors": [redact_text(str(exc))],
                })
                self.storage.finalize_run(failed, {"outcome.json": {"failure_category": "execution", "error": str(exc)}})
                self.storage.index_run(failed, suite_name=suite.name, suite_version=suite.version)
                run = failed
            self._progress("run.failed", run_id, task.task_id, variant.variant_id, error=str(exc))
            return run
        finally:
            if workspace is not None:
                try:
                    backend.cleanup(workspace)
                except Exception as cleanup_error:
                    self._progress("workspace.cleanup_failed", run_id, task.task_id, variant.variant_id, error=str(cleanup_error))

    def _commands(self, commands: list[str], cwd: Path, timeout: int, recorder: ArtifactEvalRecorder, *, event: str) -> tuple[list[CommandRecord], bool]:
        records: list[CommandRecord] = []
        for command in commands:
            started = time.monotonic()
            try:
                result = subprocess.run(command, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout)
                record = CommandRecord(
                    command=command, working_directory=".", exit_code=result.returncode,
                    duration_seconds=time.monotonic() - started,
                    stdout=redact_text(result.stdout), stderr=redact_text(result.stderr),
                    environment_variable_names=sorted(os.environ),
                )
            except subprocess.TimeoutExpired as exc:
                record = CommandRecord(
                    command=command, working_directory=".", duration_seconds=time.monotonic() - started,
                    stdout=redact_text(str(exc.stdout or "")), stderr=redact_text(str(exc.stderr or "")), timed_out=True,
                    environment_variable_names=sorted(os.environ),
                )
            recorder.record_command({"phase": event, **record.model_dump(mode="json")})
            records.append(record)
            if record.timed_out or record.exit_code != 0:
                return records, False
        return records, True

    def _tests(self, commands: list[str], cwd: Path, timeout: int, recorder: ArtifactEvalRecorder) -> tuple[list[TestResult], bool]:
        if not commands:
            return [], True
        records, success = self._commands(commands, cwd, timeout, recorder, event="test")
        tests = [TestResult(
            command=item.command, passed=item.exit_code == 0 and not item.timed_out,
            exit_code=item.exit_code, duration_seconds=item.duration_seconds,
            stdout=item.stdout, stderr=item.stderr, timed_out=item.timed_out,
        ) for item in records]
        for item in tests:
            recorder.record_test_result(item.model_dump(mode="json"))
        return tests, success

    @staticmethod
    def _route_record(result: Any, variant: EvalVariant, latency: float, prompt: str) -> RouteRecord | None:
        decision = getattr(result, "decision", None)
        payload = decision.to_dict() if callable(getattr(decision, "to_dict", None)) else (
            decision if isinstance(decision, dict) else {}
        )
        entry_route = str((getattr(result, "payload", {}) or {}).get("entry_route") or payload.get("route") or "")
        if not payload and not entry_route:
            return None
        return RouteRecord(
            route_input_hash=stable_hash(prompt),
            router_model=variant.router_model,
            router_prompt_hash=variant.prompt_hashes.get("router", ""),
            intent=str(payload.get("intent") or entry_route),
            execution_path=entry_route,
            selected_tools=list(payload.get("selected_tools") or []),
            tool_inputs=dict(payload.get("tool_inputs") or {}),
            lane_selection=str((getattr(result, "payload", {}) or {}).get("lane_id") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            reasoning_summary=str(payload.get("reasoning_summary") or payload.get("reason") or ""),
            flow_action=str(payload.get("flow_action") or "none"),
            decision_verification_result={"passed": payload.get("verifier_passed"), "summary": payload.get("verifier_summary", "")},
            decision_latency_seconds=latency,
        )

    @staticmethod
    def _capture_patch(cwd: Path, run_dir: Path, recorder: ArtifactEvalRecorder) -> PatchRecord:
        diff = subprocess.run(["git", "diff", "--binary", "--no-ext-diff"], cwd=cwd, capture_output=True, text=True).stdout
        changed = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True).stdout
        files = sorted({line[3:] for line in changed.splitlines() if len(line) > 3})
        added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
        safe_diff = redact_text(diff)
        atomic_write(run_dir / "final.patch", safe_diff)
        record = PatchRecord(changed_files=files, diff_hash=stable_hash(safe_diff), added_lines=added, removed_lines=removed)
        recorder.record_patch(record.model_dump(mode="json"))
        return record

    @staticmethod
    def _tool_records(result: Any) -> list[ToolCallRecord]:
        now = datetime.now(timezone.utc)
        records: list[ToolCallRecord] = []
        for sequence, trace in enumerate(getattr(result, "trace", []) or [], start=1):
            if not isinstance(trace, dict):
                continue
            status = str(trace.get("status") or ("failed" if trace.get("error") else "success"))
            records.append(ToolCallRecord(
                sequence=sequence,
                tool_name=str(trace.get("tool_name") or trace.get("name") or "unknown"),
                owner=str(trace.get("lane") or trace.get("agent_id") or ""),
                redacted_input=dict(trace.get("tool_input") or trace.get("input") or {}),
                started_at=now,
                ended_at=now,
                duration_seconds=float(trace.get("duration_seconds") or 0.0),
                status=status,
                redacted_output_preview=str(trace.get("result_summary") or trace.get("output_preview") or "")[:4000],
                error_type=str(trace.get("error_type") or "") or None,
                retry_number=int(trace.get("retry_number") or trace.get("retry_attempt") or 0),
                mutated_repository=bool(trace.get("mutated_repository") or trace.get("mutation")),
            ))
        return records

    @staticmethod
    def _event_records(run_dir: Path, suite: EvalSuite) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = {"model_calls": [], "usage": [], "reviews": [], "verifications": []}
        pricing = EvalRunner._pricing_registry(suite)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            payload = event.get("payload") or {}
            event_type = event.get("event_type")
            if event_type in {"model.call", "model.usage"}:
                if event_type == "model.call":
                    result["model_calls"].append(payload)
                usage = payload.get("usage")
                usage_items = usage if isinstance(usage, list) else [usage]
                for item in usage_items:
                    if not isinstance(item, dict):
                        continue
                    input_tokens = int(item.get("input_tokens") or item.get("prompt_tokens") or 0)
                    output_tokens = int(item.get("output_tokens") or item.get("completion_tokens") or 0)
                    provider = str(item.get("provider") or payload.get("provider") or "unknown")
                    model = str(item.get("model") or payload.get("model") or "unknown")
                    cached = int(item.get("cached_input_tokens") or 0)
                    reasoning = int(item.get("reasoning_tokens") or 0)
                    result["usage"].append(UsageRecord(
                        input_tokens=input_tokens,
                        cached_input_tokens=cached,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning,
                        total_tokens=int(item.get("total_tokens") or input_tokens + output_tokens),
                        latency_seconds=float(payload.get("latency_seconds") or 0.0),
                        provider=provider,
                        model=model,
                        calculated_cost=pricing.calculate(
                            provider=provider, model=model, input_tokens=input_tokens,
                            cached_input_tokens=cached, output_tokens=output_tokens,
                            reasoning_tokens=reasoning,
                        ),
                        pricing_table_version=pricing.version,
                    ))
            elif event_type == "review.finished":
                result["reviews"].append(payload)
            elif event_type == "verification.finished":
                result["verifications"].append(payload.get("result") or payload)
        return result

    @staticmethod
    def _pricing_registry(suite: EvalSuite) -> PricingRegistry:
        if not suite.pricing_overrides:
            return DEFAULT_PRICING
        registry = PricingRegistry(version=f"suite:{suite.name}:{suite.version}")
        for item in suite.pricing_overrides:
            registry.add(Price(
                provider=str(item["provider"]),
                model=str(item["model"]),
                effective_date=date.fromisoformat(str(item.get("effective_date") or date.today().isoformat())),
                input_per_million=Decimal(str(item["input_per_million"])),
                cached_input_per_million=(Decimal(str(item["cached_input_per_million"])) if item.get("cached_input_per_million") is not None else None),
                output_per_million=Decimal(str(item["output_per_million"])),
                reasoning_per_million=(Decimal(str(item["reasoning_per_million"])) if item.get("reasoning_per_million") is not None else None),
            ))
        return registry

    @staticmethod
    def _final_artifacts(run: EvalRun, route: RouteRecord | None, tests: list[TestResult], patch: PatchRecord, result: Any) -> dict[str, Any]:
        return {
            "route.json": route.model_dump(mode="json") if route else {},
            "commands.jsonl": "".join(json.dumps(item.model_dump(mode="json"), sort_keys=True, default=str) + "\n" for item in run.commands),
            "tool-calls.jsonl": "".join(json.dumps(item.model_dump(mode="json"), sort_keys=True, default=str) + "\n" for item in run.tool_calls),
            "tests.json": [item.model_dump(mode="json") for item in tests],
            "review.json": run.reviewer_results,
            "outcome.json": run.outcome.model_dump(mode="json") if run.outcome else {},
            "stdout.log": str(getattr(result, "answer", "") or ""),
            "stderr.log": "\n".join(run.errors),
        }

    def _progress(self, event: str, run_id: str, task_id: str, variant_id: str, **extra: Any) -> None:
        if self.progress_sink:
            self.progress_sink({"event": event, "run_id": run_id, "task_id": task_id, "variant_id": variant_id, **extra})
