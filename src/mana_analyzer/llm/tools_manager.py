from __future__ import annotations
import logging
import ast
import hashlib
import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence, TypeVar

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from mana_analyzer.llm.tool_worker_process import ToolRunRequest, ToolRunResponse, ToolWorkerClient
from mana_analyzer.llm.tools_executor import (
    BatchToolRequest,
    BatchExecutionResult,
    LocalToolsExecutor,
    ToolsExecutionConfig,
    ToolsExecutor,
)
from mana_analyzer.services.coding_memory_service import CodingMemoryService

logger = logging.getLogger(__name__)

PlanDecision = Literal["continue", "revise", "finalize", "stop"]
StepStatus = Literal["pending", "in_progress", "done", "blocked"]
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class ToolsPlanStep(BaseModel):
    id: str
    title: str
    tool_intent: Literal["inspect", "search", "edit", "verify", "answer"]
    args_hint: str = ""
    success_signal: str = ""
    fallback: str = ""
    status: StepStatus = "pending"


class ToolsPlan(BaseModel):
    objective: str
    steps: list[ToolsPlanStep] = Field(default_factory=list)
    current_step_id: str = ""
    decision: PlanDecision = "continue"
    decision_reason: str = ""
    stop_conditions: list[str] = Field(default_factory=list)
    finalize_action: str = ""


class ToolsManagerRequest(BaseModel):
    question: str
    tool_policy_override: dict[str, Any] | None = None
    timeout_seconds: int | None = None
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    mutating: bool = False
    strategy_hint: str = ""


class ToolsManagerBatch(BaseModel):
    planner_step_id: str = ""
    batch_reason: str = ""
    requests: list[ToolsManagerRequest] = Field(default_factory=list)
    continue_after: bool = True
    expected_progress: str = ""


class AutoExecuteResult(BaseModel):
    answer: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    passes: int = 0
    terminal_reason: str = ""
    toolsmanager_requests_count: int = 0
    pass_logs: list[dict[str, Any]] = Field(default_factory=list)
    planner_decisions: list[dict[str, Any]] = Field(default_factory=list)
    prechecklist: dict[str, Any] | None = None
    prechecklist_source: str = ""
    prechecklist_warning: str = ""
    execution_backend: str = "local"
    execution_run_id: str = ""
    execution_duration_ms: float = 0.0
    execution_requests_ok: int = 0
    execution_requests_failed: int = 0
    duplicate_request_skips: int = 0
    duplicate_semantic_search_skips: int = 0
    duplicate_tool_execution_blocks: int = 0
    request_retry_attempts: int = 0
    request_retry_exhausted: int = 0
    edit_retry_mode_activations: int = 0
    persisted_fingerprint_counts: dict[str, int] = Field(default_factory=dict)


class ToolsManagerDecisionProvider(Protocol):
    def plan_with_source(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int,
        pass_cap: int,
        previous_plan: ToolsPlan | None,
        pass_logs: Sequence[dict[str, Any]],
        warnings: Sequence[str],
        changed_files: Sequence[str],
        latest_answer: str,
    ) -> tuple[ToolsPlan, list[str], str]:
        ...

    def build_batch(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        pass_index: int,
        pass_cap: int,
        pass_logs: Sequence[dict[str, Any]],
        warnings: Sequence[str],
        changed_files: Sequence[str],
        latest_answer: str,
    ) -> tuple[ToolsManagerBatch | None, list[str]]:
        ...



class _InternalDecisionProvider:
    """Fallback decision provider that uses the orchestrator's _invoke_model directly.

    This is used when no external decision provider is attached (e.g. in tests
    that build the orchestrator with object.__new__ and monkeypatch _invoke_model).
    """

    def __init__(self, orchestrator: "ToolsManagerOrchestrator") -> None:
        self._orchestrator = orchestrator

    def plan_with_source(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int = 0,
        pass_cap: int = 4,
        previous_plan: "ToolsPlan | None" = None,
        pass_logs: "Sequence[dict[str, Any]]" = (),
        warnings: "Sequence[str]" = (),
        changed_files: "Sequence[str]" = (),
        latest_answer: str = "",
    ) -> "tuple[ToolsPlan, list[str], str]":
        import json as _json
        issues: list[str] = []
        payload = {
            "request": request,
            "flow_context": (flow_context or "none").strip(),
            "pass_index": int(pass_index),
            "pass_cap": int(pass_cap),
            "previous_plan": previous_plan.model_dump() if previous_plan is not None else None,
            "pass_logs": list(pass_logs)[-4:],
            "warnings": list(warnings)[-12:],
            "changed_files": list(changed_files),
            "latest_answer": str(latest_answer or "")[:1500],
        }
        human_prompt = _json.dumps(payload, ensure_ascii=False, indent=2)
        raw = self._orchestrator._invoke_model(
            system_prompt="tools_planner",
            human_prompt=human_prompt,
        )
        parsed = self._orchestrator.parse_tools_plan(raw, request=request, previous_plan=previous_plan)
        if parsed is not None:
            return parsed, issues, "planner"

        issues.append("head_tools_planner parse failed; attempting repair")
        repair_raw = self._orchestrator._invoke_model(
            system_prompt="tools_planner_repair",
            human_prompt=(
                "Repair this planner output to strict JSON schema.\n"
                "Do not add markdown. Return only one JSON object.\n\n"
                f"Broken output:\n{raw}\n\n"
                f"Execution payload:\n{human_prompt}"
            ),
        )
        repaired = self._orchestrator.parse_repair(
            repair_raw, "plan", request=request, previous_plan=previous_plan
        )
        if isinstance(repaired, ToolsPlan):
            return repaired, issues, "planner_repair"

        issues.append("head_tools_planner repair failed; using deterministic fallback")
        fallback = self._orchestrator._deterministic_fallback_plan(
            request=request,
            flow_context=flow_context,
            previous_plan=previous_plan,
            reason="planner_parse_failed",
        )
        return fallback, issues, "deterministic_fallback"

    def build_batch(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: "ToolsPlan",
        pass_index: int,
        pass_cap: int,
        pass_logs: "Sequence[dict[str, Any]]" = (),
        warnings: "Sequence[str]" = (),
        changed_files: "Sequence[str]" = (),
        latest_answer: str = "",
    ) -> "tuple[ToolsManagerBatch | None, list[str]]":
        import json as _json
        issues: list[str] = []
        payload = {
            "request": request,
            "flow_context": (flow_context or "").strip(),
            "planner": plan.model_dump(),
            "pass_index": int(pass_index),
            "pass_cap": int(pass_cap),
            "pass_logs": list(pass_logs)[-4:],
            "warnings": list(warnings)[-10:],
            "changed_files": list(changed_files),
            "latest_answer": str(latest_answer or "")[:1500],
        }
        human_prompt = _json.dumps(payload, ensure_ascii=False, indent=2)
        raw = self._orchestrator._invoke_model(
            system_prompt="toolsmanager",
            human_prompt=human_prompt,
        )
        batch = self._orchestrator.parse_tools_batch(raw, planner_step_id=plan.current_step_id)
        if batch is not None:
            return batch, issues

        issues.append("toolsmanager batch invalid; attempting repair")
        repair_raw = self._orchestrator._invoke_model(
            system_prompt="toolsmanager_repair",
            human_prompt=(
                "Repair this tools-manager output to strict JSON schema.\n"
                "Do not add markdown. Return only one JSON object.\n\n"
                f"Broken output:\n{raw}\n\n"
                f"Execution payload:\n{human_prompt}"
            ),
        )
        repaired = self._orchestrator.parse_repair(
            repair_raw, "batch", request=request, previous_plan=plan,
            planner_step_id=plan.current_step_id,
        )
        if isinstance(repaired, ToolsManagerBatch):
            return repaired, issues

        issues.append("toolsmanager repair failed")
        return None, issues


class ToolsManagerOrchestrator:
    """Planner-driven auto-execution loop for agent-tools chat turns."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        worker_client: ToolWorkerClient,
        repo_root: Path,
        base_url: str | None = None,
        execution_config: ToolsExecutionConfig | None = None,
        executor: ToolsExecutor | None = None,
        coding_memory_service: CodingMemoryService | None = None,
        decision_provider: ToolsManagerDecisionProvider | None = None,
    ) -> None:
        _ = (api_key, model, base_url)
        self.worker_client = worker_client
        self.repo_root = repo_root.resolve()
        self.execution_config = execution_config or ToolsExecutionConfig()
        self.executor = executor or LocalToolsExecutor(worker_client=worker_client)
        try:
            from .config import ToolsExecutionConfig
            from .executor import LocalToolsExecutor
            self.execution_config = execution_config or ToolsExecutionConfig()
            self.executor = executor or LocalToolsExecutor(worker_client=worker_client)
        except (ImportError, NameError):
            self.execution_config = execution_config
            self.executor = executor
            
        self.coding_memory_service = coding_memory_service
        self._decision_provider: ToolsManagerDecisionProvider | None = decision_provider

    def _setup_llm(self) -> None:
        """Deprecated: tools manager is deterministic and does not use an LLM."""
        return None

    @staticmethod
    def _normalize_request(req: ToolsManagerRequest) -> ToolsManagerRequest:
        normalized_tool_name = str(req.tool_name or "").strip()
        normalized_tool_args = dict(req.tool_args or {})
        normalized_strategy = str(req.strategy_hint or "").strip().lower()
        if normalized_tool_name and not req.question:
            inferred = f"run tool {normalized_tool_name}"
        else:
            inferred = str(req.question or "")
        return ToolsManagerRequest(
            question=inferred,
            tool_policy_override=dict(req.tool_policy_override or {}),
            timeout_seconds=req.timeout_seconds,
            tool_name=normalized_tool_name,
            tool_args=normalized_tool_args,
            mutating=bool(req.mutating),
            strategy_hint=normalized_strategy,
        )

    @staticmethod
    def _should_force_write_fallback(
        *,
        request: ToolsManagerRequest,
        patch_attempts: int,
        saw_no_change: bool,
        failed: bool,
    ) -> bool:
        if str(request.tool_name or "") != "apply_patch":
            return False
        if str(request.strategy_hint or "") not in ("", "auto"):
            return False
        return failed or saw_no_change or patch_attempts >= 2

    def update_model(self, new_model: str) -> None:
        """No-op: tools manager has no model dependency."""
        logger.info("Ignoring model update; ToolsManagerOrchestrator is deterministic-only.")
        _ = new_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True
    )
    def _call_llm_with_retry(self, messages: list[Any]) -> Any:
        """
        متد مرکزی برای تمام فراخوانی‌های LLM.
        این متد مجهز به Retry با Exponential Backoff است.
        """
        _ = messages
        raise RuntimeError("ToolsManagerOrchestrator no longer supports LLM calls")

    @staticmethod
    def _strip_code_fence(raw: str) -> str:
        text = str(raw or "").strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return text

    @staticmethod
    def _parse_json_or_literal(raw: str) -> Any | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    @staticmethod
    def _extract_json_object_text(text: str) -> str | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : idx + 1].strip()
        return None

    @classmethod
    def _collect_candidates(cls, raw_text: str) -> list[Any]:
        pending: list[Any] = [raw_text]
        candidates: list[Any] = []
        seen_text: set[str] = set()
        seen_ids: set[int] = set()

        while pending:
            item = pending.pop(0)
            if isinstance(item, str):
                text = item.strip()
                if not text or text in seen_text:
                    continue
                seen_text.add(text)
                candidates.append(text)

                unwrapped = cls._strip_code_fence(text)
                if unwrapped and unwrapped not in seen_text:
                    pending.append(unwrapped)

                obj_text = cls._extract_json_object_text(text)
                if obj_text and obj_text not in seen_text:
                    pending.append(obj_text)

                parsed = cls._parse_json_or_literal(text)
                if parsed is not None:
                    pending.append(parsed)
                continue

            if isinstance(item, dict):
                marker = id(item)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                candidates.append(item)

                for key in ("answer", "content", "text", "message", "output", "payload", "data", "raw"):
                    if key in item:
                        pending.append(item.get(key))
                for value in item.values():
                    if isinstance(value, (dict, list)):
                        pending.append(value)
                    elif isinstance(value, str) and len(value) <= 20000:
                        pending.append(value)
                continue

            if isinstance(item, list):
                marker = id(item)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                candidates.append(item)
                pending.extend(item)

        return candidates

    @classmethod
    def _parse_model(cls, text: str, model_cls: type[_ModelT]) -> _ModelT:
        last_error: Exception | None = None
        for candidate in cls._collect_candidates(text):
            if isinstance(candidate, dict):
                try:
                    return model_cls.model_validate(candidate)
                except Exception as exc:
                    last_error = exc
            elif isinstance(candidate, str):
                parsed = cls._parse_json_or_literal(candidate)
                if isinstance(parsed, dict):
                    try:
                        return model_cls.model_validate(parsed)
                    except Exception as exc:
                        last_error = exc
        if last_error is not None:
            raise ValueError(str(last_error)) from last_error
        raise ValueError("No valid JSON object found")

    @staticmethod
    def _status_from_text(text: str) -> StepStatus:
        lowered = str(text or "").strip().lower()
        if lowered in {"pending", "in_progress", "done", "blocked"}:
            return lowered  # type: ignore[return-value]
        return "pending"

    def _normalize_plan(self, plan: ToolsPlan, *, previous_plan: ToolsPlan | None = None) -> ToolsPlan:
        steps: list[ToolsPlanStep] = []
        seen_ids: set[str] = set()
        for idx, step in enumerate(plan.steps, start=1):
            base_id = str(step.id or "").strip() or f"s{idx}"
            step_id = base_id
            suffix = 1
            while step_id in seen_ids:
                suffix += 1
                step_id = f"{base_id}_{suffix}"
            seen_ids.add(step_id)
            steps.append(
                ToolsPlanStep(
                    id=step_id,
                    title=str(step.title or "").strip() or f"Step {idx}",
                    tool_intent=step.tool_intent,
                    args_hint=str(step.args_hint or "").strip(),
                    success_signal=str(step.success_signal or "").strip(),
                    fallback=str(step.fallback or "").strip(),
                    status=self._status_from_text(step.status),
                )
            )

        if not steps and previous_plan is not None and previous_plan.steps:
            steps = [ToolsPlanStep.model_validate(item.model_dump()) for item in previous_plan.steps]

        if not steps:
            steps = [
                ToolsPlanStep(
                    id="s1",
                    title="Inspect target files",
                    tool_intent="inspect",
                    args_hint="Choose repo_search, semantic_search, read_file, find_symbols, or call_graph for the task.",
                    success_signal="relevant file context gathered",
                    fallback="If unknown files, run targeted search once.",
                    status="in_progress",
                ),
                ToolsPlanStep(
                    id="s2",
                    title="Apply requested changes",
                    tool_intent="edit",
                    args_hint="Use apply_patch first, write_file fallback if needed.",
                    success_signal="requested edits applied",
                    fallback="Use write_file if patch loop fails twice.",
                    status="pending",
                ),
                ToolsPlanStep(
                    id="s3",
                    title="Verify and finalize",
                    tool_intent="verify",
                    args_hint="Run targeted verification and summarize.",
                    success_signal="verification complete",
                    fallback="If verification tooling unavailable, state limits and remaining risk.",
                    status="pending",
                ),
            ]

        objective = str(plan.objective or "").strip() or "Execute requested plan"
        decision: PlanDecision = str(plan.decision or "continue").strip().lower()  # type: ignore[assignment]
        if decision not in {"continue", "revise", "finalize", "stop"}:
            decision = "continue"

        current_step_id = str(plan.current_step_id or "").strip()
        if current_step_id not in {step.id for step in steps}:
            active = next((step for step in steps if step.status not in {"done", "blocked"}), None)
            current_step_id = active.id if active is not None else steps[0].id

        if decision in {"finalize", "stop"} and not str(plan.decision_reason or "").strip():
            decision_reason = "Planner marked terminal decision."
        else:
            decision_reason = str(plan.decision_reason or "").strip()

        stop_conditions = [str(item).strip() for item in plan.stop_conditions if str(item).strip()]
        if not stop_conditions:
            stop_conditions = [
                "Planner chooses finalize/stop",
                "Two consecutive non-actionable passes",
                "Pass cap reached",
            ]

        finalize_action = str(plan.finalize_action or "").strip() or "Return final answer with completed work and verification."

        return ToolsPlan(
            objective=objective,
            steps=steps,
            current_step_id=current_step_id,
            decision=decision,
            decision_reason=decision_reason,
            stop_conditions=stop_conditions,
            finalize_action=finalize_action,
        )

    def _deterministic_fallback_plan(
        self,
        *,
        request: str,
        flow_context: str | None,
        previous_plan: ToolsPlan | None,
        reason: str,
    ) -> ToolsPlan:
        if previous_plan is not None:
            base = self._normalize_plan(previous_plan, previous_plan=previous_plan)
            return ToolsPlan(
                objective=base.objective,
                steps=base.steps,
                current_step_id=base.current_step_id,
                decision="continue",
                decision_reason=f"Deterministic fallback: {reason}",
                stop_conditions=base.stop_conditions,
                finalize_action=base.finalize_action,
            )

        context_hint = ""
        if flow_context:
            for line in str(flow_context).splitlines():
                text = line.strip()
                if text.lower().startswith("current objective:"):
                    context_hint = text.split(":", 1)[1].strip()
                    break

        objective = context_hint or (" ".join((request or "").strip().split())[:220] or "Execute requested plan")
        plan = ToolsPlan(
            objective=objective,
            steps=[],
            current_step_id="",
            decision="continue",
            decision_reason=f"Deterministic fallback: {reason}",
            stop_conditions=[],
            finalize_action="Return final answer with completed work.",
        )
        return self._normalize_plan(plan, previous_plan=None)

    def parse_tools_plan(
        self,
        raw_text: str,
        *,
        request: str,
        previous_plan: ToolsPlan | None = None,
    ) -> ToolsPlan | None:
        try:
            parsed = self._parse_model(raw_text, ToolsPlan)
            return self._normalize_plan(parsed, previous_plan=previous_plan)
        except Exception:
            return None

    def _normalize_batch(self, batch: ToolsManagerBatch, *, planner_step_id: str) -> ToolsManagerBatch:
        requests: list[ToolsManagerRequest] = []
        for item in batch.requests[:8]:
            question = str(item.question or "").strip()
            if not question:
                continue
            override = item.tool_policy_override if isinstance(item.tool_policy_override, dict) else None
            timeout = item.timeout_seconds if isinstance(item.timeout_seconds, int) else None
            requests.append(
                ToolsManagerRequest(
                    question=question,
                    tool_policy_override=override,
                    timeout_seconds=timeout,
                )
            )

        resolved_step = str(batch.planner_step_id or "").strip() or planner_step_id
        return ToolsManagerBatch(
            planner_step_id=resolved_step,
            batch_reason=str(batch.batch_reason or "").strip() or "toolsmanager_batch",
            requests=requests,
            continue_after=bool(batch.continue_after),
            expected_progress=str(batch.expected_progress or "").strip(),
        )

    def parse_tools_batch(
        self,
        raw_text: str,
        *,
        planner_step_id: str,
    ) -> ToolsManagerBatch | None:
        try:
            parsed = self._parse_model(raw_text, ToolsManagerBatch)
            batch = self._normalize_batch(parsed, planner_step_id=planner_step_id)
            self._validate_batch(batch)
            return batch
        except Exception:
            return None

    def parse_repair(
        self,
        raw_text: str,
        schema_kind: Literal["plan", "batch"],
        *,
        request: str,
        previous_plan: ToolsPlan | None = None,
        planner_step_id: str = "",
    ) -> ToolsPlan | ToolsManagerBatch | None:
        if schema_kind == "plan":
            return self.parse_tools_plan(raw_text, request=request, previous_plan=previous_plan)
        return self.parse_tools_batch(raw_text, planner_step_id=planner_step_id)

    @staticmethod
    def _validate_batch(batch: ToolsManagerBatch) -> None:
        for idx, req in enumerate(batch.requests):
            if not str(req.question or "").strip():
                raise ValueError(f"request[{idx}] question must not be empty")

    def _invoke_model(self, *, system_prompt: str, human_prompt: str) -> str:
        _ = (system_prompt, human_prompt)
        raise RuntimeError("ToolsManagerOrchestrator no longer supports model invocation")

    def attach_decision_provider(self, provider: ToolsManagerDecisionProvider) -> None:
        self._decision_provider = provider

    def _decision_provider_or_raise(self) -> ToolsManagerDecisionProvider:
        provider = getattr(self, "_decision_provider", None)
        if provider is None:
            return _InternalDecisionProvider(self)
        return provider

    def _plan(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int = 0,
        pass_cap: int = 4,
        previous_plan: ToolsPlan | None = None,
        pass_logs: Sequence[dict[str, Any]] = (),
        warnings: Sequence[str] = (),
        changed_files: Sequence[str] = (),
        latest_answer: str = "",
    ) -> tuple[ToolsPlan, list[str]]:
        plan, issues, _source = self._plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=pass_index,
            pass_cap=pass_cap,
            previous_plan=previous_plan,
            pass_logs=pass_logs,
            warnings=warnings,
            changed_files=changed_files,
            latest_answer=latest_answer,
        )
        return plan, issues

    def _build_batch(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        pass_index: int,
        pass_cap: int,
        pass_logs: Sequence[dict[str, Any]],
        warnings: Sequence[str],
        changed_files: Sequence[str],
        latest_answer: str = "",
    ) -> tuple[ToolsManagerBatch | None, list[str]]:
        provider = self._decision_provider_or_raise()
        return provider.build_batch(
            request=request,
            flow_context=flow_context,
            plan=plan,
            pass_index=pass_index,
            pass_cap=pass_cap,
            pass_logs=pass_logs,
            warnings=warnings,
            changed_files=changed_files,
            latest_answer=latest_answer,
        )

    @staticmethod
    def _merge_policy(base_policy: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(base_policy)
        if isinstance(override, dict):
            for key, value in override.items():
                merged[key] = value
        return merged

    @staticmethod
    def _clip_timeout(value: int | None, *, session_timeout: int) -> int:
        base = int(value or session_timeout)
        return max(5, min(base, max(5, int(session_timeout))))

    @staticmethod
    def _fingerprint_request(question: str, policy: dict[str, Any], timeout_seconds: int) -> str:
        normalized_question = re.sub(r"\s+", " ", str(question or "").strip()).lower()
        raw = json.dumps(
            {
                "question": normalized_question,
                "policy": policy,
                "timeout_seconds": timeout_seconds,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _normalize_fingerprint_key(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip()).lower()

    @classmethod
    def _semantic_request_fingerprint(cls, question: str) -> str | None:
        normalized = cls._normalize_fingerprint_key(question)
        if not normalized:
            return None
        if not any(token in normalized for token in ("semantic_search", "search", "find", "locate", "grep")):
            return None
        semantic_key = cls._normalize_semantic_key(normalized)
        return hashlib.sha1(semantic_key.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def _normalize_semantic_key(cls, text: str) -> str:
        normalized = cls._normalize_fingerprint_key(text)
        if not normalized:
            return ""
        query_match = re.search(r"query\s*[:=]\s*['\"]?([^'\"\n]+)['\"]?", normalized)
        k_match = re.search(r"\bk\s*[:=]\s*(\d+)", normalized)
        query = cls._normalize_fingerprint_key(query_match.group(1)) if query_match else ""
        if query:
            query = re.sub(r"\bk\s*[:=]\s*\d+.*$", "", query).strip()
        k_val = str(k_match.group(1)).strip() if k_match else ""
        if query or k_val:
            return f"query={query}|k={k_val or '0'}"
        return normalized

    @classmethod
    def _semantic_trace_fingerprints(cls, rows: Sequence[dict[str, Any]]) -> set[str]:
        out: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("tool_name", "")).strip().lower() != "semantic_search":
                continue
            args_summary = cls._normalize_semantic_key(str(row.get("args_summary", "") or ""))
            if not args_summary:
                continue
            out.add(hashlib.sha1(args_summary.encode("utf-8")).hexdigest()[:12])
        return out

    @staticmethod
    def _looks_like_search_request_text(question: str) -> bool:
        lowered = str(question or "").strip().lower()
        if not lowered:
            return False
        patterns = (
            "semantic_search",
            "search",
            "find",
            "locate",
            "grep",
            "inspect repository",
        )
        return any(token in lowered for token in patterns)

    @staticmethod
    def _looks_like_edit_request_text(question: str) -> bool:
        lowered = str(question or "").strip().lower()
        if not lowered:
            return False
        patterns = (
            "apply_patch",
            "write_file",
            "edit",
            "modify",
            "mutation",
            "change file",
            "update file",
            "create",
            "generate",
            "add file",
            "write a",
            "write the",
        )
        return any(token in lowered for token in patterns)

    def _is_edit_task(self, plan: "ToolsPlan", request: str) -> bool:
        """A task is an edit task if the plan has an edit step or the request reads as one."""
        if any(step.tool_intent == "edit" for step in getattr(plan, "steps", []) or []):
            return True
        return self._looks_like_edit_request_text(request)

    def _compute_effective_pass_cap(
        self,
        *,
        configured_pass_cap: int,
        plan: "ToolsPlan",
        request: str,
        max_allowed_passes: int = 12,
    ) -> int:
        """Ensure edit tasks reserve enough passes to reach edit + verify.

        Inspect/search steps are compressed into a single pass (they can run as
        one batch), then at least one edit and one verify pass are reserved so
        an 8-step checklist with a low configured cap still reaches the
        create/write/verify stages.
        """
        configured = max(1, min(int(configured_pass_cap), max_allowed_passes))
        if not self._is_edit_task(plan, request):
            return configured

        intents = [step.tool_intent for step in getattr(plan, "steps", []) or []]
        edit_steps = sum(1 for intent in intents if intent == "edit")
        verify_steps = sum(1 for intent in intents if intent == "verify")
        inspect_like = sum(1 for intent in intents if intent in ("inspect", "search"))

        inspect_passes = 1 if inspect_like else 0
        # Always reserve >=1 edit and >=1 verify pass for edit tasks.
        minimum_passes = inspect_passes + max(1, edit_steps) + max(1, verify_steps)
        return max(configured, min(max_allowed_passes, minimum_passes + 2))

    @staticmethod
    def _looks_like_apply_patch_failure_trace(row: dict[str, Any]) -> bool:
        if str(row.get("tool_name", "")).strip().lower() != "apply_patch":
            return False
        status = str(row.get("status", "")).strip().lower()
        preview = str(row.get("output_preview", "")).strip().lower()
        if status in {"error", "timeout", "failed"}:
            return True
        if status == "ok" and not preview:
            return True
        if '"ok": false' in preview or "'ok': false" in preview or '"error"' in preview:
            return True
        return False

    @staticmethod
    def _build_failure_signature(*, question: str, code: str, detail: str) -> str:
        normalized = {
            "question": re.sub(r"\s+", " ", str(question or "").strip()).lower()[:320],
            "code": str(code or "").strip().lower(),
            "detail": re.sub(r"\s+", " ", str(detail or "").strip()).lower()[:320],
        }
        raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _mutation_retry_request(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        step: ToolsPlanStep | None,
        pass_index: int,
    ) -> ToolsManagerRequest:
        base = self._deterministic_fallback_request(
            request=request,
            flow_context=flow_context,
            plan=plan,
            step=step,
            pass_index=pass_index,
        )
        lines = [
            str(base.question if base is not None else "").strip(),
            "Mutation retry lock is active because prior apply_patch attempt failed or no-oped.",
            "Do not start new broad semantic search.",
            "Execute a direct mutation fallback now: use write_file with full content if apply_patch fails.",
            "Verify changed_files evidence before any terminal response.",
        ]
        return ToolsManagerRequest(question="\n".join(line for line in lines if line.strip()))

    def _git_status_paths(self) -> set[str]:
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return set()
            paths: set[str] = set()
            for line in proc.stdout.splitlines():
                if len(line) >= 4:
                    path = line[3:].strip()
                    if path:
                        paths.add(path.replace("\\", "/"))
            return paths
        except Exception:
            return set()

    def _resolve_step(self, plan: ToolsPlan) -> ToolsPlanStep | None:
        for step in plan.steps:
            if step.id == plan.current_step_id:
                return step
        return plan.steps[0] if plan.steps else None

    @staticmethod
    def _planner_decision_row(plan: ToolsPlan, pass_index: int) -> dict[str, Any]:
        step = next((item for item in plan.steps if item.id == plan.current_step_id), None)
        return {
            "pass_index": int(pass_index),
            "current_step_id": str(plan.current_step_id or ""),
            "current_step_title": str(getattr(step, "title", "") or ""),
            "decision": str(plan.decision or "continue"),
            "decision_reason": str(plan.decision_reason or ""),
        }

    @staticmethod
    def _planner_task_fingerprint(step: ToolsPlanStep | None) -> str:
        if step is None:
            return ""
        raw = json.dumps(
            {
                "id": str(step.id or "").strip().lower(),
                "title": re.sub(r"\s+", " ", str(step.title or "").strip()).lower(),
                "tool_intent": str(step.tool_intent or "").strip().lower(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def _recent_step_repeat_count(
        cls,
        pass_logs: Sequence[dict[str, Any]],
        *,
        step_id: str,
    ) -> int:
        target = str(step_id or "").strip()
        if not target:
            return 0
        count = 0
        for row in reversed(list(pass_logs)):
            if not isinstance(row, dict):
                continue
            if str(row.get("planner_step_id", "")).strip() != target:
                break
            if int(row.get("requests_count", 0) or 0) <= 0 and int(row.get("tool_steps", 0) or 0) <= 0:
                break
            count += 1
        return count

    @staticmethod
    def _advance_plan_to_next_unfinished_step(plan: ToolsPlan) -> ToolsPlan | None:
        current_id = str(plan.current_step_id or "").strip()
        if not current_id:
            return None
        candidate: ToolsPlanStep | None = None
        seen_current = False
        for item in plan.steps:
            if str(item.id or "").strip() == current_id:
                seen_current = True
                continue
            if str(item.status or "").strip().lower() in {"done", "blocked"}:
                continue
            if seen_current:
                candidate = item
                break
            if candidate is None:
                candidate = item
        if candidate is None:
            return None
        updated_steps: list[ToolsPlanStep] = []
        for item in plan.steps:
            if str(item.id or "").strip() == current_id and str(item.status or "").strip().lower() == "in_progress":
                updated_steps.append(item.model_copy(update={"status": "done"}))
            elif str(item.id or "").strip() == str(candidate.id or "").strip():
                updated_steps.append(item.model_copy(update={"status": "in_progress"}))
            else:
                updated_steps.append(item)
        return plan.model_copy(
            update={
                "steps": updated_steps,
                "current_step_id": str(candidate.id or "").strip(),
                "decision": "continue",
                "decision_reason": (
                    f"Auto-advanced from duplicate task {current_id} to next unresolved step {str(candidate.id or '').strip()}"
                ),
            }
        )

    @staticmethod
    def _truncate_line(value: str, *, limit: int = 220) -> str:
        text = " ".join(str(value or "").strip().split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _deterministic_fallback_request(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        step: ToolsPlanStep | None,
        pass_index: int,
    ) -> ToolsManagerRequest | None:
        if step is None:
            return None

        step_title = self._truncate_line(step.title or "current step")
        step_hint = self._truncate_line(step.args_hint or step.fallback or step.success_signal or "")
        objective = self._truncate_line(plan.objective or request or "Execute requested plan")
        flow = self._truncate_line(flow_context or "", limit=320)
        user_request = self._truncate_line(request or "", limit=320)

        if step.tool_intent == "inspect":
            directive = (
                "Inspect repository files for this step using repo_search, semantic_search, read_file, "
                "find_symbols, call_graph, or run_command as appropriate. "
                "Gather concrete evidence with file paths and line ranges."
            )
        elif step.tool_intent == "search":
            directive = (
                "Run targeted repository search for the requested behavior and gather concrete file evidence "
                "before proposing edits."
            )
        elif step.tool_intent == "edit":
            directive = (
                "Apply concrete repository edits for this step. Prefer apply_patch first; if patch chain fails or no-ops, "
                "force write_file full-content fallback, then verify changed_files evidence before terminal response. "
                "Do not emit conversational terminal text for unresolved edit-intent work."
            )
        elif step.tool_intent == "verify":
            directive = (
                "Verify relevant changes with targeted checks (tests/lint/type checks or focused run_command checks), "
                "then summarize verification evidence."
            )
        else:
            directive = (
                "Summarize current status with concrete repository evidence and identify the next actionable step."
            )

        lines = [
            f"Deterministic fallback request for planner pass {int(pass_index)}.",
            f"Objective: {objective}",
            f"Planner step: {step_title}",
            f"Intent: {step.tool_intent}",
            f"Original request: {user_request or '-'}",
        ]
        if step_hint:
            lines.append(f"Step hint: {step_hint}")
        if flow:
            lines.append(f"Flow context: {flow}")
        lines.append(f"Action: {directive}")
        return ToolsManagerRequest(question="\n".join(lines))

    @staticmethod
    def _looks_like_conversational_terminal(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        patterns = (
            "if you want",
            "reply \"yes",
            "reply 'yes",
            "let me know if you want",
            "if you want, i can",
            "if you want i can",
            "say yes",
            "type yes",
        )
        return any(token in lowered for token in patterns)

    @classmethod
    def _looks_like_hard_blocker_prompt(cls, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        patterns = (
            "missing credential",
            "missing credentials",
            "missing api key",
            "missing token",
            "missing secret",
            "permission denied",
            "insufficient permission",
            "unauthorized",
            "forbidden",
            "access denied",
            "missing target identifier",
            "target identifier required",
            "missing identifier",
            "identifier is required",
            "missing file path",
            "path is required",
            "target path required",
            "provide file path",
            "unavailable",
        )
        return any(token in lowered for token in patterns)

    @classmethod
    def _looks_like_non_hard_blocker_prompt(cls, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        if cls._looks_like_hard_blocker_prompt(lowered):
            return False
        patterns = (
            "blocker decision",
            "need one blocker decision",
            "scope choice",
            "scope decision",
            "need scope",
            "which scope",
            "choose scope",
            "choose option",
            "which option",
            "option 1",
            "option 2",
            "1 or 2",
            "one or two",
            "pick one",
            "awaiting scope decision",
            "awaiting your decision",
            "i'm blocked on",
            "i am blocked on",
            "blocked on making",
            "need to read",
            "need to inspect",
            "need to review",
            "before patching",
            "before editing",
            "before making changes",
            "share permission to proceed",
            "permission to proceed",
            "requires explicit tool execution",
            "tool access is available",
            "once tool access is available",
        )
        return any(token in lowered for token in patterns)

    @classmethod
    def _is_blocked_terminal_plan(cls, plan: ToolsPlan) -> bool:
        fields = [str(plan.decision_reason or ""), str(plan.finalize_action or "")]
        combined = " ".join(fields)
        if str(plan.decision or "").strip().lower() == "stop":
            return cls._looks_like_hard_blocker_prompt(combined)
        return cls._looks_like_hard_blocker_prompt(combined)

    @staticmethod
    def _synthesize_terminal_answer(
        *,
        terminal_reason: str,
        pass_logs: Sequence[dict[str, Any]],
        planner_decisions: Sequence[dict[str, Any]],
        toolsmanager_requests_count: int,
    ) -> str:
        reason = str(terminal_reason or "unknown").strip() or "unknown"
        passes = len(pass_logs)
        last_pass = pass_logs[-1] if pass_logs else {}
        if not isinstance(last_pass, dict):
            last_pass = {}
        last_step = str(last_pass.get("planner_step_title", "") or "").strip()
        decision_reason = ""
        if planner_decisions:
            tail = planner_decisions[-1]
            if isinstance(tail, dict):
                decision_reason = str(tail.get("decision_reason", "") or "").strip()
        if not decision_reason:
            decision_reason = str(last_pass.get("planner_decision_reason", "") or "").strip()

        lines = [
            "Auto-execute ended without a direct answer from tool runs.",
            f"terminal_reason={reason}",
            f"passes={passes}",
            f"toolsmanager_requests={int(toolsmanager_requests_count)}",
        ]
        if last_step:
            lines.append(f"last_step={last_step}")
        if decision_reason:
            lines.append(f"planner_reason={decision_reason}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_prechecklist(plan: ToolsPlan, *, source: str) -> dict[str, Any]:
        steps: list[dict[str, str]] = []
        for item in plan.steps[:20]:
            steps.append(
                {
                    "id": str(item.id or "").strip() or "step",
                    "title": str(item.title or "").strip() or "step",
                    "status": str(item.status or "pending"),
                }
            )
        return {
            "objective": str(plan.objective or "").strip(),
            "steps": steps,
            "source": str(source or ""),
        }

    def preview_plan(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_cap: int,
    ) -> dict[str, Any]:
        plan, warnings, source = self._plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=0,
            pass_cap=pass_cap,
            previous_plan=None,
            pass_logs=[],
            warnings=[],
            changed_files=[],
            latest_answer="",
        )
        warning_text = ""
        if source == "deterministic_fallback":
            warning_text = "Planner parse failed; using deterministic fallback checklist."
        return {
            "prechecklist": self._normalize_prechecklist(plan, source=source),
            "prechecklist_source": source,
            "prechecklist_warning": warning_text,
            "warnings": warnings,
        }

    def _plan_with_source(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int = 0,
        pass_cap: int = 4,
        previous_plan: ToolsPlan | None = None,
        pass_logs: Sequence[dict[str, Any]] = (),
        warnings: Sequence[str] = (),
        changed_files: Sequence[str] = (),
        latest_answer: str = "",
    ) -> tuple[ToolsPlan, list[str], str]:
        provider = self._decision_provider_or_raise()
        return provider.plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=pass_index,
            pass_cap=pass_cap,
            previous_plan=previous_plan,
            pass_logs=pass_logs,
            warnings=warnings,
            changed_files=changed_files,
            latest_answer=latest_answer,
        )

    def run(
        self,
        *,
        request: str,
        flow_context: str | None,
        index_dir: str | Path | None,
        index_dirs: Sequence[str | Path] | None,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        tool_policy: dict[str, Any],
        pass_cap: int,
        on_event: Callable[[Any], None] | None = None,
        flow_id: str | None = None,
    ) -> AutoExecuteResult:
        pass_cap = max(1, min(int(pass_cap), 12))
        all_warnings: list[str] = []
        all_trace: list[dict[str, Any]] = []
        all_sources: list[dict[str, Any]] = []
        all_pass_logs: list[dict[str, Any]] = []
        planner_decisions: list[dict[str, Any]] = []
        terminal_reason = "pass_cap_reached"
        toolsmanager_requests_count = 0
        latest_answer = ""
        stalled_passes = 0
        execution_run_id = uuid.uuid4().hex
        execution_started = time.perf_counter()
        execution_requests_ok = 0
        execution_requests_failed = 0
        duplicate_request_skips = 0
        duplicate_semantic_search_skips = 0
        duplicate_tool_execution_blocks = 0
        request_retry_attempts = 0
        request_retry_exhausted = 0
        edit_retry_mode_activations = 0
        edit_retry_mode_pending = False
        persisted_fingerprint_counts: dict[str, int] = {}
        recent_failure_summaries: list[str] = []
        seen_planner_task_fingerprints: set[str] = set()
        execution_backend = str(getattr(getattr(self, "execution_config", None), "backend", "local") or "local")
        memory_service = getattr(self, "coding_memory_service", None)
        flow_key = str(flow_id or "").strip()

        seen_request_fingerprints: set[str] = set()
        seen_semantic_search_fingerprints: set[str] = set()
        executed_tools_this_turn: set[str] = set()
        seen_failure_signatures: set[str] = set()

        if memory_service is not None and flow_key:
            try:
                seen_request_fingerprints = set(
                    memory_service.get_tool_fingerprints(
                        flow_id=flow_key,
                        kind="request_fingerprint",
                    )
                )
                seen_semantic_search_fingerprints = set(
                    memory_service.get_tool_fingerprints(
                        flow_id=flow_key,
                        kind="semantic_search_fingerprint",
                    )
                )
                seen_failure_signatures = set(
                    memory_service.get_tool_fingerprints(
                        flow_id=flow_key,
                        kind="mutation_failure_signature",
                    )
                )
                seen_planner_task_fingerprints = set(
                    memory_service.get_tool_fingerprints(
                        flow_id=flow_key,
                        kind="planner_task_fingerprint",
                    )
                )
            except Exception as exc:
                all_warnings.append(f"toolsmanager persistent fingerprint load failed: {exc}")
            persisted_fingerprint_counts = {
                "request_fingerprint": len(seen_request_fingerprints),
                "semantic_search_fingerprint": len(seen_semantic_search_fingerprints),
                "mutation_failure_signature": len(seen_failure_signatures),
                "planner_task_fingerprint": len(seen_planner_task_fingerprints),
            }

        plan, plan_warnings, _source = self._plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=0,
            pass_cap=pass_cap,
            previous_plan=None,
            pass_logs=[],
            warnings=[],
            changed_files=[],
            latest_answer="",
        )
        all_warnings.extend(plan_warnings)

        # Edit/create/verify workflows need enough passes to reach the write and
        # verify stages; a low configured cap would otherwise terminate early.
        is_edit_task = self._is_edit_task(plan, request)
        effective_pass_cap = self._compute_effective_pass_cap(
            configured_pass_cap=pass_cap,
            plan=plan,
            request=request,
        )
        if effective_pass_cap != pass_cap:
            all_warnings.append(
                f"pass_cap_raised_for_edit_task: {pass_cap} -> {effective_pass_cap}"
            )
            pass_cap = effective_pass_cap

        before = self._git_status_paths()
        changed_files: list[str] = []

        for pass_index in range(1, pass_cap + 1):
            step = self._resolve_step(plan)
            step_repeat_count = self._recent_step_repeat_count(
                all_pass_logs,
                step_id=str(getattr(step, "id", "") or ""),
            )
            task_fingerprint = self._planner_task_fingerprint(step)
            if (
                step is not None
                and not changed_files
                and not edit_retry_mode_pending
                and (step_repeat_count >= 1 or (task_fingerprint and task_fingerprint in seen_planner_task_fingerprints))
            ):
                advanced_plan = self._advance_plan_to_next_unfinished_step(plan)
                if advanced_plan is not None:
                    all_warnings.append("planner_duplicate_task_advanced")
                    plan = advanced_plan
                    step = self._resolve_step(plan)
            planner_row = self._planner_decision_row(plan, pass_index)
            planner_decisions.append(planner_row)

            if plan.decision in {"finalize", "stop"}:
                decision_text = " ".join(
                    [
                        str(plan.decision_reason or "").strip(),
                        str(plan.finalize_action or "").strip(),
                    ]
                ).strip()
                has_hard_blocker = self._looks_like_hard_blocker_prompt(decision_text)
                has_non_hard_blocker = self._looks_like_non_hard_blocker_prompt(decision_text)
                has_conversational_terminal = (
                    self._looks_like_conversational_terminal(plan.finalize_action)
                    or self._looks_like_conversational_terminal(plan.decision_reason)
                )
                should_retry_terminal = bool(
                    (not changed_files)
                    and pass_index < pass_cap
                    and (
                        has_conversational_terminal
                        or has_non_hard_blocker
                        or (plan.decision == "stop" and not has_hard_blocker)
                    )
                )
                if should_retry_terminal:
                    if has_conversational_terminal:
                        all_warnings.append(
                            "planner_finalize_conversational_without_edits; forcing another execution pass"
                        )
                    if has_non_hard_blocker or (plan.decision == "stop" and not has_hard_blocker):
                        all_warnings.append(
                            "planner_terminal_nonhard_blocker_retry; forcing another execution pass"
                        )
                    plan = self._deterministic_fallback_plan(
                        request=(
                            f"{request}\n\n"
                            "Full-auto continuation directive:\n"
                            "- Do not ask for confirmation.\n"
                            "- Do not pause for non-hard scope/option choices.\n"
                            "- Continue with concrete file inspection/edits/verification.\n"
                            "- Return blocked only for true blockers."
                        ),
                        flow_context=flow_context,
                        previous_plan=plan,
                        reason="terminal_nonhard_blocker_retry",
                    )
                    continue

                if (
                    plan.decision == "finalize"
                    and not changed_files
                    and not self._is_blocked_terminal_plan(plan)
                    and (
                        self._looks_like_conversational_terminal(plan.finalize_action)
                        or self._looks_like_conversational_terminal(plan.decision_reason)
                    )
                    and pass_index < pass_cap
                ):
                    all_warnings.append(
                        "planner_finalize_conversational_without_edits; forcing another execution pass"
                    )
                    plan = self._deterministic_fallback_plan(
                        request=(
                            f"{request}\n\n"
                            "Full-auto continuation directive:\n"
                            "- Do not ask for confirmation.\n"
                            "- Continue with concrete file inspection/edits/verification.\n"
                            "- Return blocked only for true blockers."
                        ),
                        flow_context=flow_context,
                        previous_plan=plan,
                        reason="conversational_terminal_retry",
                    )
                    continue

                # Never let an edit task finalize as "success" without file-change
                # evidence; force another pass to actually write + verify.
                if (
                    plan.decision == "finalize"
                    and is_edit_task
                    and not changed_files
                    and not self._is_blocked_terminal_plan(plan)
                    and pass_index < pass_cap
                ):
                    all_warnings.append(
                        "planner_finalize_edit_without_changed_files; forcing edit/verify pass"
                    )
                    plan = self._deterministic_fallback_plan(
                        request=(
                            f"{request}\n\n"
                            "Full-auto continuation directive:\n"
                            "- This is a file create/edit task but no files have changed yet.\n"
                            "- Use write_file (full content) or apply_patch to make the change now.\n"
                            "- Then verify the file exists before any terminal response.\n"
                            "- Do not claim success without changed_files evidence."
                        ),
                        flow_context=flow_context,
                        previous_plan=plan,
                        reason="edit_finalize_without_changes",
                    )
                    edit_retry_mode_pending = True
                    continue
                if plan.decision == "stop":
                    if has_hard_blocker:
                        all_warnings.append("planner_terminal_hard_blocker_stop")
                    else:
                        all_warnings.append(
                            "planner_terminal_nonhard_blocker_retry_exhausted; pass cap reached before retry"
                        )
                        terminal_reason = "pass_cap_reached"
                        if not latest_answer:
                            latest_answer = str(plan.decision_reason or plan.finalize_action or "").strip()
                        all_pass_logs.append(
                            {
                                "pass_index": pass_index,
                                "planner_step_id": plan.current_step_id,
                                "planner_step_title": str(getattr(step, "title", "") or ""),
                                "planner_decision": "continue",
                                "planner_decision_reason": "non-hard blocker stop downgraded to pass_cap_reached",
                                "batch_reason": "planner_terminal_nonhard_exhausted",
                                "expected_progress": "",
                                "requests_count": 0,
                                "request_fingerprints": [],
                                "tool_steps": 0,
                                "warnings_delta": 0,
                            }
                        )
                        break
                terminal_reason = "planner_finalize" if plan.decision == "finalize" else "planner_stop"
                if not latest_answer:
                    latest_answer = str(plan.finalize_action or "").strip()
                all_pass_logs.append(
                    {
                        "pass_index": pass_index,
                        "planner_step_id": plan.current_step_id,
                        "planner_step_title": str(getattr(step, "title", "") or ""),
                        "planner_decision": plan.decision,
                        "planner_decision_reason": plan.decision_reason,
                        "batch_reason": "planner_terminal",
                        "expected_progress": "",
                        "requests_count": 0,
                        "request_fingerprints": [],
                        "tool_steps": 0,
                        "warnings_delta": 0,
                    }
                )
                break

            batch_warning_context = list(all_warnings)
            if recent_failure_summaries:
                batch_warning_context.append(
                    "recent_request_failures: " + " | ".join(recent_failure_summaries[-3:])
                )

            batch, batch_warnings = self._build_batch(
                request=request,
                flow_context=flow_context,
                plan=plan,
                pass_index=pass_index,
                pass_cap=pass_cap,
                pass_logs=all_pass_logs,
                warnings=batch_warning_context,
                changed_files=changed_files,
                latest_answer=latest_answer,
            )
            all_warnings.extend(batch_warnings)

            if batch is None:
                terminal_reason = "invalid_request_batch"
                break

            edit_retry_mode_active = bool(edit_retry_mode_pending)
            if edit_retry_mode_active:
                edit_retry_mode_pending = False

            if plan.decision == "continue" and not batch.requests:
                fallback_request = (
                    self._mutation_retry_request(
                        request=request,
                        flow_context=flow_context,
                        plan=plan,
                        step=step,
                        pass_index=pass_index,
                    )
                    if edit_retry_mode_active
                    else self._deterministic_fallback_request(
                        request=request,
                        flow_context=flow_context,
                        plan=plan,
                        step=step,
                        pass_index=pass_index,
                    )
                )
                if fallback_request is not None:
                    batch = ToolsManagerBatch(
                        planner_step_id=str(batch.planner_step_id or plan.current_step_id or ""),
                        batch_reason=(
                            "edit_retry_mode_forced_mutation"
                            if edit_retry_mode_active
                            else "deterministic_empty_batch_fallback"
                        ),
                        requests=[fallback_request],
                        continue_after=bool(batch.continue_after),
                        expected_progress=(
                            str(batch.expected_progress or "").strip()
                            or (
                                "Execute deterministic mutation retry for current planner step."
                                if edit_retry_mode_active
                                else "Execute deterministic fallback request for current planner step."
                            )
                        ),
                    )
                    all_warnings.append(
                        (
                            f"toolsmanager emitted empty request batch on pass {int(pass_index)}; "
                            "edit_retry_mode active so forcing mutation retry request"
                            if edit_retry_mode_active
                            else f"toolsmanager emitted empty request batch on pass {int(pass_index)}; using deterministic fallback request"
                        )
                    )
                else:
                    all_warnings.append(
                        f"toolsmanager emitted empty request batch on pass {int(pass_index)} and no deterministic fallback could be derived"
                    )

            request_fingerprints: list[str] = []
            tool_steps_this_pass = 0
            warnings_before = len(all_warnings)
            executed_requests = 0
            retries_this_pass = 0
            retries_exhausted_this_pass = 0
            duplicate_skips_this_pass = 0
            batch_requests: list[BatchToolRequest] = []
            request_lookup: dict[int, BatchToolRequest] = {}
            pass_trace_rows: list[dict[str, Any]] = []

            for request_index, item in enumerate(batch.requests):
                merged_policy = self._merge_policy(tool_policy, item.tool_policy_override)
                clipped_timeout = self._clip_timeout(item.timeout_seconds, session_timeout=timeout_seconds)
                question = str(item.question or "").strip()
                if not question:
                    continue

                if edit_retry_mode_active and self._looks_like_search_request_text(question):
                    if not self._looks_like_edit_request_text(question):
                        duplicate_semantic_search_skips += 1
                        duplicate_skips_this_pass += 1
                        all_warnings.append("duplicate_semantic_search_skipped")
                        continue

                semantic_fingerprint = self._semantic_request_fingerprint(question)
                if semantic_fingerprint and semantic_fingerprint in seen_semantic_search_fingerprints:
                    duplicate_semantic_search_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append("duplicate_semantic_search_skipped")
                    continue

                request_fingerprint = self._fingerprint_request(question, merged_policy, clipped_timeout)
                if request_fingerprint in seen_request_fingerprints:
                    duplicate_request_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append("duplicate_request_skipped")
                    continue

                canonical_tool_name = self._canonical_tool_name_from_question(question)
                if canonical_tool_name:
                    if canonical_tool_name in executed_tools_this_turn:
                        duplicate_tool_execution_blocks += 1
                        duplicate_skips_this_pass += 1
                        all_warnings.append(
                            f"duplicate_tool_execution_blocked: tool={canonical_tool_name} turn={execution_run_id}"
                        )
                        continue
                    executed_tools_this_turn.add(canonical_tool_name)
                    all_warnings.append(
                        f"tool_execution_registered: tool={canonical_tool_name} turn={execution_run_id}"
                    )

                seen_request_fingerprints.add(request_fingerprint)
                request_fingerprints.append(request_fingerprint)
                if semantic_fingerprint:
                    seen_semantic_search_fingerprints.add(semantic_fingerprint)

                toolsmanager_requests_count += 1
                req = BatchToolRequest(
                    request_index=request_index,
                    request=ToolRunRequest(
                        question=question,
                        index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
                        index_dirs=[str(Path(p).resolve()) for p in (index_dirs or []) if str(p).strip()] or None,
                        flow_id=flow_key or None,
                        k=int(k),
                        max_steps=int(max_steps),
                        timeout_seconds=clipped_timeout,
                        tool_policy=merged_policy,
                        system_prompt=None,
                    ),
                )
                batch_requests.append(req)
                request_lookup[int(request_index)] = req

            if plan.decision == "continue" and batch.requests and not batch_requests:
                fallback_request = self._mutation_retry_request(
                    request=request,
                    flow_context=flow_context,
                    plan=plan,
                    step=step,
                    pass_index=pass_index,
                )
                merged_policy = self._merge_policy(tool_policy, fallback_request.tool_policy_override)
                clipped_timeout = self._clip_timeout(fallback_request.timeout_seconds, session_timeout=timeout_seconds)
                fallback_fp = self._fingerprint_request(fallback_request.question, merged_policy, clipped_timeout)
                if fallback_fp not in seen_request_fingerprints:
                    seen_request_fingerprints.add(fallback_fp)
                    request_fingerprints.append(fallback_fp)
                    toolsmanager_requests_count += 1
                    batch_requests = [
                        BatchToolRequest(
                            request_index=0,
                            request=ToolRunRequest(
                                question=fallback_request.question,
                                index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
                                index_dirs=[str(Path(p).resolve()) for p in (index_dirs or []) if str(p).strip()] or None,
                                flow_id=flow_key or None,
                                k=int(k),
                                max_steps=int(max_steps),
                                timeout_seconds=clipped_timeout,
                                tool_policy=merged_policy,
                                system_prompt=None,
                            ),
                        )
                    ]
                    request_lookup = {0: batch_requests[0]}
                    batch = ToolsManagerBatch(
                        planner_step_id=str(batch.planner_step_id or plan.current_step_id or ""),
                        batch_reason="deterministic_duplicate_suppression_fallback",
                        requests=[fallback_request],
                        continue_after=bool(batch.continue_after),
                        expected_progress="Execute deterministic fallback after duplicate suppression.",
                    )
                    all_warnings.append(
                        "toolsmanager duplicate suppression removed entire batch; forcing deterministic fallback request"
                    )
                else:
                    all_warnings.append(
                        "toolsmanager duplicate suppression removed entire batch and fallback was also duplicate"
                    )

            executor = getattr(self, "executor", LocalToolsExecutor(worker_client=self.worker_client))
            executor_failed = False
            batch_results: list[BatchExecutionResult] = []
            if batch_requests:
                try:
                    batch_results = executor.run_batch(
                        run_id=execution_run_id,
                        requests=batch_requests,
                        on_event=on_event,
                    )
                except Exception as exc:  # pragma: no cover - executor guardrail
                    batch_results = []
                    executor_failed = True
                    all_warnings.append(f"toolsmanager executor error: job_failed: {exc}")
                    execution_requests_failed += len(batch_requests)

            failed_request_reasons: dict[int, tuple[str, str]] = {}
            if not executor_failed:
                seen_indexes = {int(item.request_index) for item in batch_results}
                for req in batch_requests:
                    if int(req.request_index) not in seen_indexes:
                        execution_requests_failed += 1
                        msg = (
                            "result_decode_failed: missing result for request index "
                            f"{int(req.request_index)}"
                        )
                        all_warnings.append(f"toolsmanager executor error: {msg}")
                        failed_request_reasons[int(req.request_index)] = ("result_decode_failed", msg)

            for item in sorted(batch_results, key=lambda row: int(row.request_index)):
                idx = int(item.request_index)
                if idx not in request_lookup:
                    execution_requests_failed += 1
                    all_warnings.append(
                        f"toolsmanager executor error: result_decode_failed: unexpected request index {idx}"
                    )
                    continue
                if not item.ok:
                    execution_requests_failed += 1
                    code = str(item.error_code or "job_failed")
                    detail = str(item.error_message or "request failed")
                    all_warnings.append(f"toolsmanager executor error: {code}: {detail}")
                    failed_request_reasons[idx] = (code, detail)
                    continue
                if not isinstance(item.response, dict):
                    execution_requests_failed += 1
                    msg = "result_decode_failed: response payload missing"
                    all_warnings.append(f"toolsmanager executor error: {msg}")
                    failed_request_reasons[idx] = ("result_decode_failed", msg)
                    continue
                try:
                    response = ToolRunResponse.model_validate(item.response)
                except Exception as exc:
                    execution_requests_failed += 1
                    msg = f"result_decode_failed: {exc}"
                    all_warnings.append(f"toolsmanager executor error: {msg}")
                    failed_request_reasons[idx] = ("result_decode_failed", msg)
                    continue

                failed_request_reasons.pop(idx, None)
                execution_requests_ok += 1
                executed_requests += 1
                if response.answer:
                    latest_answer = str(response.answer)
                if isinstance(response.warnings, list):
                    for warning in response.warnings:
                        text = str(warning).strip()
                        if text:
                            all_warnings.append(text)
                if isinstance(response.trace, list):
                    rows = [row for row in response.trace if isinstance(row, dict)]
                    all_trace.extend(rows)
                    pass_trace_rows.extend(rows)
                    tool_steps_this_pass += len(rows)
                if isinstance(response.sources, list):
                    all_sources.extend([row for row in response.sources if isinstance(row, dict)])

            if failed_request_reasons:
                retry_requests = [
                    request_lookup[idx]
                    for idx in sorted(failed_request_reasons)
                    if idx in request_lookup
                ]
                if retry_requests:
                    retry_lookup = {int(req.request_index): req for req in retry_requests}
                    retries_this_pass = len(retry_requests)
                    request_retry_attempts += retries_this_pass
                    all_warnings.append(
                        f"toolsmanager_request_retry_once; retrying {len(retry_requests)} failed request(s)"
                    )
                    retry_results: list[BatchExecutionResult] = []
                    retry_executor_failed = False
                    try:
                        retry_results = executor.run_batch(
                            run_id=execution_run_id,
                            requests=retry_requests,
                            on_event=on_event,
                        )
                    except Exception as exc:  # pragma: no cover - executor guardrail
                        retry_results = []
                        retry_executor_failed = True
                        all_warnings.append(f"toolsmanager executor retry error: job_failed: {exc}")
                        execution_requests_failed += len(retry_requests)

                    retry_failures = dict(failed_request_reasons)
                    if not retry_executor_failed:
                        seen_retry_indexes = {int(item.request_index) for item in retry_results}
                        for req in retry_requests:
                            idx = int(req.request_index)
                            if idx not in seen_retry_indexes:
                                execution_requests_failed += 1
                                msg = f"result_decode_failed: missing retry result for request index {idx}"
                                all_warnings.append(f"toolsmanager executor retry error: {msg}")
                                retry_failures[idx] = ("result_decode_failed", msg)

                        for item in sorted(retry_results, key=lambda row: int(row.request_index)):
                            idx = int(item.request_index)
                            if idx not in retry_lookup:
                                execution_requests_failed += 1
                                msg = f"result_decode_failed: unexpected retry result index {idx}"
                                all_warnings.append(f"toolsmanager executor retry error: {msg}")
                                continue
                            if not item.ok:
                                execution_requests_failed += 1
                                code = str(item.error_code or "job_failed")
                                detail = str(item.error_message or "request failed")
                                all_warnings.append(f"toolsmanager executor retry error: {code}: {detail}")
                                retry_failures[idx] = (code, detail)
                                continue
                            if not isinstance(item.response, dict):
                                execution_requests_failed += 1
                                msg = "result_decode_failed: retry response payload missing"
                                all_warnings.append(f"toolsmanager executor retry error: {msg}")
                                retry_failures[idx] = ("result_decode_failed", msg)
                                continue
                            try:
                                response = ToolRunResponse.model_validate(item.response)
                            except Exception as exc:
                                execution_requests_failed += 1
                                msg = f"result_decode_failed: {exc}"
                                all_warnings.append(f"toolsmanager executor retry error: {msg}")
                                retry_failures[idx] = ("result_decode_failed", msg)
                                continue

                            retry_failures.pop(idx, None)
                            execution_requests_ok += 1
                            executed_requests += 1
                            if response.answer:
                                latest_answer = str(response.answer)
                            if isinstance(response.warnings, list):
                                for warning in response.warnings:
                                    text = str(warning).strip()
                                    if text:
                                        all_warnings.append(text)
                            if isinstance(response.trace, list):
                                rows = [row for row in response.trace if isinstance(row, dict)]
                                all_trace.extend(rows)
                                pass_trace_rows.extend(rows)
                                tool_steps_this_pass += len(rows)
                            if isinstance(response.sources, list):
                                all_sources.extend([row for row in response.sources if isinstance(row, dict)])

                    if retry_failures:
                        retries_exhausted_this_pass = len(retry_failures)
                        request_retry_exhausted += retries_exhausted_this_pass
                        for idx, (code, detail) in retry_failures.items():
                            req = request_lookup.get(int(idx))
                            question = str(req.request.question if req is not None else "")
                            signature = self._build_failure_signature(
                                question=question,
                                code=code,
                                detail=detail,
                            )
                            if signature not in seen_failure_signatures:
                                seen_failure_signatures.add(signature)
                                summary = f"{code}: {self._truncate_line(detail, limit=140)}"
                                recent_failure_summaries.append(summary)
                                all_warnings.append(f"toolsmanager_retry_exhausted_signature={signature}:{summary}")
                            if memory_service is not None and flow_key:
                                try:
                                    memory_service.record_tool_fingerprint(
                                        flow_id=flow_key,
                                        kind="mutation_failure_signature",
                                        fingerprint=signature,
                                    )
                                except Exception:
                                    pass

            changed_now = sorted(self._git_status_paths().difference(before))
            changed_files = changed_now

            semantic_trace_fps = self._semantic_trace_fingerprints(pass_trace_rows)
            if semantic_trace_fps:
                seen_semantic_search_fingerprints.update(semantic_trace_fps)

            if memory_service is not None and flow_key:
                try:
                    if request_fingerprints:
                        memory_service.record_tool_fingerprints(
                            flow_id=flow_key,
                            kind="request_fingerprint",
                            fingerprints=request_fingerprints,
                        )
                    if semantic_trace_fps:
                        memory_service.record_tool_fingerprints(
                            flow_id=flow_key,
                            kind="semantic_search_fingerprint",
                            fingerprints=sorted(semantic_trace_fps),
                        )
                    step_fingerprint = self._planner_task_fingerprint(step)
                    if step_fingerprint and (executed_requests > 0 or tool_steps_this_pass > 0):
                        seen_planner_task_fingerprints.add(step_fingerprint)
                        memory_service.record_tool_fingerprint(
                            flow_id=flow_key,
                            kind="planner_task_fingerprint",
                            fingerprint=step_fingerprint,
                        )
                    memory_service.prune_tool_fingerprints(
                        flow_id=flow_key,
                        kind="request_fingerprint",
                    )
                    memory_service.prune_tool_fingerprints(
                        flow_id=flow_key,
                        kind="semantic_search_fingerprint",
                    )
                    memory_service.prune_tool_fingerprints(
                        flow_id=flow_key,
                        kind="mutation_failure_signature",
                    )
                    memory_service.prune_tool_fingerprints(
                        flow_id=flow_key,
                        kind="planner_task_fingerprint",
                    )
                except Exception as exc:
                    all_warnings.append(f"toolsmanager persistent fingerprint write failed: {exc}")

            apply_patch_attempted = any(
                str(row.get("tool_name", "")).strip().lower() == "apply_patch"
                for row in pass_trace_rows
            )
            apply_patch_failed = any(self._looks_like_apply_patch_failure_trace(row) for row in pass_trace_rows)
            if changed_now:
                edit_retry_mode_pending = False
            elif apply_patch_attempted and (apply_patch_failed or not changed_now):
                edit_retry_mode_pending = True
                edit_retry_mode_activations += 1
                all_warnings.append("edit_retry_mode_activated")

            warnings_delta = max(0, len(all_warnings) - warnings_before)
            all_pass_logs.append(
                {
                    "pass_index": pass_index,
                    "planner_step_id": plan.current_step_id,
                    "planner_step_title": str(getattr(step, "title", "") or ""),
                    "planner_decision": plan.decision,
                    "planner_decision_reason": plan.decision_reason,
                    "batch_reason": str(batch.batch_reason or ""),
                    "expected_progress": str(batch.expected_progress or ""),
                    "requests_count": len(batch.requests),
                    "request_fingerprints": request_fingerprints,
                    "tool_steps": tool_steps_this_pass,
                    "warnings_delta": warnings_delta,
                    "continue_after": bool(batch.continue_after),
                    "execution_backend": execution_backend,
                    "duplicate_skips": duplicate_skips_this_pass,
                    "request_retry_attempts": retries_this_pass,
                    "request_retry_exhausted": retries_exhausted_this_pass,
                    "edit_retry_mode_active": bool(edit_retry_mode_active),
                }
            )

            if executed_requests == 0:
                stalled_passes += 1
            else:
                stalled_passes = 0

            if stalled_passes >= 2:
                terminal_reason = "stalled_no_actionable_requests"
                break

            if pass_index >= pass_cap:
                terminal_reason = "pass_cap_reached"
                break

            plan_warning_context = list(all_warnings)
            if recent_failure_summaries:
                plan_warning_context.append(
                    "recent_request_failures: " + " | ".join(recent_failure_summaries[-3:])
                )

            plan, new_plan_warnings, _source = self._plan_with_source(
                request=request,
                flow_context=flow_context,
                pass_index=pass_index,
                pass_cap=pass_cap,
                previous_plan=plan,
                pass_logs=all_pass_logs,
                warnings=plan_warning_context,
                changed_files=changed_files,
                latest_answer=latest_answer,
            )
            all_warnings.extend(new_plan_warnings)

        if (
            terminal_reason == "pass_cap_reached"
            and is_edit_task
            and not changed_files
        ):
            latest_answer = self._synthesize_terminal_answer(
                terminal_reason=terminal_reason,
                pass_logs=all_pass_logs,
                planner_decisions=planner_decisions,
                toolsmanager_requests_count=toolsmanager_requests_count,
            )
            all_warnings.append("edit_task_pass_cap_without_changed_files")

        if not str(latest_answer or "").strip():
            latest_answer = self._synthesize_terminal_answer(
                terminal_reason=terminal_reason,
                pass_logs=all_pass_logs,
                planner_decisions=planner_decisions,
                toolsmanager_requests_count=toolsmanager_requests_count,
            )

        persisted_fingerprint_counts = {
            "request_fingerprint": len(seen_request_fingerprints),
            "semantic_search_fingerprint": len(seen_semantic_search_fingerprints),
            "mutation_failure_signature": len(seen_failure_signatures),
            "planner_task_fingerprint": len(seen_planner_task_fingerprints),
        }

        return AutoExecuteResult(
            answer=latest_answer,
            sources=all_sources,
            trace=all_trace,
            warnings=all_warnings,
            changed_files=changed_files,
            plan=plan.model_dump(),
            passes=len(all_pass_logs),
            terminal_reason=terminal_reason,
            toolsmanager_requests_count=toolsmanager_requests_count,
            pass_logs=all_pass_logs,
            planner_decisions=planner_decisions,
            execution_backend=execution_backend,
            execution_run_id=execution_run_id,
            execution_duration_ms=round((time.perf_counter() - execution_started) * 1000.0, 3),
            execution_requests_ok=execution_requests_ok,
            execution_requests_failed=execution_requests_failed,
            duplicate_request_skips=duplicate_request_skips,
            duplicate_semantic_search_skips=duplicate_semantic_search_skips,
            duplicate_tool_execution_blocks=duplicate_tool_execution_blocks,
            request_retry_attempts=request_retry_attempts,
            request_retry_exhausted=request_retry_exhausted,
            edit_retry_mode_activations=edit_retry_mode_activations,
            persisted_fingerprint_counts=persisted_fingerprint_counts,
        )
        
    @staticmethod
    def _canonical_tool_name_from_question(question: str) -> str:
        normalized = re.sub(r"\s+", " ", str(question or "").strip()).lower()
        if not normalized:
            return ""
        match = re.search(r"\btool\s*[:=]\s*['\"]?([a-z0-9_\-]+)", normalized)
        if match:
            return str(match.group(1) or "").strip().lower()
        known = (
            "semantic_search",
            "repo_search",
            "read_file",
            "find_symbols",
            "call_graph",
            "run_command",
            "apply_patch",
            "write_file",
            "search_internet",
            "github_search",
        )
        for tool_name in known:
            if tool_name in normalized:
                return tool_name
        return ""


__all__ = [
    "ToolsPlan",
    "ToolsPlanStep",
    "ToolsManagerRequest",
    "ToolsManagerBatch",
    "AutoExecuteResult",
    "ToolsManagerOrchestrator",
]
