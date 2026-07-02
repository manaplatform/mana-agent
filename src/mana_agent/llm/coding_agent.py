"""
mana_agent.llm.coding_agent

Coding agent wrapper with:
- mutation tools (edit_file/multi_edit_file/apply_patch/create_file/write_file/delete_file)
- structured flow/checklist planning
- anti-loop tool policy (search/read budgets, duplicate search guards)
- flow-memory continuity integration
"""

from __future__ import annotations
from tenacity import retry, stop_after_attempt, wait_exponential
import ast
import json
import logging
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from mana_agent.llm.prompts import (
    CODING_AGENT_RECOGNITION_PROMPT,
    CODING_AGENT_LANGUAGE_TOOLING_PROMPT,
    HEAD_TOOLS_PLANNER_PROMPT,
    CODING_FLOW_MEMORY_PROMPT,
    CODING_FLOW_PLANNER_PROMPT,
    FULL_AUTO_EXECUTION_PROMPT,
    TOOLSMANAGER_PROMPT
)
from mana_agent.llm.agent_work_queue import QueueManager
from mana_agent.llm.coding_agent_models import (
    AskAgentLike,
    DynamicReadPolicy,
    ExecutionDecision,
    FlowChecklist,
    FlowStep,
    as_jsonable,
)
from mana_agent.llm.auto_chat import apply_auto_chat_tool_policy
from mana_agent.llm.coding_agent_prompt import CODING_SYSTEM_PROMPT


from mana_agent.llm.tool_worker_process import (
    ToolRunRequest,
    ToolWorkerClient,
    ToolWorkerProcessError,
)

from mana_agent.services.coding_memory_service import CodingMemoryService
from mana_agent.tools import (
    build_apply_patch_tool,
    build_create_file_tool,
    build_delete_file_tool,
    build_edit_file_tool,
    build_multi_edit_file_tool,
    build_write_file_tool,
)

logger = logging.getLogger(__name__)
_as_jsonable = as_jsonable


class CodingAgent:
    _EDIT_INTENT_TOKENS = (
        "fix",
        "bug",
        "issue",
        "patch",
        "edit",
        "update",
        "modify",
        "change",
        "implement",
        "add",
        "remove",
        "delete",
        "create",
        "write",
        "refactor",
        "rename",
        "cleanup",
    )
    _EXPLICIT_FILE_RE = re.compile(r"(?i)\b([A-Za-z0-9_\-./]+\.[A-Za-z0-9_]+)\b")
    _EXPLICIT_DOTFILE_RE = re.compile(
        r"(?i)(?<![\w/.-])"
        r"(\.(?:gitignore|dockerignore|env|env\.[A-Za-z0-9_-]+|npmrc|yarnrc|prettierrc|eslintrc|flake8))"
        r"(?![\w/.-])"
    )
    _AMBIGUOUS_FOLLOWUP_RE = re.compile(
        r"(?i)^\s*(yes|yep|ok|okay|sure|continue|go|proceed|begin|start|do it|done|next)\s*[/!.]?\s*$"
    )
    _PLAN_TRIGGER_RE = re.compile(
        r"(?i)^\s*(?:please\s+)?(?:implement|execute|run|apply|trigger)\s+"
        r"(?:the\s+|last\s+|that\s+|current\s+)?plan\s*[/!.]?\s*$"
    )

    _DEFAULT_READ_LINE_WINDOW = 400
    _MIN_READ_LINE_WINDOW = 200
    _MAX_READ_LINE_WINDOW = 2000
    _MAX_DYNAMIC_READ_BUDGET = 60

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
    def _checklist_from_plan_text(text: str, request: str = "") -> FlowChecklist | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if "\\n" in raw and "\n" not in raw:
            raw = raw.replace("\\n", "\n")
        steps: list[str] = []
        objective = ""
        for line in raw.splitlines():
            item = line.strip()
            if not item:
                continue
            lowered = item.lower()
            if lowered.startswith("objective:"):
                objective = item.split(":", 1)[1].strip()
                continue
            if lowered in {"plan:", "execution plan:", "**execution plan**"}:
                continue
            match = re.match(r"^(?:\d+[.)]\s+|[-*]\s+)(.+)$", item)
            if match:
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(step_text[:220])
        if not steps:
            return None
        resolved_objective = objective or (" ".join((request or "").strip().split())[:220] or "Implement requested change")
        flow_steps: list[FlowStep] = []
        for idx, title in enumerate(steps[:20], start=1):
            flow_steps.append(
                FlowStep(
                    id=f"s{idx}",
                    title=title,
                    reason="Derived from planner text output",
                    status="in_progress" if idx == 1 else "pending",
                    requires_tools=[],
                )
            )
        return FlowChecklist(
            objective=resolved_objective,
            requires_edit=False,
            target_files=[],
            constraints=[],
            acceptance=["Requested change is applied"],
            steps=flow_steps,
            next_action=flow_steps[0].title if flow_steps else "",
        )

    @classmethod
    def _coerce_checklist_from_obj(cls, parsed: Any, request: str = "") -> FlowChecklist | None:
        if isinstance(parsed, dict):
            if "objective" in parsed and "steps" in parsed:
                return FlowChecklist.model_validate(parsed)
            text_payload = parsed.get("text")
            if isinstance(text_payload, str):
                return cls._checklist_from_plan_text(text_payload, request=request)
            return None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "objective" in item and "steps" in item:
                    return FlowChecklist.model_validate(item)
            text_chunks: list[str] = []
            for item in parsed:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        text_chunks.append(text_value.strip())
            if text_chunks:
                return cls._checklist_from_plan_text("\n".join(text_chunks), request=request)
        return None

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        raw = str(text or "").strip()
        if not raw.startswith("```"):
            return raw
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return raw

    @classmethod
    def _collect_checklist_candidates(cls, raw_text: str) -> list[Any]:
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

    @staticmethod
    def _extract_json_object_text(text: str) -> str | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        raw = CodingAgent._strip_code_fence(raw)
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
                    candidate = raw[start : idx + 1].strip()
                    return candidate if candidate else None
        return None

    @classmethod
    def _parse_flow_checklist_json(cls, text: str, request: str = "") -> FlowChecklist:
        raw = str(text or "").strip()
        for candidate in cls._collect_checklist_candidates(raw):
            if isinstance(candidate, (dict, list)):
                try:
                    checklist = cls._coerce_checklist_from_obj(candidate, request=request)
                except (ValidationError, TypeError, ValueError):
                    checklist = None
                if checklist is not None:
                    return checklist
                continue

            if not isinstance(candidate, str):
                continue

            parsed_candidate = cls._parse_json_or_literal(candidate)
            if parsed_candidate is not None:
                try:
                    checklist = cls._coerce_checklist_from_obj(parsed_candidate, request=request)
                except (ValidationError, TypeError, ValueError):
                    checklist = None
                if checklist is not None:
                    return checklist

            text_checklist = cls._checklist_from_plan_text(candidate, request=request)
            if text_checklist is not None:
                return text_checklist

        raise json.JSONDecodeError("No checklist payload found", raw, 0)

    def _fallback_checklist(self, request: str) -> FlowChecklist:
        explicit_files = sorted(
            {
                match.group(1).strip()
                for match in self._EXPLICIT_FILE_RE.finditer(request or "")
                if match.group(1).strip()
            }
        )
        objective = " ".join((request or "").strip().split())[:220] or "Implement requested change"
        inspect_title = "Inspect target file(s)" if explicit_files else "Discover target file(s)"
        inspect_reason = (
            f"Validate current behavior in: {', '.join(explicit_files[:3])}"
            if explicit_files
            else "Collect concrete evidence before edits"
        )
        return FlowChecklist(
            objective=objective,
            requires_edit=True,
            target_files=explicit_files,
            acceptance=["Requested change is applied", "No obvious regressions in touched files"],
            steps=[
                FlowStep(
                    id="s1",
                    title=inspect_title,
                    reason=inspect_reason,
                    status="in_progress",
                    requires_tools=["semantic_search", "read_file"],
                ),
                FlowStep(
                    id="s2",
                    title="Apply requested change",
                    reason="Implement the user request in repository files",
                    status="pending",
                    requires_tools=["apply_patch", "create_file", "write_file", "delete_file"],
                ),
                FlowStep(
                    id="s3",
                    title="Verify and summarize",
                    reason="Confirm edits and report outcomes",
                    status="pending",
                    requires_tools=["run_command", "read_file"],
                ),
            ],
            next_action="Inspect file context, then apply the requested edit.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        repo_root: Path,
        ask_agent: AskAgentLike,
        base_url: str | None = None,
        allowed_prefixes: Optional[Sequence[str]] = ("src/", "tests/", ""),
        system_prompt: str = CODING_SYSTEM_PROMPT,
        coding_memory_service: CodingMemoryService | None = None,
        coding_memory_enabled: bool = True,
        plan_max_steps: int = 8,
        search_budget: int = 4,
        read_budget: int = 6,
        require_read_files: int = 2,
        tool_worker_client: ToolWorkerClient | None = None,
        full_auto_mode: bool = False,
        planner_model: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = str(base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
        self.repo_root = repo_root.resolve()
        self.ask_agent: AskAgentLike = ask_agent
        self.allowed_prefixes = allowed_prefixes
        self.system_prompt = system_prompt
        self.coding_memory_service = coding_memory_service
        self.coding_memory_enabled = bool(coding_memory_enabled)
        self._current_flow_id: str | None = None

        self.plan_max_steps = max(1, int(plan_max_steps))
        self.search_budget = max(1, int(search_budget))
        self.read_budget = max(1, int(read_budget))
        self.require_read_files = max(1, int(require_read_files))
        self.tool_worker_client = tool_worker_client
        self.full_auto_mode = bool(full_auto_mode)
        self.planner_model = str(planner_model).strip() if planner_model else None
        self.tools_manager_orchestrator: QueueManager | None = None
        self._setup_planner()

        if hasattr(self.ask_agent, "tools"):
            self.ask_agent.tools.extend(
                [
                    build_edit_file_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
                    build_multi_edit_file_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
                    build_apply_patch_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
                    build_write_file_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
                    build_create_file_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
                    build_delete_file_tool(repo_root=self.repo_root, allowed_prefixes=self.allowed_prefixes),
                ]
            )

    def update_model(self, model_name: str):
        """قابلیت تغییر مدل در زمان اجرا"""
        if hasattr(self, "ask_agent"):
            self.ask_agent.model = model_name
        
        # بازنشانی Planner با مدل جدید
        self._setup_planner()
        logger.info(f"Planner model updated to: {model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _setup_planner(self):
        """تنظیم و مقداردهی اولیه LLM با قابلیت Retry"""
        try:
            env_planner_model = str(
                os.getenv("OPENAI_CODING_PLANNER_MODEL")
                or os.getenv("CODING_AGENT_PLANNER_MODEL")
                or ""
            ).strip()
            current_model = (
                env_planner_model
                or str(self.planner_model or "").strip()
                or str(getattr(self.ask_agent, "model", "")).strip()
                or str(os.getenv("OPENAI_CHAT_MODEL") or "").strip()
                or "gpt-4.1-mini"
            )
            
            planner_kwargs = {
                "api_key": self.api_key, # فرض بر این است که self.api_key در کلاس موجود است
                "model": current_model,
            }
            
            if hasattr(self, "base_url") and self.base_url:
                planner_kwargs["base_url"] = self.base_url
                
            self.planner_llm = ChatOpenAI(**planner_kwargs)
            
            # یک تست کوچک برای اطمینان از صحت مدل (اختیاری)
            # self.planner_llm.predict("health check") 
            
        except Exception as e:
            logger.error(f"Failed to initialize planner with model {current_model}: {e}")
            # اگر خطا مربوط به پیدا نشدن مدل بود، اینجا می‌توان منطق تغییر مدل خودکار را هم اضافه کرد
            raise e

    def initialize_components(self):
        """متدی که کد اصلی شما را اجرا می‌کند"""
        # فراخوانی متد با قابلیت Retry
        self._setup_planner()

        # افزودن ابزارها به ask_agent
        if hasattr(self.ask_agent, "tools"):
            self.ask_agent.tools.extend(
                [
                    build_edit_file_tool(
                        repo_root=self.repo_root,
                        allowed_prefixes=self.allowed_prefixes
                    ),
                    build_multi_edit_file_tool(
                        repo_root=self.repo_root,
                        allowed_prefixes=self.allowed_prefixes
                    ),
                    build_apply_patch_tool(
                        repo_root=self.repo_root,
                        allowed_prefixes=self.allowed_prefixes
                    ),
                    build_write_file_tool(
                        repo_root=self.repo_root,
                        allowed_prefixes=self.allowed_prefixes
                    ),
                    build_create_file_tool(
                        repo_root=self.repo_root,
                        allowed_prefixes=self.allowed_prefixes
                    ),
                    build_delete_file_tool(
                        repo_root=self.repo_root,
                        allowed_prefixes=self.allowed_prefixes
                    ),
                ]
            )
        
    def set_tools_manager_orchestrator(self, manager: QueueManager):
        """
        Bind the queue manager as this agent's execution backend.

        The queue manager is deterministic and drives the AgentWorkQueue
        directly, so no LLM decision provider is attached.
        """
        self.tools_manager_orchestrator = manager

    def run_via_work_queue(
        self,
        request: str,
        *,
        seeds: "Sequence[Any] | None" = None,
        tool_policy: dict[str, Any] | None = None,
        index_dir: str | None = None,
        flow_id: str | None = None,
        max_steps: int = 60,
        max_reads: int = 40,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        """Drive a request through the Live Agent Work Queue.

        The coding agent sits at the top of the hierarchy here: it seeds the
        queue, the executor claims runnable jobs and runs them through the tool
        worker, the EventBus broadcasts every transition to the TaskBoard, and
        the agent's sniffer live-analyzes each result and emits follow-up jobs.

        Returns a dict with the run report, the rendered board, and the final
        job list. Falls back gracefully if no tool worker is attached.
        """
        from mana_agent.llm.agent_work_queue import (
            AgentWorkQueue,
            TaskBoard,
            WorkItem,
            WorkQueueRunner,
        )
        from mana_agent.llm.agent_work_queue_adapters import (
            CodingAgentSniffer,
            make_worker_executor,
        )
        from mana_agent.llm.goal_profiles import active_goal_profile

        if self.tool_worker_client is None:
            return {
                "ok": False,
                "error": "no_tool_worker_client",
                "report": None,
                "board": "",
            }

        queue = AgentWorkQueue()
        board = TaskBoard(queue=queue)

        profile = active_goal_profile(request)
        if profile is not None:
            def _relevant(path: str) -> bool:
                return profile.is_relevant(path, self.repo_root)
        else:
            def _relevant(path: str) -> bool:
                return True

        seed_items: list[WorkItem] = []
        if seeds:
            for seed in seeds:
                seed_items.append(seed if isinstance(seed, WorkItem) else WorkItem(**dict(seed)))
        else:
            seed_items.append(
                WorkItem(
                    kind="discover",
                    tool_name="repo_search",
                    tool_args={"query": request},
                    question=f"Locate files relevant to: {request}",
                    gate="locate_candidates",
                    priority=10,
                )
            )
        queue.submit_many(seed_items)

        execute = make_worker_executor(
            worker_client=self.tool_worker_client,
            repo_root=self.repo_root,
            on_event=self._log_worker_event,
            default_timeout=int(timeout_seconds),
            tool_policy=tool_policy,
            index_dir=index_dir,
            flow_id=flow_id,
        )
        planned_checklist, _plan_warnings = self._plan_checklist(request, flow_context=flow_id)
        sniffer = CodingAgentSniffer(
            repo_root=self.repo_root,
            request=request,
            emit_edit=self._checklist_requires_edit(planned_checklist),
            target_files=(planned_checklist.target_files if planned_checklist is not None else []),
            max_reads=int(max_reads),
            relevant=_relevant,
        )
        runner = WorkQueueRunner(
            queue=queue,
            execute=execute,
            sniffer=sniffer,
            board=board,
            max_steps=int(max_steps),
        )
        report = runner.run()
        return {
            "ok": report.failed == 0 and report.blocked == 0,
            "report": report.model_dump(),
            "board": board.render(),
            "snapshot": queue.snapshot(),
        }

    @staticmethod
    def _normalize_prechecklist(checklist: FlowChecklist, *, source: str) -> dict[str, Any]:
        steps: list[dict[str, str]] = []
        for item in checklist.steps[:20]:
            steps.append(
                {
                    "id": str(item.id or "").strip() or "step",
                    "title": str(item.title or "").strip() or "step",
                    "status": str(item.status or "pending"),
                }
            )
        return {
            "objective": str(checklist.objective or "").strip(),
            "requires_edit": bool(checklist.requires_edit),
            "target_files": [str(item).strip() for item in checklist.target_files if str(item).strip()],
            "steps": steps,
            "source": str(source or ""),
        }

    def preview_execution_checklist(
        self,
        request: str,
        *,
        flow_id: str | None = None,
        flow_context: str | None = None,
    ) -> dict[str, Any]:
        """Build and persist a pre-execution checklist preview for UI rendering."""
        before = self._git_status_paths()
        warnings: list[str] = []
        active_flow_id = flow_id
        effective_flow_context = flow_context

        if self.coding_memory_enabled and self.coding_memory_service is not None:
            try:
                active_flow_id = self.coding_memory_service.ensure_flow(flow_id=flow_id, request=request)
                self._current_flow_id = active_flow_id
                if effective_flow_context is None:
                    effective_flow_context = self.coding_memory_service.build_flow_context(
                        active_flow_id,
                        sorted(before),
                    )
            except Exception as exc:
                warnings.append(f"coding memory setup failed: {exc}")

        checklist, plan_warnings, source = self._plan_checklist_with_source(
            request,
            flow_context=effective_flow_context,
        )
        warnings.extend(plan_warnings)
        if checklist is None:
            return {
                "flow_id": active_flow_id,
                "flow_context": effective_flow_context,
                "prechecklist": {
                    "objective": "",
                    "requires_edit": True,
                    "target_files": [],
                    "steps": [],
                    "source": "deterministic_fallback",
                },
                "prechecklist_source": "deterministic_fallback",
                "prechecklist_warning": "Planner failed to produce a preview checklist.",
                "requires_edit": True,
                "target_files": [],
                "warnings": warnings,
            }

        prechecklist = self._normalize_prechecklist(checklist, source=source)
        prechecklist_warning = ""
        if source == "deterministic_fallback":
            prechecklist_warning = "Planner parse failed; using deterministic fallback checklist."

        if (
            self.coding_memory_enabled
            and self.coding_memory_service is not None
            and active_flow_id is not None
        ):
            try:
                self.coding_memory_service.persist_preview_checklist(
                    flow_id=active_flow_id,
                    user_request=request,
                    checklist=checklist.model_dump(),
                    source=source,
                    warning=prechecklist_warning,
                )
            except Exception as exc:
                warnings.append(f"coding memory preview persistence failed: {exc}")

        return {
            "flow_id": active_flow_id,
            "flow_context": effective_flow_context,
            "prechecklist": prechecklist,
            "prechecklist_source": source,
            "prechecklist_warning": prechecklist_warning,
            "requires_edit": self._checklist_requires_edit(checklist),
            "target_files": [str(item).strip() for item in checklist.target_files if str(item).strip()],
            "warnings": warnings,
        }

    def generate_auto_execute(
        self,
        request: str,
        *,
        index_dir: str | Path | None = None,
        index_dirs: Sequence[str | Path] | None = None,
        k: int | None = None,
        max_steps: int = 200,
        timeout_seconds: int = 600,
        pass_cap: int = 4,
        flow_context: str | None = None,
        flow_id: str | None = None,
        run_id: str | None = None,
        callbacks: Sequence[Any] | None = None,
        prechecklist_payload: dict[str, Any] | None = None,
        auto_chat_mode: str | None = None,
    ) -> dict[str, Any]:
        preview_payload = prechecklist_payload or self.preview_execution_checklist(
            request,
            flow_id=flow_id,
            flow_context=flow_context,
        )
        prechecklist = preview_payload.get("prechecklist") if isinstance(preview_payload.get("prechecklist"), dict) else None
        prechecklist_source = str(preview_payload.get("prechecklist_source", "") or "")
        prechecklist_warning = str(preview_payload.get("prechecklist_warning", "") or "")

        if self.tools_manager_orchestrator is None:
            return {
                "status": "warning",
                "answer": "Auto-execute orchestrator is unavailable for this session.",
                "changed_files": [],
                "diff": "",
                "warnings": ["auto_execute_orchestrator_unavailable"],
                "flow_id": flow_id,
                "plan": None,
                "progress": {"phase": "blocked", "why": "auto_execute_orchestrator_unavailable"},
                "checklist": {"done": 0, "pending": 0, "blocked": 1, "total": 0},
                "actions_taken_total": 0,
                "actions_taken_truncated": False,
                "actions_taken": [],
                "next_step": "Initialize tool worker and tools manager orchestrator, then retry.",
                "static_analysis": {"finding_count": 0, "findings": []},
                "auto_execute_passes": 0,
                "auto_execute_terminal_reason": "orchestrator_unavailable",
                "toolsmanager_requests_count": 0,
                "pass_logs": [],
                "planner_decisions": [],
                "tool_execution_backend": "",
                "tool_execution_run_id": "",
                "tool_execution_duration_ms": 0.0,
                "tool_execution_requests_ok": 0,
                "tool_execution_requests_failed": 0,
                "prechecklist": prechecklist,
                "prechecklist_source": prechecklist_source,
                "prechecklist_warning": prechecklist_warning,
            }

        before = self._git_status_paths()
        warnings: list[str] = []
        preview_warnings = preview_payload.get("warnings") if isinstance(preview_payload.get("warnings"), list) else []
        warnings.extend(str(item).strip() for item in preview_warnings if str(item).strip())
        active_flow_id = flow_id
        effective_flow_context = flow_context
        if isinstance(preview_payload.get("flow_id"), str) and str(preview_payload.get("flow_id")).strip():
            active_flow_id = str(preview_payload.get("flow_id")).strip()
        if effective_flow_context is None and isinstance(preview_payload.get("flow_context"), str):
            flow_ctx = str(preview_payload.get("flow_context") or "").strip()
            effective_flow_context = flow_ctx or None
        if self.coding_memory_enabled and self.coding_memory_service is not None:
            try:
                active_flow_id = self.coding_memory_service.ensure_flow(flow_id=flow_id, request=request)
                self._current_flow_id = active_flow_id
                if effective_flow_context is None:
                    effective_flow_context = self.coding_memory_service.build_flow_context(
                        active_flow_id,
                        sorted(before),
                    )
            except Exception as exc:
                warnings.append(f"coding memory setup failed: {exc}")

        requires_edit = bool(preview_payload.get("requires_edit", True))
        target_files = [
            str(item).strip()
            for item in (preview_payload.get("target_files") if isinstance(preview_payload.get("target_files"), list) else [])
            if str(item).strip()
        ]
        tool_policy = self._tool_policy_for_request(
            request,
            flow_context=effective_flow_context,
            auto_chat_mode=auto_chat_mode,
        )
        try:
            _ = callbacks
            orchestrated = self.tools_manager_orchestrator.run(
                request=request,
                flow_context=effective_flow_context,
                index_dir=index_dir,
                index_dirs=index_dirs,
                k=int(k if k is not None else 8),
                max_steps=int(max_steps),
                timeout_seconds=int(timeout_seconds),
                tool_policy=tool_policy,
                pass_cap=max(1, int(pass_cap)),
                requires_edit=requires_edit,
                target_files=target_files,
                on_event=self._log_worker_event,
                flow_id=active_flow_id,
                run_id=run_id,
            )
        except ToolWorkerProcessError as exc:
            warnings.append(f"toolsmanager worker failure: {exc.code}: {exc}")
            return {
                "status": "warning",
                "answer": "Auto-execute failed due to worker process error.",
                "changed_files": [],
                "diff": "",
                "warnings": warnings,
                "flow_id": active_flow_id,
                "plan": None,
                "progress": {"phase": "blocked", "why": "tool_worker_error"},
                "checklist": {"done": 0, "pending": 0, "blocked": 1, "total": 0},
                "actions_taken_total": 0,
                "actions_taken_truncated": False,
                "actions_taken": [],
                "next_step": "Retry with a more specific tool-executable request.",
                "static_analysis": {"finding_count": 0, "findings": []},
                "auto_execute_passes": 0,
                "auto_execute_terminal_reason": "tool_worker_error",
                "toolsmanager_requests_count": 0,
                "pass_logs": [],
                "planner_decisions": [],
                "tool_execution_backend": "",
                "tool_execution_run_id": "",
                "tool_execution_duration_ms": 0.0,
                "tool_execution_requests_ok": 0,
                "tool_execution_requests_failed": 0,
                "prechecklist": prechecklist,
                "prechecklist_source": prechecklist_source,
                "prechecklist_warning": prechecklist_warning,
            }

        changed_files = sorted(self._git_status_paths().difference(before))
        changed_for_result = sorted({*changed_files, *list(orchestrated.changed_files)})
        findings = self._run_static_analysis([p for p in changed_for_result if p.endswith(".py")])
        diff = self._git_diff(changed_for_result)
        warnings.extend([str(item).strip() for item in orchestrated.warnings if str(item).strip()])
        checklist_total = len((orchestrated.plan or {}).get("steps", []) if isinstance(orchestrated.plan, dict) else [])
        checklist_done = checklist_total if changed_for_result else 0

        planner_decisions = (
            list(orchestrated.planner_decisions)
            if isinstance(orchestrated.planner_decisions, list)
            else []
        )

        if (
            self.coding_memory_enabled
            and self.coding_memory_service is not None
            and active_flow_id is not None
        ):
            try:
                transitions: list[dict[str, Any]] = []
                for item in planner_decisions:
                    if not isinstance(item, dict):
                        continue
                    transitions.append(
                        {
                            "from_phase": f"pass_{int(item.get('pass_index', 0) or 0)}",
                            "to_phase": str(item.get("decision", "continue") or "continue"),
                            "reason": str(item.get("decision_reason", "") or "").strip()
                            or "planner decision",
                        }
                    )
                transitions.append(
                    {
                        "from_phase": "auto_execute",
                        "to_phase": "answer",
                        "reason": f"auto_execute_terminal_reason={orchestrated.terminal_reason}",
                    }
                )
                self.coding_memory_service.record_turn(
                    flow_id=active_flow_id,
                    user_request=request,
                    effective_prompt=self._effective_system_prompt_for(request, flow_context=effective_flow_context),
                    agent_answer=str(orchestrated.answer or ""),
                    changed_files=changed_for_result,
                    warnings=warnings,
                    static_findings=[_as_jsonable(f) for f in findings],
                    checklist=orchestrated.plan or {},
                    transitions=transitions,
                )
            except Exception as exc:
                warnings.append(f"coding memory turn persistence failed: {exc}")

        status = "ok" if not findings else "warning"
        return {
            "status": status,
            "answer": str(orchestrated.answer or ""),
            "changed_files": changed_for_result,
            "diff": diff,
            "warnings": warnings,
            "flow_id": active_flow_id,
            "plan": orchestrated.plan,
            "progress": {
                "phase": "answer",
                "why": f"auto_execute_terminal_reason={orchestrated.terminal_reason}",
                "budgets": {
                    "search_budget": self.search_budget,
                    "read_budget": int(tool_policy.get("read_budget", self.read_budget) or self.read_budget),
                    "read_budget_cap": int(tool_policy.get("read_budget_cap", self.read_budget) or self.read_budget),
                    "read_line_window": int(
                        tool_policy.get("read_line_window", self._DEFAULT_READ_LINE_WINDOW)
                        or self._DEFAULT_READ_LINE_WINDOW
                    ),
                    "dynamic_read_budget_used": bool(tool_policy.get("dynamic_read_budget_used", False)),
                    "dynamic_read_budget_fallback_used": bool(
                        tool_policy.get("dynamic_read_budget_fallback_used", False)
                    ),
                    "required_read_files": int(tool_policy.get("require_read_files", self.require_read_files) or self.require_read_files),
                },
            },
            "checklist": {
                "done": checklist_done,
                "pending": max(0, checklist_total - checklist_done),
                "blocked": 0,
                "total": checklist_total,
            },
            "actions_taken_total": len(orchestrated.trace),
            "actions_taken_truncated": len(orchestrated.trace) > 20,
            "actions_taken": orchestrated.trace[:20],
            "next_step": orchestrated.terminal_reason or "Completed.",
            "static_analysis": {
                "finding_count": len(findings),
                "findings": [_as_jsonable(f) for f in findings],
            },
            "auto_execute_passes": orchestrated.passes,
            "auto_execute_terminal_reason": orchestrated.terminal_reason,
            "toolsmanager_requests_count": orchestrated.toolsmanager_requests_count,
            "pass_logs": orchestrated.pass_logs,
            "planner_decisions": planner_decisions,
            "tool_execution_backend": orchestrated.execution_backend,
            "tool_execution_run_id": orchestrated.execution_run_id,
            "run_id": orchestrated.run_id,
            "run_status": orchestrated.run_status,
            "resume_command": orchestrated.resume_command,
            "next_action": orchestrated.next_action,
            "tool_execution_duration_ms": orchestrated.execution_duration_ms,
            "tool_execution_requests_ok": orchestrated.execution_requests_ok,
            "tool_execution_requests_failed": orchestrated.execution_requests_failed,
            "duplicate_request_skips": orchestrated.duplicate_request_skips,
            "duplicate_semantic_search_skips": orchestrated.duplicate_semantic_search_skips,
            "request_retry_attempts": orchestrated.request_retry_attempts,
            "request_retry_exhausted": orchestrated.request_retry_exhausted,
            "edit_retry_mode_activations": orchestrated.edit_retry_mode_activations,
            "persisted_fingerprint_counts": orchestrated.persisted_fingerprint_counts,
            "prechecklist": prechecklist,
            "prechecklist_source": prechecklist_source,
            "prechecklist_warning": prechecklist_warning,
        }

    def _looks_like_edit_request(self, request: str) -> bool:
        lowered = request.lower()
        return any(token in lowered for token in self._EDIT_INTENT_TOKENS)

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

    @staticmethod
    def _looks_like_model_docs_request(*values: str) -> bool:
        combined = " ".join(str(value or "") for value in values).lower()
        if "docs/models.md" in combined:
            return True
        return bool(
            "model" in combined
            and any(token in combined for token in ("doc", "document", "documentation", "update"))
        )

    @classmethod
    def _is_ambiguous_followup(cls, request: str) -> bool:
        text = (request or "").strip()
        if not text:
            return False
        if cls._PLAN_TRIGGER_RE.match(text):
            return True
        if cls._AMBIGUOUS_FOLLOWUP_RE.match(text):
            return True
        # Keep very short acknowledgements as ambiguous unless they include file/symbol hints.
        if len(text) <= 12 and not cls._EXPLICIT_FILE_RE.search(text):
            lowered = text.lower()
            return lowered in {"yes", "yes.", "ok", "ok.", "begin", "begin.", "go", "go.", "continue", "continue."}
        return False

    @classmethod
    def _is_plan_trigger_followup(cls, request: str) -> bool:
        return bool(cls._PLAN_TRIGGER_RE.match((request or "").strip()))

    @staticmethod
    def _extract_objective_from_flow_context(flow_context: str | None) -> str:
        context = (flow_context or "").strip()
        if not context:
            return ""
        for line in context.splitlines():
            text = line.strip()
            if text.lower().startswith("current objective:"):
                return text.split(":", 1)[1].strip()
        return ""

    @staticmethod
    def _extract_pending_steps_from_flow_context(flow_context: str | None) -> list[str]:
        context = (flow_context or "").strip()
        if not context:
            return []
        pending: list[str] = []
        for line in context.splitlines():
            text = line.strip()
            match = re.match(r"^- \[(pending|in_progress)\]\s+(.+)$", text, flags=re.IGNORECASE)
            if match:
                title = match.group(2).strip()
                if title:
                    pending.append(title[:220])
        return pending[:8]

    def _rewrite_ambiguous_followup(self, request: str, flow_context: str | None) -> str:
        if not self._is_ambiguous_followup(request):
            return request
        objective = self._extract_objective_from_flow_context(flow_context)
        if not objective:
            return request
        if self._is_plan_trigger_followup(request):
            pending_steps = self._extract_pending_steps_from_flow_context(flow_context)
            pending_block = ""
            if pending_steps:
                lines = "\n".join(f"- {item}" for item in pending_steps)
                pending_block = f"\nPending checklist steps:\n{lines}\n"
            return (
                "Execute the active flow checklist now.\n"
                f"Original follow-up: {request.strip()}\n"
                f"Current objective: {objective}\n"
                f"{pending_block}"
                "Rules:\n"
                "- Do not return only a new high-level plan.\n"
                "- Start executing pending checklist steps with tool calls.\n"
                "- Apply concrete edits and verification steps when required."
            )
        return (
            "Continue the active coding flow.\n"
            f"Original follow-up: {request.strip()}\n"
            f"Current objective: {objective}\n"
            "Proceed with concrete inspection/edit/verification steps for this objective."
        )

    def _effective_system_prompt_for(self, request: str, *, flow_context: str | None = None) -> str:
        prompt = self.system_prompt
        prompt = f"{prompt}\n\n{CODING_AGENT_LANGUAGE_TOOLING_PROMPT}"
        if self._looks_like_edit_request(request):
            prompt = f"{prompt}\n\n{CODING_AGENT_RECOGNITION_PROMPT}"
        if self.full_auto_mode:
            prompt = f"{prompt}\n\n{FULL_AUTO_EXECUTION_PROMPT}"
        if flow_context:
            prompt = (
                f"{prompt}\n\n{CODING_FLOW_MEMORY_PROMPT}\n\n"
                f"Flow context:\n{flow_context.strip()}"
            )
        return prompt

    @staticmethod
    def _log_worker_event(event: Any) -> None:
        """Stream worker tool events into the live tool-activity panel."""
        from mana_agent.commands.ui_helpers import emit_tool_event

        name = ""
        data: dict[str, Any] = {}

        if isinstance(event, dict):
            name = str(event.get("name", "") or "")
            maybe_data = event.get("data")
            if isinstance(maybe_data, dict):
                data = maybe_data
        else:
            name = str(getattr(event, "name", "") or "")
            maybe_data = getattr(event, "data", {})
            if isinstance(maybe_data, dict):
                data = maybe_data

        tool = str(data.get("tool", "") or "tool")
        event_id = str(data.get("event_id", "") or "").strip() or None
        if name in {"tool_start", "worker_request_start"}:
            emit_tool_event("start", tool, args=str(data.get("args", "") or "").strip(), event_id=event_id)
            return
        if name in {"tool_end", "worker_request_end"}:
            dt = data.get("duration_seconds")
            emit_tool_event("end", tool, duration=dt if isinstance(dt, (int, float)) else None, event_id=event_id)
            return
        if name in {"tool_error", "worker_request_error"}:
            emit_tool_event("error", tool, error=str(data.get("error", "") or "").strip(), event_id=event_id)
            return

    def _plan_checklist_with_source(
        self,
        request: str,
        *,
        flow_context: str | None = None,
    ) -> tuple[FlowChecklist | None, list[str], str]:
        warnings: list[str] = []
        user_prompt = (
            f"User request:\n{request}\n\n"
            f"Max steps: {self.plan_max_steps}\n\n"
            f"Flow context:\n{(flow_context or 'none').strip()}\n"
        )
        messages = [
            SystemMessage(content=CODING_FLOW_PLANNER_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            first = self.planner_llm.invoke(messages)
            raw = str(getattr(first, "content", "") or "").strip()
            checklist = self._parse_flow_checklist_json(raw, request=request)
            if len(checklist.steps) > self.plan_max_steps:
                checklist.steps = checklist.steps[: self.plan_max_steps]
            return checklist, warnings, "planner"
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            warnings.append(f"planner parse failed; attempting repair: {exc}")
            try:
                repair = self.planner_llm.invoke(
                    [
                        SystemMessage(content=CODING_FLOW_PLANNER_PROMPT),
                        HumanMessage(
                            content=(
                                "Repair this into strict schema JSON only.\n\n"
                                f"Broken output:\n{raw if 'raw' in locals() else ''}\n\n"
                                f"Original request:\n{request}"
                            )
                        ),
                    ]
                )
                repaired_raw = str(getattr(repair, "content", "") or "").strip()
                checklist = self._parse_flow_checklist_json(repaired_raw, request=request)
                if len(checklist.steps) > self.plan_max_steps:
                    checklist.steps = checklist.steps[: self.plan_max_steps]
                return checklist, warnings, "planner_repair"
            except Exception as exc2:  # pragma: no cover - deterministic fallback
                warnings.append(f"planner repair failed: {exc2}")
                warnings.append("planner fallback: using deterministic checklist")
                fallback = self._fallback_checklist(request)
                if len(fallback.steps) > self.plan_max_steps:
                    fallback.steps = fallback.steps[: self.plan_max_steps]
                return fallback, warnings, "deterministic_fallback"
        except Exception as exc:  # pragma: no cover
            warnings.append(f"planner invocation failed: {exc}")
            warnings.append("planner fallback: using deterministic checklist")
            fallback = self._fallback_checklist(request)
            if len(fallback.steps) > self.plan_max_steps:
                fallback.steps = fallback.steps[: self.plan_max_steps]
            return fallback, warnings, "deterministic_fallback"

    def _plan_checklist(self, request: str, *, flow_context: str | None = None) -> tuple[FlowChecklist | None, list[str]]:
        checklist, warnings, _source = self._plan_checklist_with_source(request, flow_context=flow_context)
        return checklist, warnings

    # Mutation tools whose presence in a planned step means the run must end in
    # an actual edit (and verify), not just discovery/reads.
    _MUTATION_TOOLS = frozenset({"edit_file", "multi_edit_file", "apply_patch", "create_file", "write_file", "delete_file"})

    @classmethod
    def _checklist_requires_edit(cls, checklist: "FlowChecklist | None") -> bool:
        """Recognize edit intent from the planner's checklist, not request text.

        The planner LLM decides whether the request needs changes with the
        structured ``requires_edit`` field. Older planner payloads are still
        accepted when they explicitly list mutation tools. When no checklist is
        available we err toward acting rather than silently refusing to edit.
        """
        if checklist is None:
            return True
        if bool(checklist.requires_edit):
            return True
        for step in checklist.steps:
            tools = {str(tool).strip().lower() for tool in (step.requires_tools or [])}
            if tools & cls._MUTATION_TOOLS:
                return True
        return False

    @staticmethod
    def _extract_payload(text: str) -> dict[str, Any] | None:
        raw = (text or "").strip()
        if not raw.startswith("{"):
            return None
        try:
            loaded = json.loads(raw)
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None

    @staticmethod
    def _extract_answer_text(answer: str) -> str:
        payload = CodingAgent._extract_payload(answer)
        if payload is not None and isinstance(payload.get("answer"), str):
            return str(payload["answer"]).strip()
        return (answer or "").strip()

    @classmethod
    def _trace_read_metrics(cls, trace: list[dict[str, Any]]) -> dict[str, int]:
        metrics = {
            "read_used": 0,
            "read_cache_hits": 0,
            "read_cache_misses": 0,
            "read_full_mode_used": 0,
            "read_full_mode_blocked": 0,
            "read_cache_invalidations": 0,
        }
        for row in trace:
            if str(row.get("tool_name", "")) != "read_file":
                continue
            payload = cls._extract_payload(str(row.get("output_preview", "") or "")) or {}
            if not payload:
                continue
            if bool(payload.get("cache_invalidated", False)):
                metrics["read_cache_invalidations"] += 1
            if bool(payload.get("cache_hit", False)):
                metrics["read_cache_hits"] += 1
            elif not str(payload.get("error", "")).strip():
                metrics["read_cache_misses"] += 1
                metrics["read_used"] += 1
            if str(payload.get("mode", "")).strip() == "full":
                if str(payload.get("error", "")).strip():
                    metrics["read_full_mode_blocked"] += 1
                else:
                    metrics["read_full_mode_used"] += 1
        return metrics

    @staticmethod
    def _tool_ok_from_trace_row(row: dict[str, Any]) -> bool | None:
        """
        Best-effort extraction of ok/failed from a tool trace row.
        Returns:
          - True/False if determinable
          - None if unknown
        """
        # Common patterns: { "status": "ok" } or { "ok": true } in the row itself
        if isinstance(row.get("ok"), bool):
            return bool(row["ok"])
        status = row.get("status")
        if isinstance(status, str):
            s = status.lower().strip()
            if s in ("ok", "success", "succeeded", "passed"):
                return True
            if s in ("error", "failed", "failure"):
                return False

        # Many agents store a JSON-ish preview under output_preview
        preview = row.get("output_preview")
        if isinstance(preview, str) and preview.strip().startswith("{"):
            try:
                payload = json.loads(preview)
                if isinstance(payload, dict) and isinstance(payload.get("ok"), bool):
                    return bool(payload["ok"])
            except Exception:
                pass

        return None

    def _compute_progress(
        self,
        *,
        checklist: FlowChecklist,
        trace: list[dict[str, Any]],
        warnings: list[str],
        changed_files: list[str],
        required_read_files: int,
    ) -> tuple[ExecutionDecision, dict[str, int], int]:
        counts: dict[str, int] = {}
        read_files: set[str] = set()
        for row in trace:
            name = str(row.get("tool_name", ""))
            counts[name] = counts.get(name, 0) + 1
            if name == "read_file":
                preview = str(row.get("output_preview", "") or "")
                payload = self._extract_payload(preview)
                if payload is not None:
                    path = str(payload.get("file_path", "")).strip()
                    if path:
                        read_files.add(path)

        read_count = len(read_files)
        if changed_files:
            phase = "verify" if any(x.endswith(".py") for x in changed_files) else "answer"
            return (
                ExecutionDecision(
                    phase=phase,
                    tool_call_allowed=True,
                    why="Edits detected; proceed to verification/answer.",
                ),
                counts,
                read_count,
            )

        if read_count < required_read_files:
            return (
                ExecutionDecision(
                    phase="blocked",
                    tool_call_allowed=False,
                    why=f"Need at least {required_read_files} unique read_file inspections before edit/answer.",
                ),
                counts,
                read_count,
            )

        if not changed_files:
            joined = "\n".join(warnings).lower()
            if "apply_patch failed" in joined or "patch did not apply cleanly" in joined or "patch-only loop" in joined:
                return (
                    ExecutionDecision(
                        phase="blocked",
                        tool_call_allowed=False,
                        why="Patch failures detected via warnings with no repo changes; stopping without duplicate mutation retry.",
                    ),
                    counts,
                    read_count,
                )

        if warnings:
            return (
                ExecutionDecision(phase="inspect", tool_call_allowed=True, why="No edits yet; continue inspection."),
                counts,
                read_count,
            )

        _ = checklist
        return (
            ExecutionDecision(phase="edit", tool_call_allowed=True, why="Evidence gate met; editing allowed."),
            counts,
            read_count,
        )

    def _checklist_counts(self, checklist: FlowChecklist) -> dict[str, int]:
        done = len([step for step in checklist.steps if step.status == "done"])
        blocked = len([step for step in checklist.steps if step.status == "blocked"])
        pending = len(checklist.steps) - done - blocked
        return {"done": done, "pending": pending, "blocked": blocked, "total": len(checklist.steps)}

    @staticmethod
    def _clamp_read_line_window(value: int) -> int:
        return max(CodingAgent._MIN_READ_LINE_WINDOW, min(int(value), CodingAgent._MAX_READ_LINE_WINDOW))

    def _dynamic_read_policy_for_request(
        self,
        request: str,
        *,
        flow_context: str | None = None,
    ) -> dict[str, Any]:
        user_cap = max(1, int(self.read_budget))
        selected: dict[str, Any] = {
            "read_budget": user_cap,
            "read_line_window": self._clamp_read_line_window(self._DEFAULT_READ_LINE_WINDOW),
            "dynamic_read_budget_used": False,
            "dynamic_read_budget_fallback_used": False,
            "dynamic_read_budget_reason": "static_default",
        }
        if not self.full_auto_mode:
            return selected
        if self._looks_like_model_docs_request(request, flow_context or ""):
            relevant_files: set[str] = set()
            try:
                for path in self.repo_root.rglob("models.py"):
                    rel = path.resolve().relative_to(self.repo_root).as_posix()
                    parts = set(Path(rel).parts)
                    if parts.intersection({".git", ".mana", ".venv", "venv", "node_modules", "__pycache__"}):
                        continue
                    relevant_files.add(rel)
                if (self.repo_root / "docs" / "models.md").exists():
                    relevant_files.add("docs/models.md")
            except Exception:
                relevant_files = set()
            if relevant_files:
                budget = max(user_cap, min(len(relevant_files), self._MAX_DYNAMIC_READ_BUDGET))
                selected.update(
                    {
                        "read_budget": budget,
                        "read_line_window": self._clamp_read_line_window(self._MAX_READ_LINE_WINDOW),
                        "dynamic_read_budget_used": True,
                        "dynamic_read_budget_fallback_used": False,
                        "dynamic_read_budget_reason": "model_docs_inventory",
                        "read_budget_cap": budget,
                    }
                )
                return selected
        dynamic_cap = max(1, min(user_cap, self._MAX_DYNAMIC_READ_BUDGET))
        selected["read_budget"] = dynamic_cap

        flow_summary = str(flow_context or "").strip()
        if len(flow_summary) > 1600:
            flow_summary = flow_summary[:1600].rstrip()
        prompt = (
            "Select safe read_file limits for one coding turn.\n"
            "Return JSON with fields: read_budget, read_line_window, reason.\n"
            f"Constraints:\n"
            f"- read_budget must be between 1 and {dynamic_cap}\n"
            f"- read_line_window must be between {self._MIN_READ_LINE_WINDOW} and {self._MAX_READ_LINE_WINDOW}\n"
            "- Increase limits only when broad discovery is required.\n"
            "- Keep limits conservative by default.\n\n"
            f"User request:\n{request}\n\n"
            f"Flow context:\n{flow_summary or 'none'}"
        )
        messages = [
            SystemMessage(content="You output only valid JSON for the requested schema."),
            HumanMessage(content=prompt),
        ]

        def _finalize(policy: DynamicReadPolicy) -> dict[str, Any]:
            budget = max(1, min(int(policy.read_budget), dynamic_cap))
            line_window = self._clamp_read_line_window(int(policy.read_line_window))
            return {
                "read_budget": budget,
                "read_line_window": line_window,
                "dynamic_read_budget_used": True,
                "dynamic_read_budget_fallback_used": False,
                "dynamic_read_budget_reason": str(policy.reason or "").strip(),
            }

        try:
            if hasattr(self.planner_llm, "with_structured_output"):
                structured_llm = self.planner_llm.with_structured_output(DynamicReadPolicy)
                structured_result = structured_llm.invoke(messages)
                policy = (
                    structured_result
                    if isinstance(structured_result, DynamicReadPolicy)
                    else DynamicReadPolicy.model_validate(structured_result)
                )
                return _finalize(policy)
        except Exception:
            pass

        try:
            response = self.planner_llm.invoke(messages)
            raw = str(getattr(response, "content", "") or "").strip()
            parsed = self._parse_json_or_literal(raw)
            if isinstance(parsed, dict):
                policy = DynamicReadPolicy.model_validate(parsed)
                return _finalize(policy)
        except Exception:
            pass

        selected["dynamic_read_budget_fallback_used"] = True
        selected["dynamic_read_budget_reason"] = "fallback_static_default"
        return selected

    def _tool_policy_for_request(
        self,
        request: str,
        *,
        flow_context: str | None = None,
        auto_chat_mode: str | None = None,
    ) -> dict[str, Any]:
        require_read_files = self.require_read_files
        explicit_files = {
            match.group(1).strip()
            for match in self._EXPLICIT_FILE_RE.finditer(request or "")
            if match.group(1).strip()
        }
        explicit_files.update(
            match.group(1).strip()
            for match in self._EXPLICIT_DOTFILE_RE.finditer(request or "")
            if match.group(1).strip()
        )
        # Single-target file edits (e.g. README.md) should not be blocked by a 2-file gate.
        if len(explicit_files) == 1:
            require_read_files = 1
        dynamic_read_policy = self._dynamic_read_policy_for_request(
            request,
            flow_context=flow_context,
        )
        policy = {
            "allowed_tools": [
                "semantic_search",
                "read_file",
                "run_command",
                "chunk_file",
                "list_tools",
                "ls",
                "repo_search",
                "list_files",
                "find_symbols",
                "call_graph",
                "git_status",
                "git_diff",
                "verify_project",
                "tool_contracts",
                "edit_file",
                "multi_edit_file",
                "apply_patch",
                "create_file",
                "write_file",
                "delete_file",
            ],
            "search_budget": self.search_budget,
            "read_budget": int(dynamic_read_policy.get("read_budget", self.read_budget) or self.read_budget),
            "read_budget_cap": int(dynamic_read_policy.get("read_budget_cap", self.read_budget) or self.read_budget),
            "read_line_window": int(
                dynamic_read_policy.get("read_line_window", self._DEFAULT_READ_LINE_WINDOW)
                or self._DEFAULT_READ_LINE_WINDOW
            ),
            "read_mode_preference": "full_preferred",
            "read_full_file_max_lines": 5000,
            "read_full_file_max_chars": 250000,
            "read_cache_scope": "flow" if (self.coding_memory_enabled and self.coding_memory_service is not None) else "run",
            "dynamic_read_budget_used": bool(dynamic_read_policy.get("dynamic_read_budget_used", False)),
            "dynamic_read_budget_fallback_used": bool(
                dynamic_read_policy.get("dynamic_read_budget_fallback_used", False)
            ),
            "dynamic_read_budget_reason": str(dynamic_read_policy.get("dynamic_read_budget_reason", "") or ""),
            "require_read_files": require_read_files,
            "search_repeat_limit": 1,
            "max_semantic_k": 50,
        }
        if auto_chat_mode:
            policy = apply_auto_chat_tool_policy(policy, auto_chat_mode)
        return policy

    def _generate_common(
        self,
        request: str,
        *,
        call_agent_fn,
        flow_context: str | None,
        flow_id: str | None,
        auto_chat_mode: str | None = None,
    ) -> tuple[dict[str, Any], str | None, str | None]:
        before = self._git_status_paths()
        warnings: list[str] = []
        active_flow_id = flow_id
        effective_flow_context = flow_context
        if self.coding_memory_enabled and self.coding_memory_service is not None:
            try:
                active_flow_id = self.coding_memory_service.ensure_flow(flow_id=flow_id, request=request)
                self._current_flow_id = active_flow_id
                if effective_flow_context is None:
                    effective_flow_context = self.coding_memory_service.build_flow_context(
                        active_flow_id,
                        sorted(before),
                    )
            except Exception as exc:
                warnings.append(f"coding memory setup failed: {exc}")
                active_flow_id = flow_id

        checklist, plan_warnings = self._plan_checklist(request, flow_context=effective_flow_context)
        warnings.extend(plan_warnings)
        if checklist is None:
            result = {
                "status": "warning",
                "answer": "Planner failed to produce valid checklist JSON after repair; stopping to avoid blind tool loop.",
                "changed_files": [],
                "diff": "",
                "warnings": warnings,
                "flow_id": active_flow_id,
                "plan": None,
                "progress": {"phase": "blocked", "why": "planner_failed"},
                "checklist": {"done": 0, "pending": 0, "blocked": 1, "total": 0},
                "actions_taken": [],
                "next_step": "Refine request or retry planner with stricter constraints.",
            }
            return result, active_flow_id, effective_flow_context

        tool_policy = self._tool_policy_for_request(
            request,
            flow_context=effective_flow_context,
            auto_chat_mode=auto_chat_mode,
        )
        required_read_files = int(tool_policy.get("require_read_files", self.require_read_files) or self.require_read_files)
        request_for_run = self._rewrite_ambiguous_followup(request, effective_flow_context)
        if request_for_run != request:
            warnings.append("followup_request_rewritten_from_flow_context")

        edit_intent = self._looks_like_edit_request(request_for_run)
        # First pass
        answer = call_agent_fn(
            request_text=request_for_run,
            tool_policy=tool_policy,
            flow_context=effective_flow_context,
        )
        changed = sorted(self._git_status_paths().difference(before))
        payload = self._extract_payload(answer) or {}
        trace = payload.get("trace", [])
        if not isinstance(trace, list):
            trace = []

        payload_warnings = payload.get("warnings", [])
        if isinstance(payload_warnings, list):
            warnings.extend(str(item) for item in payload_warnings if str(item).strip())

        trace_rows = [item for item in trace if isinstance(item, dict)]
        combined_trace_rows = list(trace_rows)

        decision, counters, read_count = self._compute_progress(
            checklist=checklist,
            trace=combined_trace_rows,
            warnings=warnings,
            changed_files=changed,
            required_read_files=required_read_files,
        )
        answer_text = self._extract_answer_text(answer)
        mutation_tools_seen = {str(row.get("tool_name", "")) for row in combined_trace_rows}
        attempted_apply_patch = "apply_patch" in mutation_tools_seen
        attempted_write_file = "write_file" in mutation_tools_seen
        attempted_mutation = bool(mutation_tools_seen.intersection({"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}))

        if edit_intent and not changed and attempted_apply_patch:
            warnings.append("mutation_noop_after_apply_patch")
        if edit_intent and not changed and attempted_write_file:
            warnings.append("mutation_noop_after_write_file")

        if (
            edit_intent
            and not changed
            and self._looks_like_conversational_terminal(answer_text)
            and not attempted_mutation
        ):
            warnings.append("edit_intent_conversational_noop_detected")
            retry_request = (
                "Do not ask for confirmation. Execute concrete repository edits now. "
                "If apply_patch fails/no-ops, use write_file full-content fallback and verify changed_files.\n\n"
                f"Original request:\n{request}"
            )
            with self._without_tool("apply_patch"):
                answer = call_agent_fn(
                    request_text=retry_request,
                    tool_policy=tool_policy,
                    flow_context=effective_flow_context,
                )
            changed = sorted(self._git_status_paths().difference(before))
            payload = self._extract_payload(answer) or {}
            trace = payload.get("trace", [])
            if not isinstance(trace, list):
                trace = []
            payload_warnings = payload.get("warnings", [])
            if isinstance(payload_warnings, list):
                warnings.extend(str(item) for item in payload_warnings if str(item).strip())
            trace_rows = [item for item in trace if isinstance(item, dict)]
            combined_trace_rows.extend(trace_rows)
            answer_text = self._extract_answer_text(answer)
            mutation_tools_seen = {str(row.get("tool_name", "")) for row in combined_trace_rows}
            attempted_apply_patch = "apply_patch" in mutation_tools_seen
            attempted_write_file = "write_file" in mutation_tools_seen
            decision, counters, read_count = self._compute_progress(
                checklist=checklist,
                trace=combined_trace_rows,
                warnings=warnings,
                changed_files=changed,
                required_read_files=required_read_files,
            )

        if edit_intent and not changed and attempted_mutation:
            warnings.append("mutation_failed_no_changes")
            decision = ExecutionDecision(
                phase="blocked",
                tool_call_allowed=False,
                why=(
                    "mutation_failed_no_changes: mutation tool returned without file changes; "
                    "stopping without duplicate mutation retry."
                ),
            )

        trace_rows = combined_trace_rows
        findings = self._run_static_analysis([p for p in changed if p.endswith(".py")])
        diff = self._git_diff(changed)
        checklist_counts = self._checklist_counts(checklist)
        transitions = [
            {
                "from_phase": "discover",
                "to_phase": decision.phase,
                "reason": decision.why,
            }
        ]
        effective_prompt = self._effective_system_prompt_for(request, flow_context=effective_flow_context)

        if (
            self.coding_memory_enabled
            and self.coding_memory_service is not None
            and active_flow_id is not None
        ):
            try:
                self.coding_memory_service.record_turn(
                    flow_id=active_flow_id,
                    user_request=request,
                    effective_prompt=effective_prompt,
                    agent_answer=answer_text,
                    changed_files=changed,
                    warnings=warnings,
                    static_findings=[_as_jsonable(f) for f in findings],
                    checklist=checklist.model_dump(),
                    transitions=transitions,
                )
            except Exception as exc:
                warnings.append(f"coding memory turn persistence failed: {exc}")

        status = "ok" if not findings else "warning"
        if decision.phase == "blocked":
            status = "warning"
        trace_rows = [item for item in trace if isinstance(item, dict)]
        read_metrics = self._trace_read_metrics(trace_rows)
        warning_text = "\n".join(str(item) for item in warnings).lower()
        tools_only_fallback = (
            ("tools_only_violation" in warning_text)
            and len(trace_rows) == 0
            and not changed
        )
        controlled_worker_fallback = ""
        if len(trace_rows) == 0 and not changed:
            for code in (
                "tools_only_violation",
                "mutation_not_attempted",
                "mutation_failed",
                "invalid_tool_args",
                "run_failed",
            ):
                if code in warning_text:
                    controlled_worker_fallback = code
                    break
        result = {
            "status": status,
            "answer": answer,
            "changed_files": changed,
            "diff": diff,
            "warnings": warnings,
            "flow_id": active_flow_id,
            "plan": checklist.model_dump(),
            "progress": {
                "phase": decision.phase,
                "why": decision.why,
                "budgets": {
                    "search_budget": self.search_budget,
                    "search_used": counters.get("semantic_search", 0),
                    "read_budget": int(tool_policy.get("read_budget", self.read_budget) or self.read_budget),
                    "read_budget_cap": int(tool_policy.get("read_budget_cap", self.read_budget) or self.read_budget),
                    "read_used": read_metrics.get("read_used", 0),
                    "read_line_window": int(
                        tool_policy.get("read_line_window", self._DEFAULT_READ_LINE_WINDOW)
                        or self._DEFAULT_READ_LINE_WINDOW
                    ),
                    "read_mode_preference": str(tool_policy.get("read_mode_preference", "full_preferred") or "full_preferred"),
                    "read_full_file_max_lines": int(tool_policy.get("read_full_file_max_lines", 5000) or 5000),
                    "read_full_file_max_chars": int(tool_policy.get("read_full_file_max_chars", 250000) or 250000),
                    "read_cache_scope": str(tool_policy.get("read_cache_scope", "run") or "run"),
                    "read_cache_hits": read_metrics.get("read_cache_hits", 0),
                    "read_cache_misses": read_metrics.get("read_cache_misses", 0),
                    "read_full_mode_used": read_metrics.get("read_full_mode_used", 0),
                    "read_full_mode_blocked": read_metrics.get("read_full_mode_blocked", 0),
                    "read_cache_invalidations": read_metrics.get("read_cache_invalidations", 0),
                    "dynamic_read_budget_used": bool(tool_policy.get("dynamic_read_budget_used", False)),
                    "dynamic_read_budget_fallback_used": bool(
                        tool_policy.get("dynamic_read_budget_fallback_used", False)
                    ),
                    "required_read_files": required_read_files,
                    "read_files_observed": read_count,
                },
            },
            "checklist": checklist_counts,
            "actions_taken_total": len(trace_rows),
            "actions_taken_truncated": len(trace_rows) > 20,
            "actions_taken": trace_rows[:20],
            "next_step": checklist.next_action or decision.why,
            "static_analysis": {
                "finding_count": len(findings),
                "findings": [_as_jsonable(f) for f in findings],
            },
            "render_mode": "answer_only" if controlled_worker_fallback or tools_only_fallback else "default",
            "fallback_reason": controlled_worker_fallback or ("tools_only_violation" if tools_only_fallback else ""),
            "fallback_retry_attempted": False,
        }
        return result, active_flow_id, effective_flow_context

    def generate(
        self,
        request: str,
        *,
        index_dir: str | Path | None = None,
        k: int | None = None,
        max_steps: int = 200,
        timeout_seconds: int = 600,
        callbacks: Sequence[Any] | None = None,
        flow_context: str | None = None,
        flow_id: str | None = None,
        auto_chat_mode: str | None = None,
    ) -> dict[str, Any]:
        def _call(*, request_text: str, tool_policy: dict[str, Any], flow_context: str | None) -> str:
            return self._call_agent_single(
                request_text,
                index_dir=index_dir,
                k=k,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                callbacks=callbacks,
                flow_context=flow_context,
                tool_policy=tool_policy,
                flow_id=self._current_flow_id or flow_id,
            )

        result, _, _ = self._generate_common(
            request,
            call_agent_fn=_call,
            flow_context=flow_context,
            flow_id=flow_id,
            auto_chat_mode=auto_chat_mode,
        )
        return result

    def generate_dir_mode(
        self,
        request: str,
        *,
        index_dirs: Sequence[str | Path],
        k: int | None = None,
        max_steps: int = 200,
        timeout_seconds: int = 600,
        callbacks: Sequence[Any] | None = None,
        flow_context: str | None = None,
        flow_id: str | None = None,
        auto_chat_mode: str | None = None,
    ) -> dict[str, Any]:
        def _call(*, request_text: str, tool_policy: dict[str, Any], flow_context: str | None) -> str:
            return self._call_agent_multi(
                request_text,
                index_dirs=index_dirs,
                k=k,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                callbacks=callbacks,
                flow_context=flow_context,
                tool_policy=tool_policy,
                flow_id=self._current_flow_id or flow_id,
            )

        result, _, _ = self._generate_common(
            request,
            call_agent_fn=_call,
            flow_context=flow_context,
            flow_id=flow_id,
            auto_chat_mode=auto_chat_mode,
        )
        return result

    def _execute_via_manager(
        self,
        tool_req: "ToolRunRequest",
        *,
        tool_policy: "dict[str, Any] | None" = None,
        index_dir: "str | Path | None" = None,
        index_dirs: "list[str] | None" = None,
        flow_id: "str | None" = None,
        pass_cap: int = 1,
        timeout_seconds: int = 60,
        max_steps: int = 6,
        k: "int | None" = None,
        flow_context: "str | None" = None,
    ) -> "ToolRunResponse":
        """Route a ToolRunRequest through the QueueManager.

        CodingAgent does not execute repository tools directly. Tool work must
        enter through QueueManager so AgentWorkQueue owns scheduling and the
        worker process owns actual tool execution.
        """
        from mana_agent.llm.tool_worker_process import ToolRunResponse as _TRR

        if self.tools_manager_orchestrator is None:
            return _TRR(
                answer="Auto-execute orchestrator is unavailable for this coding-agent session.",
                sources=[],
                mode="agent-tools",
                trace=[],
                warnings=["auto_execute_orchestrator_unavailable"],
            )

        resolved_tool_policy = tool_policy or {}
        orchestrated = self.tools_manager_orchestrator.run(
            request=tool_req.question,
            flow_context=flow_context,
            index_dir=index_dir,
            index_dirs=index_dirs,
            k=int(k if k is not None else 8),
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            tool_policy=resolved_tool_policy,
            pass_cap=max(1, pass_cap),
            on_event=self._log_worker_event,
            flow_id=flow_id,
            run_id=tool_req.run_id,
        )
        return _TRR(
            answer=str(orchestrated.answer or ""),
            sources=list(orchestrated.sources),
            mode="agent-tools",
            trace=list(orchestrated.trace),
            warnings=list(orchestrated.warnings),
        )

    def _call_agent_single(
        self,
        request: str,
        *,
        index_dir: str | Path | None,
        k: int | None,
        max_steps: int,
        timeout_seconds: int,
        callbacks: Sequence[Any] | None,
        flow_context: str | None = None,
        tool_policy: dict[str, Any] | None = None,
        flow_id: str | None = None,
    ) -> str:
        _ = callbacks
        effective_prompt = self._effective_system_prompt_for(request, flow_context=flow_context)
        tool_req = ToolRunRequest(
            question=request,
            index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
            k=int(k if k is not None else 8),
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            tool_policy=tool_policy,
            system_prompt=effective_prompt,
            flow_id=flow_id,
            run_id=flow_id,
        )
        try:
            response = self._execute_via_manager(
                tool_req,
                tool_policy=tool_policy,
                index_dir=index_dir,
                flow_id=flow_id,
                timeout_seconds=timeout_seconds,
                max_steps=max_steps,
                k=k,
                flow_context=flow_context,
            )
            return self._stringify(response.model_dump())
        except ToolWorkerProcessError as exc:
            return self._stringify(self._controlled_worker_error_payload(exc))

    def _call_agent_multi(
        self,
        request: str,
        *,
        index_dirs: Sequence[str | Path],
        k: int | None,
        max_steps: int,
        timeout_seconds: int,
        callbacks: Sequence[Any] | None,
        flow_context: str | None = None,
        tool_policy: dict[str, Any] | None = None,
        flow_id: str | None = None,
    ) -> str:
        _ = callbacks
        resolved = [str(Path(p).resolve()) for p in index_dirs if str(p).strip()]
        if not resolved:
            return "No index_dirs provided for dir-mode."
        effective_prompt = self._effective_system_prompt_for(request, flow_context=flow_context)
        tool_req = ToolRunRequest(
            question=request,
            index_dirs=resolved,
            k=int(k if k is not None else 8),
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            tool_policy=tool_policy,
            system_prompt=effective_prompt,
            flow_id=flow_id,
            run_id=flow_id,
        )
        try:
            response = self._execute_via_manager(
                tool_req,
                tool_policy=tool_policy,
                index_dirs=resolved,
                flow_id=flow_id,
                timeout_seconds=timeout_seconds,
                max_steps=max_steps,
                k=k,
                flow_context=flow_context,
            )
            return self._stringify(response.model_dump())
        except ToolWorkerProcessError as exc:
            return self._stringify(self._controlled_worker_error_payload(exc))

    @staticmethod
    def _controlled_worker_error_payload(exc: ToolWorkerProcessError) -> dict[str, Any]:
        code = str(exc.code or "worker_error")
        return {
            "answer": f"Worker stopped cleanly: {code}: {exc}",
            "sources": [],
            "mode": "agent-tools",
            "trace": [],
            "warnings": [f"{code}: {exc}"],
            "render_mode": "answer_only",
            "fallback_reason": code,
            "fallback_retry_attempted": False,
        }

    @contextmanager
    def _without_tool(self, tool_name: str):
        tools = getattr(self.ask_agent, "tools", None)
        if not isinstance(tools, list):
            yield
            return
        original = list(tools)
        tools[:] = [tool for tool in tools if str(getattr(tool, "name", "")) != tool_name]
        try:
            yield
        finally:
            tools[:] = original

    def _stringify(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        try:
            return json.dumps(_as_jsonable(result), indent=2, ensure_ascii=False)
        except Exception:
            return str(result)

    def _invoke_tools_planner(self, human_prompt: str) -> str:
        if not hasattr(self, "planner_llm"):
            raise AttributeError("planner_llm is not initialized")
        messages = [
            SystemMessage(content=HEAD_TOOLS_PLANNER_PROMPT),
            HumanMessage(content=human_prompt),
        ]
        response = self.planner_llm.invoke(messages)
        return str(getattr(response, "content", response) or "").strip()

    def _repair_tools_planner(self, raw: str, human_prompt: str) -> str:
        repair_prompt = (
            "The previous tools planner output was invalid."
            " Return corrected JSON matching the required schema."
            " Original output:\n"
            f"{raw}\n\n"
            f"Original prompt:\n{human_prompt}"
        )
        return self._invoke_tools_planner(repair_prompt)

    def _invoke_tools_batcher(self, human_prompt: str) -> str:
        if not hasattr(self, "planner_llm"):
            raise AttributeError("planner_llm is not initialized")
        messages = [
            SystemMessage(content=TOOLSMANAGER_PROMPT),
            HumanMessage(content=human_prompt),
        ]
        response = self.planner_llm.invoke(messages)
        return str(getattr(response, "content", response) or "").strip()

    def _repair_tools_batcher(self, raw: str, human_prompt: str) -> str:
        repair_prompt = (
            "The previous tools batcher output was invalid."
            " Return corrected JSON matching the ToolsManagerBatch schema."
            " Original output:\n"
            f"{raw}\n\n"
            f"Original prompt:\n{human_prompt}"
        )
        return self._invoke_tools_batcher(repair_prompt)

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
                    p = line[3:].strip()
                    if p:
                        paths.add(p.replace("\\", "/"))
            return paths
        except Exception:
            return set()

    def _git_diff(self, paths: list[str]) -> str:
        if not paths:
            return ""
        try:
            proc = subprocess.run(
                ["git", "diff", "--", *paths],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return ""
            return proc.stdout[:200_000]
        except Exception:
            return ""

    def _run_static_analysis(self, py_paths: list[str]) -> list[Any]:
        if not py_paths:
            return []
        try:
            from mana_agent.analysis.checks import PythonStaticAnalyzer  # type: ignore

            analyzer = PythonStaticAnalyzer()
            all_findings: list[Any] = []
            for rel in py_paths:
                p = (self.repo_root / rel).resolve()
                try:
                    findings = analyzer.analyze_file(p)
                    if findings:
                        all_findings.extend(findings)
                except Exception as exc:
                    all_findings.append({"path": str(rel), "error": f"Static analysis error: {exc}"})
            return all_findings
        except Exception as exc:
            logger.debug("Static analysis unavailable: %s", exc)
            return []

    def get_active_flow_id(self) -> str | None:
        if self._current_flow_id:
            return self._current_flow_id
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            return None
        return self.coding_memory_service.get_active_flow_id()

    def flow_summary(self, flow_id: str | None = None) -> dict[str, Any] | None:
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            return None
        target = flow_id or self.get_active_flow_id()
        if not target:
            return None
        summary = self.coding_memory_service.get_flow_summary(target)
        if summary is None:
            return None
        return {
            "flow_id": summary.flow_id,
            "objective": summary.objective,
            "updated_at": summary.updated_at,
            "constraints": summary.constraints,
            "acceptance": summary.acceptance,
            "open_tasks": summary.open_tasks,
            "recent_decisions": summary.recent_decisions,
            "last_changed_files": summary.last_changed_files,
            "unresolved_static_findings": summary.unresolved_static_findings,
            "checklist": summary.checklist,
            "transitions": summary.transitions,
            "last_blocked_reason": summary.last_blocked_reason,
            "recent_turns": self.coding_memory_service.list_recent_turns(summary.flow_id),
        }

    def reset_flow(self, flow_id: str | None = None) -> str | None:
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            self._current_flow_id = None
            return None
        target = flow_id or self.get_active_flow_id()
        if not target:
            self._current_flow_id = None
            return None
        self.coding_memory_service.reset_flow(target)
        self._current_flow_id = None
        return target

    def checkpoint_flow(self, flow_id: str | None = None) -> str | None:
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            return None
        target = flow_id or self.get_active_flow_id()
        if not target:
            return None
        summary = self.flow_summary(target) or {}
        self.coding_memory_service.checkpoint(target, snapshot=summary)
        return target

    def is_conflicting_request(self, request: str, flow_id: str | None = None) -> bool:
        if not self.coding_memory_enabled or self.coding_memory_service is None:
            return False
        target = flow_id or self.get_active_flow_id()
        if not target:
            return False
        return self.coding_memory_service.is_conflicting_request(target, request)
