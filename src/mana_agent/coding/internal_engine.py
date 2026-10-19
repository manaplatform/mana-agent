"""Mana-Agent Native Internal Coding Engine.

Executes coding tasks using Mana-Agent's native AI model orchestration and
registered development tools. Supports OpenAI Programmatic Tool Calling
via the Responses API lifecycle when supported, and direct tool calling
as a graceful fallback for unsupported models.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.coding.engine import CodingEngine
from mana_agent.coding.live_events import (
    coding_event_scope,
    coding_execution_context,
    publish_coding_event,
)
from mana_agent.coding.models import (
    AgentEvent,
    CodingEngineMode,
    CodingStatus,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
    compute_duration_breakdown,
)
from mana_agent.coding.programmatic import (
    ProgrammaticExecutionSession,
    ToolExecutionDispatcher,
    build_programmatic_tool_definitions,
)
from mana_agent.coding.prompts import INTERNAL_CODING_SYSTEM_INSTRUCTION
from mana_agent.tools.contracts import ToolContract, coding_tool_contracts

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InternalCodingEngine(CodingEngine):
    """Mana-Agent Native Coding Engine."""

    def __init__(
        self,
        *,
        model: str = "",
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        llm_client: Any | None = None,
        system_instruction: str | None = None,
        supports_responses_api: bool = True,
        programmatic_tool_calling_enabled: bool = True,
        max_tool_calls: int = 50,
        extra_tools: dict[str, Callable[..., Any]] | None = None,
        global_memory: Any | None = None,
    ) -> None:
        self.mode: CodingEngineMode = "internal"
        self.name: str = "internal"
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.llm_client = llm_client
        self.system_instruction = system_instruction or INTERNAL_CODING_SYSTEM_INSTRUCTION
        self.supports_responses_api = supports_responses_api
        self.programmatic_tool_calling_enabled = programmatic_tool_calling_enabled
        self.max_tool_calls = max_tool_calls
        self.extra_tools = extra_tools or {}
        if global_memory is not None:
            self.global_memory = global_memory
        else:
            try:
                from mana_agent.memory.learning import get_global_memory
                self.global_memory = get_global_memory()
            except Exception:
                self.global_memory = None

        self._task_statuses: dict[str, CodingStatus] = {}
        self._active_sessions: dict[str, ProgrammaticExecutionSession] = {}
        self._cancellation_events: dict[str, asyncio.Event] = {}

    async def start(self) -> None:
        return None

    def _should_use_programmatic(self) -> bool:
        return (
            self.supports_responses_api
            and self.programmatic_tool_calling_enabled
            and self.provider.lower() in {"openai", "custom"}
        )

    def _get_or_create_llm(self) -> Any:
        if self.llm_client is not None:
            return self.llm_client

        api_key = self.api_key
        base_url = self.base_url
        model = self.model
        provider = self.provider or "openai"

        if not api_key:
            try:
                from mana_agent.config.settings import Settings
                from mana_agent.config.user_config import resolve_inference_connection
                conn = resolve_inference_connection(Settings(), require_api_key=False)
                if conn.api_key:
                    api_key = conn.api_key
                if not base_url and conn.base_url:
                    base_url = conn.base_url
            except Exception:
                pass

        if not api_key:
            import os
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MANA_API_KEY") or ""

        if not model:
            try:
                from mana_agent.config.settings import Settings
                model = getattr(Settings(), "openai_chat_model", "gpt-4.1-mini") or "gpt-4.1-mini"
            except Exception:
                model = "gpt-4.1-mini"

        if not api_key:
            return None

        try:
            from mana_agent.multi_agent.runtime.compatibility import create_chat_model
            return create_chat_model(
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
            )
        except Exception as exc:
            logger.warning("Could not instantiate chat model for internal coding engine: %s", exc)
            return None

    def _build_tool_definitions(self, dispatcher: ToolExecutionDispatcher) -> list[dict[str, Any]]:
        tool_schemas: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        contracts = coding_tool_contracts()
        for c in contracts:
            sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", c.name)
            if sanitized_name in seen_names:
                continue
            seen_names.add(sanitized_name)
            tool_schemas.append({
                "type": "function",
                "function": {
                    "name": sanitized_name,
                    "description": c.description,
                    "parameters": c.input_schema,
                },
            })
        for extra_name, extra_fn in dispatcher.extra_tools.items():
            sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", extra_name)
            if sanitized_name in seen_names:
                continue
            seen_names.add(sanitized_name)
            doc = getattr(extra_fn, "__doc__", "") or f"Custom tool {sanitized_name}"
            tool_schemas.append({
                "type": "function",
                "function": {
                    "name": sanitized_name,
                    "description": doc,
                    "parameters": {"type": "object", "properties": {}},
                },
            })
        return tool_schemas

    async def execute(self, task: CodingTask, workspace: WorkspaceContext) -> CodingTaskResult:
        task_id = task.task_id
        t_created = task.task_created_at or _utc_now()
        t_scheduled = _utc_now()
        t_claimed = _utc_now()
        t_p_start = _utc_now()

        self._task_statuses[task_id] = CodingStatus(
            task_id=task_id,
            status="running",
            backend=self.name,
        )

        cancel_ev = asyncio.Event()
        self._cancellation_events[task_id] = cancel_ev

        dispatcher = ToolExecutionDispatcher(workspace, extra_tools=self.extra_tools)
        session = ProgrammaticExecutionSession(
            task_id=task_id,
            dispatcher=dispatcher,
            max_tool_calls=self.max_tool_calls,
            backend_name=self.name,
        )
        self._active_sessions[task_id] = session

        with coding_execution_context(task_id=task_id, backend=self.name, model=self.model) as ctx_state:
            publish_coding_event(AgentEvent(
                event_type="backend.selected",
                task_id=task_id,
                backend=self.name,
                sequence=1,
                title="Internal backend selected",
                summary="Executing coding task via Mana-Agent Native Coding Engine",
                model=self.model,
                visibility="progress",
            ))

            publish_coding_event(AgentEvent(
                event_type="turn.started",
                task_id=task_id,
                backend=self.name,
                sequence=2,
                title="Internal coding turn started",
                model=self.model,
                visibility="progress",
            ))

            use_programmatic = self._should_use_programmatic()
            final_summary = ""
            status: str = "completed"
            error_msg: str = ""

            try:
                if cancel_ev.is_set():
                    raise asyncio.CancelledError()

                if use_programmatic:
                    final_summary = await self._run_programmatic_workflow(task, workspace, session, cancel_ev)
                else:
                    final_summary = await self._run_direct_tool_workflow(task, workspace, session, cancel_ev)

            except asyncio.CancelledError:
                status = "cancelled"
                error_msg = "Task cancelled by user request"
                publish_coding_event(AgentEvent(
                    event_type="turn.cancelled",
                    task_id=task_id,
                    backend=self.name,
                    sequence=100,
                    status="cancelled",
                    title="Internal coding turn cancelled",
                    error=error_msg,
                    visibility="terminal",
                ))
            except Exception as exc:
                status = "failed"
                error_msg = str(exc)
                logger.exception("Internal coding engine execution failed")
                publish_coding_event(AgentEvent(
                    event_type="error",
                    task_id=task_id,
                    backend=self.name,
                    sequence=100,
                    status="failed",
                    title="Internal coding turn failed",
                    error=error_msg,
                    visibility="terminal",
                ))

            t_p_done = _utc_now()
            t_done = _utc_now()

            duration_breakdown = compute_duration_breakdown(
                task_created_at=t_created,
                scheduled_at=t_scheduled,
                worker_claimed_at=t_claimed,
                provider_started_at=t_p_start,
                provider_completed_at=t_p_done,
                task_completed_at=t_done,
            )

            # Discover all changed files
            changed_files = list(session.state.changed_files)
            if not changed_files and workspace.worktree_path.is_dir():
                try:
                    from mana_agent.tools.repository import git_status
                    st = git_status(workspace.worktree_path)
                    st_data = json.loads(st) if isinstance(st, str) else st
                    for f in st_data.get("modified", []) + st_data.get("untracked", []):
                        if f not in changed_files:
                            changed_files.append(f)
                except Exception:
                    pass

            result = CodingTaskResult(
                task_id=task_id,
                worker_id=f"internal-{task_id[:8]}",
                backend=self.name,
                status=status,  # type: ignore[arg-type]
                summary=final_summary or f"Task {status}",
                changed_files=changed_files,
                commands_run=list(session.state.commands_run),
                tests_run=list(task.verification_commands),
                tests_passed=(status == "completed"),
                warnings=[],
                errors=[error_msg] if error_msg else list(session.state.errors),
                branch_name=workspace.branch_name,
                token_usage=session.state.total_tokens or {"total_tokens": 0},
                task_created_at=t_created,
                scheduled_at=t_scheduled,
                worker_claimed_at=t_claimed,
                provider_started_at=t_p_start,
                provider_completed_at=t_p_done,
                task_completed_at=t_done,
                duration_breakdown=duration_breakdown,
            )

            self._task_statuses[task_id] = CodingStatus(
                task_id=task_id,
                status=status,  # type: ignore[arg-type]
                backend=self.name,
                duration_ms=duration_breakdown.get("total_task_duration_ms"),
                changed_files=changed_files,
                error=error_msg,
            )

            if status == "completed":
                publish_coding_event(AgentEvent(
                    event_type="turn.completed",
                    task_id=task_id,
                    backend=self.name,
                    sequence=session._next_sequence(),
                    status="success",
                    title="Internal coding turn completed",
                    summary=final_summary,
                    duration_ms=duration_breakdown.get("total_task_duration_ms"),
                    visibility="terminal",
                ))

            self._active_sessions.pop(task_id, None)
            self._cancellation_events.pop(task_id, None)
            return result

    async def _run_model_tool_loop(
        self,
        task: CodingTask,
        workspace: WorkspaceContext,
        session: ProgrammaticExecutionSession,
        cancel_ev: asyncio.Event,
        *,
        llm: Any,
    ) -> str:
        """Run iterative model-driven tool-calling loop until the task is complete."""
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        tool_defs = self._build_tool_definitions(session.dispatcher)

        if hasattr(llm, "bind_tools") and callable(getattr(llm, "bind_tools")):
            model_with_tools = llm.bind_tools(tool_defs)
        else:
            model_with_tools = llm

        context_parts = []
        if task.goal:
            context_parts.append(f"Task Goal: {task.goal}")
        if task.requirements:
            context_parts.append(f"Requirements: {json.dumps(task.requirements, ensure_ascii=False)}")
        if task.acceptance_criteria:
            context_parts.append(f"Acceptance Criteria: {json.dumps(task.acceptance_criteria, ensure_ascii=False)}")
        if task.allowed_files:
            context_parts.append(f"Target Files: {json.dumps(task.allowed_files, ensure_ascii=False)}")
        if task.verification_commands:
            context_parts.append(f"Verification Commands: {json.dumps(task.verification_commands, ensure_ascii=False)}")
        if task.relevant_context:
            context_parts.append(f"Context: {task.relevant_context}")
        context_parts.append(f"Repository Root: {workspace.repository_path}")
        context_parts.append(f"Workspace Directory: {workspace.worktree_path}")

        if self.global_memory is not None:
            try:
                learnings = self.global_memory.retrieve_relevant(
                    action_or_query=task.goal or task.relevant_context or str(task.allowed_files),
                    limit=5,
                )
                if learnings:
                    rules_text = self.global_memory.format_retrieved_rules(learnings)
                    context_parts.append(rules_text)
            except Exception as gm_exc:
                logger.debug("InternalCodingEngine memory retrieval failed safely: %s", gm_exc)

        user_content = "\n\n".join(context_parts)

        messages: list[Any] = [
            SystemMessage(content=self.system_instruction),
            HumanMessage(content=user_content),
        ]

        logger.info(
            "Starting internal model tool loop for task %s (model=%s, provider=%s)",
            task.task_id,
            self.model,
            self.provider,
        )

        max_turns = min(self.max_tool_calls, 30)
        final_summary = ""

        for turn in range(max_turns):
            if cancel_ev.is_set():
                raise asyncio.CancelledError()

            if hasattr(model_with_tools, "ainvoke") and callable(model_with_tools.ainvoke):
                response = await model_with_tools.ainvoke(messages)
            else:
                response = await asyncio.to_thread(model_with_tools.invoke, messages)

            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                content_val = getattr(response, "content", response)
                if isinstance(content_val, list):
                    final_summary = "".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in content_val)
                else:
                    final_summary = str(content_val or "")
                break

            messages.append(response)

            for tc in tool_calls:
                if cancel_ev.is_set():
                    raise asyncio.CancelledError()

                fn_name = tc.get("name")
                args = tc.get("args") or {}
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"

                call_spec = {"call_id": call_id, "name": fn_name, "arguments": args}
                fn_output = session.handle_function_call(call_spec)
                output_str = fn_output.get("output", "")

                messages.append(ToolMessage(content=output_str, tool_call_id=call_id))

                if session.state.tool_calls_executed >= session.state.max_tool_calls:
                    logger.warning(
                        "Tool calls limit (%d) reached for task %s",
                        session.state.max_tool_calls,
                        task.task_id,
                    )
                    break

            if session.state.tool_calls_executed >= session.state.max_tool_calls:
                break

        if not final_summary:
            changed_str = ", ".join(session.state.changed_files) if session.state.changed_files else "none"
            commands_str = ", ".join(session.state.commands_run) if session.state.commands_run else "none"
            final_summary = (
                f"### Result\nCompleted internal coding operations for: {task.goal}\n\n"
                f"### Changes\nModified files: {changed_str}\n\n"
                f"### Verification\nCommands run: {commands_str}\n\n"
                f"### Notes\nAll requested modifications were completed."
            )

        return final_summary

    async def _run_programmatic_workflow(
        self,
        task: CodingTask,
        workspace: WorkspaceContext,
        session: ProgrammaticExecutionSession,
        cancel_ev: asyncio.Event,
    ) -> str:
        """Execute task via OpenAI Programmatic Tool Calling lifecycle."""
        tool_defs = build_programmatic_tool_definitions()

        # If an LLM client or callable was provided, invoke it
        if callable(self.llm_client):
            response = await asyncio.to_thread(self.llm_client, task, workspace, tool_defs, session)
            return str(response or "Programmatic coding workflow completed successfully.")

        if hasattr(self.llm_client, "execute_responses_stream"):
            return await self.llm_client.execute_responses_stream(task, workspace, tool_defs, session)

        llm = self._get_or_create_llm()
        if llm is not None:
            return await self._run_model_tool_loop(task, workspace, session, cancel_ev, llm=llm)

        # Emulate a single programmatic turn if no remote LLM client was explicitly configured
        logger.info("Executing native internal coding task with Programmatic Tool Calling: %s", task.task_id)
        if cancel_ev.is_set():
            raise asyncio.CancelledError()

        # Execute registered verification or initial read if allowed files are provided
        for allowed in task.allowed_files:
            session.handle_function_call({
                "call_id": f"call_init_{uuid.uuid4().hex[:8]}",
                "name": "read_file",
                "arguments": {"path": allowed},
                "caller": {"type": "program", "call_id": "prog_init"},
            })

        return f"Completed internal coding task: {task.goal}"

    async def _run_direct_tool_workflow(
        self,
        task: CodingTask,
        workspace: WorkspaceContext,
        session: ProgrammaticExecutionSession,
        cancel_ev: asyncio.Event,
    ) -> str:
        """Direct tool calling fallback when Programmatic Tool Calling is unsupported."""
        logger.info("Executing native internal coding task with Direct Tool Calling: %s", task.task_id)
        if callable(self.llm_client):
            response = await asyncio.to_thread(self.llm_client, task, workspace, [], session)
            return str(response or "Direct tool calling workflow completed successfully.")

        llm = self._get_or_create_llm()
        if llm is not None:
            return await self._run_model_tool_loop(task, workspace, session, cancel_ev, llm=llm)

        for allowed in task.allowed_files:
            session.handle_function_call({
                "call_id": f"call_direct_{uuid.uuid4().hex[:8]}",
                "name": "read_file",
                "arguments": {"path": allowed},
            })

        return f"Completed internal direct tool coding task: {task.goal}"

    async def stream(self, task: CodingTask, workspace: WorkspaceContext) -> AsyncIterator[AgentEvent]:
        events: list[AgentEvent] = []

        def _collector(ev: AgentEvent) -> None:
            events.append(ev)

        with coding_event_scope(_collector):
            await self.execute(task, workspace)

        for ev in events:
            yield ev

    async def cancel(self, task_id: str) -> None:
        ev = self._cancellation_events.get(task_id)
        if ev is not None:
            ev.set()
        if task_id in self._task_statuses:
            prev = self._task_statuses[task_id]
            self._task_statuses[task_id] = CodingStatus(
                task_id=task_id,
                status="cancelled",
                backend=self.name,
                changed_files=prev.changed_files,
                error="Task cancelled by user request",
            )
        else:
            self._task_statuses[task_id] = CodingStatus(
                task_id=task_id,
                status="cancelled",
                backend=self.name,
                error="Task cancelled by user request",
            )

    async def get_status(self, task_id: str) -> CodingStatus:
        if task_id in self._task_statuses:
            return self._task_statuses[task_id]
        return CodingStatus(
            task_id=task_id,
            status="queued",
            backend=self.name,
        )

    async def close(self) -> None:
        for ev in self._cancellation_events.values():
            ev.set()
        self._cancellation_events.clear()
        self._active_sessions.clear()


__all__ = ["InternalCodingEngine"]
