"""Mana-Agent Native Internal Coding Engine.

Executes coding tasks using Mana-Agent's native AI model orchestration and
registered development tools. Supports OpenAI Programmatic Tool Calling
via the Responses API lifecycle when supported, and direct tool calling
as a graceful fallback for unsupported models.
"""

from __future__ import annotations

import asyncio
import hashlib
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
from mana_agent.coding.scheduler import (
    DependencyGraph,
    OperationKind,
    ParallelToolExecutor,
    classify_tool_operation,
    extract_target_paths,
)
from mana_agent.tools.batch_file_ops import (
    batch_read_files,
    batch_write_files,
    batch_apply_patches,
    batch_search_files,
    batch_list_files,
    batch_move_files,
    batch_delete_files,
)
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
        execution_supervisor: Any | None = None,
        taskboard: Any | None = None,
        max_concurrency: int = 10,
        batch_operations_enabled: bool = True,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300,
        parallel_scheduling_enabled: bool = True,
        batch_manager: Any | None = None,
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
        self.execution_supervisor = execution_supervisor
        self.taskboard = taskboard
        self.max_concurrency = max(1, min(int(max_concurrency or 10), 32))
        self.batch_operations_enabled = batch_operations_enabled
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self.parallel_scheduling_enabled = parallel_scheduling_enabled

        self.executor = ParallelToolExecutor(
            max_concurrency=self.max_concurrency,
            cache_enabled=self.cache_enabled,
            cache_ttl_seconds=self.cache_ttl_seconds,
        )

        if global_memory is not None:
            self.global_memory = global_memory
        else:
            try:
                from mana_agent.memory.learning import get_global_memory
                self.global_memory = get_global_memory()
            except Exception:
                self.global_memory = None

        if batch_manager is not None:
            self.batch_manager = batch_manager
        else:
            try:
                from mana_agent.batch.manager import BatchLifecycleManager
                self.batch_manager = BatchLifecycleManager()
            except Exception:
                self.batch_manager = None

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

            if self.parallel_scheduling_enabled and len(tool_calls) > 1:
                # Parallel tool execution with dependency-aware scheduling
                graph = DependencyGraph.build(tool_calls)

                has_writes = any(n.kind == OperationKind.WRITE for n in graph.nodes.values())
                has_exec = any(n.kind == OperationKind.EXECUTE for n in graph.nodes.values())
                if has_exec:
                    current_phase = "verification"
                elif has_writes:
                    current_phase = "implementation"
                else:
                    current_phase = "exploration"

                publish_coding_event(AgentEvent(
                    event_type="parallel.group.started",
                    task_id=task.task_id,
                    backend=self.name,
                    sequence=session._next_sequence(),
                    title=f"Parallel tool group ({len(tool_calls)} operations)",
                    summary=f"Executing {len(tool_calls)} operations in parallel · Phase: {current_phase}",
                    visibility="progress",
                    semantic_kind="tool_execution",
                    payload={"phase": current_phase, "count": len(tool_calls), "concurrency": self.max_concurrency},
                ))

                def _on_parallel_event(event_name: str, data: dict[str, Any]) -> None:
                    if event_name == "tool.started":
                        cid = data["call_id"]
                        tname = data["tool_name"]
                        args = data.get("arguments", {})
                        session.state.tool_calls_executed += 1
                        try:
                            from mana_agent.commands.ui_helpers import emit_tool_event
                            emit_tool_event("start", tname, args=json.dumps(args, ensure_ascii=False, default=str), event_id=cid)
                        except Exception:
                            pass
                        publish_coding_event(AgentEvent(
                            event_type="tool.call.started",
                            task_id=task.task_id,
                            backend=self.name,
                            sequence=session._next_sequence(),
                            title=f"Tool call: {tname}",
                            tool_name=tname,
                            visibility="progress",
                            semantic_kind="tool_execution",
                            payload={"call_id": cid, "arguments": args, "active_concurrency": data.get("active_concurrency", 1)},
                        ))
                    elif event_name == "tool.completed":
                        cid = data["call_id"]
                        tname = data["tool_name"]
                        dur = data.get("duration_ms", 0)
                        res = data.get("result", {})
                        if tname in {"write_file", "create_file", "edit_file", "multi_edit_file", "delete_file"}:
                            cp = str(data.get("arguments", {}).get("path") or "")
                            if cp and cp not in session.state.changed_files:
                                session.state.changed_files.append(cp)
                        elif tname in {"apply_patch", "apply_patch_batch", "batch_write_files", "batch_apply_patches", "batch_move_files"}:
                            for p in res.get("touched_files") or res.get("changed_files") or res.get("files_changed") or []:
                                if p not in session.state.changed_files:
                                    session.state.changed_files.append(str(p))
                        elif tname == "batch_delete_files":
                            for p in res.get("deleted_files") or []:
                                if p not in session.state.changed_files:
                                    session.state.changed_files.append(str(p))
                        elif tname in {"run_command", "run_script_once", "shell"}:
                            cmd = str(data.get("arguments", {}).get("command") or data.get("arguments", {}).get("script") or "")
                            if cmd:
                                session.state.commands_run.append(cmd)

                        try:
                            from mana_agent.commands.ui_helpers import emit_tool_event
                            emit_tool_event("end", tname, duration=dur / 1000.0, event_id=cid)
                        except Exception:
                            pass
                        publish_coding_event(AgentEvent(
                            event_type="tool.call.completed",
                            task_id=task.task_id,
                            backend=self.name,
                            sequence=session._next_sequence(),
                            status="success",
                            title=f"Tool call: {tname} (success)",
                            tool_name=tname,
                            duration_ms=dur,
                            visibility="progress",
                            semantic_kind="tool_execution",
                            payload={"call_id": cid, "result": res},
                        ))
                    elif event_name in {"tool.failed", "tool.skipped"}:
                        cid = data["call_id"]
                        tname = data["tool_name"]
                        dur = data.get("duration_ms", 0)
                        err = data.get("error", "")
                        session.state.errors.append(f"{tname}: {err}")
                        try:
                            from mana_agent.commands.ui_helpers import emit_tool_event
                            emit_tool_event("error", tname, duration=dur / 1000.0, error=err, event_id=cid)
                        except Exception:
                            pass
                        publish_coding_event(AgentEvent(
                            event_type="tool.call.failed",
                            task_id=task.task_id,
                            backend=self.name,
                            sequence=session._next_sequence(),
                            status="failed",
                            title=f"Tool call: {tname} (failed)",
                            tool_name=tname,
                            duration_ms=dur,
                            error=err,
                            visibility="progress",
                            semantic_kind="tool_execution",
                            payload={"call_id": cid, "error": err},
                        ))
                        if self.global_memory is not None:
                            try:
                                self.global_memory.record_failure(
                                    category="tool",
                                    attempted_action=f"{tname} {cid}",
                                    error=err,
                                    metadata={"tool_name": tname, "task_id": task.task_id},
                                )
                            except Exception:
                                pass

                turn_results = await self.executor.execute_turn(
                    graph,
                    session.dispatcher,
                    on_event=_on_parallel_event,
                    cancel_ev=cancel_ev,
                )

                publish_coding_event(AgentEvent(
                    event_type="parallel.group.completed",
                    task_id=task.task_id,
                    backend=self.name,
                    sequence=session._next_sequence(),
                    title=f"Parallel tool group finished ({len(tool_calls)} operations)",
                    summary=f"Completed {len(tool_calls)} operations in parallel",
                    visibility="progress",
                    semantic_kind="tool_execution",
                    payload={"count": len(tool_calls)},
                ))

                # Append ToolMessages preserving original LLM call order
                for tc in tool_calls:
                    call_id = tc.get("id") or str(tc.get("call_id") or "")
                    res_entry = turn_results.get(call_id)
                    output_str = res_entry.get("output", "") if res_entry else json.dumps({"ok": False, "error": "Not executed"})
                    messages.append(ToolMessage(content=output_str, tool_call_id=call_id))

                # Checkpoint with supervisor if present
                if self.execution_supervisor is not None and hasattr(self.execution_supervisor, "record_action"):
                    try:
                        self.execution_supervisor.record_action(
                            task_id=task.task_id,
                            action_name="parallel_tools_turn",
                            changed_files=list(session.state.changed_files),
                        )
                    except Exception:
                        pass

                # Update TaskBoard progress if present
                if self.taskboard is not None and hasattr(self.taskboard, "update_task"):
                    try:
                        self.taskboard.update_task(
                            task_id=task.task_id,
                            progress_summary=f"Executed {len(tool_calls)} operations ({current_phase})",
                            changed_files=list(session.state.changed_files),
                        )
                    except Exception:
                        pass

            else:
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
        if len(task.allowed_files) > 1 and self.batch_operations_enabled:
            session.handle_function_call({
                "call_id": f"call_batch_init_{uuid.uuid4().hex[:8]}",
                "name": "batch_read_files",
                "arguments": {"files": list(task.allowed_files)},
                "caller": {"type": "program", "call_id": "prog_init"},
            })
        else:
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

        if len(task.allowed_files) > 1 and self.batch_operations_enabled:
            session.handle_function_call({
                "call_id": f"call_direct_batch_{uuid.uuid4().hex[:8]}",
                "name": "batch_read_files",
                "arguments": {"files": list(task.allowed_files)},
            })
        else:
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

    def _get_current_git_commit(self, repo_root: str | Path) -> str:
        try:
            import subprocess
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return ""

    def submit_code_analysis_batch(
        self,
        task: CodingTask,
        files: list[str],
        prompt: str,
        *,
        workspace: WorkspaceContext | None = None,
        instructions: str = "",
        model: str = "",
    ) -> Any:
        """Submit a bulk code analysis/review batch via OpenAI Batch API."""
        if self.batch_manager is None:
            raise RuntimeError("BatchLifecycleManager is not available on InternalCodingEngine")

        from mana_agent.batch.models import BatchItemRequest

        target_model = model or self.model or "gpt-4o"
        raw_root = (
            getattr(workspace, "worktree_path", None)
            or getattr(workspace, "repository_path", None)
            or getattr(workspace, "repo_root", None)
            or getattr(task, "workspace_context", None)
            or "."
        )
        if hasattr(raw_root, "repo_root"):
            raw_root = raw_root.repo_root
        elif hasattr(raw_root, "worktree_path"):
            raw_root = raw_root.worktree_path
        repo_root = Path(raw_root)
        current_commit = self._get_current_git_commit(repo_root)

        items: list[BatchItemRequest] = []
        for idx, file_rel in enumerate(files):
            file_path = repo_root / file_rel
            content = ""
            file_sha = ""
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    file_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                except Exception as exc:
                    logger.warning("Could not read file %s for batch request: %s", file_path, exc)

            custom_id = f"{task.task_id}-review-{idx}-{Path(file_rel).name}"
            req_prompt = (
                f"{instructions}\n\nFile: {file_rel}\n```\n{content}\n```\n\nTask: {prompt}"
                if content
                else f"{instructions}\n\nFile: {file_rel}\n\nTask: {prompt}"
            )

            # Build request body matching the configured endpoint (/v1/responses preferred)
            if self.batch_manager.config.endpoint == "/v1/chat/completions":
                body = {
                    "model": target_model,
                    "messages": [
                        {"role": "system", "content": self.system_instruction},
                        {"role": "user", "content": req_prompt},
                    ],
                }
                url = "/v1/chat/completions"
            else:
                body = {
                    "model": target_model,
                    "input": req_prompt,
                }
                url = "/v1/responses"

            items.append(
                BatchItemRequest(
                    custom_id=custom_id,
                    body=body,
                    url=url,
                    origin_task_id=task.task_id,
                    file_path=file_rel,
                    expected_sha256=file_sha,
                    target_revision=current_commit,
                )
            )

        job = self.batch_manager.submit_batch(
            items,
            model=target_model,
            endpoint=self.batch_manager.config.endpoint,
            task_id=task.task_id,
            idempotency_key=f"analysis-{task.task_id}-{len(files)}",
            git_commit_sha=current_commit,
        )

        if self.taskboard is not None:
            try:
                tb_item = self.taskboard.get_task(task.task_id)
                if tb_item and hasattr(tb_item, "queue_job_ids"):
                    if job.batch_id not in tb_item.queue_job_ids:
                        tb_item.queue_job_ids.append(job.batch_id)
                        self.taskboard.save()
            except Exception:
                pass

        return job

    def retrieve_and_apply_batch_results(
        self,
        batch_id: str,
        workspace: WorkspaceContext | Any,
    ) -> list[dict[str, Any]]:
        """Retrieve results from OpenAI Batch API, enforcing revision fencing against file/repo drift."""
        if self.batch_manager is None:
            raise RuntimeError("BatchLifecycleManager is not available on InternalCodingEngine")

        from mana_agent.batch.manager import BatchLifecycleManager

        job, results = self.batch_manager.finalize_batch(batch_id)
        items_map = {it.custom_id: it for it in self.batch_manager.store.get_items(batch_id)}

        raw_root = (
            getattr(workspace, "worktree_path", None)
            or getattr(workspace, "repository_path", None)
            or getattr(workspace, "repo_root", None)
            or "."
        )
        repo_root = Path(raw_root)
        current_commit = self._get_current_git_commit(repo_root)

        applied: list[dict[str, Any]] = []

        for res in results:
            item = items_map.get(res.custom_id)
            if not item:
                continue

            current_content: str | None = None
            if item.file_path:
                fp = repo_root / item.file_path
                if fp.is_file():
                    try:
                        current_content = fp.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        pass

            # Enforce revision fencing check
            fencing_ok = BatchLifecycleManager.verify_revision_fencing(
                item,
                current_file_content=current_content,
                current_git_commit=current_commit,
            )

            if not fencing_ok:
                logger.warning(
                    "Revision fencing rejected batch result for %s: concurrent workspace drift detected.",
                    item.file_path,
                )
                applied.append(
                    {
                        "custom_id": res.custom_id,
                        "file_path": item.file_path,
                        "status": "stale_revision_rejected",
                        "error": "Workspace or file revision changed while batch was processing.",
                    }
                )
                continue

            # Extract output text
            output_text = ""
            if res.response_body:
                # /v1/responses format
                output_text = res.response_body.get("output_text") or ""
                # /v1/chat/completions format
                if not output_text and "choices" in res.response_body:
                    choices = res.response_body.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        msg = choices[0].get("message") or {}
                        output_text = msg.get("content") or ""

            applied.append(
                {
                    "custom_id": res.custom_id,
                    "file_path": item.file_path,
                    "status": "verified_and_ready",
                    "output_text": output_text,
                    "response_body": res.response_body,
                    "ok": res.ok,
                }
            )

        return applied


__all__ = ["InternalCodingEngine"]
