"""Turn orchestration for AgentChatGateway.

Extracted from the old chat CLI console path so auto-chat, coding agent, and
model-driven routing exist once in the gateway for every frontend.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mana_agent.multi_agent.routing.agent_decision import AgentDecision, AgentDecisionEngine
from mana_agent.multi_agent.runtime.agent_session import route_for_turn
from mana_agent.multi_agent.runtime.auto_chat import (
    AutoChatMode,
    AutoChatSessionState,
    classify_auto_chat_intent,
    is_plan_execution_request,
    load_auto_chat_state,
    resolve_auto_followup,
    save_auto_chat_state,
)
from mana_agent.multi_agent.runtime.small_direct_edit import handle_small_direct_edit
from mana_agent.search.config import SearchConfig
from mana_agent.search.models import SearchDecision, SearchQuery
from mana_agent.search.router import SearchRouter
from mana_agent.workspaces.preparation import RepositoryPreparationError

logger = logging.getLogger(__name__)


def _conversation_prompt(session_state: dict[str, Any], current_message: str) -> str:
    """Build one chronological conversation prompt with the current message once."""
    messages = list(session_state.get("messages") or [])
    prior = messages[:-1] if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == current_message else messages
    prior = [item for item in prior if item.get("role") in {"user", "assistant", "tool"}][-40:]
    if not prior:
        return current_message
    lines = ["Active conversation history (chronological):"]
    labels = {"user": "User", "assistant": "Assistant", "tool": "Tool result"}
    for item in prior:
        lines.append(f"{labels[str(item.get('role'))]}: {str(item.get('content') or '')}")
    followup_memory = str(session_state.get("followup_memory_context") or "").strip()
    if followup_memory:
        lines.extend(["", "Relevant shared memory:", followup_memory])
    lines.extend(["", "Current user message:", current_message])
    return "\n".join(lines)[-40000:]


@dataclass
class ChatTurnResult:
    """Structured result of one gateway-owned chat turn."""

    answer: str
    sources: list[Any] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    mode: str | None = None
    flow_id: str | None = None
    decision: Any | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    trace: list[Any] = field(default_factory=list)
    used_coding_agent: bool = False
    auto_chat_mode: str | None = None


def agent_decision_llm(ask_service: Any) -> Any:
    ask_agent = getattr(ask_service, "ask_agent", None)
    if ask_agent is not None and hasattr(ask_agent, "llm"):
        return getattr(ask_agent, "llm")
    qna_chain = getattr(ask_service, "qna_chain", None)
    return getattr(qna_chain, "llm", None)


def decide_chat_route(
    *,
    ask_service: Any,
    question: str,
    root: Path,
    memory_context: str = "",
) -> AgentDecision:
    llm = agent_decision_llm(ask_service)
    engine = AgentDecisionEngine(llm=llm, enable_fallback=False)
    return engine.decide(
        user_request=question,
        repo_context=f"Repository root: {root}",
        memory_context=memory_context,
        command_hint="chat",
    )


def auto_chat_mode_from_agent_decision(
    decision: AgentDecision, fallback: AutoChatMode
) -> AutoChatMode:
    if decision.intent == "edit" or decision.code_editing_needed:
        return AutoChatMode.EDIT
    if decision.intent == "plan":
        return AutoChatMode.PLAN_ONLY
    if decision.intent == "analyze":
        return AutoChatMode.ANALYZE
    if decision.intent == "verify":
        return AutoChatMode.VERIFY
    if decision.intent == "review":
        return AutoChatMode.REVIEW
    return fallback


# Tools that live on ChatService / AskAgent (connectors, MCP, research), not CodingAgent.
_CONNECTOR_OR_AUTO_CHAT_TOOLS = frozenset(
    {
        "web_search",
        "github_search",
        "email_accounts_list",
        "email_search",
        "email_read",
        "email_thread_read",
    }
)
_MUTATION_TOOLS = frozenset(
    {
        "edit_file",
        "multi_edit_file",
        "apply_patch",
        "apply_patch_batch",
        "write_file",
        "create_file",
        "delete_file",
        "document_create",
        "document_update",
        "document_delete",
        "move_file",
    }
)


def _selected_tools(decision: AgentDecision | None) -> list[str]:
    if decision is None:
        return []
    return [str(t).strip() for t in (decision.selected_tools or []) if str(t).strip()]


def is_auto_chat_connector_turn(
    *,
    decision: AgentDecision | None,
    auto_chat_mode: AutoChatMode,
    question: str = "",
) -> bool:
    """True when the turn should use auto-chat / ChatService.ask (email, MCP, etc.).

    Matches TUI behavior: general queries like "check my latest gmail" must not
    enter the CodingAgent planner; they go through agent-tools ask which exposes
    email_* / browser_* / MCP tools.
    """
    tools = _selected_tools(decision)
    if any(t.startswith("browser_") or t.startswith("email_") for t in tools):
        return True
    if any(t in _CONNECTOR_OR_AUTO_CHAT_TOOLS for t in tools):
        return True
    if any(t.startswith("mcp_") or t == "mcp" for t in tools):
        return True
    # Mode-based: answer/review/verify/analyze are auto-chat, not coding planner
    if auto_chat_mode in (
        AutoChatMode.ANSWER_ONLY,
        AutoChatMode.REVIEW,
        AutoChatMode.VERIFY,
        AutoChatMode.ANALYZE,
    ):
        # Still coding if model explicitly selected mutation tools
        if any(t in _MUTATION_TOOLS for t in tools):
            return False
        if decision is not None and (
            decision.code_editing_needed or decision.intent in ("edit", "plan")
        ):
            return False
        return True
    return False


def should_use_coding_agent_turn(
    *,
    coding_agent_available: bool,
    agent_tools: bool,
    edit_request: bool,
    plan_trigger_request: bool,
    force_plan_only_response: bool,
    has_pending_prechecklist: bool,
    coding_agent_is_custom: bool,
    general_coding_agent_turns: bool = False,
    decision: AgentDecision | None = None,
    auto_chat_mode: AutoChatMode | None = None,
    question: str = "",
) -> bool:
    """Decide coding-agent vs auto-chat (ChatService.ask) path.

    Default ``general_coding_agent_turns=False`` matches TUI: only edit/plan
    (and explicit mutation) use CodingAgent. Connector work (Gmail, MCP, web)
    stays on auto-chat agent-tools.
    """
    if not coding_agent_available:
        return False

    mode = auto_chat_mode or AutoChatMode.ANSWER_ONLY
    if is_auto_chat_connector_turn(
        decision=decision, auto_chat_mode=mode, question=question
    ):
        return False

    if edit_request or plan_trigger_request or force_plan_only_response or has_pending_prechecklist:
        return True

    tools = _selected_tools(decision)
    if any(t in _MUTATION_TOOLS for t in tools):
        return True
    if decision is not None and (
        decision.code_editing_needed or decision.intent in ("edit", "plan")
    ):
        return True
    if mode in (AutoChatMode.EDIT, AutoChatMode.PLAN_ONLY):
        return True

    if not general_coding_agent_turns:
        return False
    route = route_for_turn(
        coding_agent_available=coding_agent_available,
        agent_tools=agent_tools,
        coding_agent_is_custom=coding_agent_is_custom,
        reason="gateway turn routing",
    )
    return route.uses_coding_agent


def load_analysis_context(root: Path | str) -> str | None:
    """Load compact agent_context.json produced by /analyze."""
    import json as _json

    try:
        from mana_agent.workspaces.paths import repository_analysis_dir, repository_id_for_path
    except Exception:
        return None

    root_path = Path(root)
    ctx_path = repository_analysis_dir(repository_id_for_path(root_path)) / "agent_context.json"
    if not ctx_path.exists():
        return None
    try:
        context = _json.loads(ctx_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(context, dict):
        return None

    lines: list[str] = ["Project analysis (from .mana/analyze/agent_context.json):"]
    summary = str(context.get("project_summary", "") or "").strip()
    if summary:
        lines.append(f"- Summary: {summary}")
    stack = ", ".join(context.get("detected_stack", []) or [])
    if stack:
        lines.append(f"- Stack: {stack}")
    arch = str(context.get("architecture_summary", "") or "").strip()
    if arch:
        lines.append(f"- Architecture: {arch[:600]}")
    workflow = str(context.get("agent_workflow", "") or "").strip()
    if workflow:
        lines.append(f"- Agent workflow: {workflow[:400]}")
    risks = context.get("risks", []) or []
    if risks:
        lines.append("- Top risks:")
        for item in risks[:5]:
            if isinstance(item, dict):
                loc = f" ({item.get('file')}:{item.get('line')})" if item.get("file") else ""
                lines.append(f"  - [{item.get('severity', 'info')}] {item.get('title', '')}{loc}")
    tasks = context.get("recommended_tasks", []) or []
    if tasks:
        lines.append("- Recommended tasks:")
        for item in tasks[:5]:
            if isinstance(item, dict):
                lines.append(f"  - {item.get('title', '')}")
    when = str(context.get("last_analyzed_at", "") or "").strip()
    if when:
        lines.append(f"- Last analyzed at: {when}")
    return "\n".join(lines)


def run_web_research_answer(
    *,
    ask_service: Any,
    question: str,
    root: Path,
    decision: AgentDecision,
) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    config = SearchConfig.from_env()
    router = SearchRouter(root=str(root), llm=agent_decision_llm(ask_service), config=config)
    queries: list[SearchQuery] = []
    web_query = str((decision.tool_inputs.get("web_search") or {}).get("query") or "").strip()
    if "web_search" in decision.selected_tools:
        queries.append(SearchQuery(query=web_query or question, target="web"))
    github_input = decision.tool_inputs.get("github_search") or {}
    github_query = str(github_input.get("query") or web_query or question).strip()
    if "github_search" in decision.selected_tools:
        queries.append(
            SearchQuery(
                query=github_query,
                target="github",
                github_kind=str(github_input.get("github_kind") or "repositories"),  # type: ignore[arg-type]
                repo=str(github_input.get("repo") or "").strip() or None,
            )
        )
    if not queries:
        queries.append(SearchQuery(query=question, target="web"))
    targets = list(dict.fromkeys(query.target for query in queries))
    forced_decision = SearchDecision(
        needs_search=True,
        targets=targets,  # type: ignore[arg-type]
        reason=decision.reasoning_summary,
        confidence=decision.confidence,
        queries=queries,
        reuse_memory_first=True,
        max_results=config.max_results,
        mode="both" if set(targets) == {"web", "github"} else targets[0],  # type: ignore[arg-type]
    )
    result = router.run(
        user_query=question,
        repo_context=f"Repository root: {root}",
        task_id=None,
        decision_override=forced_decision,
    )
    context = result.context_block(
        max_results=config.max_injected_results,
        max_words=config.max_summary_words,
    )
    if not context:
        warning_text = "; ".join(result.warnings)
        return (
            (
                "No external search results were available for that request."
                + (f"\n\nSearch warnings: {warning_text}" if warning_text else "")
            ),
            [],
            [
                {
                    "tool_name": "+".join(decision.selected_tools) or "external_search",
                    "status": "ok",
                    "args_summary": "; ".join(query.query for query in queries),
                    "result_summary": "0 result(s)",
                }
            ],
        )
    qna_chain = getattr(ask_service, "qna_chain", None)
    if qna_chain is not None and hasattr(qna_chain, "run"):
        answer = qna_chain.run(question=question, context=context)
    else:
        answer = context
    sources: list[dict[str, str]] = []
    for item in [*result.results, *result.memory_hits]:
        title = str(getattr(item, "title", "") or "").strip()
        url = str(getattr(item, "url", "") or "").strip()
        if url:
            sources.append({"title": title, "url": url})
    trace = [
        {
            "tool_name": "web_search",
            "status": "ok",
            "args_summary": "; ".join(query.query for query in queries),
            "result_summary": (
                f"{len(result.results)} result(s), {len(result.memory_hits)} memory hit(s)"
                + (f", warnings={len(result.warnings)}" if result.warnings else "")
            ),
        }
    ]
    return str(answer or "").strip(), sources, trace


def _save_auto_chat_turn_state(
    root: Path,
    *,
    mode: AutoChatMode,
    task: str,
    answer_text: str,
    sources: list[Any] | None = None,
    changed_files: list[str] | None = None,
    verification: str = "",
) -> AutoChatSessionState:
    source_paths: list[str] = []
    for item in sources or []:
        if isinstance(item, dict):
            value = item.get("path") or item.get("file") or item.get("source")
        else:
            value = str(item)
        text = str(value or "").strip()
        if text and text not in source_paths:
            source_paths.append(text)
    state = AutoChatSessionState(
        last_mode=mode.value,
        last_task=task,
        relevant_files=source_paths[:12],
        changed_files=[str(item) for item in (changed_files or []) if str(item).strip()][:12],
        verification=verification,
        summary=str(answer_text or "").strip()[:800],
    )
    try:
        save_auto_chat_state(root, state)
    except Exception as exc:
        logger.debug("Failed to save auto-chat state: %s", exc)
    return state


def _serialize_tool_traces(resp: Any) -> list[dict[str, Any]]:
    """Normalize AskResponseWithTrace / list traces into JSON-serializable dicts."""
    raw = getattr(resp, "trace", None)
    if raw is None and isinstance(resp, dict):
        raw = resp.get("trace")
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for item in list(raw)[:80]:
        if item is None:
            continue
        if hasattr(item, "to_dict"):
            try:
                payload = item.to_dict()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                out.append(payload)
                continue
        if isinstance(item, dict):
            out.append(dict(item))
            continue
        name = str(getattr(item, "tool_name", None) or getattr(item, "name", "") or "tool")
        out.append(
            {
                "tool_name": name,
                "args_summary": str(getattr(item, "args_summary", "") or "")[:500],
                "duration_ms": float(getattr(item, "duration_ms", 0.0) or 0.0),
                "status": str(getattr(item, "status", "ok") or "ok"),
                "output_preview": str(getattr(item, "output_preview", "") or "")[:4000],
            }
        )
    return out


def process_chat_turn(
    *,
    root: Path,
    text: str,
    chat_service: Any,
    ask_service: Any,
    coding_agent: Any | None,
    config: Any,
    session_state: dict[str, Any],
    coding_agent_is_custom: bool = False,
    resolved_k: int = 6,
    coding_agent_max_steps: int = 200,
    index_dir: str | Path | None = None,
    index_dirs: list[str | Path] | None = None,
    event_sink: Callable[..., None] | None = None,
    callbacks: list[Any] | None = None,
    agent_decision: AgentDecision | None = None,
    coding_workspace_preparer: Callable[[], Any] | None = None,
) -> ChatTurnResult:
    """Run one model-driven chat turn (non-UI).

    Session state keys (mutated):
      history, active_flow_id, auto_chat_state, analysis_context
    """
    root = Path(root).expanduser().resolve()
    original_question = str(text or "").strip()
    if not original_question:
        return ChatTurnResult(answer="", error="empty_question")

    def _emit(event_type: str, title: str, **kwargs: Any) -> None:
        if callable(event_sink):
            try:
                event_sink(event_type, title, **kwargs)
            except Exception:
                pass

    # Auto-chat follow-up resolution
    auto_chat_state = session_state.get("auto_chat_state")
    if not isinstance(auto_chat_state, AutoChatSessionState):
        try:
            auto_chat_state = load_auto_chat_state(root)
        except Exception:
            auto_chat_state = AutoChatSessionState()
        session_state["auto_chat_state"] = auto_chat_state

    question = resolve_auto_followup(original_question, auto_chat_state)
    model_question = _conversation_prompt(session_state, question)
    active_flow_id = session_state.get("active_flow_id")
    if isinstance(active_flow_id, str):
        active_flow_id = active_flow_id.strip() or None
    else:
        active_flow_id = None

    agent_tools = bool(getattr(config, "agent_tools", True))
    auto_execute_plan = bool(getattr(config, "auto_execute_plan", True))
    auto_execute_max_passes = int(getattr(config, "auto_execute_max_passes", 4) or 4)
    auto_continue = bool(getattr(config, "auto_continue", True))
    dir_mode = bool(getattr(config, "dir_mode", False))
    agent_timeout_seconds = int(getattr(config, "agent_timeout_seconds", 30) or 30)
    # Flow routing context for decision
    flow_routing_context = ""
    if active_flow_id and coding_agent is not None and hasattr(coding_agent, "flow_summary"):
        try:
            flow_summary = coding_agent.flow_summary(active_flow_id)
            if flow_summary:
                flow_routing_context = (
                    "Active coding flow (decide flow_action explicitly):\n"
                    + json.dumps(flow_summary, ensure_ascii=False, sort_keys=True, default=str)[:4000]
                )
        except Exception:
            pass

    analysis_context = session_state.get("analysis_context")
    if analysis_context is None:
        analysis_context = load_analysis_context(root)
        session_state["analysis_context"] = analysis_context

    _emit("agent.decision", "Decision routing", status="running")
    if agent_decision is None:
        try:
            agent_decision = decide_chat_route(
                ask_service=ask_service,
                question=question,
                root=root,
                memory_context="\n\n".join(
                    part for part in (flow_routing_context, _conversation_prompt(session_state, question)) if part
                ),
            )
        except Exception as exc:
            logger.exception("gateway process_turn decision failed")
            return ChatTurnResult(
                answer="",
                error=f"Model decision failed: {exc}. No fallback action was executed.",
            )

    if (
        active_flow_id
        and (agent_decision.intent == "edit" or agent_decision.code_editing_needed)
        and agent_decision.flow_action == "none"
    ):
        return ChatTurnResult(
            answer=(
                "Model decision failed: flow_action. "
                "No repository action was executed. The active coding flow requires "
                "a validated continue or new decision."
            ),
            error="flow_action_required",
            decision=agent_decision,
        )

    if active_flow_id and agent_decision.flow_action == "new":
        if coding_agent is not None and hasattr(coding_agent, "reset_flow"):
            try:
                coding_agent.reset_flow(active_flow_id)
            except Exception:
                pass
        active_flow_id = None
        session_state["active_flow_id"] = None

    auto_chat_mode = auto_chat_mode_from_agent_decision(
        agent_decision,
        classify_auto_chat_intent(question),
    )
    if is_plan_execution_request(question):
        auto_chat_mode = AutoChatMode.EDIT

    # Small direct edit (docs-only)
    small = handle_small_direct_edit(root, question)
    if small.handled:
        _save_auto_chat_turn_state(
            root,
            mode=auto_chat_mode,
            task=original_question,
            answer_text=small.answer,
            sources=[{"path": p} for p in small.changed_files],
            changed_files=list(small.changed_files),
            verification="skipped_docs_only" if small.ok else "minimal_check_failed",
        )
        return ChatTurnResult(
            answer=str(small.answer or ""),
            sources=[{"path": p} for p in small.changed_files],
            changed_files=list(small.changed_files),
            mode="small-direct-edit",
            flow_id=active_flow_id,
            decision=agent_decision,
            auto_chat_mode=auto_chat_mode.value,
            warnings=[] if small.ok else [str(small.error or "edit failed")],
            payload={"small_direct_edit": True},
        )

    # External research tools selected by model
    selected = list(agent_decision.selected_tools or [])
    if any(t in selected for t in ("web_search", "github_search")) and not any(
        str(t).startswith("browser_") for t in selected
    ):
        try:
            answer, sources, trace = run_web_research_answer(
                ask_service=ask_service,
                question=question,
                root=root,
                decision=agent_decision,
            )
            _save_auto_chat_turn_state(
                root,
                mode=auto_chat_mode,
                task=original_question,
                answer_text=answer,
                sources=sources,
            )
            return ChatTurnResult(
                answer=answer,
                sources=sources,
                mode="web-research",
                flow_id=active_flow_id,
                decision=agent_decision,
                auto_chat_mode=auto_chat_mode.value,
                trace=trace,
            )
        except Exception as exc:
            logger.debug("web research path failed: %s", exc)

    edit_request = bool(
        agent_decision.intent == "edit"
        or agent_decision.code_editing_needed
        or auto_chat_mode == AutoChatMode.EDIT
    )
    plan_trigger_request = bool(
        agent_decision.intent == "plan" or auto_chat_mode == AutoChatMode.PLAN_ONLY
    )
    force_plan_only = plan_trigger_request and not is_plan_execution_request(question)
    force_auto_execute_edit = bool(coding_agent is not None and auto_execute_plan and edit_request)

    # Auto-chat routing (TUI parity): Gmail / MCP / general Q&A → ChatService.ask
    # with agent tools. Only edit/plan/mutation → CodingAgent.
    use_coding = should_use_coding_agent_turn(
        coding_agent_available=coding_agent is not None,
        agent_tools=agent_tools,
        edit_request=edit_request,
        plan_trigger_request=plan_trigger_request,
        force_plan_only_response=force_plan_only,
        has_pending_prechecklist=isinstance(session_state.get("pending_prechecklist"), dict),
        coding_agent_is_custom=coding_agent_is_custom,
        # Do NOT force all turns through CodingAgent; that breaks connector auto-chat
        # (e.g. "check my latest gmail") which needs AskAgent email_* tools.
        general_coding_agent_turns=False,
        decision=agent_decision,
        auto_chat_mode=auto_chat_mode,
        question=question,
    )

    if is_auto_chat_connector_turn(
        decision=agent_decision, auto_chat_mode=auto_chat_mode, question=question
    ):
        use_coding = False
        _emit(
            "agent.decision",
            "Auto-chat routing",
            status="success",
            message=f"mode={auto_chat_mode.value}; path=chat_service.ask",
        )

    # Classic / agent-tools ask path (auto-chat: email, MCP, web, general Q&A)
    if not use_coding or coding_agent is None:
        ask_callbacks = list(callbacks or [])
        try:
            conversation_only = bool(
                agent_decision.intent == "answer"
                and not list(agent_decision.selected_tools or [])
                and not agent_decision.code_editing_needed
            )
            if conversation_only:
                answer_text = chat_service.ask_conversation(model_question)
                resp = {
                    "answer": str(answer_text or ""),
                    "sources": [],
                    "warnings": [],
                    "mode": "route-conversation",
                    "trace": [],
                }
            elif ask_callbacks:
                resp = chat_service.ask(model_question, k=resolved_k, callbacks=ask_callbacks)
            else:
                resp = chat_service.ask(model_question, k=resolved_k)
        except TypeError:
            try:
                resp = chat_service.ask(model_question, callbacks=ask_callbacks) if ask_callbacks else chat_service.ask(model_question)
            except TypeError:
                resp = chat_service.ask(model_question)
        except Exception as exc:
            return ChatTurnResult(
                answer="",
                error=f"Chat request failed: {exc}",
                decision=agent_decision,
                auto_chat_mode=auto_chat_mode.value,
            )
        answer = str((resp.get("answer") if isinstance(resp, dict) else getattr(resp, "answer", resp)) or "").strip()
        sources = list((resp.get("sources") if isinstance(resp, dict) else getattr(resp, "sources", [])) or [])
        raw_warnings = resp.get("warnings") if isinstance(resp, dict) else getattr(resp, "warnings", [])
        warnings = [str(w).strip() for w in (raw_warnings or []) if str(w).strip()]
        raw_mode = resp.get("mode") if isinstance(resp, dict) else getattr(resp, "mode", "")
        mode_name = str(raw_mode or "").strip() or (
            "agent-tools" if agent_tools else "classic"
        )
        tool_traces = _serialize_tool_traces(resp)
        _save_auto_chat_turn_state(
            root,
            mode=auto_chat_mode,
            task=original_question,
            answer_text=answer,
            sources=sources,
        )
        hist = session_state.setdefault("history", [])
        hist.append((original_question, answer))
        session_state["history"] = hist[-12:]
        return ChatTurnResult(
            answer=answer or "(No response from agent)",
            sources=sources,
            mode=mode_name if mode_name != "classic" else (
                f"auto-chat:{auto_chat_mode.value}" if agent_tools else "classic"
            ),
            flow_id=active_flow_id,
            decision=agent_decision,
            auto_chat_mode=auto_chat_mode.value,
            warnings=warnings,
            used_coding_agent=False,
            # Real tool invocations (email_read, web_search, MCP, …) for TUI ToolCards.
            trace=tool_traces,
            payload={
                "route": "auto_chat",
                "auto_chat_mode": auto_chat_mode.value,
                "selected_tools": list(agent_decision.selected_tools or []),
                "trace": tool_traces,
                "actions_taken": tool_traces,
            },
        )

    # Coding agent path
    if coding_workspace_preparer is not None:
        try:
            coding_workspace_preparer()
        except RepositoryPreparationError as exc:
            logger.exception("gateway repository preparation failed before coding turn")
            return ChatTurnResult(
                answer="",
                error=str(exc),
                decision=agent_decision,
                auto_chat_mode=auto_chat_mode.value,
                used_coding_agent=False,
            )
    execute_plan_now = bool(
        auto_execute_plan
        and not force_plan_only
        and (plan_trigger_request or force_auto_execute_edit)
    )
    auto_execute_available = bool(
        execute_plan_now and hasattr(coding_agent, "generate_auto_execute")
    )
    request_for_generation = model_question
    if analysis_context:
        request_for_generation = f"{analysis_context}\n\n{request_for_generation}"

    timeout = min(max(agent_timeout_seconds, 60), 600)
    pending_prechecklist = session_state.get("pending_prechecklist")
    pending_source = str(session_state.get("pending_prechecklist_source") or "")
    pending_warning = str(session_state.get("pending_prechecklist_warning") or "")

    try:
        if dir_mode:
            dirs = list(index_dirs or [])
            if not dirs:
                return ChatTurnResult(
                    answer="No indexes available in dir-mode.",
                    error="missing_indexes",
                    decision=agent_decision,
                    auto_chat_mode=auto_chat_mode.value,
                )

            def _call() -> dict[str, Any]:
                if auto_execute_available:
                    return coding_agent.generate_auto_execute(
                        request_for_generation,
                        index_dirs=dirs,
                        k=resolved_k,
                        max_steps=coding_agent_max_steps,
                        timeout_seconds=timeout,
                        pass_cap=auto_execute_max_passes,
                        flow_id=active_flow_id,
                        auto_chat_mode=auto_chat_mode.value,
                        prechecklist_payload=(
                            {
                                "flow_id": active_flow_id,
                                "prechecklist": pending_prechecklist,
                                "prechecklist_source": pending_source,
                                "prechecklist_warning": pending_warning,
                            }
                            if isinstance(pending_prechecklist, dict)
                            else None
                        ),
                    )
                return coding_agent.generate_dir_mode(
                    request_for_generation,
                    index_dirs=dirs,
                    k=resolved_k,
                    max_steps=coding_agent_max_steps,
                    timeout_seconds=timeout,
                    flow_id=active_flow_id,
                    auto_chat_mode=auto_chat_mode.value,
                )
        else:
            target_index = index_dir
            if target_index is None:
                # ChatService may already own a resolved index
                target_index = getattr(chat_service, "_index_dir", None) or getattr(
                    config, "index_dir", None
                )

            def _call() -> dict[str, Any]:
                if auto_execute_available:
                    return coding_agent.generate_auto_execute(
                        request_for_generation,
                        index_dir=target_index,
                        k=resolved_k,
                        max_steps=coding_agent_max_steps,
                        timeout_seconds=timeout,
                        pass_cap=auto_execute_max_passes,
                        flow_id=active_flow_id,
                        auto_chat_mode=auto_chat_mode.value,
                        prechecklist_payload=(
                            {
                                "flow_id": active_flow_id,
                                "prechecklist": pending_prechecklist,
                                "prechecklist_source": pending_source,
                                "prechecklist_warning": pending_warning,
                            }
                            if isinstance(pending_prechecklist, dict)
                            else None
                        ),
                    )
                return coding_agent.generate(
                    request_for_generation,
                    index_dir=target_index,
                    k=resolved_k,
                    max_steps=coding_agent_max_steps,
                    timeout_seconds=timeout,
                    flow_id=active_flow_id,
                    auto_chat_mode=auto_chat_mode.value,
                )

        result: dict[str, Any] = {}
        resume_cycles = 0
        while True:
            result = _call() or {}
            if not (auto_continue and execute_plan_now and isinstance(result, dict)):
                break
            terminal_reason = str(
                result.get("auto_execute_terminal_reason", "") or ""
            ).strip().lower()
            run_status = str(result.get("run_status", "") or "").strip().lower()
            flow_from = result.get("flow_id")
            if isinstance(flow_from, str) and flow_from.strip():
                active_flow_id = flow_from.strip()
                session_state["active_flow_id"] = active_flow_id
            if terminal_reason != "pass_cap_reached" and run_status != "needs_resume":
                break
            resume_cycles += 1
            if resume_cycles > 8:
                break
            request_for_generation = (
                f"{question}\n\n[full-auto resume cycle {resume_cycles}; "
                f"continue remaining work after {terminal_reason or run_status}]"
            )

        answer = str((result or {}).get("answer", "") or "").strip()
        changed = [str(c) for c in ((result or {}).get("changed_files") or []) if str(c).strip()]
        warns = [str(w) for w in ((result or {}).get("warnings") or []) if str(w).strip()]
        flow_from = (result or {}).get("flow_id")
        if isinstance(flow_from, str) and flow_from.strip():
            active_flow_id = flow_from.strip()
            session_state["active_flow_id"] = active_flow_id

        # Clear consumed prechecklist
        if execute_plan_now:
            session_state["pending_prechecklist"] = None
            session_state["pending_prechecklist_source"] = ""
            session_state["pending_prechecklist_warning"] = ""

        _save_auto_chat_turn_state(
            root,
            mode=auto_chat_mode,
            task=original_question,
            answer_text=answer,
            changed_files=changed,
            verification=str((result or {}).get("auto_execute_terminal_reason", "") or ""),
        )
        hist = session_state.setdefault("history", [])
        hist.append((original_question, answer))
        session_state["history"] = hist[-12:]

        return ChatTurnResult(
            answer=answer or "(No response from coding agent)",
            sources=[],
            changed_files=changed,
            mode="coding-agent" + ("-auto-execute" if auto_execute_available else ""),
            flow_id=active_flow_id,
            decision=agent_decision,
            auto_chat_mode=auto_chat_mode.value,
            payload=dict(result or {}),
            warnings=warns,
            used_coding_agent=True,
        )
    except Exception as exc:
        logger.exception("gateway coding agent turn failed")
        return ChatTurnResult(
            answer="",
            error=f"Coding agent failed: {exc}",
            decision=agent_decision,
            auto_chat_mode=auto_chat_mode.value,
            used_coding_agent=True,
        )
