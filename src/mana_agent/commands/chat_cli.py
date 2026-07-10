from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any
import uuid

from .cli_internal import *
from .cli_internal import _build_project_llm_analyzer, _record_multi_agent_request
from .chat_analyze_command import (
    analyze_command_args,
    handle_analyze_command,
    is_analyze_command,
)
from mana_agent.multi_agent.runtime.auto_chat import (
    AutoChatMode,
    AutoChatSessionState,
    apply_auto_chat_tool_policy,
    classify_auto_chat_intent,
    is_plan_execution_request,
    load_auto_chat_state,
    resolve_auto_followup,
    save_auto_chat_state,
)
from mana_agent.multi_agent.runtime.agent_session import route_for_turn
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.runtime.model_levels import resolve_model_for_role
from mana_agent.multi_agent.runtime.small_direct_edit import handle_small_direct_edit
from mana_agent.multi_agent.runtime.tools_executor import build_tools_executor_with_fallback
from mana_agent.multi_agent.routing.agent_decision import AgentDecision, AgentDecisionEngine
from mana_agent.search.config import SearchConfig
from mana_agent.search.models import SearchDecision, SearchQuery
from mana_agent.search.router import SearchRouter
from mana_agent.cli.chat_ui import (
    ChatUIState,
    default_ui_mode,
    detect_skills_status,
    render_startup_header,
    render_status,
)
from mana_agent.cli.events import make_event
from mana_agent.cli.renderers import InlineChatRenderer
from mana_agent.tui.menu import NonInteractivePromptError
from mana_agent.tui.wizard import ensure_setup


_NEW_TOPIC_COMMANDS = {"/new", "/new-topic", "new topic", "new topic chat"}


def _is_new_topic_command(text: str) -> bool:
    return str(text or "").strip().lower() in _NEW_TOPIC_COMMANDS


def _load_analysis_context(root) -> str | None:
    """Load the compact ``agent_context.json`` produced by ``/analyze``.

    Returns a bounded text block suitable for prepending to chat/coding-agent
    context so later questions ("explain architecture") are grounded in the most
    recent analysis. Returns ``None`` when no analysis exists or it cannot be read.
    """
    import json
    from pathlib import Path as _Path

    from mana_agent.workspaces.paths import repository_analysis_dir, repository_id_for_path

    ctx_path = repository_analysis_dir(repository_id_for_path(root)) / "agent_context.json"
    if not ctx_path.exists():
        return None
    try:
        context = json.loads(ctx_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - context load is best-effort
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


def _planning_questions(max_questions: int) -> list[str]:
    questions = [
        "What is the concrete goal and the success criteria?",
        "What should be in scope and out of scope?",
        "What constraints, risks, or preferences should shape the implementation?",
        "What acceptance tests or verification steps should be included?",
        "Are there rollout, migration, or compatibility requirements?",
        "What output format should the final plan use?",
    ]
    limit = max(1, min(int(max_questions or 3), len(questions)))
    return questions[:limit]


def _generate_planning_question_llm(
    *,
    ask_service,
    planning_request: str,
    prior_questions: list[str],
    prior_answers: list[str],
    asked_count: int,
    max_questions: int,
) -> str:
    qna_chain = getattr(ask_service, "qna_chain", None)
    llm = getattr(qna_chain, "llm", None)
    if llm is None or not hasattr(llm, "invoke"):
        raise RuntimeError("planning question LLM is unavailable")
    prompt = (
        "Generate exactly one concise clarification question for a coding plan.\n"
        f"Request: {planning_request}\n"
        f"Question number: {asked_count + 1} of {max_questions}\n"
        "Prior Q&A:\n"
    )
    for idx, (prior_q, prior_a) in enumerate(zip(prior_questions, prior_answers), start=1):
        prompt += f"Q{idx}: {prior_q}\nA{idx}: {prior_a}\n"
    prompt += "Ask only for information that materially changes the implementation plan."
    response = llm.invoke(prompt)
    content = str(getattr(response, "content", response) or "").strip()
    if not content:
        raise RuntimeError("planning question LLM returned an empty question")
    return content


def _agent_decision_llm(ask_service):
    ask_agent = getattr(ask_service, "ask_agent", None)
    if ask_agent is not None and hasattr(ask_agent, "llm"):
        return getattr(ask_agent, "llm")
    qna_chain = getattr(ask_service, "qna_chain", None)
    return getattr(qna_chain, "llm", None)


def _decide_chat_route(*, ask_service, question: str, root: Path) -> AgentDecision:
    llm = _agent_decision_llm(ask_service)
    engine = AgentDecisionEngine(llm=llm, enable_fallback=False)
    return engine.decide(
        user_request=question,
        repo_context=f"Repository root: {root}",
        command_hint="chat",
    )


def _auto_chat_mode_from_agent_decision(decision: AgentDecision, fallback: AutoChatMode) -> AutoChatMode:
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


def _explicit_mcp_server_request(*, ask_service: Any, question: str) -> str | None:
    resolver = getattr(ask_service, "_requested_mcp_server", None)
    if not callable(resolver):
        return None
    value = resolver(question)
    return str(value).strip() or None


def _run_web_research_answer(*, ask_service, question: str, root: Path, decision: AgentDecision) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    config = SearchConfig.from_env()
    router = SearchRouter(root=str(root), llm=_agent_decision_llm(ask_service), config=config)
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


def _is_planning_question_auth_failure(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return any(marker in text for marker in ("error code: 401", "status code: 401", "incorrect api key", "unauthorized"))


def _should_use_coding_agent_turn(
    *,
    coding_agent_available: bool,
    agent_tools: bool,
    edit_request: bool,
    plan_trigger_request: bool,
    force_plan_only_response: bool,
    has_pending_prechecklist: bool,
    coding_agent_is_custom: bool,
    general_coding_agent_turns: bool = True,
) -> bool:
    """Route turns consistently with the chat banner when CodingAgent is active."""
    if not coding_agent_available:
        return False
    if edit_request or plan_trigger_request or force_plan_only_response or has_pending_prechecklist:
        return True
    if not general_coding_agent_turns:
        return False
    route = route_for_turn(
        coding_agent_available=coding_agent_available,
        agent_tools=agent_tools,
        coding_agent_is_custom=coding_agent_is_custom,
        reason="chat turn routing",
    )
    return route.uses_coding_agent


def _build_planning_instruction(
    planning_request: str,
    planning_answers: list[str],
    max_questions: int,
    *,
    questions: list[str] | None = None,
) -> str:
    lines = [
        "You are in planning mode.",
        "Produce a decision-complete implementation plan. Do not edit files.",
        "",
        f"Original request: {planning_request}",
        "",
        "Clarifications:",
    ]
    effective_questions = questions or _planning_questions(max_questions)
    for idx, answer in enumerate(planning_answers, start=1):
        question = effective_questions[idx - 1] if idx - 1 < len(effective_questions) else f"Question {idx}"
        lines.append(f"Q{idx}: {question}")
        lines.append(f"A{idx}: {answer}")
    return "\n".join(lines)


def _render_auto_execute_pass_status(
    console,
    *,
    objective: str,
    pass_index: int,
    pass_cap: int,
    planner_step_id: str,
    planner_step_title: str,
    planner_decision: str,
    planner_decision_reason: str,
    batch_reason: str,
    expected_progress: str,
) -> None:
    details = []
    if objective:
        details.append(f"objective={objective}")
    if planner_step_id or planner_step_title:
        details.append(f"step={planner_step_id or planner_step_title}")
    if planner_decision:
        details.append(f"decision={planner_decision}")
    if planner_decision_reason:
        details.append(f"reason={planner_decision_reason}")
    if batch_reason:
        details.append(f"batch={batch_reason}")
    if expected_progress:
        details.append(f"expected={expected_progress}")
    suffix = "; ".join(details)
    console.print(
        f"[cyan]Auto-execute pass {max(0, int(pass_index))}/{max(1, int(pass_cap))}[/cyan]"
        + (f" - {suffix}" if suffix else "")
    )



def chat(
    prompt: str | None = typer.Argument(None, help="Optional first chat prompt."),
    model: str | None = typer.Option(None, "--model"),
    index_dir: str | None = typer.Option(None, "--index-dir"),
    k: int | None = typer.Option(None, "--k"),
    ephemeral_index: bool = typer.Option(
        False,
        "--ephemeral-index",
        help="Use temporary index(es) and delete them when chat exits (ignored if --index-dir is set).",
    ),
    dir_mode: bool = typer.Option(
        False,
        "--dir-mode",
        help="Enable directory-aware chat mode (uses subproject indexes).",
    ),
    root_dir: str | None = typer.Option(
        None,
        "--root-dir",
        "--repo",
        help="Project root used for tool execution and default index paths.",
    ),
    mcp_server_json: list[str] = typer.Option(
        [],
        "--mcp-server-json",
        help="Inline JSON MCP server definition; may be repeated for ad-hoc tool/resource providers.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session",
        help="Resume an isolated session whose primary repository matches --root-dir.",
    ),
    max_indexes: int = typer.Option(
        0,
        "--max-indexes",
        help="Maximum discovered indexes to use in dir-mode (0 means unlimited).",
    ),
    auto_index_missing: bool = typer.Option(
        True,
        "--auto-index-missing/--no-auto-index-missing",
    ),
    agent_tools: bool | None = typer.Option(
        None,
        "--agent-tools/--no-agent-tools",
        help="Use agent tool mode for chat answers.",
    ),
    coding_agent: bool | None = typer.Option(
        None,
        "--coding-agent/--no-coding-agent",
        help="Allow coding-agent file-edit workflows.",
    ),
    tool_worker_process: bool = typer.Option(
        True,
        "--tool-worker-process/--no-tool-worker-process",
        help="Run coding-agent tool execution in a persistent worker subprocess.",
    ),
    tool_worker_strict: bool = typer.Option(
        True,
        "--tool-worker-strict/--no-tool-worker-strict",
        help="Require at least one successful tool call in worker mode.",
    ),
    tool_exec_backend: str = typer.Option(
        "local",
        "--tool-exec-backend",
        help="ToolsManager executor backend: local or redis.",
    ),
    redis_url: str | None = typer.Option(
        None,
        "--redis-url",
        help="Redis URL for --tool-exec-backend redis.",
    ),
    toolsmanager_parallel_requests: int = typer.Option(
        3,
        "--toolsmanager-parallel-requests",
        min=1,
        help="Maximum concurrent per-pass ToolsManager requests.",
    ),
    redis_queue_name: str = typer.Option(
        "mana-tools",
        "--redis-queue-name",
        help="Redis queue name used by ToolsManager redis backend.",
    ),
    redis_ttl_seconds: int = typer.Option(
        86_400,
        "--redis-ttl-seconds",
        min=60,
        help="Redis TTL for ToolsManager runtime status/event keys.",
    ),
    coding_memory: bool = typer.Option(
        True,
        "--coding-memory/--no-coding-memory",
        help="Persist coding-agent flow memory across turns and chat restarts.",
    ),
    flow_id: str | None = typer.Option(
        None,
        "--flow-id",
        help="Optional flow ID to resume or pin coding-agent continuity.",
    ),
    coding_plan_max_steps: int = typer.Option(
        8,
        "--coding-plan-max-steps",
        help="Maximum checklist steps generated by coding planner.",
    ),
    coding_search_budget: int = typer.Option(
        4,
        "--coding-search-budget",
        help="Max semantic_search calls per coding turn.",
    ),
    coding_read_budget: int = typer.Option(
        6,
        "--coding-read-budget",
        help="Max read_file calls per coding turn (full-auto uses this as dynamic per-turn cap).",
    ),
    coding_require_read_files: int = typer.Option(
        2,
        "--coding-require-read-files",
        help="Minimum unique read_file inspections required before edits/answer.",
    ),
    planning_mode: bool = typer.Option(
        False,
        "--planning-mode",
        help="Enable multi-step planning Q&A before generating plan answers.",
        hidden=True,
    ),
    planning_max_questions: int = typer.Option(
        3,
        "--planning-max-questions",
        help="Maximum planning clarification questions to ask (1-6).",
    ),
    auto_execute_plan: bool | None = typer.Option(
        None,
        "--auto-execute-plan/--no-auto-execute-plan",
        help="Automatically execute plan-producing turns in agent-tools mode.",
        hidden=True,
    ),
    auto_execute_max_passes: int = typer.Option(
        4,
        "--auto-execute-max-passes",
        min=1,
        max=12,
        help="Maximum planner->toolsmanager execution passes per turn.",
    ),
    auto_continue: bool = typer.Option(
        True,
        "--auto-continue/--no-auto-continue",
        help="Automatically resume auto-execute checkpoints until work completes or blocks.",
    ),
    execution_profile: str = typer.Option(
        "balanced",
        "--execution-profile",
        help="Execution profile: full-auto, balanced, conservative.",
    ),
    full_auto: bool = typer.Option(
        False,
        "--full-auto",
        help="Alias for --execution-profile full-auto.",
    ),
    full_auto_status_every: int = typer.Option(
        10,
        "--full-auto-status-every",
        min=0,
        help="In full-auto profile, print a compact checkpoint every N auto-execute passes (0 disables).",
    ),
    agent_max_steps: int = typer.Option(6, "--agent-max-steps"),
    agent_unlimited: bool = typer.Option(
        False,
        "--agent-unlimited/--no-agent-unlimited",
        help="Use effectively unlimited agent tool steps (subject to timeout/resources).",
    ),
    agent_timeout_seconds: int = typer.Option(30, "--agent-timeout-seconds"),
    multiline_input: bool = typer.Option(
        True,
        "--multiline-input/--no-multiline-input",
        help="Enable multiline chat input (`/paste` trigger or buffered paste burst detection).",
    ),
    multiline_terminator: str = typer.Option(
        ".end",
        "--multiline-terminator",
        help="Terminator line used to submit multiline input.",
    ),
    diagram_render_images: bool = typer.Option(
        True,
        "--diagram-render-images/--no-diagram-render-images",
        help="Render Mermaid diagram blocks to SVG/PNG artifacts when possible.",
    ),
    diagram_output_dir: str | None = typer.Option(
        None,
        "--diagram-output-dir",
        help="Directory for rendered diagram artifacts (default: <root>/.mana/diagrams).",
    ),
    diagram_format: str = typer.Option(
        "svg",
        "--diagram-format",
        help="Diagram artifact format: svg or png.",
    ),
    diagram_open: bool = typer.Option(
        False,
        "--diagram-open/--no-diagram-open",
        help="Open rendered diagram artifacts with the system default app.",
    ),
    diagram_timeout_seconds: int = typer.Option(
        25,
        "--diagram-timeout-seconds",
        help="Timeout in seconds for Mermaid artifact rendering.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit responses as JSON objects."),
    simple: bool = typer.Option(False, "--simple", help="Use the plain chat renderer."),
    welcome: str = typer.Option(
        "compact",
        "--welcome",
        help="Startup welcome detail: compact or full.",
    ),
) -> None:
    output_file = _resolve_output_file()
    if mcp_server_json:
        # Worker subprocesses inherit this explicit per-invocation setting. It
        # is parsed and validated by the MCP tool registry before discovery.
        os.environ["MANA_MCP_SERVER_OVERRIDES"] = json.dumps(mcp_server_json)
    try:
        ensure_setup(command_needs_llm=True, console=console)
    except NonInteractivePromptError as exc:
        raise typer.BadParameter(str(exc)) from exc
    # Keep noisy indexing/parsing logs off the interactive console (the file log
    # still captures everything) so the prompt is usable while the index builds.
    _install_quiet_chat_console_logging()
    agent_tools_explicit = agent_tools is not None
    coding_agent_explicit = coding_agent is not None
    agent_tools = True if agent_tools is None else bool(agent_tools)
    coding_agent = True if coding_agent is None else bool(coding_agent)
    if agent_tools_explicit and not coding_agent_explicit:
        coding_agent = False
    logger.info(
        "Chat command started",
        extra={
            "model_override": model,
            "index_dir": index_dir,
            "k": k,
            "dir_mode": dir_mode,
            "root_dir": root_dir,
            "session_id": session_id,
            "max_indexes": max_indexes,
            "auto_index_missing": auto_index_missing,
            "agent_tools": agent_tools,
            "coding_agent": coding_agent,
            "tool_worker_process": tool_worker_process,
            "tool_worker_strict": tool_worker_strict,
            "tool_exec_backend": tool_exec_backend,
            "redis_url": redis_url,
            "toolsmanager_parallel_requests": toolsmanager_parallel_requests,
            "redis_queue_name": redis_queue_name,
            "redis_ttl_seconds": redis_ttl_seconds,
            "coding_memory": coding_memory,
            "flow_id": flow_id,
            "coding_plan_max_steps": coding_plan_max_steps,
            "coding_search_budget": coding_search_budget,
            "coding_read_budget": coding_read_budget,
            "coding_require_read_files": coding_require_read_files,
            "planning_mode": planning_mode,
            "planning_max_questions": planning_max_questions,
            "auto_execute_plan": auto_execute_plan,
            "auto_execute_max_passes": auto_execute_max_passes,
            "auto_continue": auto_continue,
            "execution_profile": execution_profile,
            "full_auto": full_auto,
            "full_auto_status_every": full_auto_status_every,
            "agent_max_steps": agent_max_steps,
            "agent_unlimited": agent_unlimited,
            "agent_timeout_seconds": agent_timeout_seconds,
            "multiline_input": multiline_input,
            "multiline_terminator": multiline_terminator,
            "diagram_render_images": diagram_render_images,
            "diagram_output_dir": diagram_output_dir,
            "diagram_format": diagram_format,
            "diagram_open": diagram_open,
            "diagram_timeout_seconds": diagram_timeout_seconds,
            "as_json": as_json,
            "ephemeral_index": ephemeral_index,
        },
    )
    if full_auto:
        execution_profile = "full-auto"
    execution_profile = str(execution_profile or "balanced").strip().lower()
    if execution_profile not in {"full-auto", "balanced", "conservative"}:
        raise typer.BadParameter("--execution-profile must be one of: full-auto, balanced, conservative.")

    legacy_auto_execute_plan_requested = auto_execute_plan is True
    legacy_auto_execute_plan_disabled = auto_execute_plan is False
    auto_execute_plan = not legacy_auto_execute_plan_disabled
    if execution_profile == "full-auto":
        auto_execute_plan = True
        # Keep user override when they pass a non-default value.
        if int(auto_execute_max_passes) == 4:
            auto_execute_max_passes = 10
    chat_auto_continue = bool(auto_continue and auto_execute_plan)
    full_auto_status_every = max(0, int(full_auto_status_every))
    planning_question_limit = max(1, min(planning_max_questions, 6))
    auto_execute_max_passes = max(1, min(int(auto_execute_max_passes), 12))
    chat_agent_max_steps = _resolve_agent_max_steps(
        agent_max_steps,
        agent_unlimited=agent_unlimited,
        min_steps=1,
    )
    coding_agent_max_steps = _resolve_agent_max_steps(
        agent_max_steps,
        agent_unlimited=agent_unlimited,
        min_steps=8,
        cap=200,
    )
    settings = _public_symbol("Settings", Settings)()
    resolved_tool_exec_backend = str(
        (tool_exec_backend or getattr(settings, "tool_exec_backend", "local")) or "local"
    ).strip().lower()
    if resolved_tool_exec_backend not in {"local", "redis"}:
        raise typer.BadParameter("--tool-exec-backend must be 'local' or 'redis'.")
    resolved_redis_url = str(
        (redis_url or getattr(settings, "redis_url", "redis://127.0.0.1:6379/0"))
        or "redis://127.0.0.1:6379/0"
    ).strip()
    resolved_parallel_requests = max(
        1,
        int(
            toolsmanager_parallel_requests
            or getattr(settings, "toolsmanager_parallel_requests", 3)
            or 3
        ),
    )
    resolved_redis_queue_name = str(
        (redis_queue_name or getattr(settings, "redis_queue_name", "mana-tools")) or "mana-tools"
    ).strip() or "mana-tools"
    resolved_redis_ttl_seconds = max(
        60,
        int(redis_ttl_seconds or getattr(settings, "redis_ttl_seconds", 86_400) or 86_400),
    )
    tools_execution_config = ToolsExecutionConfig(
        backend=resolved_tool_exec_backend,
        redis_url=resolved_redis_url,
        queue_name=resolved_redis_queue_name,
        parallel_requests=resolved_parallel_requests,
        ttl_seconds=resolved_redis_ttl_seconds,
    )
    tools_execution_boot_warnings: list[str] = []
    diagram_format = str(diagram_format or "svg").strip().lower()
    if diagram_format not in {"svg", "png"}:
        raise typer.BadParameter("--diagram-format must be 'svg' or 'png'.")
    diagram_timeout_seconds = max(5, int(diagram_timeout_seconds))
    multiline_terminator = str(multiline_terminator or ".end").strip()
    if not multiline_terminator:
        raise typer.BadParameter("--multiline-terminator must be a non-empty line token.")
    resolved_k = k or settings.default_top_k

    root = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
    if root.is_file():
        root = root.parent
    _record_multi_agent_request(root, "chat command", entrypoint="chat", command_scope=True, session_id=session_id)

    recorded_initial_prompt = False
    if prompt:
        _record_multi_agent_request(root, prompt, entrypoint="chat", session_id=session_id)
        recorded_initial_prompt = True
        direct_edit_result = handle_small_direct_edit(root, prompt)
        if direct_edit_result.handled:
            console.print(f"[bold cyan]mana ❯[/bold cyan] {prompt}")
            _render_answer_header(console)
            console.print(Markdown(direct_edit_result.answer))
            return

    logger.debug("Resolved chat root", extra={"root": str(root)})
    run_logger_cls = _public_symbol("LlmRunLogger", LlmRunLogger)
    run_logger = run_logger_cls(
        log_file=(
            default_llm_logs_dir(root)
            / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-{(root.name or 'project')}-runs.jsonl"
        )
    )
    resolved_diagram_output_dir = (
        Path(diagram_output_dir).expanduser().resolve()
        if diagram_output_dir
        else default_diagrams_dir(root).resolve()
    )

    ask_service = _build_ask_service_compat(settings, model_override=model, project_root=root)

    chat_service = ChatService(
        ask_service=ask_service,
        settings=settings,
        model_override=model,
        index_dir=index_dir,
        dir_mode=dir_mode,
        root_dir=str(root),
        k=resolved_k,
        agent_tools=bool(agent_tools_explicit and agent_tools),
        agent_max_steps=chat_agent_max_steps,
        agent_timeout_seconds=agent_timeout_seconds,
        max_indexes=max_indexes,
        auto_index_missing=auto_index_missing,
    )

    # ✅ Initialize CodingAgent (only when enabled)
    coding_agent_instance: CodingAgent | None = None
    coding_memory_service: CodingMemoryService | None = None
    tool_worker_client: ToolWorkerClient | None = None
    tools_manager_orchestrator: QueueManager | None = None
    tools_executor_instance: LocalToolsExecutor | RedisRQToolsExecutor | None = None
    coding_agent_cls = _public_symbol("CodingAgent", CodingAgent)
    coding_agent_is_custom = coding_agent_cls is not CodingAgent
    effective_model = resolve_model_for_role(
        AgentRole.MAIN,
        global_model=model or settings.openai_chat_model,
    ).resolved_model
    coding_model_assignment = resolve_model_for_role(AgentRole.CODING, global_model=effective_model)
    planner_model_assignment = resolve_model_for_role(
        AgentRole.PLANNER,
        global_model=settings.openai_coding_planner_model or effective_model,
    )
    tool_worker_model_assignment = resolve_model_for_role(
        AgentRole.TOOL_WORKER,
        global_model=settings.openai_tool_worker_model or effective_model,
    )
    effective_tool_worker_model = tool_worker_model_assignment.resolved_model
    effective_base_url = settings.openai_base_url or os.getenv("OPENAI_BASE_URL")
    chat_log_path = default_logs_dir(root) / f"mana_agent_{datetime.now().strftime('%Y%m%d')}.log"
    chat_ui_state = ChatUIState(
        repo_root=root,
        provider="openai-compatible",
        model=effective_model,
        mode="chat",
        tools_enabled=bool(agent_tools),
        approvals="auto",
        memory_enabled=bool(coding_memory and coding_agent),
        skills_status=detect_skills_status(root),
        ui_mode=("plain" if simple else default_ui_mode(console, as_json=as_json)),
        index_path=str(index_dir or default_index_dir(root)),
        k_value=resolved_k,
        ephemeral_index=bool(ephemeral_index),
        coding_agent=bool(coding_agent),
        coding_memory=bool(coding_memory),
        flow_memory=bool(coding_memory),
        auto_execute=bool(auto_execute_plan),
        max_passes=int(auto_execute_max_passes),
        auto_continue=bool(chat_auto_continue),
        execution_profile=execution_profile,
        diagram_rendering=f"{'on' if diagram_render_images else 'off'} ({diagram_format})",
        tool_worker_backend=resolved_tool_exec_backend,
        log_path=chat_log_path,
        session_id=session_id or f"sess-{uuid.uuid4().hex}",
    )

    def _build_tools_executor(worker_client: ToolWorkerClient):
        helper = _public_symbol("build_tools_executor_with_fallback", build_tools_executor_with_fallback)
        init_payload = (
            worker_client.init_payload_dict()
            if hasattr(worker_client, "init_payload_dict")
            else {}
        )
        return helper(
            worker_client=worker_client,
            config=tools_execution_config,
            worker_init_payload=init_payload,
            warnings=tools_execution_boot_warnings,
            warning_key=f"chat:{root}:{tools_execution_config.redis_url}:{tools_execution_config.queue_name}",
            local_executor_cls=_public_symbol("LocalToolsExecutor", LocalToolsExecutor),
            redis_executor_cls=_public_symbol("RedisRQToolsExecutor", RedisRQToolsExecutor),
        )

    if coding_agent:
        if not agent_tools:
            raise typer.BadParameter("--coding-agent requires --agent-tools (needs tool loop).")
        if getattr(ask_service, "ask_agent", None) is None:
            raise typer.BadParameter("--coding-agent requires AskService.ask_agent to be configured.")
        if coding_memory:
            coding_memory_service = CodingMemoryService(
                project_root=root,
                max_turns=settings.coding_flow_max_turns,
                max_tasks=settings.coding_flow_max_tasks,
                session_id=chat_ui_state.session_id,
            )
        if hasattr(ask_service.ask_agent, "update_model"):
            ask_service.ask_agent.update_model(coding_model_assignment.resolved_model)
        elif hasattr(ask_service.ask_agent, "model"):
            ask_service.ask_agent.model = coding_model_assignment.resolved_model
        if tool_worker_process:
            tool_worker_client_cls = _public_symbol("ToolWorkerClient", ToolWorkerClient)
            tool_worker_client = tool_worker_client_cls(
                api_key=settings.openai_api_key,
                model=effective_tool_worker_model,
                base_url=effective_base_url,
                repo_root=root,
                project_root=root,
                allowed_prefixes=None,
                tools_only_strict=tool_worker_strict,
                model_level=tool_worker_model_assignment.model_level,
                workspace_id=chat_ui_state.workspace_id,
                repository_id=chat_ui_state.repository_id,
            )

        # ✅ IMPORTANT: allow_prefixes=None => unrestricted under repo_root
        coding_agent_instance = coding_agent_cls(
            api_key=settings.openai_api_key,
            base_url=effective_base_url,
            repo_root=root,
            ask_agent=ask_service.ask_agent,
            allowed_prefixes=None,
            coding_memory_service=coding_memory_service,
            coding_memory_enabled=coding_memory,
            plan_max_steps=max(1, int(coding_plan_max_steps or settings.coding_plan_max_steps)),
            search_budget=max(1, int(coding_search_budget or settings.coding_search_budget)),
            read_budget=max(1, int(coding_read_budget or settings.coding_read_budget)),
            require_read_files=max(1, int(coding_require_read_files or settings.coding_require_read_files)),
            tool_worker_client=tool_worker_client,
            full_auto_mode=(execution_profile == "full-auto"),
            planner_model=planner_model_assignment.resolved_model,
        )

    if coding_agent_instance is not None and auto_execute_plan and tool_worker_client is not None:
        tools_executor_instance = _build_tools_executor(tool_worker_client)
        tools_manager_orchestrator_cls = _public_symbol("QueueManager", QueueManager)
        tools_manager_orchestrator = tools_manager_orchestrator_cls(
            api_key=settings.openai_api_key,
            model=effective_model,
            base_url=settings.openai_base_url,
            worker_client=tool_worker_client,
            repo_root=root,
            execution_config=tools_execution_config,
            executor=tools_executor_instance,
            coding_memory_service=coding_memory_service,
            workspace_id=chat_ui_state.workspace_id,
            repository_id=chat_ui_state.repository_id,
            session_id=chat_ui_state.session_id,
        )
        if hasattr(coding_agent_instance, "set_tools_manager_orchestrator"):
            coding_agent_instance.set_tools_manager_orchestrator(tools_manager_orchestrator)
        # Give preview_plan access to the coding agent's LLM planner + memory so
        # the pre-execution checklist is accurate, not deterministic-only.
        if hasattr(tools_manager_orchestrator, "attach_decision_provider"):
            tools_manager_orchestrator.attach_decision_provider(coding_agent_instance)

    tmp_root: tempfile.TemporaryDirectory | None = None
    tmp_base: Path | None = None
    dir_mode_index_dirs: list[Path] = []
    background_index_state: dict[str, Any] = {"status": "idle", "announced": False, "error": ""}
    try:
        set_active_chat_ui_state(chat_ui_state)
        # -----------------------------
        # Resolve indexes (dir-mode vs classic)
        # -----------------------------
        resolved_index_dir: Path | None = None

        if dir_mode:
            discovered_subprojects = _public_symbol("discover_subprojects", discover_subprojects)(root)
            discovered_indexes = _public_symbol("discover_index_dirs", discover_index_dirs)(root)
            discovered_index_set = {item.resolve() for item in discovered_indexes}

            if ephemeral_index and not index_dir:
                tmp_root, tmp_base = _make_ephemeral_index_dir(prefix="mana_chat_indexes_")

            index_service = _public_symbol("build_index_service", build_index_service)(settings)
            auto_indexed_count = 0
            skipped_missing_count = 0
            warnings: list[str] = []
            selected_indexes: list[Path] = []

            def _auto_index(target_path: Path, idx_dir: Path) -> bool:
                logger.info("Chat auto-index attempt", extra={"target_path": str(target_path), "idx_dir": str(idx_dir)})
                try:
                    _index_service_index_compat(
                        index_service,
                        target_path=target_path,
                        index_dir=idx_dir,
                        rebuild=False,
                        vectors=True,
                    )
                    logger.info("Chat auto-index vectors success", extra={"idx_dir": str(idx_dir)})
                    return True
                except Exception as exc:
                    warning = f"Vector index failed for {target_path} (fallback to chunks-only): {exc}"
                    logger.warning(warning)
                    warnings.append(warning)
                    try:
                        _index_service_index_compat(
                            index_service,
                            target_path=target_path,
                            index_dir=idx_dir,
                            rebuild=False,
                            vectors=False,
                        )
                        logger.info("Chat auto-index chunks-only success", extra={"idx_dir": str(idx_dir)})
                        return True
                    except Exception as exc2:
                        warning2 = f"Chunks-only index failed for {target_path}: {exc2}"
                        logger.warning(warning2)
                        warnings.append(warning2)
                        return False

            if discovered_subprojects:
                for subproject in discovered_subprojects:
                    if tmp_base is not None:
                        expected_index = (tmp_base / _stable_subdir_name(subproject.root_path)).resolve()
                        has_index_dir = expected_index.exists()
                    else:
                        expected_index = default_index_dir(subproject.root_path).resolve()
                        has_index_dir = expected_index in discovered_index_set

                    has_search_data = has_index_dir and _index_has_search_data(expected_index)

                    if has_search_data:
                        selected_indexes.append(expected_index)
                        continue

                    if auto_index_missing:
                        ok = _auto_index(subproject.root_path, expected_index)
                        if ok:
                            auto_indexed_count += 1
                            selected_indexes.append(expected_index)
                        else:
                            if _index_has_chunks(expected_index):
                                selected_indexes.append(expected_index)
                    else:
                        skipped_missing_count += 1
                        warning = f"Skipped missing or empty index for subproject {subproject.root_path}"
                        logger.warning(warning)
                        warnings.append(warning)

            else:
                if tmp_base is not None:
                    root_index = (tmp_base / _stable_subdir_name(root)).resolve()
                else:
                    root_index = default_index_dir(root).resolve()

                if root_index.exists() and _index_has_search_data(root_index):
                    selected_indexes = [root_index]
                elif auto_index_missing:
                    ok = _auto_index(root, root_index)
                    if ok:
                        auto_indexed_count = 1
                        selected_indexes = [root_index]
                    else:
                        if _index_has_chunks(root_index):
                            selected_indexes = [root_index]

            selected_indexes = sorted({item.resolve() for item in selected_indexes}, key=lambda p: str(p))
            if max_indexes > 0:
                selected_indexes = selected_indexes[:max_indexes]

            if not selected_indexes:
                msg = (
                    f"No usable indexes found under {root}. "
                    f"Try running: mana-agent index {root} "
                    f"or re-run chat with --auto-index-missing."
                )
                _emit_text(msg, output_file=output_file)
                logger.error("Chat aborted: no usable indexes", extra={"root": str(root)})
                raise typer.Exit(code=2)

            chat_service.set_index_dirs(selected_indexes)
            dir_mode_index_dirs = selected_indexes
            chat_ui_state.index_path = f"{len(selected_indexes)} dir-mode index(es)"
            if warnings:
                _emit_text("Warnings:\n" + "\n".join(f"- {w}" for w in warnings), output_file=output_file)

        else:
            if ephemeral_index and not index_dir:
                tmp_root, resolved_index_dir = _make_ephemeral_index_dir(prefix="mana_chat_index_")
                index_service = _public_symbol("build_index_service", build_index_service)(settings)
                _index_service_index_compat(
                    index_service,
                    target_path=root,
                    index_dir=resolved_index_dir,
                    rebuild=False,
                    vectors=True,
                )
            else:
                resolved_index_dir = Path(index_dir).resolve() if index_dir else default_index_dir(root)

            # Build the semantic index in the BACKGROUND when it is missing so chat
            # is usable immediately (direct project-search fallback) and semantic
            # search activates automatically once the index is ready. Indexing is
            # never blocking and never prompts.
            if (
                auto_index_missing
                and not ephemeral_index
                and resolved_index_dir is not None
                and not _index_has_search_data(resolved_index_dir)
            ):
                _start_background_index(
                    settings=settings,
                    target_root=root,
                    index_dir=resolved_index_dir,
                    state=background_index_state,
                )

            if background_index_state["status"] == "building":
                index_state = "building in background; using direct project search until ready"
            elif _index_has_search_data(resolved_index_dir):
                index_state = "ready"
            else:
                index_state = "missing (run `mana-agent index`; using direct project search fallback)"
            chat_ui_state.index_path = f"{resolved_index_dir} ({index_state})"

        # -----------------------------
        # Tool-first retry logic
        # -----------------------------
        def _looks_like_guess(answer: str) -> bool:
            a = (answer or "").lower()
            triggers = [
                "i can't",
                "cannot",
                "can't",
                "no source",
                "no actual source",
                "snippets you provided",
                "from the snippets",
                "i don’t have",
                "i don't have",
                "not enough",
                "insufficient",
                "based on the repository name",
                "infer",
                "guess",
                "might be",
                "appears to",
                "only clues",
                "doesn’t contain application logic",
                "does not contain application logic",
                "path is outside project root",
            ]
            return any(t in a for t in triggers)

        def _tool_first_instruction(user_question: str) -> str:
            return (
                f"{user_question}\n\n"
                "TOOL-FIRST INSTRUCTIONS (MANDATORY):\n"
                "- Do NOT guess.\n"
                "- You MUST use tools to search the repository and inspect real source files.\n"
                "- You MUST open at least TWO real source files before concluding.\n"
                "- Avoid caches/build output: node_modules/, .next/, .angular/, dist/, build/, .cache/, .npm-cache/, generated/.\n"
                "- Final answer MUST include evidence (file path + line ranges).\n"
            )

        max_attempts = 3
        min_sources = 2

        render_startup_header(console, chat_ui_state)

        # Collect the per-session configuration so it renders as one tidy panel
        # instead of a scattered list of single-line prints.
        status_rows: list[tuple[str, str]] = []
        status_rows.append(("coding agent", "active — all decisions routed through CodingAgent"))
        _ca_flow = flow_id or (coding_agent_instance.get_active_flow_id() if coding_agent_instance else None)
        if coding_memory:
            if _ca_flow:
                status_rows.append(("flow memory", f"active — resuming flow {_ca_flow}"))
            else:
                status_rows.append(("flow memory", "active — new flow on first request"))
        auto_execute_status = "automatic" if auto_execute_plan else "disabled by legacy override"
        status_rows.append(("auto-execute", f"{auto_execute_status} (max passes: {auto_execute_max_passes})"))
        status_rows.append(("auto-continue", "on" if chat_auto_continue else "off"))
        status_rows.append(("execution profile", execution_profile))
        if execution_profile == "full-auto":
            if full_auto_status_every > 0:
                status_rows.append(("full-auto checkpoint", f"every {full_auto_status_every} pass(es)"))
            else:
                status_rows.append(("full-auto checkpoint", "disabled"))
        if diagram_render_images:
            status_rows.append(("diagram rendering", f"{diagram_format} → {resolved_diagram_output_dir}"))
        if tool_worker_client is not None and (coding_agent_instance is not None or tools_manager_orchestrator is not None):
            try:
                start = getattr(tool_worker_client, "start", None)
                if callable(start):
                    start()
                health = getattr(tool_worker_client, "health", None)
                if callable(health):
                    health()
                status_rows.append(
                    (
                        "tool worker",
                        f"active — strict={tool_worker_strict}, "
                        f"backend={tools_execution_config.backend}, "
                        f"parallel={tools_execution_config.parallel_requests}",
                    )
                )
            except Exception as exc:
                _log_exception("tool_worker_client.start", exc)
                console.print(
                    "[yellow]Tool worker failed to initialize. Auto-execute and coding-agent paths are disabled for this session.[/yellow]"
                )
                coding_agent_instance = None
                tools_manager_orchestrator = None

        if str(welcome or "compact").strip().lower() == "full":
            console.print(render_status(chat_ui_state, full=True))

        planning_request: str | None = None
        planning_answers: list[str] = []
        planning_questions: list[str] = []
        planning_question_source: str = "none"
        planning_questions_asked_count: int = 0
        planning_question_llm_disabled = False
        planning_question_failure_logged = False
        active_flow_id: str | None = _ca_flow if isinstance(_ca_flow, str) and _ca_flow.strip() else None
        pending_conflict_question: str | None = None
        pending_ui_selection: dict[str, Any] | None = None
        pending_prechecklist: dict[str, Any] | None = None
        pending_prechecklist_source: str = ""
        pending_prechecklist_warning: str = ""
        session_turns: list[ChatTurnTelemetry] = []
        # Most recent /analyze output, loaded so later chat answers are grounded in
        # the analysis. Refreshed whenever /analyze runs during the session.
        analysis_context_text: str | None = _load_analysis_context(root)
        auto_chat_state: AutoChatSessionState = load_auto_chat_state(root)
        full_auto_pass_window_logs: list[dict[str, Any]] = []
        full_auto_pass_window_decisions: list[dict[str, Any]] = []
        full_auto_latest_checklist_counts: dict[str, int] | None = None
        full_auto_passes_total: int = 0
        full_auto_pass_checkpoints_emitted: int = 0

        inline_renderer = InlineChatRenderer(console, mode=chat_ui_state.ui_mode)

        def _finish_ui_turn(turn_id: str, message: str = "Final response rendered.") -> None:
            chat_ui_state.finish_turn(turn_id, message=message)
            summary = chat_ui_state.execution_summary(turn_id=turn_id)
            if (
                summary
                and "\n- subagent_" in summary
                and chat_ui_state.ui_mode != "json"
                and str(os.getenv("MANA_CHAT_EXECUTION_SUMMARY", "1") or "").strip().lower()
                not in {"0", "false", "no", "off"}
            ):
                console.print("\n[bold]Execution summary[/bold]")
                console.print(summary)
            detailed_timeline_enabled = bool(
                chat_ui_state.verbose_logs
                or _cli_verbose_enabled()
                or chat_ui_state.trace_mode == "full"
            )
            if not detailed_timeline_enabled:
                return
            if chat_ui_state.trace_mode == "off" or chat_ui_state.active_panel != "timeline":
                return
            turn_events = [
                event
                for event in chat_ui_state.normalized_events
                if event.turn_id == turn_id and not event.type.startswith("tool.")
            ]
            if turn_events:
                console.print(chat_ui_state.renderer.render_timeline(turn_events))

        def _run_chat_event_step(title: str, worker, *, event_type: str = "step.started", tool_name: str | None = None):
            event = chat_ui_state.record_event(
                make_event(
                    event_type,
                    title=title.rstrip("."),
                    message="Running.",
                    status="running",
                    session_id=chat_ui_state.session_id,
                    turn_id=chat_ui_state.tracker.current_turn_id,
                    metadata={"tool_name": tool_name} if tool_name else None,
                )
            )
            inline_renderer.render_event(event)
            try:
                result = worker()
                completed = chat_ui_state.update_event_status(event.event_id, status="success", message="Completed.")
                inline_renderer.render_event(completed)
                return result
            except BaseException as exc:
                failed = chat_ui_state.update_event_status(event.event_id, status="failed", message=str(exc) or "Failed.")
                inline_renderer.render_event(failed)
                raise

        def _start_new_topic() -> str | None:
            nonlocal active_flow_id, pending_conflict_question
            reset_id: str | None = None
            if coding_agent_instance is not None:
                target_flow = active_flow_id or coding_agent_instance.get_active_flow_id()
                if isinstance(target_flow, str) and target_flow.strip():
                    if hasattr(coding_agent_instance, "reset_flow"):
                        reset_id = coding_agent_instance.reset_flow(target_flow.strip())
                    else:
                        reset_id = target_flow.strip()
            active_flow_id = None
            pending_conflict_question = None
            return reset_id

        def _base_auto_execute_tool_policy(user_question: str, *, auto_chat_mode: AutoChatMode | None = None) -> dict[str, Any]:
            if coding_agent_instance is not None:
                return coding_agent_instance._tool_policy_for_request(  # type: ignore[attr-defined]
                    user_question,
                    auto_chat_mode=(auto_chat_mode.value if auto_chat_mode is not None else None),
                )
            policy = {
                "allowed_tools": [
                    "semantic_search",
                    "read_file",
                    "run_command",
                    "apply_patch",
                    "create_file",
                    "write_file",
                ],
                "search_budget": max(1, int(coding_search_budget or settings.coding_search_budget)),
                "read_budget": max(1, int(coding_read_budget or settings.coding_read_budget)),
                "read_line_window": 400,
                "require_read_files": max(1, int(coding_require_read_files or settings.coding_require_read_files)),
                "search_repeat_limit": 1,
                "max_semantic_k": 50,
            }
            if auto_chat_mode is not None:
                policy = apply_auto_chat_tool_policy(policy, auto_chat_mode)
            return policy

        def _ensure_tools_manager_orchestrator() -> QueueManager | None:
            nonlocal tool_worker_client
            nonlocal tools_manager_orchestrator
            nonlocal tools_executor_instance
            if not (agent_tools and auto_execute_plan and tool_worker_process):
                return None
            if tools_manager_orchestrator is not None:
                return tools_manager_orchestrator
            if tool_worker_client is None:
                tool_worker_client_cls = _public_symbol("ToolWorkerClient", ToolWorkerClient)
                tool_worker_client = tool_worker_client_cls(
                    api_key=settings.openai_api_key,
                    model=effective_tool_worker_model,
                    base_url=effective_base_url,
                    repo_root=root,
                    project_root=root,
                    allowed_prefixes=None,
                    tools_only_strict=tool_worker_strict,
                    model_level=tool_worker_model_assignment.model_level,
                    workspace_id=chat_ui_state.workspace_id,
                    repository_id=chat_ui_state.repository_id,
                )
            try:
                start = getattr(tool_worker_client, "start", None)
                if callable(start):
                    start()
                health = getattr(tool_worker_client, "health", None)
                if callable(health):
                    health()
            except Exception as exc:
                _log_exception("tool_worker_client.start", exc)
                return None
            if tools_executor_instance is None:
                tools_executor_instance = _build_tools_executor(tool_worker_client)
            tools_manager_orchestrator_cls = _public_symbol("QueueManager", QueueManager)
            tools_manager_orchestrator = tools_manager_orchestrator_cls(
                api_key=settings.openai_api_key,
                model=effective_model,
                base_url=effective_base_url,
                worker_client=tool_worker_client,
                repo_root=root,
                execution_config=tools_execution_config,
                executor=tools_executor_instance,
                coding_memory_service=(
                    getattr(coding_agent_instance, "coding_memory_service", None)
                    if coding_agent_instance is not None
                    else None
                ),
                workspace_id=chat_ui_state.workspace_id,
                repository_id=chat_ui_state.repository_id,
                session_id=chat_ui_state.session_id,
            )
            if coding_agent_instance is not None and hasattr(coding_agent_instance, "set_tools_manager_orchestrator"):
                coding_agent_instance.set_tools_manager_orchestrator(tools_manager_orchestrator)
            # Accurate, memory-backed preview planning: route preview_plan through
            # the coding agent's LLM planner when one is available.
            if coding_agent_instance is not None and hasattr(tools_manager_orchestrator, "attach_decision_provider"):
                tools_manager_orchestrator.attach_decision_provider(coding_agent_instance)
            return tools_manager_orchestrator

        def _run_auto_execute_pipeline(
            user_question: str,
            *,
            render_progress: bool = True,
            run_id: str | None = None,
            auto_chat_mode: AutoChatMode | None = None,
        ) -> tuple[dict[str, Any], str]:
            # We may adopt the flow id the preview attached to, so this name is
            # rebound here rather than only read from the enclosing scope.
            nonlocal active_flow_id
            if not agent_tools or not auto_execute_plan:
                return {}, ""
            orchestrator = _ensure_tools_manager_orchestrator()
            if orchestrator is None:
                return {
                    "answer": "Auto-execute requested but tools manager worker is unavailable.",
                    "warnings": [
                        "auto_execute_worker_unavailable",
                        *[str(item).strip() for item in tools_execution_boot_warnings if str(item).strip()],
                    ],
                    "trace": [],
                    "sources": [],
                    "changed_files": [],
                    "plan": None,
                    "passes": 0,
                    "terminal_reason": "worker_unavailable",
                    "toolsmanager_requests_count": 0,
                    "pass_logs": [],
                    "planner_decisions": [],
                    "prechecklist": None,
                    "prechecklist_source": "",
                    "prechecklist_warning": "",
                }, ""
            if dir_mode:
                if not dir_mode_index_dirs:
                    return {
                        "answer": "Auto-execute unavailable: no dir-mode indexes resolved.",
                        "warnings": [
                            "auto_execute_missing_index_dirs",
                            *[str(item).strip() for item in tools_execution_boot_warnings if str(item).strip()],
                        ],
                        "trace": [],
                        "sources": [],
                        "changed_files": [],
                        "plan": None,
                        "passes": 0,
                        "terminal_reason": "missing_indexes",
                        "toolsmanager_requests_count": 0,
                        "pass_logs": [],
                        "planner_decisions": [],
                        "prechecklist": None,
                        "prechecklist_source": "",
                        "prechecklist_warning": "",
                    }, ""
                target_index_dir: Path | None = None
                target_index_dirs: list[Path] | None = list(dir_mode_index_dirs)
            else:
                target_index_dir = resolved_index_dir
                target_index_dirs = None

            flow_context_text: str | None = None
            if coding_agent_instance is not None and active_flow_id:
                try:
                    summary = coding_agent_instance.flow_summary(active_flow_id)
                except Exception:
                    summary = None
                if isinstance(summary, dict):
                    lines: list[str] = []
                    objective = str(summary.get("objective", "") or "").strip()
                    if objective:
                        lines.append(f"Current objective: {objective}")
                    checklist = summary.get("checklist")
                    if isinstance(checklist, dict):
                        steps = checklist.get("steps") if isinstance(checklist.get("steps"), list) else []
                        if steps:
                            lines.append("Current checklist:")
                            for step in steps[:20]:
                                if not isinstance(step, dict):
                                    continue
                                status = str(step.get("status", "pending") or "pending")
                                title = str(step.get("title", "step") or "step")
                                lines.append(f"- [{status}] {title}")
                    if lines:
                        flow_context_text = "\n".join(lines)

            # Ground the turn in the most recent /analyze output when available.
            if analysis_context_text:
                flow_context_text = (
                    f"{analysis_context_text}\n\n{flow_context_text}"
                    if flow_context_text
                    else analysis_context_text
                )

            preview_payload: dict[str, Any] = {}
            if hasattr(orchestrator, "preview_plan"):
                try:
                    preview_payload = orchestrator.preview_plan(
                        request=user_question,
                        flow_context=flow_context_text,
                        flow_id=active_flow_id,
                        pass_cap=auto_execute_max_passes,
                    )
                except Exception as exc:
                    _log_exception("tools_manager.preview_plan", exc)
                    preview_payload = {
                        "prechecklist": None,
                        "prechecklist_source": "",
                        "prechecklist_warning": f"Planner preview failed: {exc}",
                        "warnings": [],
                    }
            else:
                preview_payload = {
                    "prechecklist": None,
                    "prechecklist_source": "",
                    "prechecklist_warning": "",
                    "warnings": [],
                }
            # Adopt the flow the preview attached to so the run, its memory turn,
            # and the todo ledger all share one flow id (preview may have created
            # the flow when none was active yet).
            preview_flow_id = preview_payload.get("flow_id")
            if isinstance(preview_flow_id, str) and preview_flow_id.strip():
                active_flow_id = preview_flow_id.strip()
            preview_checklist = (
                preview_payload.get("prechecklist")
                if isinstance(preview_payload.get("prechecklist"), dict)
                else None
            )
            preview_warning = str(preview_payload.get("prechecklist_warning", "") or "").strip()
            preview_steps = (
                preview_checklist.get("steps", [])
                if isinstance(preview_checklist, dict) and isinstance(preview_checklist.get("steps"), list)
                else []
            )
            chat_ui_state.record_event(
                make_event(
                    "agent.planning",
                    title="Planner preview",
                    message=f"Prepared {len(preview_steps)} planned step(s).",
                    status="success" if not preview_warning else "skipped",
                    session_id=chat_ui_state.session_id,
                    turn_id=chat_ui_state.tracker.current_turn_id,
                    step_id="06",
                    metadata={
                        "flow_id": active_flow_id or "",
                        "prechecklist_source": str(preview_payload.get("prechecklist_source", "") or ""),
                        "warning": preview_warning,
                    },
                ).finish(status="success" if not preview_warning else "skipped")
            )
            if render_progress and preview_checklist is not None:
                _render_prechecklist_preview(
                    console,
                    prechecklist=preview_checklist,
                    warning=preview_warning,
                )

            if render_progress:
                console.print(
                    f"[cyan]Auto-executing plan:[/cyan] max passes {auto_execute_max_passes} (same turn, no extra confirmation)."
                )

            def _call(callbacks: list[BaseCallbackHandler]):
                _ = callbacks
                # Carry the accurate preview decision into execution so the run
                # and the previewed plan agree on edit intent and target files.
                preview_requires_edit = preview_payload.get("requires_edit")
                preview_targets = preview_payload.get("target_files")
                return orchestrator.run(
                    request=user_question,
                    flow_context=flow_context_text,
                    index_dir=target_index_dir,
                    index_dirs=target_index_dirs,
                    k=resolved_k,
                    max_steps=chat_agent_max_steps,
                    timeout_seconds=agent_timeout_seconds,
                    tool_policy=_base_auto_execute_tool_policy(user_question, auto_chat_mode=auto_chat_mode),
                    pass_cap=auto_execute_max_passes,
                    on_event=CodingAgent._log_worker_event,
                    flow_id=active_flow_id,
                    run_id=run_id,
                    requires_edit=preview_requires_edit if isinstance(preview_requires_edit, bool) else None,
                    target_files=tuple(preview_targets) if isinstance(preview_targets, list) else (),
                )

            result_obj, debug_tail = _run_with_live_buffer(
                console,
                spinner_text="Auto-executing…",
                fn=_call,
                callbacks=[],
                show_all_logs=_cli_verbose_enabled(),
            )
            if hasattr(result_obj, "model_dump"):
                payload = result_obj.model_dump()
            elif isinstance(result_obj, dict):
                payload = dict(result_obj)
            else:
                payload = {"answer": str(result_obj)}
            merged_executor_warnings = [str(item).strip() for item in tools_execution_boot_warnings if str(item).strip()]
            existing_payload_warnings = (
                [str(item).strip() for item in payload.get("warnings", []) if str(item).strip()]
                if isinstance(payload.get("warnings"), list)
                else []
            )
            payload["warnings"] = [*existing_payload_warnings, *merged_executor_warnings]
            payload["prechecklist"] = preview_checklist
            payload["prechecklist_source"] = str(preview_payload.get("prechecklist_source", "") or "")
            payload["prechecklist_warning"] = preview_warning
            plan_payload = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
            objective = str(plan_payload.get("objective", "")).strip()
            if render_progress:
                for item in payload.get("pass_logs", []) if isinstance(payload.get("pass_logs"), list) else []:
                    if not isinstance(item, dict):
                        continue
                    _render_auto_execute_pass_status(
                        console,
                        objective=objective,
                        pass_index=int(item.get("pass_index", 0) or 0),
                        pass_cap=auto_execute_max_passes,
                        planner_step_id=str(item.get("planner_step_id", "") or ""),
                        planner_step_title=str(item.get("planner_step_title", "") or ""),
                        planner_decision=str(item.get("planner_decision", "") or ""),
                        planner_decision_reason=str(item.get("planner_decision_reason", "") or ""),
                        batch_reason=str(item.get("batch_reason", "") or ""),
                        expected_progress=str(item.get("expected_progress", "") or ""),
                    )
            for item in payload.get("pass_logs", []) if isinstance(payload.get("pass_logs"), list) else []:
                if not isinstance(item, dict):
                    continue
                pass_index = int(item.get("pass_index", 0) or 0)
                step_title = str(item.get("planner_step_title", "") or item.get("planner_step_id", "") or "planner pass")
                decision = str(item.get("planner_decision", "") or "").strip()
                expected = str(item.get("expected_progress", "") or "").strip()
                chat_ui_state.record_event(
                    make_event(
                        "agent.routing",
                        title=f"Pass {pass_index}",
                        message="; ".join(part for part in (step_title, decision, expected) if part),
                        status="success",
                        session_id=chat_ui_state.session_id,
                        turn_id=chat_ui_state.tracker.current_turn_id,
                        step_id="07",
                        metadata=dict(item),
                    ).finish(status="success")
                )
            return payload, debug_tail

        def _build_full_auto_resume_request(
            *,
            original_question: str,
            resume_cycle: int,
            terminal_reason: str,
        ) -> str:
            reason = str(terminal_reason or "").strip() or "pass_cap_reached"
            return (
                f"{original_question}\n\n"
                "FULL-AUTO RESUME DIRECTIVE:\n"
                f"- resume_cycle: {int(resume_cycle)}\n"
                f"- prior_terminal_reason: {reason}\n"
                "- Continue from current repository/flow state.\n"
                "- Do not ask for confirmation or scope-choice questions.\n"
                "- Execute inspect/edit/verify loops until complete or hard-blocked.\n"
            )

        def _ingest_full_auto_pass_payload(payload: dict[str, Any]) -> int:
            nonlocal full_auto_pass_window_logs
            nonlocal full_auto_pass_window_decisions
            nonlocal full_auto_latest_checklist_counts
            nonlocal full_auto_passes_total

            pass_logs = payload.get("pass_logs") if isinstance(payload.get("pass_logs"), list) else []
            normalized_logs = [item for item in pass_logs if isinstance(item, dict)]
            if normalized_logs:
                full_auto_pass_window_logs.extend(normalized_logs)
                full_auto_passes_total += len(normalized_logs)

            planner_rows = (
                payload.get("planner_decisions")
                if isinstance(payload.get("planner_decisions"), list)
                else []
            )
            normalized_planner_rows = [item for item in planner_rows if isinstance(item, dict)]
            if normalized_planner_rows:
                full_auto_pass_window_decisions.extend(normalized_planner_rows)

            counts = _resolve_payload_checklist_counts(payload)
            if counts is not None:
                full_auto_latest_checklist_counts = counts
            return len(normalized_logs)

        def _emit_full_auto_pass_checkpoints(*, resume_cycles: int) -> int:
            nonlocal full_auto_pass_window_logs
            nonlocal full_auto_pass_window_decisions
            nonlocal full_auto_pass_checkpoints_emitted

            if str(execution_profile or "").strip().lower() != "full-auto":
                return 0
            every = max(0, int(full_auto_status_every))
            if every <= 0:
                return 0

            emitted = 0
            while len(full_auto_pass_window_logs) >= every:
                checkpoint_logs = full_auto_pass_window_logs[:every]
                full_auto_pass_window_logs = full_auto_pass_window_logs[every:]

                take_rows = min(len(full_auto_pass_window_decisions), every)
                checkpoint_planner_rows = full_auto_pass_window_decisions[:take_rows]
                full_auto_pass_window_decisions = full_auto_pass_window_decisions[take_rows:]

                decision_rows = _checkpoint_decisions_from_pass_window(
                    checkpoint_planner_rows,
                    checkpoint_logs,
                )
                _render_full_auto_checkpoint(
                    console,
                    decision_rows=decision_rows,
                    checklist_counts=full_auto_latest_checklist_counts,
                    window_passes=len(checkpoint_logs),
                    pass_total=full_auto_passes_total,
                    resume_cycles=resume_cycles,
                )
                full_auto_pass_checkpoints_emitted += 1
                emitted += 1
            return emitted

        def _run_auto_execute_pipeline_with_resume(
            user_question: str,
            *,
            render_progress: bool = True,
            auto_chat_mode: AutoChatMode | None = None,
        ) -> tuple[dict[str, Any], str]:
            run_full_auto = str(execution_profile or "").strip().lower() == "full-auto"
            resume_cycles = 0
            resumed_from_pass_cap = False
            turn_passes_total = 0
            turn_checkpoints_emitted = 0
            effective_question = user_question
            resume_run_id: str | None = None
            last_debug_tail = ""

            while True:
                payload, debug_tail = _run_auto_execute_pipeline(
                    effective_question,
                    render_progress=(render_progress and resume_cycles == 0),
                    run_id=resume_run_id,
                    auto_chat_mode=auto_chat_mode,
                )
                last_debug_tail = debug_tail
                payload_run_id = str(payload.get("run_id", "") or "").strip()
                if payload_run_id:
                    resume_run_id = payload_run_id
                if run_full_auto:
                    turn_passes_total += _ingest_full_auto_pass_payload(payload)
                terminal_reason = str(payload.get("terminal_reason", "") or "").strip().lower()
                run_status = str(payload.get("run_status", "") or "").strip().lower()
                if chat_auto_continue and (terminal_reason == "pass_cap_reached" or run_status == "needs_resume"):
                    resumed_from_pass_cap = True
                    resume_cycles += 1
                    turn_checkpoints_emitted += _emit_full_auto_pass_checkpoints(resume_cycles=resume_cycles)
                    effective_question = _build_full_auto_resume_request(
                        original_question=user_question,
                        resume_cycle=resume_cycles,
                        terminal_reason=terminal_reason,
                    )
                    continue

                payload["full_auto_resume_cycles"] = int(resume_cycles)
                payload["full_auto_passes_total"] = int(turn_passes_total)
                payload["full_auto_pass_checkpoints_emitted"] = int(turn_checkpoints_emitted)
                payload["resumed_from_pass_cap"] = bool(resumed_from_pass_cap)
                return payload, last_debug_tail

        def _emit_auto_execute_terminal(
            *,
            user_question: str,
            payload: dict[str, Any],
            debug_tail: str,
            auto_chat_mode: AutoChatMode,
        ) -> None:
            nonlocal pending_ui_selection
            answer_raw = str(payload.get("answer", "") or "")
            answer_text, parsed_payload = _extract_structured_answer(answer_raw)
            payload_sources = payload.get("sources", []) if isinstance(payload.get("sources"), list) else []
            payload_trace = payload.get("trace", []) if isinstance(payload.get("trace"), list) else []
            payload_warnings = payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else []
            ui_blocks = _effective_ui_blocks(answer_text, parsed_payload if isinstance(parsed_payload, dict) else None)
            rendered_dynamic = _render_dynamic_blocks(
                console,
                ui_blocks,
                diagram_render_images=diagram_render_images,
                diagram_output_dir=resolved_diagram_output_dir,
                diagram_format=diagram_format,
                diagram_open_artifact=diagram_open,
                diagram_timeout_seconds=diagram_timeout_seconds,
                project_root=root,
            )
            selection_block = _pending_ui_selection_from_blocks(ui_blocks)
            if selection_block is not None:
                pending_ui_selection = selection_block

            warnings_merged = [str(item).strip() for item in payload_warnings if str(item).strip()]
            if isinstance(parsed_payload, dict):
                warnings_merged = _merge_warnings(warnings_merged, parsed_payload)
            auto_trace = _coerce_trace_items(payload_trace)
            changed_files = [str(item) for item in payload.get("changed_files", []) if str(item).strip()]
            if execution_profile == "full-auto":
                answer_text = _sanitize_full_auto_answer_text(
                    answer_text,
                    changed_files_count=len(changed_files),
                    terminal_reason=str(payload.get("terminal_reason", "") or ""),
                )
            turn_record = ChatTurnTelemetry(
                turn_index=len(session_turns) + 1,
                timestamp=_now_iso(),
                question=user_question,
                answer_text=answer_text,
                sources=payload_sources if isinstance(payload_sources, list) else [],
                warnings=warnings_merged,
                trace=auto_trace,
                tool_steps_total=len(auto_trace),
                decisions=_extract_decisions(
                    answer_text=answer_text,
                    warnings=warnings_merged,
                    payload=parsed_payload if isinstance(parsed_payload, dict) else None,
                ),
                changed_files=changed_files,
                has_diff=bool(changed_files),
                coding_state={
                    "plan": payload.get("plan"),
                    "progress": {
                        "phase": "answer",
                        "why": payload.get("terminal_reason", ""),
                    },
                    "checklist": None,
                    "next_step": payload.get("terminal_reason", ""),
                    "flow_id": active_flow_id,
                    "duplicate_request_skips": int(payload.get("duplicate_request_skips", 0) or 0),
                    "duplicate_semantic_search_skips": int(
                        payload.get("duplicate_semantic_search_skips", 0) or 0
                    ),
                    "request_retry_attempts": int(payload.get("request_retry_attempts", 0) or 0),
                    "request_retry_exhausted": int(payload.get("request_retry_exhausted", 0) or 0),
                    "edit_retry_mode_activations": int(payload.get("edit_retry_mode_activations", 0) or 0),
                    "persisted_fingerprint_counts": (
                        dict(payload.get("persisted_fingerprint_counts", {}))
                        if isinstance(payload.get("persisted_fingerprint_counts"), dict)
                        else {}
                    ),
                },
            )
            session_turns.append(turn_record)
            chat_ui_state.add_conversation_turn(turn_record.question, turn_record.answer_text)
            _ = rendered_dynamic
            _render_answer_header(console)
            console.print(Markdown(answer_text) if answer_text else "[dim](no answer text)[/dim]")
            if warnings_merged:
                warning_lines = "\n".join(f"- {w}" for w in warnings_merged[:12])
                console.print(Panel(warning_lines, title="Warnings", border_style="yellow"))
            if _cli_verbose_enabled() and debug_tail:
                console.print("\n[bold]Debug tail[/bold]\n" + debug_tail)
            chat_ui_state.record_event(
                make_event(
                    "agent.decision",
                    title="Agent decision",
                    message="Auto-execute completed and rendered the final answer.",
                    status="success",
                    session_id=chat_ui_state.session_id,
                    turn_id=current_turn_id,
                    step_id="05",
                    metadata={
                        "next_action": "final response",
                        "verification_target": str(payload.get("terminal_reason", "") or ""),
                    },
                ).finish(status="success")
            )
            _finish_ui_turn(current_turn_id)
            _log_chat_turn(
                run_logger,
                turn=turn_record,
                mode="tools-manager-auto-exec",
                dir_mode=dir_mode,
                coding_agent=bool(coding_agent_instance is not None),
                flow_id=active_flow_id,
                render_mode=str(payload.get("render_mode", "default") or "default"),
                fallback_reason=str(payload.get("fallback_reason", "") or ""),
                planning_mode=planning_mode,
                planning_question_source=planning_question_source,
                planning_question_index=planning_questions_asked_count,
                auto_execute_plan=True,
                auto_execute_passes=int(payload.get("passes", 0) or 0),
                auto_execute_terminal_reason=str(payload.get("terminal_reason", "") or ""),
                toolsmanager_requests_count=int(payload.get("toolsmanager_requests_count", 0) or 0),
                auto_execute_pass_logs=payload.get("pass_logs", []) if isinstance(payload.get("pass_logs"), list) else [],
                planner_decisions=payload.get("planner_decisions", []) if isinstance(payload.get("planner_decisions"), list) else [],
                prechecklist_source=str(payload.get("prechecklist_source", "") or ""),
                prechecklist_steps_count=(
                    len(payload.get("prechecklist", {}).get("steps", []))
                    if isinstance(payload.get("prechecklist"), dict)
                    and isinstance(payload.get("prechecklist", {}).get("steps"), list)
                    else 0
                ),
                prechecklist_warning=str(payload.get("prechecklist_warning", "") or ""),
                tool_execution_backend=str(payload.get("execution_backend", "") or ""),
                tool_execution_run_id=str(payload.get("execution_run_id", "") or ""),
                tool_execution_duration_ms=float(payload.get("execution_duration_ms", 0.0) or 0.0),
                tool_execution_requests_ok=int(payload.get("execution_requests_ok", 0) or 0),
                tool_execution_requests_failed=int(payload.get("execution_requests_failed", 0) or 0),
                full_auto_resume_cycles=int(payload.get("full_auto_resume_cycles", 0) or 0),
                full_auto_passes_total=int(payload.get("full_auto_passes_total", 0) or 0),
                full_auto_pass_checkpoints_emitted=int(payload.get("full_auto_pass_checkpoints_emitted", 0) or 0),
                resumed_from_pass_cap=bool(payload.get("resumed_from_pass_cap", False)),
                multiline_input=multiline_input,
                multiline_terminator=multiline_terminator,
            )
            _save_auto_chat_turn_state(
                mode=auto_chat_mode,
                task=user_question,
                answer_text=answer_text,
                sources=payload_sources if isinstance(payload_sources, list) else [],
                changed_files=changed_files,
                verification=str(payload.get("terminal_reason", "") or ""),
            )

        def _save_auto_chat_turn_state(
            *,
            mode: AutoChatMode,
            task: str,
            answer_text: str,
            sources: list[Any] | None = None,
            changed_files: list[str] | None = None,
            verification: str = "",
        ) -> None:
            nonlocal auto_chat_state
            source_paths: list[str] = []
            for item in sources or []:
                if isinstance(item, dict):
                    value = item.get("path") or item.get("file") or item.get("source")
                else:
                    value = str(item)
                text = str(value or "").strip()
                if text and text not in source_paths:
                    source_paths.append(text)
            auto_chat_state = AutoChatSessionState(
                last_mode=mode.value,
                last_task=task,
                relevant_files=source_paths[:12],
                changed_files=[str(item) for item in (changed_files or []) if str(item).strip()][:12],
                verification=verification,
                summary=str(answer_text or "").strip()[:800],
            )
            try:
                save_auto_chat_state(root, auto_chat_state)
            except Exception as exc:
                logger.debug("Failed to save auto-chat state: %s", exc)

        queued_questions = [prompt] if prompt else []

        while True:
            try:
                if queued_questions:
                    question = queued_questions.pop(0)
                    console.print(f"[bold cyan]mana ❯[/bold cyan] {question}")
                else:
                    question = _read_chat_input(
                        console,
                        prompt=CHAT_PROMPT,
                        multiline_enabled=multiline_input,
                        multiline_terminator=multiline_terminator,
                    )
            except (EOFError, KeyboardInterrupt):
                console.print("\nExiting chat.")
                logger.info("Chat session ended by user interrupt/EOF")
                break

            # Announce background-index completion once, so the user knows when
            # semantic search becomes active.
            if not background_index_state.get("announced", False):
                bg_status = background_index_state.get("status")
                if bg_status == "ready":
                    console.print("[green]Semantic index ready — semantic search is now active.[/green]")
                    background_index_state["announced"] = True
                elif bg_status == "failed":
                    console.print(
                        "[yellow]Background indexing failed; continuing with direct project search.[/yellow]"
                    )
                    background_index_state["announced"] = True

            if not question:
                continue
            if question.lower() in {"exit", "quit", "/exit", "/quit"}:
                console.print("Goodbye!")
                logger.info("Chat session ended by user command", extra={"command": question.lower()})
                break
            if question.lower() == "/clear":
                console.clear()
                console.print("[green]Chat history cleared. Session preserved.[/green]")
                continue
            if pending_conflict_question is None and _is_new_topic_command(question):
                reset_id = _start_new_topic()
                if reset_id:
                    console.print(f"[green]Started new chat topic; flow reset: {reset_id}[/green]")
                else:
                    console.print("[green]Started new chat topic.[/green]")
                continue
            if question.strip().startswith("/session"):
                from mana_agent.workspaces.service import WorkspaceService

                service = WorkspaceService()
                parts = question.strip().split()
                action = parts[1].lower() if len(parts) > 1 else "show"
                if action == "list":
                    rows = [
                        item
                        for item in service.store.list_sessions()
                        if item.workspace_id == chat_ui_state.workspace_id
                    ]
                    if not rows:
                        console.print("[yellow]No workspace sessions found.[/yellow]")
                    for item in rows[:30]:
                        marker = "*" if item.session_id == chat_ui_state.session_id else " "
                        console.print(f"{marker} {item.session_id}  {item.status}  {item.cwd}")
                    continue
                if action == "show":
                    console.print_json(
                        json.dumps(service.store.get_session(chat_ui_state.session_id).model_dump(mode="json"))
                    )
                    continue
                if action == "new":
                    created = service.create_session(root)
                    chat_ui_state.activate_session(created.session_id)
                    session_turns.clear()
                    active_flow_id = None
                    pending_conflict_question = None
                    console.print(f"[green]New isolated session:[/green] {created.session_id}")
                    continue
                if action == "switch" and len(parts) > 2:
                    try:
                        chat_ui_state.activate_session(parts[2])
                    except (FileNotFoundError, ValueError) as exc:
                        console.print(f"[red]{exc}[/red]")
                        continue
                    session_turns.clear()
                    active_flow_id = None
                    pending_conflict_question = None
                    console.print(f"[green]Switched session:[/green] {chat_ui_state.session_id}")
                    continue
                if action == "archive":
                    archived = service.archive_session(chat_ui_state.session_id)
                    console.print(f"[green]Archived session:[/green] {archived.session_id}")
                    continue
                console.print("[yellow]Use /session new|list|show|switch <id>|archive.[/yellow]")
                continue
            if question.strip() == "/workspace show":
                from mana_agent.workspaces.service import WorkspaceService

                service = WorkspaceService()
                console.print_json(
                    json.dumps(service.store.get_workspace(chat_ui_state.workspace_id).model_dump(mode="json"))
                )
                continue
            if question.strip() == "/repo list":
                from mana_agent.workspaces.service import WorkspaceService

                context = WorkspaceService().context_for_session(chat_ui_state.session_id)
                for item in context.repositories.values():
                    marker = "*" if item.repository_id == chat_ui_state.repository_id else " "
                    console.print(f"{marker} {item.repository_id}  {item.name}  {item.canonical_path}")
                continue
            current_turn_id = f"turn-{len(session_turns) + 1}-{uuid.uuid4().hex[:8]}"
            chat_ui_state.start_turn(current_turn_id)
            if question.lower() == "/help":
                question = "help"
            if question.strip().startswith("/plan"):
                plan_args = question.strip()[len("/plan"):].strip()
                question = f"plan {plan_args}" if plan_args else "plan the next repository change"

            if recorded_initial_prompt and question == prompt:
                recorded_initial_prompt = False
            else:
                _record_multi_agent_request(root, question, entrypoint="chat", session_id=chat_ui_state.session_id)

            # -----------------------------
            # /analyze slash command (read-only; writes only .mana/ artifacts).
            # Handled before any LLM/CodingAgent routing.
            # -----------------------------
            if is_analyze_command(question):
                outcome = handle_analyze_command(
                    analyze_command_args(question),
                    root_dir=root,
                    input_func=lambda prompt: _read_chat_input(
                        console,
                        prompt=prompt,
                        multiline_enabled=False,
                        multiline_terminator=multiline_terminator,
                    ),
                    llm_analyzer=_build_project_llm_analyzer(),
                )
                # Refresh the in-session analysis grounding so the next chat
                # questions can use the freshly generated report.
                if outcome.status == "generated":
                    refreshed = _load_analysis_context(root)
                    if refreshed:
                        analysis_context_text = refreshed
                _render_answer_header(console, title="Analyze")
                style = "yellow" if outcome.status == "error" else "white"
                console.print(f"[{style}]{outcome.message}[/{style}]")
                turn_record = ChatTurnTelemetry(
                    turn_index=len(session_turns) + 1,
                    timestamp=_now_iso(),
                    question=question,
                    answer_text=outcome.message,
                    sources=[],
                    warnings=list(outcome.result.errors) if outcome.result else [],
                    trace=[],
                    tool_steps_total=0,
                    decisions=[],
                    changed_files=(
                        [str(p) for p in outcome.result.written] if outcome.result else []
                    ),
                    has_diff=bool(outcome.result and outcome.result.written),
                    coding_state={"flow_id": active_flow_id},
                )
                session_turns.append(turn_record)
                chat_ui_state.add_conversation_turn(turn_record.question, turn_record.answer_text)
                _log_chat_turn(
                    run_logger,
                    turn=turn_record,
                    mode=f"slash-command:analyze:{outcome.status}",
                    dir_mode=dir_mode,
                    coding_agent=False,
                    flow_id=active_flow_id,
                    multiline_input=multiline_input,
                    multiline_terminator=multiline_terminator,
                )
                chat_ui_state.record_event(
                    make_event(
                        "verification.finished",
                        title="Analyze",
                        message=outcome.message,
                        status="failed" if outcome.status == "error" else "success",
                        session_id=chat_ui_state.session_id,
                        turn_id=current_turn_id,
                        step_id="09",
                    ).finish(status="failed" if outcome.status == "error" else "success")
                )
                _finish_ui_turn(current_turn_id)
                continue

            # -----------------------------
            # Direct command fast-path (no FAISS / RAG / CodingAgent).
            # -----------------------------
            direct_command = _classify_direct_command(question)
            if direct_command is not None:
                if dir_mode:
                    index_available = any(
                        _index_has_search_data(item) for item in dir_mode_index_dirs
                    )
                else:
                    index_available = bool(
                        resolved_index_dir is not None
                        and _index_has_search_data(resolved_index_dir)
                    )
                tool_worker_active = bool(
                    tool_worker_client is not None
                    and getattr(tool_worker_client, "_proc", None) is not None
                    and tool_worker_client._proc is not None
                    and tool_worker_client._proc.poll() is None
                )
                answer_text = _render_direct_command(
                    console,
                    direct_command,
                    project_root=root,
                    index_available=index_available,
                    coding_agent_active=coding_agent_instance is not None,
                    tool_worker_active=tool_worker_active,
                    ui_state=chat_ui_state,
                    raw_question=question,
                )
                chat_ui_state.record_event(
                    make_event(
                        "agent.decision",
                        title="Agent decision",
                        message="Handled by direct chat command without repository tool routing.",
                        status="success",
                        session_id=chat_ui_state.session_id,
                        turn_id=current_turn_id,
                        step_id="05",
                        metadata={
                            "next_action": "render direct command response",
                            "verification_target": "command output generated",
                        },
                    ).finish(status="success")
                )
                turn_record = ChatTurnTelemetry(
                    turn_index=len(session_turns) + 1,
                    timestamp=_now_iso(),
                    question=question,
                    answer_text=answer_text,
                    sources=[],
                    warnings=[],
                    trace=[],
                    tool_steps_total=0,
                    decisions=[],
                    changed_files=[],
                    has_diff=False,
                    coding_state={"flow_id": active_flow_id},
                )
                session_turns.append(turn_record)
                chat_ui_state.add_conversation_turn(turn_record.question, turn_record.answer_text)
                _finish_ui_turn(current_turn_id)
                _log_chat_turn(
                    run_logger,
                    turn=turn_record,
                    mode=f"direct-command:{direct_command}",
                    dir_mode=dir_mode,
                    coding_agent=False,
                    flow_id=active_flow_id,
                    multiline_input=multiline_input,
                    multiline_terminator=multiline_terminator,
                )
                continue

            original_auto_question = question
            question = resolve_auto_followup(question, auto_chat_state)
            agent_decision = _run_chat_event_step(
                "Decision routing...",
                lambda: _decide_chat_route(ask_service=ask_service, question=question, root=root),
                event_type="RoutingStarted",
            )
            auto_chat_mode = _auto_chat_mode_from_agent_decision(
                agent_decision,
                classify_auto_chat_intent(question),
            )
            required_mcp_server = _explicit_mcp_server_request(
                ask_service=ask_service,
                question=question,
            )
            if is_plan_execution_request(question):
                auto_chat_mode = AutoChatMode.EDIT

            small_direct_edit_result = _run_chat_event_step(
                "Checking direct edit...",
                lambda: handle_small_direct_edit(root, question),
                event_type="ReasoningStarted",
            )
            if small_direct_edit_result.handled:
                answer_text = small_direct_edit_result.answer
                _render_answer_header(console)
                console.print(Markdown(answer_text))
                turn_record = ChatTurnTelemetry(
                    turn_index=len(session_turns) + 1,
                    timestamp=_now_iso(),
                    question=question,
                    answer_text=answer_text,
                    sources=[{"path": path} for path in small_direct_edit_result.changed_files],
                    warnings=[] if small_direct_edit_result.ok else [small_direct_edit_result.error],
                    trace=_coerce_trace_items(small_direct_edit_result.trace),
                    tool_steps_total=len(small_direct_edit_result.trace),
                    decisions=[],
                    changed_files=list(small_direct_edit_result.changed_files),
                    has_diff=bool(small_direct_edit_result.changed_files),
                    coding_state={
                        "flow_id": active_flow_id,
                        "small_direct_edit": True,
                        "verification": "skipped_docs_only" if small_direct_edit_result.ok else "minimal_check_failed",
                    },
                )
                session_turns.append(turn_record)
                chat_ui_state.add_conversation_turn(turn_record.question, turn_record.answer_text)
                _log_chat_turn(
                    run_logger,
                    turn=turn_record,
                    mode="small-direct-edit",
                    dir_mode=dir_mode,
                    coding_agent=False,
                    flow_id=active_flow_id,
                    multiline_input=multiline_input,
                    multiline_terminator=multiline_terminator,
                )
                _save_auto_chat_turn_state(
                    mode=auto_chat_mode,
                    task=original_auto_question,
                    answer_text=answer_text,
                    sources=[{"path": path} for path in small_direct_edit_result.changed_files],
                    changed_files=small_direct_edit_result.changed_files,
                    verification="Verification skipped: docs-only one-line edit.",
                )
                chat_ui_state.record_event(
                    make_event(
                        "verification.finished",
                        title="Verification",
                        message="Skipped docs-only one-line edit." if small_direct_edit_result.ok else "Minimal edit check failed.",
                        status="success" if small_direct_edit_result.ok else "failed",
                        session_id=chat_ui_state.session_id,
                        turn_id=current_turn_id,
                        step_id="09",
                    ).finish(status="success" if small_direct_edit_result.ok else "failed")
                )
                _finish_ui_turn(current_turn_id)
                continue

            # -----------------------------
            # Model-selected immediate tools for read-only research/search turns.
            # -----------------------------
            if (
                any(tool in agent_decision.selected_tools for tool in ("web_search", "github_search"))
                and agent_decision.web_search_needed
                and not agent_decision.code_editing_needed
                and not required_mcp_server
            ):
                answer_text, sources, trace = _run_chat_event_step(
                    "Web search...",
                    lambda: _run_web_research_answer(
                        ask_service=ask_service,
                        question=question,
                        root=root,
                        decision=agent_decision,
                    ),
                    event_type="ToolStarted",
                    tool_name="web_search",
                )
                _render_answer_header(console, title="Web search")
                console.print(Markdown(answer_text))
                chat_ui_state.record_event(
                    make_event(
                        "agent.decision",
                        title="Agent decision",
                        message=agent_decision.reasoning_summary,
                        status="success" if agent_decision.verifier_passed else "failed",
                        session_id=chat_ui_state.session_id,
                        turn_id=current_turn_id,
                        step_id="05",
                        metadata={
                            "intent": agent_decision.intent,
                            "selected_tools": agent_decision.selected_tools,
                            "verifier": agent_decision.verifier_summary,
                        },
                    ).finish(status="success" if agent_decision.verifier_passed else "failed")
                )
                turn_record = ChatTurnTelemetry(
                    turn_index=len(session_turns) + 1,
                    timestamp=_now_iso(),
                    question=question,
                    answer_text=answer_text,
                    sources=sources,
                    warnings=[] if agent_decision.verifier_passed else [agent_decision.verifier_summary],
                    trace=_coerce_trace_items(trace),
                    tool_steps_total=len(trace),
                    decisions=[agent_decision.to_dict()],
                    changed_files=[],
                    has_diff=False,
                    coding_state={"flow_id": active_flow_id},
                )
                session_turns.append(turn_record)
                chat_ui_state.add_conversation_turn(turn_record.question, turn_record.answer_text)
                _finish_ui_turn(current_turn_id)
                _log_chat_turn(
                    run_logger,
                    turn=turn_record,
                    mode="web-search",
                    dir_mode=dir_mode,
                    coding_agent=False,
                    flow_id=active_flow_id,
                    multiline_input=multiline_input,
                    multiline_terminator=multiline_terminator,
                )
                _save_auto_chat_turn_state(
                    mode=auto_chat_mode,
                    task=original_auto_question,
                    answer_text=answer_text,
                    sources=sources,
                    changed_files=[],
                )
                continue

            if (
                "repo_search" in agent_decision.selected_tools
                and agent_decision.repo_context_needed
                and not agent_decision.code_editing_needed
                and coding_agent_instance is None
                and not required_mcp_server
            ):
                tool_query = str((agent_decision.tool_inputs.get("repo_search") or {}).get("query") or question).strip()
                search_result = _run_chat_event_step(
                    "Repository search...",
                    lambda: project_search(tool_query, root),
                    event_type="ToolStarted",
                    tool_name="repo_search",
                )
                if search_result.matches:
                    body = search_result.format(root)
                    if search_result.truncated:
                        body += "\n… (results truncated)"
                    answer_text = (
                        f"Found {len(search_result.matches)} match(es) "
                        f"for '{tool_query}' via {search_result.backend}:\n\n{body}"
                    )
                else:
                    answer_text = f"No matches for '{tool_query}' in {root}."
                console.print("\n[bold]Search results[/bold]")
                console.print(answer_text)
                chat_ui_state.record_event(
                    make_event(
                        "tool.finished",
                        title="project_search",
                        message=f"Exact search completed via {search_result.backend}.",
                        status="success",
                        session_id=chat_ui_state.session_id,
                        turn_id=current_turn_id,
                        step_id="08",
                        token_usage=chat_ui_state.tracker.record_tool_result(
                            f"{current_turn_id}-exact-search",
                            answer_text,
                            step_id="08",
                            turn_id=current_turn_id,
                        ),
                        metadata={
                            "tool_name": "project_search",
                            "args_summary": tool_query,
                            "result_summary": f"{len(search_result.matches)} match(es)",
                            "agent_decision": agent_decision.to_dict(),
                        },
                    ).finish(status="success")
                )
                turn_record = ChatTurnTelemetry(
                    turn_index=len(session_turns) + 1,
                    timestamp=_now_iso(),
                    question=question,
                    answer_text=answer_text,
                    sources=[],
                    warnings=[],
                    trace=[],
                    tool_steps_total=0,
                    decisions=[agent_decision.to_dict()],
                    changed_files=[],
                    has_diff=False,
                    coding_state={"flow_id": active_flow_id},
                )
                session_turns.append(turn_record)
                chat_ui_state.add_conversation_turn(turn_record.question, turn_record.answer_text)
                _finish_ui_turn(current_turn_id)
                _log_chat_turn(
                    run_logger,
                    turn=turn_record,
                    mode=f"repo-search:{search_result.backend}",
                    dir_mode=dir_mode,
                    coding_agent=False,
                    flow_id=active_flow_id,
                    multiline_input=multiline_input,
                    multiline_terminator=multiline_terminator,
                )
                _save_auto_chat_turn_state(
                    mode=auto_chat_mode,
                    task=original_auto_question,
                    answer_text=answer_text,
                    sources=[
                        {"path": item.file_path}
                        for item in search_result.matches[:12]
                        if getattr(item, "file_path", None)
                    ],
                    changed_files=[],
                )
                continue

            if pending_ui_selection is not None:
                if execution_profile == "full-auto":
                    option = _auto_select_ui_option(pending_ui_selection)
                    selection_id = str(pending_ui_selection.get("id", "selection") or "selection")
                    if option is not None:
                        option_id = str(option.get("id", "") or "").replace('"', '\\"')
                        value = str(option.get("value", option_id) or "").replace('"', '\\"')
                        question = (
                            f'User selected "{option_id}" for selection "{selection_id}" '
                            f'(value="{value}") in full-auto mode. Continue accordingly.'
                        )
                        logger.info(
                            "Full-auto selection resolution",
                            extra={"selection_id": selection_id, "option_id": option_id},
                        )
                        pending_ui_selection = None
                    else:
                        logger.info(
                            "Full-auto selection dropped (no options)",
                            extra={"selection_id": selection_id},
                        )
                        pending_ui_selection = None
                if pending_ui_selection is None:
                    pass
                else:
                    selection_kind, selection_payload = _resolve_ui_selection_input(pending_ui_selection, question)
                    selection_id = str(pending_ui_selection.get("id", "selection") or "selection")

                    if selection_kind == "invalid":
                        console.print("[yellow]Invalid selection. Choose by number, option id, or label.[/yellow]")
                        _render_selection_block(console, pending_ui_selection)
                        continue

                    if selection_kind == "free_text":
                        raw_text = str((selection_payload or {}).get("text", "") or "").replace('"', '\\"')
                        question = (
                            f'User provided free-text response "{raw_text}" '
                            f'for selection "{selection_id}". Continue accordingly.'
                        )
                        pending_ui_selection = None
                    else:
                        option = selection_payload or {}
                        option_id = str(option.get("id", "") or "").replace('"', '\\"')
                        value = str(option.get("value", option_id) or "").replace('"', '\\"')
                        question = (
                            f'User selected "{option_id}" for selection "{selection_id}" '
                            f'(value="{value}"). Continue accordingly.'
                        )
                        pending_ui_selection = None

            if coding_agent_instance is not None and question.startswith("/flow"):
                parts = question.strip().split()
                action = parts[1].lower() if len(parts) > 1 else "show"
                if action == "show":
                    summary = coding_agent_instance.flow_summary(active_flow_id)
                    if not summary:
                        console.print("[yellow]No active coding flow.[/yellow]")
                        continue
                    console.print("[bold]Flow memory active[/bold]")
                    _render_flow_summary(
                        console,
                        summary,
                        include_checklist=False,
                        include_transitions=False,
                        include_recent_turns=False,
                    )
                    continue
                if action == "checklist":
                    summary = coding_agent_instance.flow_summary(active_flow_id)
                    checklist = summary.get("checklist") if isinstance(summary, dict) else None
                    if not isinstance(checklist, dict):
                        console.print("[yellow]No checklist stored for the active flow.[/yellow]")
                        continue
                    _render_flow_checklist(console, checklist)
                    continue
                if action == "checkpoint":
                    checkpointed = coding_agent_instance.checkpoint_flow(active_flow_id)
                    if checkpointed:
                        console.print(f"[green]Checkpoint saved for flow {checkpointed}.[/green]")
                    else:
                        console.print("[yellow]No active flow to checkpoint.[/yellow]")
                    continue
                if action == "reset":
                    reset_id = coding_agent_instance.reset_flow(active_flow_id)
                    active_flow_id = None
                    pending_conflict_question = None
                    if reset_id:
                        console.print(f"[green]Flow reset: {reset_id}[/green]")
                    else:
                        console.print("[yellow]No active flow to reset.[/yellow]")
                    continue
                console.print(
                    "[yellow]Unknown /flow command. Use /flow show, /flow checklist, /flow checkpoint, or /flow reset.[/yellow]"
                )
                continue

            plan_execution_request = is_plan_execution_request(question)
            plan_trigger_request = bool(auto_chat_mode == AutoChatMode.PLAN_ONLY or plan_execution_request)
            edit_request = bool(auto_chat_mode == AutoChatMode.EDIT)
            auto_planning_turn = bool(
                planning_request is not None
                or planning_mode
                or (plan_trigger_request and not agent_tools_explicit)
            )
            force_plan_only_response = bool(auto_chat_mode == AutoChatMode.PLAN_ONLY)
            force_auto_execute_edit = bool(
                coding_agent_instance is not None
                and auto_execute_plan
                and edit_request
            )

            if (
                coding_agent_instance is not None
                and auto_execute_plan
                and hasattr(coding_agent_instance, "generate_auto_execute")
                and not auto_planning_turn
                and force_auto_execute_edit
            ):
                pending_prechecklist = None
                pending_prechecklist_source = ""
                pending_prechecklist_warning = ""
                if hasattr(coding_agent_instance, "preview_execution_checklist"):
                    try:
                        preview_payload = coding_agent_instance.preview_execution_checklist(
                            question,
                            flow_id=active_flow_id,
                        )
                    except Exception as exc:
                        _log_exception("coding_agent.preview_execution_checklist", exc)
                        preview_payload = {}
                    if isinstance(preview_payload, dict):
                        flow_id_from_preview = preview_payload.get("flow_id")
                        if isinstance(flow_id_from_preview, str) and flow_id_from_preview.strip():
                            active_flow_id = flow_id_from_preview.strip()
                        preview = preview_payload.get("prechecklist")
                        if isinstance(preview, dict):
                            pending_prechecklist = preview
                            pending_prechecklist_source = str(preview_payload.get("prechecklist_source", "") or "")
                            pending_prechecklist_warning = str(preview_payload.get("prechecklist_warning", "") or "")
                target_flow = active_flow_id or coding_agent_instance.get_active_flow_id()
                if isinstance(target_flow, str) and target_flow.strip():
                    active_flow_id = target_flow.strip()

            if auto_planning_turn:
                if planning_request is None:
                    planning_request = question
                    planning_answers = []
                    planning_questions = []
                    planning_questions_asked_count = 0
                    if not planning_question_llm_disabled:
                        try:
                            llm_question = _generate_planning_question_llm(
                                ask_service=ask_service,
                                planning_request=planning_request,
                                prior_questions=planning_questions,
                                prior_answers=planning_answers,
                                asked_count=0,
                                max_questions=planning_question_limit,
                            )
                            planning_question_source = "llm"
                        except Exception as exc:
                            if _is_planning_question_auth_failure(exc):
                                planning_question_llm_disabled = True
                            if not planning_question_failure_logged:
                                logger.warning("Planning question generation failed; using static fallback: %s", exc)
                                planning_question_failure_logged = True
                            llm_question = _planning_questions(planning_question_limit)[0]
                            planning_question_source = "fallback_static"
                    else:
                        llm_question = _planning_questions(planning_question_limit)[0]
                        planning_question_source = "fallback_static"
                    planning_questions.append(llm_question)
                    console.print(
                        f"[cyan]Planning request:[/cyan] {planning_request}\n"
                        f"[bold]Planning question 1/{planning_question_limit}[/bold]\n"
                        f"{llm_question}"
                    )
                    continue

                planning_answers.append(question)
                if len(planning_answers) < planning_question_limit:
                    asked_count = len(planning_answers)
                    if not planning_question_llm_disabled:
                        try:
                            llm_question = _generate_planning_question_llm(
                                ask_service=ask_service,
                                planning_request=planning_request,
                                prior_questions=planning_questions,
                                prior_answers=planning_answers,
                                asked_count=asked_count,
                                max_questions=planning_question_limit,
                            )
                            planning_question_source = "llm"
                        except Exception as exc:
                            if _is_planning_question_auth_failure(exc):
                                planning_question_llm_disabled = True
                            if not planning_question_failure_logged:
                                logger.warning("Planning question generation failed; using static fallback: %s", exc)
                                planning_question_failure_logged = True
                            llm_question = _planning_questions(planning_question_limit)[asked_count]
                            planning_question_source = "fallback_static"
                    else:
                        llm_question = _planning_questions(planning_question_limit)[asked_count]
                        planning_question_source = "fallback_static"
                    planning_questions.append(llm_question)
                    console.print(
                        f"[cyan]Planning request:[/cyan] {planning_request}\n"
                        f"[bold]Planning question {asked_count + 1}/{planning_question_limit}[/bold]\n"
                        f"{llm_question}"
                    )
                    continue

                question = _build_planning_instruction(
                    planning_request,
                    planning_answers,
                    planning_question_limit,
                    questions=planning_questions,
                )
                logger.info(
                    "Planning Q&A complete; generating plan response",
                    extra={
                        "planning_request": planning_request,
                        "answers_count": len(planning_answers),
                        "question_source": planning_question_source,
                    },
                )
                planning_questions_asked_count = len(planning_answers)
                planning_request = None
                planning_answers = []
                planning_questions = []
                if agent_tools and legacy_auto_execute_plan_requested:
                    auto_payload, auto_debug_tail = _run_auto_execute_pipeline_with_resume(
                        question,
                        auto_chat_mode=auto_chat_mode,
                    )
                    _emit_auto_execute_terminal(
                        user_question=question,
                        payload=auto_payload,
                        debug_tail=auto_debug_tail,
                        auto_chat_mode=auto_chat_mode,
                    )
                    continue
                force_plan_only_response = True
                console.print("[cyan]Generating decision-complete plan...[/cyan]")

            logger.info("Chat question received", extra={"question": question, "dir_mode": dir_mode, "agent_tools": agent_tools})

            if (
                coding_agent_instance is not None
                and coding_memory
                and pending_conflict_question is not None
                and not plan_trigger_request
            ):
                choice = question.strip().lower()
                if choice in {"continue", "c", "1"}:
                    question = pending_conflict_question
                elif choice in {"new", "n", "2"} or _is_new_topic_command(question):
                    conflict_question = pending_conflict_question
                    _start_new_topic()
                    question = conflict_question
                elif edit_request:
                    logger.info(
                        "Pending flow conflict replaced by new edit request",
                        extra={
                            "flow_id": active_flow_id,
                            "pending_question": pending_conflict_question,
                            "question": question,
                        },
                    )
                    _start_new_topic()
                else:
                    console.print("[yellow]Reply 'continue' or 'new topic'.[/yellow]")
                    continue
                pending_conflict_question = None

            if (
                edit_request
                and coding_agent_instance is None
                and not (agent_tools and auto_execute_plan and tool_worker_process and plan_trigger_request)
            ):
                console.print(
                    "[yellow]This chat session is read-only for file edits.[/yellow] "
                    "Re-run with [bold]--agent-tools --coding-agent[/bold] to allow create_file/write_file/apply_patch/delete_file."
                )
                continue

            if (
                coding_agent_instance is not None
                and coding_memory
                and edit_request
                and not plan_trigger_request
            ):
                if active_flow_id and coding_agent_instance.is_conflicting_request(question, active_flow_id):
                    if execution_profile == "full-auto":
                        logger.info(
                            "Full-auto flow conflict auto-continued",
                            extra={"flow_id": active_flow_id, "question": question},
                        )
                    else:
                        pending_conflict_question = question
                        console.print(
                            "[yellow]This request appears to diverge from the active flow.[/yellow] "
                            "Type [bold]continue[/bold] to keep current flow or [bold]new/new topic[/bold] to start a new flow."
                        )
                        continue

            if (
                coding_agent_instance is None
                and agent_tools
                and auto_execute_plan
                and (
                    edit_request
                    or plan_execution_request
                    or legacy_auto_execute_plan_requested
                    or execution_profile == "full-auto"
                )
                and (plan_trigger_request or edit_request)
            ):
                auto_payload, auto_debug_tail = _run_auto_execute_pipeline_with_resume(
                    question,
                    render_progress=False,
                    auto_chat_mode=auto_chat_mode,
                )
                _emit_auto_execute_terminal(
                    user_question=question,
                    payload=auto_payload,
                    debug_tail=auto_debug_tail,
                    auto_chat_mode=auto_chat_mode,
                )
                continue

            use_coding_agent_turn = _should_use_coding_agent_turn(
                coding_agent_available=coding_agent_instance is not None,
                agent_tools=bool(agent_tools),
                edit_request=edit_request,
                plan_trigger_request=plan_trigger_request,
                force_plan_only_response=force_plan_only_response,
                has_pending_prechecklist=pending_prechecklist is not None,
                coding_agent_is_custom=coding_agent_is_custom,
                general_coding_agent_turns=bool(coding_agent_explicit or coding_agent_is_custom),
            )

            if not use_coding_agent_turn:
                try:
                    response = chat_service.ask(question)
                except Exception as exc:
                    _log_exception("chat_service.ask", exc)
                    console.print("[red]Chat request failed.[/red]")
                    continue
                if response is None:
                    continue

                answer_raw = str(getattr(response, "answer", "") or "")
                answer_text, parsed_payload = _extract_structured_answer(answer_raw)
                sources = list(getattr(response, "sources", []) or [])
                warnings = [str(item).strip() for item in getattr(response, "warnings", []) or [] if str(item).strip()]
                trace = _coerce_trace_items(list(getattr(response, "trace", []) or []))
                if isinstance(parsed_payload, dict):
                    warnings = _merge_warnings(warnings, parsed_payload)
                    if isinstance(parsed_payload.get("sources"), list) and not sources:
                        sources = list(parsed_payload.get("sources") or [])
                    if isinstance(parsed_payload.get("trace"), list) and not trace:
                        trace = _coerce_trace_items(parsed_payload.get("trace"))
                try:
                    ui_blocks = _effective_ui_blocks(answer_text, parsed_payload if isinstance(parsed_payload, dict) else None)
                except Exception as exc:
                    _log_exception("chat.effective_ui_blocks.normal", exc)
                    ui_blocks = []
                    warnings.append("ui_blocks_render_fallback: failed to process ui_blocks payload")
                _render_dynamic_blocks(
                    console,
                    ui_blocks,
                    diagram_render_images=diagram_render_images,
                    diagram_output_dir=resolved_diagram_output_dir,
                    diagram_format=diagram_format,
                    diagram_open_artifact=diagram_open,
                    diagram_timeout_seconds=diagram_timeout_seconds,
                    project_root=root,
                )
                selection_block = _pending_ui_selection_from_blocks(ui_blocks)
                if selection_block is not None:
                    pending_ui_selection = selection_block
                mode_name = str(getattr(response, "mode", "") or "").strip()
                if not mode_name:
                    mode_name = "agent-tools" if (agent_tools_explicit and agent_tools) else "classic"
                turn_record = ChatTurnTelemetry(
                    turn_index=len(session_turns) + 1,
                    timestamp=_now_iso(),
                    question=question,
                    answer_text=answer_text,
                    sources=sources,
                    warnings=warnings,
                    trace=trace,
                    tool_steps_total=len(trace),
                    decisions=_extract_decisions(answer_text=answer_text, warnings=warnings),
                    changed_files=[],
                    has_diff=False,
                    coding_state={"flow_id": active_flow_id},
                )
                session_turns.append(turn_record)
                chat_ui_state.add_conversation_turn(turn_record.question, turn_record.answer_text)
                _render_turn_transparency(console, turn=turn_record, history=session_turns)
                _log_chat_turn(
                    run_logger,
                    turn=turn_record,
                    mode=mode_name,
                    dir_mode=dir_mode,
                    coding_agent=False,
                    flow_id=active_flow_id,
                    planning_mode=planning_mode,
                    planning_question_source=planning_question_source,
                    planning_question_index=planning_questions_asked_count,
                    multiline_input=multiline_input,
                    multiline_terminator=multiline_terminator,
                )
                _save_auto_chat_turn_state(
                    mode=auto_chat_mode,
                    task=original_auto_question,
                    answer_text=answer_text,
                    sources=sources,
                    changed_files=[],
                )
                chat_ui_state.record_event(
                    make_event(
                        "agent.decision",
                        title="Agent decision",
                        message="Answered through the standard chat service path.",
                        status="success",
                        session_id=chat_ui_state.session_id,
                        turn_id=current_turn_id,
                        step_id="05",
                        metadata={
                            "next_action": "final response",
                            "verification_target": "answer rendered with transparency sections",
                        },
                    ).finish(status="success")
                )
                _finish_ui_turn(current_turn_id)
                continue

            # ==========================================================
            # ✅ CODING AGENT PATH (classic + dir-mode supported)
            # ==========================================================
            if coding_agent_instance is not None:
                cb = RichToolCallbackHandler(show_inputs=True)
                execute_plan_now = bool(
                    auto_execute_plan
                    and not force_plan_only_response
                    and (plan_trigger_request or force_auto_execute_edit)
                )
                auto_execute_available = bool(execute_plan_now and hasattr(coding_agent_instance, "generate_auto_execute"))
                request_for_generation = question
                turn_full_auto_resume_cycles = 0
                turn_full_auto_passes_total = 0
                turn_full_auto_pass_checkpoints_emitted = 0
                turn_resumed_from_pass_cap = False
                turn_resume_run_id: str | None = None

                try:
                    if dir_mode:
                        index_dirs = dir_mode_index_dirs
                        if not index_dirs:
                            console.print("[red]No indexes available in dir-mode.[/red]")
                            continue

                        def _call(callbacks: list[BaseCallbackHandler]):
                            if auto_execute_available:
                                return coding_agent_instance.generate_auto_execute(
                                    request_for_generation,
                                    index_dirs=index_dirs,
                                    k=resolved_k,
                                    max_steps=coding_agent_max_steps,
                                    timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                    pass_cap=auto_execute_max_passes,
                                    callbacks=callbacks,
                                    flow_id=active_flow_id,
                                    run_id=turn_resume_run_id,
                                    auto_chat_mode=auto_chat_mode.value,
                                    prechecklist_payload=(
                                        {
                                            "flow_id": active_flow_id,
                                            "prechecklist": pending_prechecklist,
                                            "prechecklist_source": pending_prechecklist_source,
                                            "prechecklist_warning": pending_prechecklist_warning,
                                        }
                                        if isinstance(pending_prechecklist, dict)
                                        else None
                                    ),
                                )
                            return coding_agent_instance.generate_dir_mode(
                                request_for_generation,
                                index_dirs=index_dirs,
                                k=resolved_k,
                                max_steps=coding_agent_max_steps,
                                timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                callbacks=callbacks,
                                flow_id=active_flow_id,
                                auto_chat_mode=auto_chat_mode.value,
                            )
                    else:
                        assert resolved_index_dir is not None

                        def _call(callbacks: list[BaseCallbackHandler]):
                            if auto_execute_available:
                                return coding_agent_instance.generate_auto_execute(
                                    request_for_generation,
                                    index_dir=resolved_index_dir,
                                    k=resolved_k,
                                    max_steps=coding_agent_max_steps,
                                    timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                    pass_cap=auto_execute_max_passes,
                                    callbacks=callbacks,
                                    flow_id=active_flow_id,
                                    run_id=turn_resume_run_id,
                                    auto_chat_mode=auto_chat_mode.value,
                                    prechecklist_payload=(
                                        {
                                            "flow_id": active_flow_id,
                                            "prechecklist": pending_prechecklist,
                                            "prechecklist_source": pending_prechecklist_source,
                                            "prechecklist_warning": pending_prechecklist_warning,
                                        }
                                        if isinstance(pending_prechecklist, dict)
                                        else None
                                    ),
                                )
                            return coding_agent_instance.generate(
                                request_for_generation,
                                index_dir=resolved_index_dir,
                                k=resolved_k,
                                max_steps=coding_agent_max_steps,
                                timeout_seconds=min(max(agent_timeout_seconds, 60), 600),
                                callbacks=callbacks,
                                flow_id=active_flow_id,
                                auto_chat_mode=auto_chat_mode.value,
                            )

                    activity = LiveToolActivity(
                        spinner_text="Coding…",
                        show_all_logs=_cli_verbose_enabled(),
                    )
                    set_active_tool_activity(activity)
                    use_live_activity = bool(_use_live_tool_activity(console))
                    cycle_result: dict[str, object] = {}

                    def _run_generation_cycles() -> None:
                        nonlocal active_flow_id
                        nonlocal request_for_generation
                        nonlocal turn_resumed_from_pass_cap
                        nonlocal turn_full_auto_resume_cycles
                        nonlocal turn_full_auto_pass_checkpoints_emitted
                        nonlocal turn_full_auto_passes_total
                        nonlocal turn_resume_run_id

                        while True:
                            cycle_payload, cycle_debug_tail = _run_with_live_buffer(
                                console,
                                spinner_text="Coding…",
                                fn=_call,
                                callbacks=[cb],
                                show_all_logs=_cli_verbose_enabled(),
                                activity=activity,
                                manage_live=False,
                            )
                            cycle_result["result"] = cycle_payload
                            cycle_result["debug_tail"] = cycle_debug_tail
                            if not (
                                chat_auto_continue
                                and execute_plan_now
                                and isinstance(cycle_payload, dict)
                            ):
                                break
                            turn_full_auto_passes_total += _ingest_full_auto_pass_payload(cycle_payload)
                            terminal_reason = str((cycle_payload or {}).get("auto_execute_terminal_reason", "") or "").strip().lower()
                            flow_from_cycle = (cycle_payload or {}).get("flow_id")
                            if isinstance(flow_from_cycle, str) and flow_from_cycle.strip():
                                active_flow_id = flow_from_cycle.strip()
                            run_from_cycle = (cycle_payload or {}).get("run_id")
                            if isinstance(run_from_cycle, str) and run_from_cycle.strip():
                                turn_resume_run_id = run_from_cycle.strip()
                            run_status = str((cycle_payload or {}).get("run_status", "") or "").strip().lower()
                            if terminal_reason != "pass_cap_reached" and run_status != "needs_resume":
                                break
                            turn_resumed_from_pass_cap = True
                            turn_full_auto_resume_cycles += 1
                            turn_full_auto_pass_checkpoints_emitted += _emit_full_auto_pass_checkpoints(
                                resume_cycles=turn_full_auto_resume_cycles
                            )
                            request_for_generation = _build_full_auto_resume_request(
                                original_question=question,
                                resume_cycle=turn_full_auto_resume_cycles,
                                terminal_reason=terminal_reason,
                            )
                            continue

                    try:
                        live_context = (
                            Live(activity, console=console, refresh_per_second=12, transient=True)
                            if use_live_activity
                            else nullcontext()
                        )
                        with live_context:
                            _run_generation_cycles()
                        result = cycle_result.get("result")
                        debug_tail = str(cycle_result.get("debug_tail", "") or "")
                    finally:
                        set_active_tool_activity(None)
                        console.print(activity)

                except ToolWorkerProcessError as exc:
                    _log_exception("coding_agent.generate.worker", exc)
                    if exc.code == "tools_only_violation":
                        console.print(
                            "[yellow]Tools-only policy blocked this request:[/yellow] "
                            "no successful tool calls were executed. "
                            "Ask for specific file/tool actions and retry."
                        )
                        continue
                    console.print(
                        "[red]Coding agent worker failed.[/red] "
                        "Retrying this turn read-only is recommended."
                    )
                    continue
                except Exception as exc:
                    _log_exception("coding_agent.generate", exc)
                    console.print("[red]Coding agent failed.[/red]")
                    continue

                # result schema expected from CodingAgent
                answer = str((result or {}).get("answer", "") or "")
                changed = (result or {}).get("changed_files", []) or []
                diff = str((result or {}).get("diff", "") or "")
                warns = (result or {}).get("warnings", []) or []
                result_flow_id = (result or {}).get("flow_id")
                if isinstance(result_flow_id, str) and result_flow_id.strip():
                    active_flow_id = result_flow_id.strip()
                if execute_plan_now and isinstance(result, dict):
                    if isinstance(pending_prechecklist, dict) and not isinstance(result.get("prechecklist"), dict):
                        result["prechecklist"] = pending_prechecklist
                    if pending_prechecklist_source and not str(result.get("prechecklist_source", "")).strip():
                        result["prechecklist_source"] = pending_prechecklist_source
                    if pending_prechecklist_warning and not str(result.get("prechecklist_warning", "")).strip():
                        result["prechecklist_warning"] = pending_prechecklist_warning
                    result["full_auto_resume_cycles"] = int(turn_full_auto_resume_cycles)
                    result["full_auto_passes_total"] = int(turn_full_auto_passes_total)
                    result["full_auto_pass_checkpoints_emitted"] = int(turn_full_auto_pass_checkpoints_emitted)
                    result["resumed_from_pass_cap"] = bool(turn_resumed_from_pass_cap)
                answer_text, parsed_payload = _extract_structured_answer(answer)
                if execution_profile == "full-auto" and execute_plan_now:
                    answer_text = _sanitize_full_auto_answer_text(
                        answer_text,
                        changed_files_count=len([str(item) for item in changed if str(item).strip()]),
                        terminal_reason=(
                            str((result or {}).get("auto_execute_terminal_reason", "") or "")
                            if isinstance(result, dict)
                            else ""
                        ),
                    )
                payload_sources = []
                payload_warnings = []
                payload_trace = []
                payload_ui_blocks: list[dict[str, Any]] = []
                if isinstance(parsed_payload, dict):
                    if isinstance(parsed_payload.get("sources"), list):
                        payload_sources = parsed_payload["sources"]
                    if isinstance(parsed_payload.get("warnings"), list):
                        payload_warnings = [str(w).strip() for w in parsed_payload["warnings"] if str(w).strip()]
                    if isinstance(parsed_payload.get("trace"), list):
                        payload_trace = parsed_payload["trace"]
                try:
                    payload_ui_blocks = _effective_ui_blocks(answer_text, parsed_payload if isinstance(parsed_payload, dict) else None)
                except Exception as exc:
                    _log_exception("chat.effective_ui_blocks", exc)
                    payload_ui_blocks = []
                    payload_warnings.append("ui_blocks_render_fallback: failed to process ui_blocks payload")
                merged_warns = list(warns)
                for warning in payload_warnings:
                    if warning not in merged_warns:
                        merged_warns.append(warning)
                # CodingAgent outputs actions_taken, not trace
                result_actions = (result or {}).get("actions_taken", []) or []
                result_trace = (result or {}).get("trace", []) or []  # legacy/compat
                effective_trace = result_actions or result_trace or payload_trace
                raw_actions_total = (result or {}).get("actions_taken_total")
                if isinstance(raw_actions_total, int):
                    actions_total = raw_actions_total
                else:
                    actions_total = len(effective_trace)
                if isinstance(result, dict):
                    existing_actions = result.get("actions_taken")
                    if effective_trace:
                        result["actions_taken"] = effective_trace
                    elif isinstance(existing_actions, list):
                        result["actions_taken"] = existing_actions
                    else:
                        result["actions_taken"] = []
                    result["actions_taken_total"] = actions_total
                    result["actions_taken_truncated"] = actions_total > len(result.get("actions_taken", []))
                    result["warnings"] = merged_warns
                render_mode = (
                    str((result or {}).get("render_mode", "")).strip().lower()
                    if isinstance(result, dict)
                    else ""
                )
                fallback_reason = (
                    str((result or {}).get("fallback_reason", "")).strip().lower()
                    if isinstance(result, dict)
                    else ""
                )
                answer_only_fallback = render_mode == "answer_only" and fallback_reason == "tools_only_violation"
                answer_only_auto_execute = bool(auto_execute_available)
                edit_completed = bool(changed) or bool(diff.strip())
                answer_only_no_edit = (not answer_only_fallback) and (not edit_completed) and (not answer_only_auto_execute)

                rendered_dynamic: dict[str, bool] = {}
                if not answer_only_fallback:
                    rendered_dynamic = _render_dynamic_blocks(
                        console,
                        payload_ui_blocks,
                        diagram_render_images=diagram_render_images,
                        diagram_output_dir=resolved_diagram_output_dir,
                        diagram_format=diagram_format,
                        diagram_open_artifact=diagram_open,
                        diagram_timeout_seconds=diagram_timeout_seconds,
                        project_root=root,
                    )
                selection_block = _pending_ui_selection_from_blocks(payload_ui_blocks)
                if selection_block is not None:
                    pending_ui_selection = selection_block
                turn_record = ChatTurnTelemetry(
                    turn_index=len(session_turns) + 1,
                    timestamp=_now_iso(),
                    question=question,
                    answer_text=answer_text,
                    sources=list(payload_sources),
                    warnings=list(merged_warns),
                    trace=_coerce_trace_items(effective_trace),
                    tool_steps_total=actions_total,
                    decisions=_extract_decisions(
                        answer_text=answer_text,
                        warnings=merged_warns,
                        payload=parsed_payload if isinstance(parsed_payload, dict) else None,
                        result_payload=result if isinstance(result, dict) else None,
                    ),
                    changed_files=[str(item) for item in changed if str(item).strip()],
                    has_diff=bool(diff.strip()),
                    coding_state={
                        "plan": (result or {}).get("plan") if isinstance(result, dict) else None,
                        "progress": (result or {}).get("progress") if isinstance(result, dict) else None,
                        "checklist": (result or {}).get("checklist") if isinstance(result, dict) else None,
                        "next_step": (result or {}).get("next_step") if isinstance(result, dict) else None,
                        "flow_id": active_flow_id,
                        "duplicate_request_skips": (
                            int((result or {}).get("duplicate_request_skips", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "duplicate_semantic_search_skips": (
                            int((result or {}).get("duplicate_semantic_search_skips", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "request_retry_attempts": (
                            int((result or {}).get("request_retry_attempts", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "request_retry_exhausted": (
                            int((result or {}).get("request_retry_exhausted", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "edit_retry_mode_activations": (
                            int((result or {}).get("edit_retry_mode_activations", 0) or 0)
                            if isinstance(result, dict)
                            else 0
                        ),
                        "persisted_fingerprint_counts": (
                            dict((result or {}).get("persisted_fingerprint_counts", {}))
                            if isinstance(result, dict)
                            and isinstance((result or {}).get("persisted_fingerprint_counts"), dict)
                            else {}
                        ),
                    },
                )
                session_turns.append(turn_record)
                chat_ui_state.add_conversation_turn(turn_record.question, turn_record.answer_text)
                if answer_only_fallback or answer_only_no_edit or answer_only_auto_execute:
                    _render_answer_header(console)
                    if answer_text:
                        console.print(Markdown(answer_text))
                    else:
                        console.print("[dim](no answer text)[/dim]")
                else:
                    _render_turn_transparency(
                        console,
                        turn=turn_record,
                        history=session_turns,
                    )
                _log_chat_turn(
                    run_logger,
                    turn=turn_record,
                    mode="coding-agent",
                    dir_mode=dir_mode,
                    coding_agent=True,
                    flow_id=active_flow_id,
                    render_mode=render_mode,
                    fallback_reason=fallback_reason,
                    planning_mode=planning_mode,
                    planning_question_source=planning_question_source,
                    planning_question_index=planning_questions_asked_count,
                    auto_execute_plan=execute_plan_now,
                    auto_execute_passes=int((result or {}).get("auto_execute_passes", 0) or 0) if isinstance(result, dict) else 0,
                    auto_execute_terminal_reason=str((result or {}).get("auto_execute_terminal_reason", "") or "") if isinstance(result, dict) else "",
                    toolsmanager_requests_count=int((result or {}).get("toolsmanager_requests_count", 0) or 0) if isinstance(result, dict) else 0,
                    auto_execute_pass_logs=(result or {}).get("pass_logs", []) if isinstance(result, dict) else [],
                    planner_decisions=(result or {}).get("planner_decisions", []) if isinstance(result, dict) else [],
                    prechecklist_source=(
                        str((result or {}).get("prechecklist_source", "") or "")
                        if isinstance(result, dict)
                        else pending_prechecklist_source
                    ),
                    prechecklist_steps_count=(
                        len((result or {}).get("prechecklist", {}).get("steps", []))
                        if isinstance(result, dict)
                        and isinstance((result or {}).get("prechecklist"), dict)
                        and isinstance((result or {}).get("prechecklist", {}).get("steps"), list)
                        else (len(pending_prechecklist.get("steps", [])) if isinstance(pending_prechecklist, dict) and isinstance(pending_prechecklist.get("steps"), list) else 0)
                    ),
                    prechecklist_warning=(
                        str((result or {}).get("prechecklist_warning", "") or "")
                        if isinstance(result, dict)
                        else pending_prechecklist_warning
                    ),
                    tool_execution_backend=(
                        str((result or {}).get("tool_execution_backend", "") or "")
                        if isinstance(result, dict)
                        else ""
                    ),
                    tool_execution_run_id=(
                        str((result or {}).get("tool_execution_run_id", "") or "")
                        if isinstance(result, dict)
                        else ""
                    ),
                    tool_execution_duration_ms=(
                        float((result or {}).get("tool_execution_duration_ms", 0.0) or 0.0)
                        if isinstance(result, dict)
                        else 0.0
                    ),
                    tool_execution_requests_ok=(
                        int((result or {}).get("tool_execution_requests_ok", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    tool_execution_requests_failed=(
                        int((result or {}).get("tool_execution_requests_failed", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    full_auto_resume_cycles=(
                        int((result or {}).get("full_auto_resume_cycles", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    full_auto_passes_total=(
                        int((result or {}).get("full_auto_passes_total", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    full_auto_pass_checkpoints_emitted=(
                        int((result or {}).get("full_auto_pass_checkpoints_emitted", 0) or 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    resumed_from_pass_cap=(
                        bool((result or {}).get("resumed_from_pass_cap", False))
                        if isinstance(result, dict)
                        else False
                    ),
                    multiline_input=multiline_input,
                    multiline_terminator=multiline_terminator,
                )
                _save_auto_chat_turn_state(
                    mode=auto_chat_mode,
                    task=original_auto_question,
                    answer_text=answer_text,
                    sources=list(payload_sources),
                    changed_files=[str(item) for item in changed if str(item).strip()],
                    verification=str((result or {}).get("auto_execute_terminal_reason", "") or ""),
                )
                pending_prechecklist = None
                pending_prechecklist_source = ""
                pending_prechecklist_warning = ""
                if not (answer_only_fallback or answer_only_no_edit or answer_only_auto_execute):
                    _render_coding_sections(
                        console,
                        result if isinstance(result, dict) else {},
                        rendered_dynamic=rendered_dynamic,
                        show_actions=False,
                    )
                    if payload_sources:
                        _render_answer_sections(
                            console,
                            answer="",
                            title="Sources",
                            sources=payload_sources,
                            warnings=[],
                            trace=[],
                            show_trace=False,
                        )
                    if _cli_verbose_enabled() and debug_tail:
                        console.print("\n[bold]Debug tail[/bold]\n" + debug_tail)

                # Optional: if you want quick diff visibility without full diff spam:
                # console.print("\n[dim]Tip: run with your own :diff command if you add history later.[/dim]")

                chat_ui_state.record_event(
                    make_event(
                        "agent.decision",
                        title="Agent decision",
                        message="Coding-agent turn completed and rendered the final response.",
                        status="success",
                        session_id=chat_ui_state.session_id,
                        turn_id=current_turn_id,
                        step_id="05",
                        metadata={
                            "next_action": "final response",
                            "verification_target": str((result or {}).get("auto_execute_terminal_reason", "") or ""),
                        },
                    ).finish(status="success")
                )
                _finish_ui_turn(current_turn_id)
                continue
    finally:
        set_active_chat_ui_state(None)
        if tool_worker_client is not None:
            stop = getattr(tool_worker_client, "stop", None)
            if callable(stop):
                stop()
        if tmp_root is not None:
            tmp_root.cleanup()
        if tmp_base is not None:
            tmp_base.cleanup()
