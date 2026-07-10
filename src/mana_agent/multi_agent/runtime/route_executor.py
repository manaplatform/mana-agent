from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from mana_agent.analysis.models import AskResponse, AskResponseWithTrace, SearchHit, ToolInvocationTrace
from mana_agent.multi_agent.runtime.entry_router import EntryRouter, RouteDecision, RouteRuntimeState
from mana_agent.search.config import SearchConfig
from mana_agent.search.models import SearchDecision, SearchQuery
from mana_agent.search.router import SearchRouter
from mana_agent.services.search_service import SearchService
from mana_agent.services.structure_service import StructureService
from mana_agent.utils.project_search import project_search


NO_INDEX_MESSAGE = (
    "Semantic index is unavailable for the selected route. Run `mana-agent index` "
    "or ask for a repository search route."
)
NO_INDEX_WARNING = "Semantic index unavailable for model-selected semantic route."


@dataclass(slots=True)
class RouteExecutionContext:
    question: str
    index_dir: Path | None
    index_dirs: list[Path] = field(default_factory=list)
    project_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    k: int = 8
    max_steps: int = 6
    timeout_seconds: int = 30
    callbacks: Sequence[Any] | None = None
    dir_mode: bool = False
    root_dir: Path | None = None
    index_available: bool = True
    required_mcp_server: str | None = None


class RouteExecutor:
    def __init__(
        self,
        *,
        router: EntryRouter,
        store: Any,
        qna_chain: Any,
        ask_agent: Any | None = None,
        search_service: SearchService | None = None,
        command_handler: Callable[[str, list[str]], AskResponseWithTrace] | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.router = router
        self.store = store
        self.qna_chain = qna_chain
        self.ask_agent = ask_agent
        self.search_service = search_service
        self.command_handler = command_handler
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()

    def execute(self, decision: RouteDecision, context: RouteExecutionContext) -> AskResponseWithTrace:
        validation = self._validate(decision, context)
        if validation:
            rerouted = self._reroute_once(decision, context, validation)
            if rerouted is not None:
                second_validation = self._validate(rerouted, context)
                if not second_validation:
                    return self._attach_route_trace(
                        self._execute_validated(rerouted, context),
                        rerouted,
                        validation=f"re-routed after validation error: {validation}",
                    )
                return self._route_error(rerouted, context, validation=second_validation)
            return self._route_error(decision, context, validation=validation)
        return self._attach_route_trace(
            self._execute_validated(decision, context),
            decision,
            validation="ok",
        )

    def _execute_validated(self, decision: RouteDecision, context: RouteExecutionContext) -> AskResponseWithTrace:
        if decision.kind == "semantic_qa":
            return self._semantic_qa(context)
        if decision.kind == "repo_search":
            return self._repo_search(context)
        if decision.kind == "tool_execution":
            return self._tool_execution(decision, context)
        if decision.kind in {"web_search", "github_search"}:
            return self._external_search(decision, context)
        if decision.kind == "command":
            return self._command(decision)
        if decision.kind in {"clarification", "unsupported"}:
            return AskResponseWithTrace(
                answer=decision.user_visible_message or decision.reason,
                sources=[],
                warnings=[],
                mode=f"route-{decision.kind}",
                trace=[],
            )
        if decision.kind in {"gitops", "coding_task", "analysis_task"}:
            return self._agent_tools(context, mode=f"route-{decision.kind}")
        return AskResponseWithTrace(
            answer=f"Selected route is not executable: {decision.kind}",
            sources=[],
            warnings=[f"Route cannot run: {decision.kind}"],
            mode="route-error",
            trace=[],
        )

    def _semantic_qa(self, context: RouteExecutionContext) -> AskResponseWithTrace:
        if context.dir_mode:
            if self.search_service is None:
                return self._plain_error("Search service is not configured for dir-mode semantic Q&A.")
            sources, warnings = self.search_service.search_multi(
                index_dirs=context.index_dirs,
                query=context.question,
                k=context.k,
            )
        else:
            if context.index_dir is None:
                return self._plain_error("No semantic index was supplied.")
            sources = self.store.search(context.index_dir, query=context.question, k=context.k)
            warnings = []
        if not sources:
            return AskResponseWithTrace(
                answer="The selected semantic route found no indexed context.",
                sources=[],
                warnings=list(warnings),
                mode="route-semantic_qa",
                trace=[],
            )
        answer = self.qna_chain.run(question=context.question, context=_render_context(sources))
        return AskResponseWithTrace(
            answer=answer,
            sources=sources,
            warnings=list(warnings),
            mode="route-semantic_qa",
            trace=[],
        )

    def _repo_search(self, context: RouteExecutionContext) -> AskResponseWithTrace:
        root = (context.root_dir or context.project_root).resolve()
        result = project_search(context.question, root, max_results=max(5, int(context.k) * 4))
        sources = [
            SearchHit(
                score=0.0,
                file_path=match.file_path,
                start_line=match.line_number,
                end_line=match.line_number,
                symbol_name="match",
                snippet=match.line_text[:500],
            )
            for match in result.matches
        ]
        if not result.matches:
            return AskResponseWithTrace(
                answer=f"No direct project matches for that query under {root}.",
                sources=[],
                warnings=[],
                mode="route-repo_search",
                trace=[],
            )
        try:
            answer = self.qna_chain.run(question=context.question, context=_render_project_context(result.matches))
        except Exception:
            answer = result.format(root)
        return AskResponseWithTrace(
            answer=answer,
            sources=sources,
            warnings=[],
            mode="route-repo_search",
            trace=[],
        )

    def _tool_execution(self, decision: RouteDecision, context: RouteExecutionContext) -> AskResponseWithTrace:
        if not decision.tool_plan:
            return self._agent_tools(context, mode="route-tool_execution")
        first_tool = str(decision.tool_plan[0].get("tool") or "")
        if first_tool == "command_inventory":
            response = command_inventory_response(context.project_root)
            return AskResponseWithTrace(
                answer=response.answer,
                sources=response.sources,
                warnings=response.warnings,
                mode="route-tool_execution",
                trace=[
                    ToolInvocationTrace(
                        tool_name="command_inventory",
                        args_summary=str(context.project_root),
                        duration_ms=0.0,
                        status="ok",
                        output_preview=response.answer[:300],
                    )
                ],
            )
        return self._agent_tools(context, mode="route-tool_execution")

    def _agent_tools(self, context: RouteExecutionContext, *, mode: str) -> AskResponseWithTrace:
        if self.ask_agent is None:
            return self._plain_error("Ask agent is not configured for the selected route.")
        if context.dir_mode:
            result = self._run_agent_multi(context)
        else:
            result = self._run_agent_single(context)
        result.mode = mode
        return result

    def _run_agent_single(self, context: RouteExecutionContext) -> AskResponseWithTrace:
        try:
            return self.ask_agent.run(
                question=context.question,
                index_dir=context.index_dir,
                k=context.k,
                max_steps=context.max_steps,
                timeout_seconds=context.timeout_seconds,
                callbacks=context.callbacks,
            )
        except TypeError:
            return self.ask_agent.run(
                question=context.question,
                index_dir=context.index_dir,
                k=context.k,
                max_steps=context.max_steps,
                timeout_seconds=context.timeout_seconds,
            )

    def _run_agent_multi(self, context: RouteExecutionContext) -> AskResponseWithTrace:
        try:
            return self.ask_agent.run_multi(
                question=context.question,
                index_dirs=context.index_dirs,
                k=context.k,
                max_steps=context.max_steps,
                timeout_seconds=context.timeout_seconds,
                callbacks=context.callbacks,
            )
        except TypeError:
            return self.ask_agent.run_multi(
                question=context.question,
                index_dirs=context.index_dirs,
                k=context.k,
                max_steps=context.max_steps,
                timeout_seconds=context.timeout_seconds,
            )

    def _external_search(self, decision: RouteDecision, context: RouteExecutionContext) -> AskResponseWithTrace:
        target = "github" if decision.kind == "github_search" else "web"
        search_router = SearchRouter(root=str(context.project_root), llm=getattr(self.router, "llm", None))
        search_decision = SearchDecision(
            needs_search=True,
            targets=[target],  # type: ignore[list-item]
            reason=decision.reason,
            confidence=decision.confidence,
            queries=[SearchQuery(query=context.question, target=target)],  # type: ignore[arg-type]
            mode=target,  # type: ignore[arg-type]
        )
        result = search_router.run(user_query=context.question, decision_override=search_decision)
        context_block = result.context_block()
        answer = context_block or f"No {target} search results were available."
        return AskResponseWithTrace(
            answer=answer,
            sources=[],
            warnings=result.warnings,
            mode=f"route-{decision.kind}",
            trace=[
                ToolInvocationTrace(
                    tool_name=decision.kind,
                    args_summary=context.question[:200],
                    duration_ms=0.0,
                    status="ok" if context_block else "empty",
                    output_preview=answer[:300],
                )
            ],
        )

    def _command(self, decision: RouteDecision) -> AskResponseWithTrace:
        if self.command_handler is None:
            return self._plain_error("Command route selected, but no command handler is configured.")
        return self.command_handler(str(decision.command_name or ""), list(decision.command_args))

    def _validate(self, decision: RouteDecision, context: RouteExecutionContext) -> str | None:
        required_mcp_server = str(getattr(context, "required_mcp_server", "") or "").strip()
        if required_mcp_server:
            selected = str((decision.tool_plan or [{}])[0].get("tool") or "") if decision.tool_plan else ""
            required_prefix = f"mcp__{required_mcp_server}__"
            if decision.kind != "tool_execution" or not selected.startswith(required_prefix):
                return f"user explicitly required MCP provider '{required_mcp_server}', but no tool from {required_prefix} was selected"
        if decision.kind == "semantic_qa" and not _index_available(context):
            return "semantic index unavailable"
        if decision.kind == "command":
            commands = available_command_names(context.project_root)
            if not decision.command_name or decision.command_name not in commands:
                return f"unknown command: {decision.command_name or '<missing>'}"
        search_config = SearchConfig.from_env()
        if decision.kind == "web_search" and not search_config.enable_web:
            return "web search disabled"
        if decision.kind == "github_search" and not search_config.enable_github:
            return "github search disabled"
        return None

    def _reroute_once(self, decision: RouteDecision, context: RouteExecutionContext, validation_error: str) -> RouteDecision | None:
        try:
            return self.router.route(
                question=context.question,
                index_dir=context.index_dir,
                project_root=context.project_root,
                available_commands=available_command_names(context.project_root),
                available_tools=available_tool_names(),
                runtime_state=RouteRuntimeState(
                    index_available=_index_available(context),
                    dir_mode=context.dir_mode,
                    validation_error=validation_error,
                    required_mcp_server=str(getattr(context, "required_mcp_server", "") or "") or None,
                ),
            )
        except Exception:
            return None

    def _route_error(self, decision: RouteDecision, context: RouteExecutionContext, *, validation: str) -> AskResponseWithTrace:
        message = NO_INDEX_MESSAGE if validation == "semantic index unavailable" else f"Selected route cannot run: {validation}."
        return self._attach_route_trace(
            AskResponseWithTrace(
                answer=decision.user_visible_message or message,
                sources=[],
                warnings=[NO_INDEX_WARNING if validation == "semantic index unavailable" else validation],
                mode="route-error",
                trace=[],
            ),
            decision,
            validation=validation,
        )

    def _attach_route_trace(self, response: AskResponseWithTrace, decision: RouteDecision, *, validation: str) -> AskResponseWithTrace:
        response.route_trace = {
            "route_kind": decision.kind,
            "router_model": self.router.router_model,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "validation": validation,
            "executed_tools": [item.tool_name for item in response.trace],
        }
        return response

    @staticmethod
    def _plain_error(message: str) -> AskResponseWithTrace:
        return AskResponseWithTrace(
            answer=message,
            sources=[],
            warnings=[message],
            mode="route-error",
            trace=[],
        )


def _index_available(context: RouteExecutionContext) -> bool:
    if not context.index_available:
        return False
    if context.dir_mode:
        return bool(context.index_dirs)
    return True


def _render_context(sources: list[SearchHit]) -> str:
    blocks: list[str] = []
    for src in sources:
        blocks.append(
            "\n".join(
                [
                    f"source: {src.file_path}:{src.start_line}-{src.end_line}",
                    f"symbol: {src.symbol_name}",
                    "snippet:",
                    src.snippet,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _render_project_context(matches: list[Any]) -> str:
    blocks: list[str] = []
    for match in matches:
        blocks.append(
            "\n".join(
                [
                    f"source: {match.file_path}:{match.line_number}",
                    "snippet:",
                    match.line_text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def available_tool_names() -> list[str]:
    names = [
        "command_inventory",
        "repo_search",
        "repo_read",
        "semantic_search",
        "web_search",
        "github_search",
        "git_status",
        "git_diff",
        "git_help",
        "git_generic",
        "git_log",
        "git_branch",
        "git_remote",
        "git_fetch",
        "git_switch",
        "git_checkout",
        "git_create_branch",
        "git_add",
        "git_commit",
        "git_push",
        "git_pull",
        "git_merge",
        "git_rebase",
        "git_reset",
        "git_restore",
        "git_revert",
        "git_tag",
        "git.status",
        "git.diff",
        "git.help",
        "git.generic",
        "git.log",
        "git.branch",
        "git.remote",
        "git.fetch",
        "git.switch",
        "git.checkout",
        "git.create_branch",
        "git.add",
        "git.commit",
        "git.push",
        "git.pull",
        "git.merge",
        "git.rebase",
        "git.reset",
        "git.restore",
        "git.revert",
        "git.tag",
        "structure_analysis",
        "code_edit",
        "test_verification",
    ]
    try:
        from mana_agent.mcp.tools import discovered_mcp_tool_names
        names.extend(discovered_mcp_tool_names())
    except Exception:
        # MCP discovery failures are surfaced by the provider itself; routing
        # must not invent alternate tools for unavailable providers.
        pass
    return sorted(set(names))


def available_command_names(project_root: Path) -> list[str]:
    scripts = read_console_scripts(project_root)
    report = StructureService(include_tests=False).analyze_project(project_root)
    return sorted(set(scripts) | set(report.commands))


def read_console_scripts(project_root: Path) -> dict[str, str]:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return {}
    scripts: dict[str, str] = {}
    in_scripts = False
    for raw_line in pyproject.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_scripts = line == "[project.scripts]"
            continue
        if not in_scripts or "=" not in line:
            continue
        name, target = line.split("=", 1)
        scripts[name.strip().strip('"').strip("'")] = target.strip().strip('"').strip("'")
    return scripts


def command_inventory_response(search_root: Path) -> AskResponse:
    scripts = read_console_scripts(search_root)
    report = StructureService(include_tests=False).analyze_project(search_root)
    command_names = sorted(set(report.commands))

    lines = ["Command surface:"]
    if scripts:
        for script_name, target in sorted(scripts.items()):
            lines.append(f"- `{script_name}` console script -> `{target}`")
    else:
        lines.append("- No console scripts found in `pyproject.toml`.")

    if command_names:
        lines.append("")
        lines.append("Detected CLI subcommands:")
        primary_script = sorted(scripts)[0] if scripts else "<console-script>"
        for command in command_names:
            lines.append(f"- `{primary_script} {command}`")
        lines.append("")
        lines.append("Use `--help` for details, for example:")
        lines.append(f"- `{primary_script} --help`")
        for command in command_names:
            lines.append(f"- `{primary_script} {command} --help`")
    else:
        lines.append("")
        lines.append("No CLI-style command declarations were detected.")

    source_matches = []
    for query in ("[project.scripts]", "@app.command"):
        source_matches.extend(project_search(query, search_root, max_results=20).matches)
    sources = [
        SearchHit(
            score=0.0,
            file_path=match.file_path,
            start_line=match.line_number,
            end_line=match.line_number,
            symbol_name="command-surface",
            snippet=match.line_text[:500],
        )
        for match in source_matches
    ]
    return AskResponse(answer="\n".join(lines), sources=sources, warnings=[])
