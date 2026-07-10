from __future__ import annotations

import os
import sys
import logging
import tempfile
import shutil
import hashlib
import json
from pathlib import Path
from typing import Any
import threading
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
import typer
import asyncio
from asyncio.subprocess import PIPE
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain_openai import ChatOpenAI
from mana_agent.config.settings import (
    Settings,
    default_diagrams_dir,  # noqa: F401 - consumed by chat_cli through wildcard command wiring
    default_index_dir,
    default_llm_logs_dir,  # noqa: F401 - consumed by chat_cli through wildcard command wiring
    default_logs_dir,
    mana_root_dir,
)
from mana_agent.config.user_config import load_effective_settings
from mana_agent.vector_store.embeddings import build_embeddings
from mana_agent.commands import ui_helpers as _ui_helpers
from mana_agent.commands.ui_helpers import *  # noqa: F401,F403
from mana_agent.commands.ui_helpers import _resolve_agent_max_steps  # noqa: F401 - consumed by chat_cli through wildcard command wiring
from mana_agent.analysis.chunker import CodeChunker
from mana_agent.parsers.multi_parser import MultiLanguageParser
from mana_agent.vector_store.faiss_store import FaissStore
from mana_agent.services.ask_service import AskService
from mana_agent.services.chat_service import ChatService  # noqa: F401 - consumed by chat_cli through wildcard command wiring
from mana_agent.services.coding_memory_service import CodingMemoryService
from mana_agent.services.index_service import IndexService
from mana_agent.services.search_service import SearchService
from mana_agent.services.dependency_service import DependencyService
from mana_agent.services.project_analyze_service import ProjectAnalyzeOptions, ProjectAnalyzeService
from mana_agent.services.project_llm_analyze_service import ModelConfig, build_llm_analyzer
from mana_agent.utils.index_discovery import discover_index_dirs  # noqa: F401 - consumed by chat_cli through wildcard command wiring
from mana_agent.utils.project_discovery import discover_subprojects  # noqa: F401 - consumed by chat_cli through wildcard command wiring
from mana_agent.multi_agent.runtime.ask_agent import AskAgent
from mana_agent.multi_agent.runtime.entry_router import EntryRouter
from mana_agent.multi_agent.runtime.qna_chain import QnAChain
from mana_agent.multi_agent.runtime.coding_agent import CodingAgent
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.runtime.model_levels import resolve_model_for_role
from mana_agent.multi_agent.runtime.tool_worker_process import ToolWorkerClient, ToolWorkerProcessError  # noqa: F401 - error class consumed by chat_cli through wildcard command wiring
from mana_agent.multi_agent.runtime.tools_executor import LocalToolsExecutor, RedisRQToolsExecutor, ToolsExecutionConfig  # noqa: F401 - executor types consumed by chat_cli through wildcard command wiring
from mana_agent.multi_agent.runtime.agent_work_queue import QueueManager
from mana_agent.multi_agent.runtime.run_logger import LlmRunLogger  # noqa: F401 - consumed by chat_cli through wildcard command wiring
from mana_agent.utils.project_search import project_search  # noqa: F401 - consumed by chat_cli through wildcard command wiring
from mana_agent.multi_agent.tools import git_tools
from mana_agent.skills import SkillManager
from mana_agent.tui.menu import NonInteractivePromptError
from mana_agent.tui.wizard import ensure_setup
from mana_agent.ui.banner import render_mode_header
from mana_agent.multi_agent import MainAgent
from .output import build_output_sink, get_shared_console
from .workspace_cli import impact_command, repo_app, search_command, session_app, workspace_app
from mana_agent.workspaces.paths import repository_analysis_dir, repository_dir, repository_id_for_path

logger = logging.getLogger(__name__)
console = get_shared_console()
app = typer.Typer(help="mana-agent CLI", invoke_without_command=True, no_args_is_help=False)
skills_app = typer.Typer(help="Manage Mana Agent skills.")
automation_app = typer.Typer(help="Create, deploy, inspect, and run persistent automations.")
mcp_app = typer.Typer(help="Connect to or serve Model Context Protocol tools and resources.")
app.add_typer(skills_app, name="skills")
app.add_typer(automation_app, name="automation")
app.add_typer(automation_app, name="cron")
app.add_typer(mcp_app, name="mcp")
app.add_typer(workspace_app, name="workspace")
app.add_typer(repo_app, name="repo")
app.add_typer(session_app, name="session")
app.command("search")(search_command)
app.command("impact")(impact_command)

OUTPUT_DIR: Path | None = None
CLI_VERBOSE_MODE = False
LLM_DEBUG_MODE = False
_SKIP_NEXT_COMMAND_ROUTE: ContextVar[bool] = ContextVar(
    "_SKIP_NEXT_COMMAND_ROUTE",
    default=False,
)

for _name in dir(_ui_helpers):
    if _name.startswith("_") and not _name.startswith("__"):
        globals().setdefault(_name, getattr(_ui_helpers, _name))


def _make_ephemeral_index_dir(prefix: str = "mana_index_") -> tuple[tempfile.TemporaryDirectory, Path]:
    tmp = tempfile.TemporaryDirectory(prefix=prefix)
    return tmp, Path(tmp.name).resolve()


def _public_symbol(name: str, default: Any) -> Any:
    public_cli = sys.modules.get("mana_agent.commands.cli")
    if public_cli is None or not hasattr(public_cli, name):
        return default
    public_value = getattr(public_cli, name)
    original_value = globals().get(name, None)
    if default is not original_value and public_value is original_value:
        return default
    return public_value


def _set_cli_runtime_flags(*, verbose: bool, debug_llm: bool) -> None:
    global CLI_VERBOSE_MODE, LLM_DEBUG_MODE
    CLI_VERBOSE_MODE = bool(verbose)
    LLM_DEBUG_MODE = bool(debug_llm)
    public_cli = sys.modules.get("mana_agent.commands.cli")
    if public_cli is not None:
        setattr(public_cli, "CLI_VERBOSE_MODE", CLI_VERBOSE_MODE)
        setattr(public_cli, "LLM_DEBUG_MODE", LLM_DEBUG_MODE)


def _cli_verbose_enabled() -> bool:
    return bool(_public_symbol("CLI_VERBOSE_MODE", CLI_VERBOSE_MODE))


def _stable_subdir_name(path: str | Path) -> str:
    resolved = str(Path(path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    return f"{Path(path).name or 'root'}-{digest}"


def _index_has_chunks(index_dir: str | Path) -> bool:
    return (Path(index_dir) / "chunks.jsonl").exists()


def _index_has_search_data(index_dir: str | Path) -> bool:
    root = Path(index_dir)
    return (root / "faiss").exists() or (root / "chunks.jsonl").exists()


def _build_project_llm_analyzer():
    """Build the analyzer from persisted Mana configuration, or ``None``.

    Analyze deliberately does not load a repository ``.env``.  Its connection
    settings are the user-level ``~/.mana/config.toml`` and ``secrets.toml`` so
    a target repository cannot silently change the model used for its report.
    Returns ``None`` (deterministic-only analyze) when no API key is configured.
    """
    try:
        config = load_effective_settings(include_env=False)
    except Exception as exc:  # noqa: BLE001 - unavailable configuration should not break analyze
        logger.warning("Project analyze LLM disabled (user configuration unavailable): %s", exc)
        return None
    api_key = str(config.get("OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        return None
    return build_llm_analyzer(
        ModelConfig(
            api_key=api_key,
            model=str(config.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"),
            base_url=str(config.get("OPENAI_BASE_URL", "") or "").strip() or None,
        )
    )


def _build_main_agent_routing_llm() -> Any | None:
    """Build the lightweight LLM used by the mandatory MainAgent route."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    try:
        settings = _public_symbol("Settings", Settings)()
    except Exception as exc:  # noqa: BLE001 - missing env should not break CLI routing
        logger.debug("MainAgent model routing disabled (settings unavailable): %s", exc)
        return None
    api_key = str(getattr(settings, "openai_api_key", "") or "").strip()
    if not api_key:
        return None
    model = resolve_model_for_role(
        AgentRole.HEAD_DECISION,
        global_model=getattr(settings, "openai_chat_model", "gpt-4.1-mini"),
    ).resolved_model
    chat_openai_cls = _public_symbol("ChatOpenAI", ChatOpenAI)
    return chat_openai_cls(
        api_key=api_key,
        model=model,
        base_url=getattr(settings, "openai_base_url", None) or os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
    )


def _split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _resolve_repo(repo: str | None, fallback: str | Path = ".") -> Path:
    root = Path(repo or fallback).expanduser().resolve()
    return root.parent if root.is_file() else root


def _record_multi_agent_request(
    root: str | Path,
    request: str,
    *,
    entrypoint: str,
    command_scope: bool = False,
    session_id: str | None = None,
) -> str:
    """Record a mandatory MainAgent route before a legacy entrypoint continues."""
    if command_scope and _SKIP_NEXT_COMMAND_ROUTE.get():
        return ""
    main_kwargs: dict[str, Any] = {"routing_llm": _build_main_agent_routing_llm()}
    if session_id:
        main_kwargs["session_id"] = session_id
    result = MainAgent(root, **main_kwargs).run_user_request(
        request,
        entrypoint=entrypoint,
    )
    return result.task_id


def _invoke_with_multi_agent_route(ctx, name: str, args: list[str] | None, *, root: str | Path, request: str, entrypoint: str) -> None:
    """Invoke a Typer subcommand after recording the root-level command route once."""
    _record_multi_agent_request(root, request, entrypoint=entrypoint, command_scope=True)
    token = _SKIP_NEXT_COMMAND_ROUTE.set(True)
    try:
        command = ctx.command.commands[name]
        with command.make_context(name, args or [], parent=ctx) as sub_ctx:
            command.invoke(sub_ctx)
    finally:
        _SKIP_NEXT_COMMAND_ROUTE.reset(token)


def _analyze_report_markdown(result, *, root: Path, focus: str | None = None) -> str:
    report = result.report
    inventory = report.get("inventory", {})
    architecture = report.get("architecture", {})
    risks = report.get("risks", {})
    recommendations = report.get("recommendations", {})
    dependencies = report.get("dependencies", {})
    symbols = report.get("symbols", {})
    llm = report.get("llm_analysis", {}) or {}

    def _items(value: Any, limit: int = 8) -> list[Any]:
        return list(value or [])[:limit] if isinstance(value, list) else []

    lines = ["# Mana Agent Analyze Report", ""]
    if focus:
        lines += [f"Focus: {focus}", ""]
    lines += [
        "## 1. Project overview",
        str(llm.get("project_summary") or report.get("project_summary") or f"Repository at {root}."),
        "",
        "## 2. Architecture map",
        str(llm.get("architecture_explanation") or architecture.get("summary") or "Architecture was derived from repository folders and imports."),
        "",
        "## 3. Important modules",
    ]
    important = _items(llm.get("important_files") or inventory.get("important_config_files") or inventory.get("entrypoints"))
    lines.extend(
        f"- `{item.get('file')}` - {item.get('why', 'Important repository file.')}"
        if isinstance(item, dict)
        else f"- `{item}`"
        for item in important
    )
    if not important:
        lines.append("- No important modules detected.")
    lines += ["", "## 4. Main data flows"]
    flows = _items(architecture.get("edges") or architecture.get("area_dependencies"))
    lines.extend(f"- {json.dumps(item, ensure_ascii=False)}" for item in flows)
    if not flows:
        lines.append("- No explicit data-flow edges detected.")
    lines += ["", "## 5. Risk areas"]
    risk_items = _items(llm.get("risk_analysis") or risks.get("items"))
    lines.extend(
        f"- {item.get('severity', 'info')}: {item.get('title') or item.get('message') or item}"
        if isinstance(item, dict)
        else f"- {item}"
        for item in risk_items
    )
    if not risk_items:
        lines.append("- No high-signal risks detected.")
    lines += [
        "",
        "## 6. Bugs or code smells",
        str(llm.get("risk_summary") or f"Static analysis found {len(risks.get('items', []) or [])} risk item(s)."),
        "",
        "## 7. Security notes",
        str(llm.get("security_notes") or "Review auth, secrets handling, file-path validation, and logging before sensitive deployments."),
        "",
        "## 8. Performance notes",
        str(llm.get("performance_notes") or f"Repository scan covered {inventory.get('total_files', 0)} file(s). Watch large generated folders and dependency-heavy paths."),
        "",
        "## 9. Missing tests",
    ]
    testing = dependencies.get("testing_packages", [])
    lines.append(
        "Testing packages detected: " + ", ".join(testing)
        if testing
        else "No testing packages were detected from dependency manifests."
    )
    lines += ["", "## 10. Recommended next actions"]
    recs = _items(llm.get("recommended_tasks") or recommendations.get("items"))
    lines.extend(
        f"- {item.get('title', item)}"
        if isinstance(item, dict)
        else f"- {item}"
        for item in recs
    )
    if not recs:
        lines.append("- Add focused tests around the highest-risk modules, then address top static-analysis findings.")
    lines += ["", "## Supporting facts"]
    lines.append(f"- Languages: {', '.join(inventory.get('detected_languages', []) or []) or 'not detected'}")
    lines.append(f"- Frameworks: {', '.join(inventory.get('detected_frameworks', []) or []) or 'not detected'}")
    lines.append(f"- Important symbols: {len(symbols.get('important_symbols', []) or [])}")
    return "\n".join(lines).rstrip() + "\n"


def _write_analyze_report(result, *, root: Path, output: str | None, focus: str | None) -> Path:
    target = (
        Path(output).expanduser().resolve()
        if output
        else repository_dir(repository_id_for_path(root)) / "reports" / "analyze.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_analyze_report_markdown(result, root=root, focus=focus), encoding="utf-8")
    return target


def _slug(text: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:80] or "plan"


def _project_snapshot(root: Path) -> dict[str, Any]:
    interesting = []
    for name in ("pyproject.toml", "package.json", "requirements.txt", "README.md", "src", "tests", "docs"):
        path = root / name
        if path.exists():
            interesting.append(name)
    return {"root": str(root), "items": interesting}


def _plan_markdown(*, task: str, root: Path, skills: list[Any], no_code: bool, yes: bool) -> str:
    snapshot = _project_snapshot(root)
    skill_names = [skill.name for skill in skills]
    skill_summary = "\n".join(f"- {skill.name} ({skill.source})" for skill in skills) or "- No specific skills matched."
    affected = ", ".join(snapshot["items"]) or "repository files discovered during implementation"
    return f"""# Implementation Plan

## 1. Goal

Build the requested change: {task}

## 2. Current System Understanding

Repository root: `{root}`

Detected project anchors: {affected}

## 3. Relevant Skills Loaded

{skill_summary}

## 4. Required Changes

- Inspect the code paths related to the task before editing.
- Reuse existing services, command wiring, tests, and conventions.
- Keep changes focused and preserve unrelated user work.

## 5. Files Likely Affected

- Files will be selected after repository inspection for `{task}`.
- `CHANGELOG.md` must be updated for the repository change.

## 6. Data Model / Migration Changes

No migration is assumed unless inspection finds model or schema changes are necessary.

## 7. Backend Logic

Update the shared service or command owner for the behavior so parallel paths do not drift.

## 8. Frontend / CLI / Bot Changes

Apply user-facing changes only where the task requires them, preserving existing text and flow conventions.

## 9. Edge Cases

- Existing user changes must not be reverted.
- Missing configuration should fail clearly.
- Existing public behavior should remain compatible unless the task explicitly changes it.

## 10. Tests

- Add focused tests for changed behavior.
- Run existing regression tests that cover the touched subsystem.

## 11. Verification Commands

```bash
PYTHONPATH=src .venv/bin/python -m compileall src
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## 12. Risks

- Broad tasks may need additional clarification before implementation.
- Over-broad edits can break CLI/chat compatibility.

## 13. Implementation Order

1. Inspect relevant files and tests.
2. Make the smallest implementation change that satisfies the task.
3. Update tests and `CHANGELOG.md`.
4. Run focused verification, then broader tests as time allows.

## 14. Approval

{"Implementation is approved by `--yes`." if yes else "Plan only; implementation requires explicit approval."}
{"`--no-code` is active, so implementation must not run." if no_code else ""}

Loaded skill names: {", ".join(skill_names) or "none"}
"""


@app.command("analyze")
def analyze_command(
    path_or_task: str = typer.Argument(".", help="Repository path or analysis focus."),
    repo: str | None = typer.Option(None, "--repo", "--root-dir", help="Repository root to analyze."),
    model: str | None = typer.Option(None, "--model", help="Model override for LLM analysis."),
    focus: str | None = typer.Option(None, "--focus", help="Focus area for the Markdown report."),
    depth: str = typer.Option("normal", "--depth", help="Analysis depth: quick, normal, or full."),
    output_format: str = typer.Option("both", "--format", help="Output format: md, json, or both."),
    output: str | None = typer.Option(None, "--output", help="Markdown report path. Defaults to .mana/reports/analyze.md."),
    artifact_dir: str | None = typer.Option(None, "--artifact-dir", help="Artifact directory. Defaults to .mana/analyze."),
    include: str | None = typer.Option(None, "--include", help="Comma-separated paths to include."),
    exclude: str | None = typer.Option(None, "--exclude", help="Comma-separated paths or folders to exclude."),
    max_files: int = typer.Option(5000, "--max-files", help="Maximum files to scan."),
    max_file_size_kb: int = typer.Option(512, "--max-file-size-kb", help="Maximum text file size to scan."),
) -> None:
    """Generate reusable repository intelligence artifacts."""
    _ = model
    if depth not in {"quick", "normal", "full"}:
        raise typer.BadParameter("depth must be quick, normal, or full")
    normalized_format = {"markdown": "md"}.get(output_format, output_format)
    if normalized_format not in {"md", "json", "both"}:
        raise typer.BadParameter("format must be md, json, or both")
    candidate = Path(path_or_task).expanduser()
    if repo:
        root = _resolve_repo(repo)
        inferred_focus = path_or_task if path_or_task != "." else None
    elif candidate.exists():
        root = _resolve_repo(candidate)
        inferred_focus = None
    else:
        root = Path.cwd().resolve()
        inferred_focus = path_or_task
    effective_focus = focus or inferred_focus
    _record_multi_agent_request(
        root,
        effective_focus or path_or_task or "analyze",
        entrypoint="analyze",
        command_scope=True,
    )
    out_dir = (
        Path(artifact_dir).expanduser().resolve()
        if artifact_dir
        else repository_analysis_dir(repository_id_for_path(root))
    )
    render_mode_header("Analyze", "Scanning repository and generating report", console)
    result = ProjectAnalyzeService().run(
        root,
        out_dir,
        options=ProjectAnalyzeOptions(
            depth=depth,
            output_format=normalized_format,
            include=_split_csv(include),
            exclude=_split_csv(exclude),
            max_files=max_files,
            max_file_size_kb=max_file_size_kb,
        ),
        llm_analyzer=_build_project_llm_analyzer(),
    )
    report_path = _write_analyze_report(result, root=root, output=output, focus=effective_focus)
    console.print(f"[green]report[/green] {report_path}")
    for path_item in result.artifacts.values():
        console.print(f"[green]wrote[/green] {path_item}")
    if result.errors:
        nonfatal = [
            error
            for error in result.errors
            if str(error).startswith("LLM analysis unavailable:")
        ]
        fatal = [error for error in result.errors if error not in nonfatal]
        for warning in nonfatal:
            console.print(f"[yellow]warning[/yellow] {warning}")
        if not fatal:
            return
        for error in result.errors:
            console.print(f"[red]error[/red] {error}")
        raise typer.Exit(code=1)


@app.command("plan")
def plan_command(
    task: str = typer.Argument("", help="Task to plan."),
    repo: str | None = typer.Option(None, "--repo", "--root-dir", help="Repository root to inspect."),
    model: str | None = typer.Option(None, "--model", help="Model override for planning."),
    yes: bool = typer.Option(False, "--yes", help="Approve implementation after the plan is generated."),
    no_code: bool = typer.Option(False, "--no-code", help="Never implement; only generate and save the plan."),
    save: bool = typer.Option(True, "--save/--no-save", help="Save the generated plan."),
    output: str | None = typer.Option(None, "--output", help="Plan output path. Defaults to .mana/plans/<task>.md."),
    skill: list[str] | None = typer.Option(None, "--skill", help="Force-load a skill by name."),
) -> None:
    """Create an approval-gated implementation plan."""
    _ = model
    try:
        ensure_setup(command_needs_llm=True, console=console)
    except NonInteractivePromptError as exc:
        raise typer.BadParameter(str(exc)) from exc
    root = _resolve_repo(repo)
    render_mode_header("Plan", "Build a safe implementation plan first", console)
    if not task.strip():
        try:
            task = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt):
            task = ""
        if not task.strip():
            raise typer.BadParameter("Plan mode requires a task.")
    _record_multi_agent_request(root, task, entrypoint="plan", command_scope=True)
    manager = SkillManager(root)
    console.print("[cyan]Inspecting repository...[/cyan]")
    console.print("[cyan]Loading relevant skills...[/cyan]")
    try:
        loaded_skills = manager.load_for_task(task, skill or [])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if loaded_skills:
        console.print("[green]Loaded skills:[/green] " + ", ".join(item.name for item in loaded_skills))
    else:
        console.print("[yellow]Loaded skills:[/yellow] none")
    plan_text = _plan_markdown(task=task, root=root, skills=loaded_skills, no_code=no_code, yes=yes)
    console.print(plan_text)
    plan_path: Path | None = None
    if save:
        plan_path = (
            Path(output).expanduser().resolve()
            if output
            else repository_dir(repository_id_for_path(root)) / "plans" / f"{_slug(task)}.md"
        )
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan_text, encoding="utf-8")
        console.print(f"[green]Saved:[/green] {plan_path}")
    if no_code:
        console.print("[yellow]No-code mode is active; implementation skipped.[/yellow]")
        return
    if yes:
        console.print("[green]Implementation approved by --yes.[/green]")
        console.print("[yellow]Use chat full-auto for execution:[/yellow] mana-agent chat --full-auto " + repr(task))
        return
    if not sys.stdin.isatty():
        console.print("[yellow]Plan ready. Re-run with --yes to approve implementation.[/yellow]")
        return
    choice = input("Approve implementation? [yes/save/edit]: ").strip().lower()
    if choice in {"yes", "y"}:
        console.print("[green]Implementation approved.[/green]")
        console.print("[yellow]Use chat full-auto for execution:[/yellow] mana-agent chat --full-auto " + repr(task))
    elif choice in {"edit", "e"}:
        console.print("[yellow]Edit the task and rerun plan mode.[/yellow]")
    else:
        if plan_path:
            console.print(f"[green]Plan saved only:[/green] {plan_path}")
        else:
            console.print("[green]Plan generated only.[/green]")


@app.command("api")
def api_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface for the FastAPI server."),
    port: int = typer.Option(8000, "--port", help="Port for the FastAPI server."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn reload mode."),
) -> None:
    """Run the Mana-Agent FastAPI server."""
    import uvicorn

    uvicorn.run("mana_agent.api.app:app", host=host, port=port, reload=reload)


@mcp_app.command("serve")
def mcp_serve_command(
    transport: str = typer.Option("streamable-http", "--transport", help="stdio or streamable-http"),
    root_dir: str = typer.Option(".", "--root-dir", "--repo", help="Repository root to expose."),
    host: str = typer.Option("127.0.0.1", "--host", help="Streamable HTTP bind host."),
    port: int = typer.Option(8765, "--port", help="Streamable HTTP port."),
) -> None:
    """Serve Mana-Agent's registered repository tools through MCP."""
    from mana_agent.mcp.server import create_mcp_server, protected_http_app

    root = Path(root_dir).expanduser().resolve()
    normalized = transport.strip().lower().replace("_", "-")
    if normalized == "stdio":
        create_mcp_server(repo_root=root).run(transport="stdio")
        return
    if normalized != "streamable-http":
        raise typer.BadParameter("transport must be stdio or streamable-http")
    token = str(os.getenv("MANA_MCP_SERVER_TOKEN") or load_effective_settings().get("MANA_MCP_SERVER_TOKEN") or "")
    if not token.strip():
        raise typer.BadParameter("MANA_MCP_SERVER_TOKEN is required for streamable-http")
    import uvicorn
    uvicorn.run(protected_http_app(repo_root=root, token=token), host=host, port=port)


@mcp_app.command("token-set")
def mcp_token_set_command(
    server_id: str | None = typer.Argument(None, help="Optional MCP server id; choose interactively when omitted."),
    token: str | None = typer.Option(None, "--token", hide_input=True),
) -> None:
    """Store a remote MCP bearer token, selecting a configured server if needed."""
    from mana_agent.mcp.config import load_mcp_servers, save_mcp_token
    from mana_agent.tui.menu import MenuOption, NonInteractivePromptError, select_option

    selected_id = str(server_id or "").strip()
    if not selected_id:
        servers = load_mcp_servers()
        if not servers:
            raise typer.BadParameter("No MCP servers are configured. Run the MCP add command first.")
        try:
            selected_id = select_option(
                title="MCP server",
                text="Use arrow keys and Enter to choose the server whose bearer token you want to store.",
                options=[MenuOption(value=server.id, label=f"{server.id} ({server.transport})") for server in servers],
            )
        except NonInteractivePromptError as exc:
            raise typer.BadParameter("Specify SERVER_ID when stdin is not interactive.") from exc
    selected_token = str(token or "").strip()
    if not selected_token:
        selected_token = typer.prompt(f"Bearer token for {selected_id}", hide_input=True, confirmation_prompt=True)
    save_mcp_token(selected_id, selected_token)
    console.print(f"[green]Stored token for MCP server '{selected_id}' in ~/.mana/mcp_secrets.toml.[/green]")


@mcp_app.command("add")
def mcp_add_command(
    server_id: str = typer.Argument(..., help="Stable local MCP provider id."),
    transport: str = typer.Option(..., "--transport", help="stdio, streamable_http, or sse."),
    command: str = typer.Option("", "--command", help="stdio server executable."),
    arg: list[str] = typer.Option([], "--arg", help="Repeat for each stdio argument."),
    url: str = typer.Option("", "--url", help="HTTP MCP endpoint."),
    token_env: str = typer.Option("", "--token-env", help="Environment variable containing the bearer token."),
    replace: bool = typer.Option(False, "--replace", help="Replace an existing server with this id."),
) -> None:
    """Register an MCP provider for future chat sessions."""
    from mana_agent.mcp.config import McpServerConfig, save_mcp_server

    normalized = transport.strip().lower().replace("-", "_")
    server = McpServerConfig(id=server_id, transport=normalized, command=command, args=arg, url=url, token_env=token_env)
    path = save_mcp_server(server, replace=replace)
    console.print(f"[green]Registered MCP server '{server.id}' in {path}.[/green]")


@app.command("dashboard")
def dashboard_command(
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root to pass to dashboard."),
    port: int = typer.Option(8501, "--port", help="Streamlit server port."),
    headless: bool = typer.Option(True, "--headless/--no-headless", help="Run Streamlit in headless mode (default for servers)."),
) -> None:
    """Launch the Mana Agent Web Dashboard (Streamlit).

    Requires the optional extra: pip install "mana-agent[dashboard]"

    This command is a thin, lazy wrapper. The real UI lives in
    dashboard/app.py and uses read-only views over .mana/ + runtime artifacts.
    """
    import sys
    from pathlib import Path

    repo_root = _resolve_repo(root)
    # Lazy import guard so core package never requires streamlit
    try:
        import streamlit  # noqa: F401 - presence check
    except ImportError:
        console.print("[red]Dashboard requires optional dependencies.[/red]")
        console.print("Install with: pip install 'mana-agent[dashboard]'")
        console.print("Then run: mana-agent dashboard  or  streamlit run dashboard/app.py")
        raise typer.Exit(code=1)

    import importlib.util

    dashboard_spec = importlib.util.find_spec("mana_agent.dashboard.app")
    if dashboard_spec is None or not dashboard_spec.origin:
        console.print("[red]Installed dashboard entrypoint was not found.[/red]")
        raise typer.Exit(code=1)
    dashboard_path = Path(dashboard_spec.origin).resolve()

    # Launch the dashboard from the installed package, independent of cwd.
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
        "--",
        # Pass root via env for the app to pick up
    ]
    env = os.environ.copy()
    env["MANA_DASHBOARD_ROOT"] = str(repo_root)
    # Also set for helpers
    os.environ["MANA_DASHBOARD_ROOT"] = str(repo_root)

    console.print(f"[cyan]Launching dashboard for[/cyan] {repo_root}")
    console.print(f"[dim]streamlit run {dashboard_path} --server.port {port}[/dim]")

    # Execute (blocking)
    import subprocess

    try:
        subprocess.run(cmd, env=env, check=False)
    except KeyboardInterrupt:
        console.print("\nDashboard stopped.")


def _automation_root(root: str | None) -> Path:
    return _resolve_repo(root)


@automation_app.command("create")
def automation_create_command(
    name: str = typer.Option(..., "--name", help="Readable schedule name."),
    action: str = typer.Option(..., "--action", help="analyze, daily_report, self_improvement, or custom."),
    cron: str = typer.Option(..., "--cron", help="Five-field POSIX cron expression."),
    target: list[str] = typer.Option(..., "--target", help="Deployment target: local and/or github."),
    command: str | None = typer.Option(None, "--command", help="Required for action=custom."),
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root."),
) -> None:
    """Create and immediately deploy an explicitly requested schedule."""
    from mana_agent.automations.service import AutomationValidationError, ScheduleDefinition, deploy_schedule

    try:
        schedule = ScheduleDefinition.create(name=name, action=action, cron=cron, targets=target, command=command)
        deployed = deploy_schedule(schedule, _automation_root(root))
    except AutomationValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(deployed.to_dict()))


@automation_app.command("list")
def automation_list_command(root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root.")) -> None:
    """List saved schedules and their most recent deployment state."""
    from mana_agent.automations.service import list_schedules

    console.print_json(json.dumps([schedule.to_dict() for schedule in list_schedules(_automation_root(root))]))


@automation_app.command("status")
def automation_status_command(
    schedule_id: str = typer.Argument(..., help="Schedule id."),
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root."),
) -> None:
    """Detect local-cron and managed-workflow deployment drift."""
    from mana_agent.automations.service import AutomationValidationError, deployment_status, get_schedule

    try:
        status = deployment_status(get_schedule(_automation_root(root), schedule_id), _automation_root(root))
    except AutomationValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(status))


@automation_app.command("deploy")
def automation_deploy_command(
    schedule_id: str = typer.Argument(..., help="Schedule id."),
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root."),
) -> None:
    """Reconcile a saved schedule with its selected deployment targets."""
    from mana_agent.automations.service import AutomationValidationError, deploy_schedule, get_schedule

    try:
        schedule = deploy_schedule(get_schedule(_automation_root(root), schedule_id), _automation_root(root))
    except AutomationValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(schedule.to_dict()))


@automation_app.command("enable")
def automation_enable_command(
    schedule_id: str = typer.Argument(..., help="Schedule id."),
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root."),
) -> None:
    """Enable or disable and immediately reconcile a schedule."""
    from mana_agent.automations.service import AutomationValidationError, deploy_schedule, get_schedule

    try:
        schedule = get_schedule(_automation_root(root), schedule_id)
        schedule.enabled = True
        schedule = deploy_schedule(schedule, _automation_root(root))
    except AutomationValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(schedule.to_dict()))


@automation_app.command("disable")
def automation_disable_command(
    schedule_id: str = typer.Argument(..., help="Schedule id."),
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root."),
) -> None:
    """Disable and immediately reconcile a schedule."""
    from mana_agent.automations.service import AutomationValidationError, deploy_schedule, get_schedule

    try:
        schedule = get_schedule(_automation_root(root), schedule_id)
        schedule.enabled = False
        schedule = deploy_schedule(schedule, _automation_root(root))
    except AutomationValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(schedule.to_dict()))


@automation_app.command("remove")
def automation_remove_command(
    schedule_id: str = typer.Argument(..., help="Schedule id."),
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root."),
) -> None:
    """Remove a saved schedule and its local deployment artifacts."""
    from mana_agent.automations.service import AutomationValidationError, delete_schedule, remove_deployment

    try:
        root_path = _automation_root(root)
        schedule = delete_schedule(root_path, schedule_id)
        remove_deployment(schedule, root_path)
    except AutomationValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Removed {schedule_id}")


@automation_app.command("run-now")
def automation_run_now_command(
    schedule_id: str = typer.Argument(..., help="Schedule id."),
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root."),
) -> None:
    """Run an explicitly selected saved schedule immediately."""
    from mana_agent.automations.service import AutomationValidationError, get_schedule, run_schedule_now

    try:
        root_path = _automation_root(root)
        result = run_schedule_now(get_schedule(root_path, schedule_id), root_path)
    except AutomationValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(result))


@automation_app.command("execute", hidden=True)
def automation_execute_command(
    action: str = typer.Option(..., "--action", help="Built-in action selected by a saved schedule."),
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository root."),
) -> None:
    """Execute a built-in action for a deployed schedule."""
    from mana_agent.automations.service import AutomationValidationError, execute_builtin_action

    try:
        result = execute_builtin_action(action, _automation_root(root))
    except AutomationValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(result))


@app.command(
    "git",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def git_command(
    ctx: typer.Context,
    repo: str | None = typer.Option(None, "--repo", "--root-dir", help="Repository root."),
    timeout: int = typer.Option(120, "--timeout", help="Git command timeout in seconds."),
    allow_protected: bool = typer.Option(False, "--allow-protected", help="Allow explicitly requested protected Git commands."),
) -> None:
    """Run a Git argv passthrough through Mana-Agent's shared Git safety policy."""
    root = _resolve_repo(repo)
    args = list(ctx.args)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        args = ["help"]
    _record_multi_agent_request(root, "git passthrough", entrypoint="git", command_scope=True)
    result = git_tools.generic(args=args, repo_path=root, timeout=timeout, allow_protected=allow_protected)
    console.print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        raise typer.Exit(code=1)


@skills_app.command("init")
def skills_init(
    repo: str | None = typer.Option(None, "--repo", "--root-dir", help="Repository root."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing root skill files."),
) -> None:
    """Create ./skills/ and copy built-in templates into it."""
    root = _resolve_repo(repo)
    _record_multi_agent_request(root, "skills init", entrypoint="skills", command_scope=True)
    written = SkillManager(root).init_project_skills(force=force)
    console.print(f"[green]Skills directory:[/green] {root / 'skills'}")
    if written:
        for path in written:
            console.print(f"[green]wrote[/green] {path}")
    else:
        console.print("[yellow]No files written; existing skills were preserved.[/yellow]")


@skills_app.command("list")
def skills_list(
    repo: str | None = typer.Option(None, "--repo", "--root-dir", help="Repository root."),
) -> None:
    """List skills by priority source."""
    root = _resolve_repo(repo)
    _record_multi_agent_request(root, "skills list", entrypoint="skills", command_scope=True)
    groups = SkillManager(root).list_by_source()
    console.print("[bold cyan]Available Skills[/bold cyan]\n")
    for title, names in groups.items():
        console.print(f"[bold]{title}:[/bold]")
        if names:
            for name in names:
                console.print(f"- {name}")
        else:
            console.print("- (none)")
        console.print("")


@skills_app.command("show")
def skills_show(
    name: str = typer.Argument(..., help="Skill name to show."),
    repo: str | None = typer.Option(None, "--repo", "--root-dir", help="Repository root."),
) -> None:
    """Print the selected skill content."""
    root = _resolve_repo(repo)
    _record_multi_agent_request(root, f"skills show {name}", entrypoint="skills", command_scope=True)
    skill = SkillManager(root).get(name)
    if skill is None:
        raise typer.BadParameter(f"Unknown skill: {name}")
    source = skill.path if skill.path is not None else skill.source
    console.print(f"[bold cyan]{skill.name}[/bold cyan] ({source})\n")
    console.print(skill.content)


@app.command("continue")
def continue_command(
    run_id: str = typer.Option(..., "--run-id", help="Run ID under .mana/runs to resume."),
    root_dir: str | None = typer.Option(None, "--root-dir", help="Repository root that owns .mana/runs/<run_id>."),
    pass_cap: int = typer.Option(4, "--pass-cap", help="Maximum passes for this continuation window."),
    auto_continue: bool = typer.Option(
        True,
        "--auto-continue/--no-auto-continue",
        help="Automatically start the next continuation pass while checkpointed work remains.",
    ),
    max_passes: int = typer.Option(
        12,
        "--max-passes",
        help="Hard cap for total continuation passes across automatic resume cycles.",
    ),
    max_total_tool_calls: int = typer.Option(
        80,
        "--max-tool-calls",
        "--max-total-tool-calls",
        help="Hard cap for total tool calls across automatic resume cycles.",
    ),
    max_runtime_minutes: float = typer.Option(
        0.0,
        "--max-runtime-minutes",
        help="Hard cap for total continuation runtime in minutes; 0 disables.",
    ),
    max_cost: float = typer.Option(
        0.0,
        "--max-cost",
        help="Reserved hard cap for provider cost telemetry; 0 disables.",
    ),
    max_no_progress_passes: int = typer.Option(
        2,
        "--max-no-progress-passes",
        help="Stop after this many continuation passes make no measurable progress.",
    ),
    timeout_seconds: int = typer.Option(30, "--timeout", help="Per-request tool timeout in seconds."),
    k: int = typer.Option(8, "--k", help="Retrieval result count for tool requests."),
    max_steps: int = typer.Option(6, "--max-steps", help="Maximum worker tool steps per request."),
    max_resume_cycles: int = typer.Option(
        0,
        "--max-resume-cycles",
        help="Safety cap for continuation cycles; 0 means keep going until completion or a non-resumable blocker.",
    ),
) -> None:
    """Resume a persisted auto-execute run from .mana/runs/<run_id>."""
    root = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
    pre_registration_id = repository_id_for_path(root)
    _record_multi_agent_request(root, f"continue run {run_id}", entrypoint="continue", command_scope=True)
    store_dir = repository_dir(repository_id_for_path(root)) / "runs" / str(run_id).strip()
    legacy_global_dir = repository_dir(pre_registration_id) / "runs" / str(run_id).strip()
    if not store_dir.exists() and legacy_global_dir.exists():
        store_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(legacy_global_dir, store_dir)
    state_path = store_dir / "state.json"
    if not state_path.exists():
        raise typer.BadParameter(f"Run state not found: {state_path}")

    settings_cls = _public_symbol("Settings", Settings)
    worker_client_cls = _public_symbol("ToolWorkerClient", ToolWorkerClient)
    tools_manager_orchestrator_cls = _public_symbol("QueueManager", QueueManager)
    settings = settings_cls()
    index_dir = default_index_dir(root)
    worker = worker_client_cls(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        project_root=root,
        repo_root=root,
    )
    # Resume must reconnect flow memory: without it the continuation loses the
    # flow read-cache and stops recording turns/todos, so resumed passes re-read
    # files the original run already cached. Keyed on project root + flow_id, the
    # disk-backed store transparently reattaches to the same flow.
    coding_memory_service = CodingMemoryService(
        project_root=root,
        max_turns=settings.coding_flow_max_turns,
        max_tasks=settings.coding_flow_max_tasks,
    )
    orchestrator = tools_manager_orchestrator_cls(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        worker_client=worker,
        repo_root=root,
        executor=LocalToolsExecutor(worker_client=worker),
        coding_memory_service=coding_memory_service,
    )
    try:
        result = None
        resume_cycles = 0
        total_passes = 0
        total_tool_calls = 0
        no_progress_cycles = 0
        started = datetime.now(timezone.utc)
        while True:
            result = orchestrator.resume_run(
                run_id=run_id,
                index_dir=index_dir,
                index_dirs=None,
                k=k,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                tool_policy={},
                pass_cap=pass_cap,
                on_event=CodingAgent._log_worker_event,
                max_no_progress_passes=max_no_progress_passes,
            )
            result_passes = int(getattr(result, "passes", 0) or 0)
            result_tool_calls = int(getattr(result, "toolsmanager_requests_count", 0) or 0)
            total_passes += result_passes
            total_tool_calls += result_tool_calls
            made_progress = any(
                bool((row or {}).get("made_progress"))
                for row in (getattr(result, "pass_logs", []) or [])
                if isinstance(row, dict)
            )
            no_progress_cycles = 0 if made_progress else no_progress_cycles + 1
            if result.run_status != "needs_resume" and result.terminal_reason != "pass_cap_reached":
                break
            if not auto_continue:
                break
            resume_cycles += 1
            console.print(
                f"[cyan]Continuation checkpoint:[/cyan] run_id={result.run_id} "
                f"cycle={resume_cycles} next_action={result.next_action or 'continue'}"
            )
            if total_passes >= max(1, int(max_passes)):
                console.print(
                    f"[yellow]Continuation stopped:[/yellow] max passes reached "
                    f"({total_passes}/{max_passes})."
                )
                break
            if total_tool_calls >= max(1, int(max_total_tool_calls)):
                console.print(
                    f"[yellow]Continuation stopped:[/yellow] max total tool calls reached "
                    f"({total_tool_calls}/{max_total_tool_calls})."
                )
                break
            if max_runtime_minutes > 0:
                elapsed_seconds = (datetime.now(timezone.utc) - started).total_seconds()
                if elapsed_seconds >= float(max_runtime_minutes) * 60.0:
                    console.print(
                        f"[yellow]Continuation stopped:[/yellow] max runtime reached "
                        f"({elapsed_seconds / 60.0:.2f}/{float(max_runtime_minutes):.2f} minutes)."
                    )
                    break
            if max_cost > 0:
                console.print(
                    "[yellow]Continuation cost cap requested but provider cost telemetry is not available in this path; "
                    "continuing with pass/tool/runtime caps.[/yellow]"
                )
                max_cost = 0.0
            if no_progress_cycles >= max(1, int(max_no_progress_passes)):
                console.print(
                    f"[yellow]Continuation stopped:[/yellow] no progress for "
                    f"{no_progress_cycles} cycle(s)."
                )
                break
            if max_resume_cycles > 0 and resume_cycles >= max_resume_cycles:
                break
    finally:
        stop = getattr(worker, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass

    assert result is not None
    console.print(str(result.answer or "").strip() or "Continuation finished without a direct answer.")
    if result.run_status == "needs_resume":
        console.print(
            f"[yellow]Resume still pending:[/yellow] mana-agent continue --root-dir {root} --run-id {result.run_id}"
        )


# Loggers whose DEBUG/INFO chatter would flood the interactive chat console.
# They stay fully intact in the file log — only the console handler is quieted.
_NOISY_CHAT_LOG_PREFIXES: tuple[str, ...] = (
    "mana_agent.parsers",
    "mana_agent.analysis",
    "mana_agent.services.index_service",
    "mana_agent.services.describe_service",
    "mana_agent.describe",
    "mana_agent.vector_store",
)


class _QuietChatConsoleFilter(logging.Filter):
    """Drop normal INFO/DEBUG records from the visible chat console."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.levelno >= logging.WARNING:
            return True
        if bool(CLI_VERBOSE_MODE) or bool(_public_symbol("CLI_VERBOSE_MODE", CLI_VERBOSE_MODE)):
            return not record.name.startswith(_NOISY_CHAT_LOG_PREFIXES)
        return False


def _install_quiet_chat_console_logging() -> None:
    """Keep normal Python logs off the interactive chat console.

    Targets only console stream handlers (stderr/stdout), never FileHandlers, so
    the prompt is not interrupted by INFO/DEBUG records while the file log keeps
    full details.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if not isinstance(handler, logging.StreamHandler):
            continue
        if any(isinstance(f, _QuietChatConsoleFilter) for f in handler.filters):
            continue
        handler.addFilter(_QuietChatConsoleFilter())


def _start_background_index(
    *,
    settings: Any,
    target_root: str | Path,
    index_dir: str | Path,
    state: dict[str, Any],
) -> "threading.Thread":
    """Build a semantic index in a daemon thread, updating ``state["status"]``.

    Non-blocking: chat stays usable via the direct project-search fallback and
    semantic search activates automatically once the index files exist. Tries a
    full vector index first, then a chunks-only index, before giving up.
    ``state["status"]`` transitions idle -> building -> ready|failed.
    """
    state["status"] = "building"

    def _worker() -> None:
        try:
            index_service = _public_symbol("build_index_service", build_index_service)(settings)
            try:
                _index_service_index_compat(
                    index_service,
                    target_path=target_root,
                    index_dir=index_dir,
                    rebuild=False,
                    vectors=True,
                )
            except Exception as exc:
                logger.warning("Background vector index failed (%s); trying chunks-only", exc)
                _index_service_index_compat(
                    index_service,
                    target_path=target_root,
                    index_dir=index_dir,
                    rebuild=False,
                    vectors=False,
                )
            state["status"] = "ready"
            logger.info("Background index ready", extra={"index_dir": str(index_dir)})
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = str(exc)
            logger.warning("Background indexing failed: %s", exc)

    thread = threading.Thread(target=_worker, name="mana-bg-index", daemon=True)
    thread.start()
    return thread


def _index_service_index_compat(
    index_service: Any,
    *,
    target_path: str | Path,
    index_dir: str | Path,
    rebuild: bool = False,
    vectors: bool = True,
) -> dict:
    try:
        return index_service.index(target_path=target_path, index_dir=index_dir, rebuild=rebuild, vectors=vectors)
    except TypeError:
        return index_service.index(target_path=target_path, index_dir=index_dir, rebuild=rebuild)



# ----------------------------------------------------------------------
# subprocess helper
# ----------------------------------------------------------------------

async def _spawn_command(cmd: list[str]) -> tuple[int, str, str]:
    """
    Spawn a subprocess, capture stdout/stderr, return (returncode, stdout, stderr).
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=PIPE,
        stderr=PIPE,
    )
    out_bytes, err_bytes = await proc.communicate()
    stdout = out_bytes.decode("utf-8", errors="ignore")
    stderr = err_bytes.decode("utf-8", errors="ignore")
    return proc.returncode, stdout, stderr
# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False, log_dir: Path | None = None) -> Path | None:
    log_level = logging.DEBUG if verbose else logging.INFO

    effective_log_dir = log_dir or default_logs_dir(Path.cwd())
    effective_log_dir.mkdir(parents=True, exist_ok=True)
    # Use a per-day tag (not per-second) so every run appends to the same
    # daily log file instead of creating a new file each invocation.
    date_tag = datetime.now().strftime("%Y%m%d")
    log_file = effective_log_dir / f"mana_agent_{date_tag}.log"

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    return log_file

def _resolve_output_file(
    output_dir: str | Path | None = None,
    *,
    default_name: str = "output",
    extension: str = "json",
    include_timestamp: bool = True,
) -> Path:
    # Backward-compatible: older command handlers pass positional `path` values.
    base_root = Path(output_dir) if output_dir is not None else (OUTPUT_DIR or Path.cwd())
    if base_root.suffix:
        base_root = base_root.parent
    base_dir = mana_root_dir(base_root)
    base_dir.mkdir(parents=True, exist_ok=True)

    if include_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{default_name}_{ts}.{extension}"
    else:
        filename = f"{default_name}.{extension}"

    return base_dir / filename

def _resolve_analyze_artifact_paths(path: str | Path) -> tuple[Path, Path, Path]:
    root = Path(path).resolve()
    out_dir = repository_dir(repository_id_for_path(root))
    return out_dir / "analyze.json", out_dir / "analyze.md", out_dir / "analyze.html"


def _resolve_describe_artifact_paths(path: str | Path) -> tuple[Path, Path, Path]:
    root = Path(path).resolve()
    out_dir = mana_root_dir(root)
    return out_dir / "describe.json", out_dir / "describe.md", out_dir / "describe.html"

def _emit_json(payload, output_file: Path | None = None) -> None:
    sink = build_output_sink(command_name="cli", json_mode=True, output_file=output_file, console=console)
    sink.emit_json(payload)

def _emit_text(text: str, output_file: Path | None = None) -> None:
    sink = build_output_sink(command_name="cli", json_mode=False, output_file=output_file, console=console)
    sink.emit_text(text)

def _render_findings_markdown(findings) -> str:
    if not findings:
        return "## Findings\n\nNo findings.\n"

    lines = ["## Findings\n"]
    for f in findings:
        lines.append(
            f"- **{f.severity.upper()}** `{f.rule_id}` "
            f"{f.file_path}:{f.line}:{f.column} — {f.message}"
        )
    return "\n".join(lines) + "\n"

def _render_repository_summary_markdown(summary: dict) -> str:
    lines = ["## Repository Summary\n"]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"

def _log_exception(context: str, exc: Exception, **extra) -> None:
    logger.error(
        f"Error during {context}: {exc}",
        extra={"context": context, **extra},
    )
    logger.debug(traceback.format_exc())

def _clamp_detail_line_target(detail_line_target: int) -> int:
    MIN_LINES = 100
    MAX_LINES = 2000

    if detail_line_target < MIN_LINES:
        return MIN_LINES
    if detail_line_target > MAX_LINES:
        return MAX_LINES
    return detail_line_target

def _resolve_report_artifact_paths(path: str) -> tuple[Path, Path, Path]:
    target = Path(path)
    if target.is_dir():
        default_json = target / "report.json"
        default_md   = target / "report.md"
        default_html = target / "report.html"
    else:
        base = target.with_suffix("") 
        default_json = base.with_suffix(".report.json")
        default_md   = base.with_suffix(".report.md")
        default_html = base.with_suffix(".report.html")
    return default_json, default_md, default_html

def _resolve_out_path(user_out: str | None, default: Path, suffix: str) -> Path:
    if user_out:
        p = Path(user_out)
        if p.is_dir():
            return p / default.name
        return p.with_suffix(suffix) if p.suffix == "" else p
    return default


# ---------------------------------------------------------------------------
# File agent (safe LangChain tools)
# ---------------------------------------------------------------------------

def build_file_agent(llm: Any, *, root_dir: Path | str = ".") -> Any:
    root = Path(root_dir).resolve()

    tools = [
        ReadFileTool(root_dir=str(root)),
        WriteFileTool(root_dir=str(root)),
        ListDirectoryTool(root_dir=str(root)),
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Vector store + search
# ---------------------------------------------------------------------------

def build_store(settings: Settings) -> FaissStore:
    store_cls = _public_symbol("FaissStore", FaissStore)
    embeddings = build_embeddings(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_embed_model,
    )
    return store_cls(embeddings=embeddings)

def build_index_service(settings: Settings) -> IndexService:
    index_service_cls = _public_symbol("IndexService", IndexService)
    return index_service_cls(
        parser=_public_symbol("MultiLanguageParser", MultiLanguageParser)(),
        chunker=_public_symbol("CodeChunker", CodeChunker)(),
        store=build_store(settings),
    )

def build_search_service(settings: Settings) -> SearchService:
    return _public_symbol("SearchService", SearchService)(store=build_store(settings))


# ---------------------------------------------------------------------------
# Ask service
# ---------------------------------------------------------------------------

def build_ask_service(
    settings: Settings,
    model_override: str | None,
    *,
    project_root: Path | None = None,
) -> AskService:
    model = resolve_model_for_role(
        AgentRole.MAIN,
        global_model=model_override or settings.openai_chat_model,
    ).resolved_model
    root = project_root.resolve() if project_root else Path.cwd().resolve()
    qna_chain_cls = _public_symbol("QnAChain", QnAChain)
    ask_agent_cls = _public_symbol("AskAgent", AskAgent)
    ask_service_cls = _public_symbol("AskService", AskService)
    entry_router_cls = _public_symbol("EntryRouter", EntryRouter)
    router_model = resolve_model_for_role(
        AgentRole.HEAD_DECISION,
        global_model=model,
    ).resolved_model

    qna_chain = qna_chain_cls(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
    )
    ask_agent = ask_agent_cls(
        api_key=settings.openai_api_key,
        model=model,
        search_service=build_search_service(settings),
        base_url=settings.openai_base_url,
        project_root=root,
    )
    router_kwargs = {"api_key": settings.openai_api_key, "model": router_model}
    if settings.openai_base_url:
        router_kwargs["base_url"] = settings.openai_base_url
    router_llm = _public_symbol("ChatOpenAI", ChatOpenAI)(**router_kwargs)

    return ask_service_cls(
        store=build_store(settings),
        qna_chain=qna_chain,
        ask_agent=ask_agent,
        search_service=build_search_service(settings),
        project_root=root,
        entry_router=entry_router_cls(llm=router_llm, router_model=router_model),
    )


def _build_ask_service_compat(
    settings: Settings,
    model_override: str | None = None,
    *,
    project_root: Path | None = None,
) -> AskService:
    public_cli = sys.modules.get("mana_agent.commands.cli")
    builder = getattr(public_cli, "build_ask_service", build_ask_service) if public_cli is not None else build_ask_service
    try:
        return builder(settings, model_override=model_override, project_root=project_root)
    except TypeError:
        return builder(settings, model_override=model_override)


# ---------------------------------------------------------------------------
# Dependency service
# ---------------------------------------------------------------------------

def build_dependency_service() -> DependencyService:
    return DependencyService()


# ---------------------------------------------------------------------------
# Export symbols
# ---------------------------------------------------------------------------

__all__ = [
    name
    for name in globals()
    if not name.startswith("__")
]
