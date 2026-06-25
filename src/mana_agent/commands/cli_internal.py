from __future__ import annotations

import os
import sys
import json
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Any
import threading
import traceback
from datetime import datetime, timezone
import typer
import asyncio
from asyncio.subprocess import PIPE
from rich.console import Console
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from mana_agent.config.settings import (
    Settings,
    default_diagrams_dir,
    default_index_dir,
    default_llm_logs_dir,
    default_logs_dir,
    mana_root_dir,
)
from mana_agent.vector_store.embeddings import build_embeddings
from mana_agent.commands import ui_helpers as _ui_helpers
from mana_agent.commands.ui_helpers import *  # noqa: F401,F403
from mana_agent.commands.ui_helpers import _resolve_agent_max_steps
from mana_agent.analysis.checks import PythonStaticAnalyzer
from mana_agent.analysis.chunker import CodeChunker
from mana_agent.parsers.multi_parser import MultiLanguageParser
from mana_agent.vector_store.faiss_store import FaissStore
from mana_agent.services.analyze_service import AnalyzeService
from mana_agent.services.ask_service import AskService
from mana_agent.services.chat_service import ChatService
from mana_agent.services.coding_memory_service import CodingMemoryService
from mana_agent.describe.build import build_describe_service
from mana_agent.services.index_service import IndexService
from mana_agent.services.search_service import SearchService
from mana_agent.services.structure_service import StructureService
from mana_agent.services.dependency_service import DependencyService
from mana_agent.services.vulnerability_service import VulnerabilityService
from mana_agent.services.report_service import ReportService
from mana_agent.utils.index_discovery import discover_index_dirs
from mana_agent.utils.project_discovery import discover_subprojects
from mana_agent.llm.analyze_chain import AnalyzeChain
from mana_agent.llm.ask_agent import AskAgent
from mana_agent.llm.qna_chain import QnAChain
from mana_agent.llm.coding_agent import CodingAgent
from mana_agent.llm.tool_worker_process import ToolWorkerClient, ToolWorkerProcessError
from mana_agent.llm.tools_executor import LocalToolsExecutor, RedisRQToolsExecutor, ToolsExecutionConfig
from mana_agent.llm.tools_manager import QueueManager
from mana_agent.llm.run_logger import LlmRunLogger
from mana_agent.tools.search_internet import build_search_internet_tool
from mana_agent.utils.project_search import project_search
# The deep‐flow LLM chain:
from mana_agent.describe.llm_chains.deep_flow import DeepFlowChain
from .output import build_output_sink, get_shared_console

logger = logging.getLogger(__name__)
console = get_shared_console()
app = typer.Typer(help="mana-agent CLI", invoke_without_command=True, no_args_is_help=False)

OUTPUT_DIR: Path | None = None
CLI_VERBOSE_MODE = False
LLM_DEBUG_MODE = False

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
    store_dir = root / ".mana" / "runs" / str(run_id).strip()
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
        try:
            worker.stop()
        except Exception:
            pass

    assert result is not None
    console.print(str(result.answer or "").strip() or "Continuation finished without a direct answer.")
    if result.run_status == "needs_resume":
        console.print(
            f"[yellow]Resume still pending:[/yellow] mana-agent continue --root-dir {root} --run-id {result.run_id}"
        )


# Loggers whose DEBUG/INFO chatter would flood the interactive chat console
# (especially while a background index build is running). They stay fully
# intact in the file log — only the console handler is quieted.
_NOISY_CHAT_LOG_PREFIXES: tuple[str, ...] = (
    "mana_agent.parsers",
    "mana_agent.analysis",
    "mana_agent.services.index_service",
    "mana_agent.services.describe_service",
    "mana_agent.describe",
    "mana_agent.vector_store",
)


class _QuietChatConsoleFilter(logging.Filter):
    """Drop sub-WARNING records from noisy indexing loggers on the console."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.levelno >= logging.WARNING:
            return True
        return not record.name.startswith(_NOISY_CHAT_LOG_PREFIXES)


def _install_quiet_chat_console_logging() -> None:
    """Keep noisy indexing logs off the interactive console (file log unchanged).

    Targets only console stream handlers (stderr/stdout), never FileHandlers, so
    the prompt is not buried under per-file parse/chunk DEBUG lines while the
    semantic index builds in the background.
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
    out_dir = root / ".mana"
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
# Analyze services
# ---------------------------------------------------------------------------

def build_analyze_service() -> AnalyzeService:
    return _public_symbol("AnalyzeService", AnalyzeService)(
        analyzer=_public_symbol("PythonStaticAnalyzer", PythonStaticAnalyzer)()
    )


def build_llm_analyze_service(
    settings: Settings,
    model_override: str | None,
):
    model_name = model_override or settings.openai_chat_model

    chat_openai_cls = _public_symbol("ChatOpenAI", ChatOpenAI)
    analyze_chain_cls = _public_symbol("AnalyzeChain", AnalyzeChain)
    llm = chat_openai_cls(
        api_key=settings.openai_api_key,
        model=model_name,
        base_url=settings.openai_base_url,
        temperature=0,
    )

    chain = analyze_chain_cls(
        api_key=settings.openai_api_key,
        model=model_name,
        base_url=settings.openai_base_url,
    )

    file_agent = build_file_agent(llm)

    return chain, file_agent


# ---------------------------------------------------------------------------
# Ask service
# ---------------------------------------------------------------------------

def build_ask_service(
    settings: Settings,
    model_override: str | None,
    *,
    project_root: Path | None = None,
) -> AskService:
    model = model_override or settings.openai_chat_model
    root = project_root.resolve() if project_root else Path.cwd().resolve()
    qna_chain_cls = _public_symbol("QnAChain", QnAChain)
    chat_openai_cls = _public_symbol("ChatOpenAI", ChatOpenAI)
    ask_agent_cls = _public_symbol("AskAgent", AskAgent)
    ask_service_cls = _public_symbol("AskService", AskService)

    qna_chain = qna_chain_cls(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
    )

    llm = chat_openai_cls(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
        temperature=0,
    )

    ask_agent = ask_agent_cls(
        api_key=settings.openai_api_key,
        model=model,
        search_service=build_search_service(settings),
        base_url=settings.openai_base_url,
        project_root=root,
    )
    tools = getattr(ask_agent, "tools", None)
    if isinstance(tools, list) and not any(getattr(tool, "name", "") == "search_internet" for tool in tools):
        tools.append(_public_symbol("build_search_internet_tool", build_search_internet_tool)())

    return ask_service_cls(
        store=build_store(settings),
        qna_chain=qna_chain,
        ask_agent=ask_agent,
        search_service=build_search_service(settings),
        project_root=root,
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
# Report service (with deep‐flow LLM injection)
# ---------------------------------------------------------------------------

def build_report_service(
    *,
    use_llm: bool,
    model_override: str | None,
    include_tests: bool,
) -> ReportService:
    dependency_service = build_dependency_service()
    analyze_service = build_analyze_service()
    
    settings = Settings()
    
    # Always build LLM components since LLM is now mandatory
    final_model = model_override or settings.openai_chat_model
    
    # 1) Build the DeepFlowChain for deep-profile flow synthesis
    llm_chain = DeepFlowChain(
        model_name=final_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    
    # 2) Build describe_service with LLM chain
    describe_service = build_describe_service(
        dependency_service=dependency_service,
        llm_chain=llm_chain,
        include_tests=include_tests,
    )
    
    structure_service = StructureService(include_tests=include_tests)
    vulnerability_service = VulnerabilityService()
    
    return ReportService(
        dependency_service=dependency_service,
        analyze_service=analyze_service,
        llm_analyze_service=None,
        describe_service=describe_service,
        structure_service=structure_service,
        vulnerability_service=vulnerability_service,
    )

# ---------------------------------------------------------------------------
# Export symbols
# ---------------------------------------------------------------------------

__all__ = [name for name in globals()]
