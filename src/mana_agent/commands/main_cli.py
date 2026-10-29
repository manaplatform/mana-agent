from __future__ import annotations

import logging
import warnings

from .cli_internal import *
from .output import build_output_sink
from mana_agent.tui.menu import NonInteractivePromptError
from mana_agent.tui.wizard import ensure_setup, run_setup_wizard
from mana_agent.ui.banner import render_banner

logger = logging.getLogger(__name__)


def _is_interactive_terminal() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def configure() -> None:
    """Open the full configuration TUI and return to the terminal."""
    if not _is_interactive_terminal():
        raise typer.BadParameter(
            "Configuration requires an interactive terminal. Run `mana-agent --configure` from a TTY."
        )
    run_setup_wizard(console=console)


def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs (console + file)."),
    debug: bool = typer.Option(False, "--debug", help="Alias for --verbose."),
    chat_mode: bool = typer.Option(False, "--chat", help="Start Chat Mode."),
    analyze_mode: bool = typer.Option(False, "--analyze", help="Start Analyze Mode."),
    plan_mode: bool = typer.Option(False, "--plan", help="Start Plan Mode."),
    act_mode: bool = typer.Option(False, "--act", help="Start Act Mode."),
    repo: str | None = typer.Option(None, "--repo", help="Repository root for the selected mode."),
    model: str | None = typer.Option(None, "--model", help="Model override for the selected mode."),
    no_banner: bool = typer.Option(False, "--no-banner", help="Hide the Mana Agent banner."),
    debug_llm: bool = typer.Option(
        False,
        "--debug-llm/--no-debug-llm",
        help="Show internal LLM transport/request logs in live chat panels.",
    ),
    log_dir: str | None = typer.Option(None, "--log-dir", help="Directory for application log files."),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Directory for saving command output logs."),
    no_interactive: bool = typer.Option(False, "--no-interactive", help="Disable interactive setup and menus."),
    configure_now: bool = typer.Option(False, "--configure", help="Open the complete configuration TUI and exit."),
) -> None:
    global OUTPUT_DIR, LLM_DEBUG_MODE
    verbose = bool(verbose or debug)
    OUTPUT_DIR = Path(output_dir).resolve() if output_dir else None
    LLM_DEBUG_MODE = debug_llm
    _set_cli_runtime_flags(verbose=verbose, debug_llm=debug_llm)
    log_file = setup_logging(verbose=verbose, log_dir=log_dir)
    if not debug_llm:
        for noisy_logger in ("openai", "httpx", "httpcore"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logger.debug(
        "CLI initialized",
        extra={
            "verbose": verbose,
            "debug_llm": debug_llm,
            "log_file": str(log_file),
            "output_dir": str(OUTPUT_DIR) if OUTPUT_DIR else None,
        },
    )
    if tuple(sys.version_info[:2]) >= (3, 14):
        warning_msg = (
            "Python 3.14+ may emit LangChain/Pydantic v1 compatibility warnings. "
            "Recommended runtime: Python 3.12 or 3.13."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        # Only emit the visual panel at the interactive root (no subcommand).
        # Subcommands (e.g. `automation list`) must keep clean/JSON output.
        # The warnings.warn still fires for all paths (used by tests and visible to users).
        if ctx.invoked_subcommand is None:
            sink = build_output_sink(command_name="main", json_mode=False, console=console)
            sink.emit_warning(warning_msg)
    if ctx.invoked_subcommand is None:
        if configure_now:
            configure()
            return
        root = Path(repo).expanduser().resolve() if repo else Path.cwd().resolve()
        if root.is_file():
            root = root.parent

        def _invoke(name: str, args: list[str] | None = None) -> None:
            command = ctx.command.commands[name]
            with command.make_context(name, args or [], parent=ctx) as sub_ctx:
                command.invoke(sub_ctx)

        selected_flags = [chat_mode, analyze_mode, plan_mode, act_mode]
        if sum(1 for item in selected_flags if item) > 1:
            raise typer.BadParameter("Choose only one of --chat, --analyze, --plan, or --act.")
        if not no_banner:
            render_banner(console)
        try:
            ensure_setup(no_interactive=no_interactive, command_needs_llm=True, console=console)
        except NonInteractivePromptError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if chat_mode:
            args = ["--root-dir", str(root)]
            if model:
                args += ["--model", model]
            _invoke_with_multi_agent_route(
                ctx,
                "chat",
                args,
                root=root,
                request="root --chat",
                entrypoint="root",
                route_in_subcommand=True,
            )
            return
        if analyze_mode:
            args = ["--repo", str(root)]
            if model:
                args += ["--model", model]
            _invoke_with_multi_agent_route(ctx, "analyze", args, root=root, request="root --analyze", entrypoint="root")
            return
        if plan_mode:
            args = ["--repo", str(root)]
            if model:
                args += ["--model", model]
            _invoke_with_multi_agent_route(ctx, "plan", args, root=root, request="root --plan", entrypoint="root")
            return
        if act_mode:
            args = ["--root-dir", str(root), "--act"]
            if model:
                args += ["--model", model]
            _invoke_with_multi_agent_route(
                ctx,
                "chat",
                args,
                root=root,
                request="root --act",
                entrypoint="root",
                route_in_subcommand=True,
            )
            return

        if no_interactive or not _is_interactive_terminal():
            raise typer.BadParameter(
                "Interactive chat requires a TTY. Use a direct command for automation; "
                "if configuration is missing, run `mana-agent --configure` from an interactive terminal."
            )
        args = ["--root-dir", str(root)]
        if model:
            args += ["--model", model]
        _invoke_with_multi_agent_route(
            ctx,
            "chat",
            args,
            root=root,
            request="open chat",
            entrypoint="root",
            route_in_subcommand=True,
        )
