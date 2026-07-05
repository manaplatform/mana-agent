from __future__ import annotations

import warnings

from .cli_internal import *
from .output import build_output_sink
from mana_agent.ui.banner import render_banner, render_repository


def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs (console + file)."),
    debug: bool = typer.Option(False, "--debug", help="Alias for --verbose."),
    chat_mode: bool = typer.Option(False, "--chat", help="Start Chat Mode."),
    analyze_mode: bool = typer.Option(False, "--analyze", help="Start Analyze Mode."),
    plan_mode: bool = typer.Option(False, "--plan", help="Start Plan Mode."),
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
        sink = build_output_sink(command_name="main", json_mode=False, console=console)
        sink.emit_warning(warning_msg)
    if ctx.invoked_subcommand is None:
        root = Path(repo).expanduser().resolve() if repo else Path.cwd().resolve()
        if root.is_file():
            root = root.parent

        def _invoke(name: str, args: list[str] | None = None) -> None:
            command = ctx.command.commands[name]
            with command.make_context(name, args or [], parent=ctx) as sub_ctx:
                command.invoke(sub_ctx)

        selected_flags = [chat_mode, analyze_mode, plan_mode]
        if sum(1 for item in selected_flags if item) > 1:
            raise typer.BadParameter("Choose only one of --chat, --analyze, or --plan.")
        if chat_mode:
            args = ["--root-dir", str(root)]
            if model:
                args += ["--model", model]
            _invoke_with_multi_agent_route(ctx, "chat", args, root=root, request="root --chat", entrypoint="root")
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

        if not no_banner:
            render_banner(console)
        render_repository(root, console)
        console.print("\n[bold cyan]Choose what you want to do:[/bold cyan]\n")
        console.print("1. Chat with repo (mana-agent chat)")
        console.print("2. Analyze repo")
        console.print("3. Create implementation plan")
        console.print("4. Exit")
        try:
            choice = input("\nMana Agent ❯ ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            return
        if choice in {"1", "chat", "c"}:
            _invoke_with_multi_agent_route(ctx, "chat", ["--root-dir", str(root)], root=root, request="root menu chat", entrypoint="root")
        elif choice in {"2", "analyze", "a"}:
            _invoke_with_multi_agent_route(ctx, "analyze", ["--repo", str(root)], root=root, request="root menu analyze", entrypoint="root")
        elif choice in {"3", "plan", "p"}:
            _invoke_with_multi_agent_route(ctx, "plan", ["--repo", str(root)], root=root, request="root menu plan", entrypoint="root")
        else:
            console.print("Goodbye!")
