from __future__ import annotations

import warnings

from .cli_internal import *
from .output import build_output_sink

@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs (console + file)."),
    debug_llm: bool = typer.Option(
        False,
        "--debug-llm/--no-debug-llm",
        help="Show internal LLM transport/request logs in live chat panels.",
    ),
    log_dir: str | None = typer.Option(None, "--log-dir", help="Directory for application log files."),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Directory for saving command output logs."),
) -> None:
    global OUTPUT_DIR, LLM_DEBUG_MODE
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
        chat_command = ctx.command.commands["chat"]
        with chat_command.make_context("chat", [], parent=ctx) as chat_ctx:
            chat_command.invoke(chat_ctx)
