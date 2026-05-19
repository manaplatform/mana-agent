from __future__ import annotations

import sys

from .cli_internal import *
from .output import build_output_sink

@app.command()
def ask(
    question: str,
    k: int | None = typer.Option(None, "--k"),
    model: str | None = typer.Option(None, "--model"),
    index_dir: str | None = typer.Option(None, "--index-dir"),
    ephemeral_index: bool = typer.Option(
        False,
        "--ephemeral-index",
        help="Use temporary index(es) and delete them after answering (ignored if --index-dir is set).",
    ),
    dir_mode: bool = typer.Option(False, "--dir-mode", help="Enable directory-aware ask mode."),
    root_dir: str | None = typer.Option(None, "--root-dir", help="Project root used for tool execution and default index paths."),
    max_indexes: int = typer.Option(0, "--max-indexes", help="Maximum discovered indexes to use (0 means no limit)."),
    auto_index_missing: bool = typer.Option(
        True,
        "--auto-index-missing/--no-auto-index-missing",
        help="Automatically create missing subproject indexes in --dir-mode.",
    ),
    agent_tools: bool = typer.Option(True, "--agent-tools"),
    agent_max_steps: int = typer.Option(6, "--agent-max-steps"),
    agent_unlimited: bool = typer.Option(
        False,
        "--agent-unlimited/--no-agent-unlimited",
        help="Use effectively unlimited agent tool steps (subject to timeout/resources).",
    ),
    agent_timeout_seconds: int = typer.Option(30, "--agent-timeout-seconds"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    output_file = _resolve_output_file()
    sink = build_output_sink(command_name="ask", json_mode=as_json, output_file=output_file, console=console)
    logger.info(
        "Ask command started",
        extra={"question": question, "k": k, "model_override": model, "dir_mode": dir_mode, "index_dir": index_dir, "ephemeral_index": ephemeral_index},
    )
    settings = Settings()
    root = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
    if root.is_file():
        root = root.parent
    resolved_k = k or settings.default_top_k
    effective_agent_max_steps = _resolve_agent_max_steps(
        agent_max_steps,
        agent_unlimited=agent_unlimited,
        min_steps=1,
    )

    # For agent tools, project_root matters (file reads/commands).
    public_cli = sys.modules.get("mana_analyzer.commands.cli")
    build_ask = getattr(public_cli, "_build_ask_service_compat", _build_ask_service_compat) if public_cli is not None else _build_ask_service_compat
    service = build_ask(settings, model_override=model, project_root=root)

    tmp_single: tempfile.TemporaryDirectory | None = None
    tmp_dir_mode_root: tempfile.TemporaryDirectory | None = None

    try:
        if dir_mode:
            logger.debug(
                "Resolved ask dir-mode parameters",
                extra={
                    "k": resolved_k,
                    "root_dir": str(root),
                    "max_indexes": max_indexes,
                    "auto_index_missing": auto_index_missing,
                    "agent_tools": agent_tools,
                    "agent_unlimited": agent_unlimited,
                    "ephemeral_index": ephemeral_index,
                },
            )

            discovered_subprojects = discover_subprojects(root)
            discovered_indexes = discover_index_dirs(root)
            discovered_index_set = {item.resolve() for item in discovered_indexes}

            index_service = build_index_service(settings)
            auto_indexed_count = 0
            skipped_missing_count = 0
            warnings: list[str] = []
            selected_indexes: list[Path] = []

            tmp_base: Path | None = None
            if ephemeral_index and not index_dir:
                tmp_dir_mode_root, tmp_base = _make_ephemeral_index_dir(prefix="mana_indexes_")

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
                        try:
                            logger.info("Auto-indexing missing/empty index", extra={"subproject_root": str(subproject.root_path), "index_dir": str(expected_index)})
                            _index_service_index_compat(
                                index_service,
                                target_path=subproject.root_path,
                                index_dir=expected_index,
                                rebuild=False,
                                vectors=True,
                            )
                            auto_indexed_count += 1
                            selected_indexes.append(expected_index)
                        except Exception as exc:
                            warning = f"Failed to auto-index {subproject.root_path}: {exc}"
                            logger.warning(warning)
                            warnings.append(warning)
                            if _index_has_chunks(expected_index):
                                selected_indexes.append(expected_index)
                    else:
                        skipped_missing_count += 1
                        warning = f"Skipped missing or empty index for subproject {subproject.root_path}"
                        warnings.append(warning)
                        logger.warning(warning)
            else:
                if tmp_base is not None:
                    root_index = (tmp_base / _stable_subdir_name(root)).resolve()
                else:
                    root_index = default_index_dir(root).resolve()

                if root_index.exists() and _index_has_search_data(root_index):
                    selected_indexes = [root_index]
                elif auto_index_missing:
                    try:
                        logger.info("Auto-indexing root", extra={"root": str(root), "index_dir": str(root_index)})
                        _index_service_index_compat(
                            index_service,
                            target_path=root,
                            index_dir=root_index,
                            rebuild=False,
                            vectors=True,
                        )
                        auto_indexed_count = 1
                        selected_indexes = [root_index]
                    except Exception as exc:
                        warning = f"Failed to auto-index {root}: {exc}"
                        logger.warning(warning)
                        warnings.append(warning)
                        if _index_has_chunks(root_index):
                            selected_indexes = [root_index]

            selected_indexes = sorted({item.resolve() for item in selected_indexes}, key=lambda item: str(item))
            if max_indexes > 0:
                selected_indexes = selected_indexes[:max_indexes]

            logger.info(
                "Ask dir-mode index selection completed",
                extra={
                    "root_dir": str(root),
                    "discovered_indexes": len(discovered_indexes),
                    "selected_indexes": len(selected_indexes),
                    "auto_indexed_count": auto_indexed_count,
                    "skipped_missing_count": skipped_missing_count,
                    "ephemeral_index": ephemeral_index,
                },
            )

            if agent_tools:
                response = service.ask_with_tools_dir_mode(
                    index_dirs=selected_indexes,
                    question=question,
                    k=resolved_k,
                    max_steps=effective_agent_max_steps,
                    timeout_seconds=agent_timeout_seconds,
                    root_dir=root,
                )
            else:
                response = service.ask_dir_mode(index_dirs=selected_indexes, question=question, k=resolved_k, root_dir=root)
            if warnings:
                response.warnings.extend(warnings)

        else:
            if ephemeral_index and not index_dir:
                tmp_single, resolved_index_dir = _make_ephemeral_index_dir()
                index_service = build_index_service(settings)
                _index_service_index_compat(
                    index_service,
                    target_path=root,
                    index_dir=resolved_index_dir,
                    rebuild=False,
                    vectors=True,
                )
            else:
                resolved_index_dir = Path(index_dir).resolve() if index_dir else default_index_dir(root)

            logger.debug(
                "Resolved ask parameters",
                extra={"k": resolved_k, "index_dir": str(resolved_index_dir), "agent_tools": agent_tools, "ephemeral_index": ephemeral_index},
            )

            if agent_tools:
                response = service.ask_with_tools(
                    index_dir=resolved_index_dir,
                    question=question,
                    k=resolved_k,
                    max_steps=effective_agent_max_steps,
                    timeout_seconds=agent_timeout_seconds,
                )
            else:
                response = service.ask(index_dir=resolved_index_dir, question=question, k=resolved_k)

        logger.info(
            "Ask command completed",
            extra={"sources": len(response.sources), "mode": getattr(response, "mode", "classic")},
        )

        if as_json:
            sink.emit_json(response.to_dict())
            return

        lines: list[str] = [response.answer]
        if hasattr(response, "mode"):
            lines.append("")
            lines.append(f"Mode: {response.mode}")
        if hasattr(response, "trace") and response.trace:
            lines.append("")
            lines.append("Tool Trace:")
            for item in response.trace:
                lines.append(
                    f"- {item.tool_name} [{item.status}] {item.duration_ms:.1f}ms args={item.args_summary}"
                )
        if getattr(response, "warnings", None):
            lines.append("")
            lines.append("Warnings:")
            for warning in response.warnings:
                lines.append(f"- {warning}")
        lines.append("")
        lines.append("Sources:")
        if not response.sources:
            lines.append("- none")
            sink.emit_text("\n".join(lines))
            return

        if getattr(response, "source_groups", None):
            for group in response.source_groups:
                lines.append(f"- subproject={group.subproject_root} index={group.index_dir}")
                for source in group.sources:
                    lines.append(f"  - {source.file_path}:{source.start_line}-{source.end_line}")
        else:
            for source in response.sources:
                lines.append(f"- {source.file_path}:{source.start_line}-{source.end_line}")
        sink.emit_text("\n".join(lines))

    finally:
        if tmp_single is not None:
            tmp_single.cleanup()
        if tmp_dir_mode_root is not None:
            tmp_dir_mode_root.cleanup()
