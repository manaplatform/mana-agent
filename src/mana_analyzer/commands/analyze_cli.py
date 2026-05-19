from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from .cli_internal import *
from .output import build_output_sink
from .ui_helpers import _build_flow_summary_payload
from mana_analyzer.renderers.html_report import render_analyze_html
from mana_analyzer.services.coding_memory_service import CodingMemoryService
from mana_analyzer.services.vulnerability_service import VulnerabilityService
from mana_analyzer.utils.guards import guard_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {}


def _to_list_of_dicts(values: list[Any]) -> list[dict[str, Any]]:
    return [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in values]


def _merge_findings(static_findings: list[Any], llm_findings: list[Any]) -> list[Any]:
    return sorted(
        {
            (
                getattr(f, "rule_id", ""),
                getattr(f, "severity", ""),
                getattr(f, "file_path", ""),
                getattr(f, "line", 0),
                getattr(f, "column", 0),
                getattr(f, "message", ""),
            ): f
            for f in [*static_findings, *llm_findings]
        }.values(),
        key=lambda f: (getattr(f, "file_path", ""), getattr(f, "line", 0), getattr(f, "column", 0), getattr(f, "rule_id", "")),
    )


def _severity_counts(findings: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        severity = str(getattr(finding, "severity", "unknown") or "unknown").lower()
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _resolve_model(settings: Settings, model: str | None) -> str:
    resolved = model or os.environ.get("LLM_MODEL") or getattr(settings, "openai_chat_model", None)
    if not resolved:
        raise typer.BadParameter("No LLM model configured. Pass --model or set LLM_MODEL/OPENAI_CHAT_MODEL.")
    return str(resolved)


def _require_llm_settings(settings: Settings) -> None:
    if not getattr(settings, "openai_api_key", None):
        raise typer.BadParameter("OPENAI_API_KEY is required for unified analyze.")


def _build_llm_findings_service(settings: Settings, model: str) -> Any:
    built = _public_attr("build_llm_analyze_service", build_llm_analyze_service)(settings, model_override=model)
    if isinstance(built, tuple):
        chain = built[0]

        class _ChainService:
            def analyze(self, path: str, static_findings: list[Any], max_files: int = 10) -> list[Any]:
                root = Path(path).resolve()
                files = list(getattr(build_dependency_service().analyze(root), "files", []) or [])[:max_files]
                findings: list[Any] = []
                for file_path in files:
                    try:
                        source = file_path.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    findings.extend(
                        chain.run(
                            file_path=str(file_path.relative_to(root)),
                            source=source[:12000],
                            static_findings=static_findings,
                        )
                    )
                return findings

        return _ChainService()
    return built


def _public_attr(name: str, fallback: Any) -> Any:
    public_cli = sys.modules.get("mana_analyzer.commands.cli")
    return getattr(public_cli, name, fallback) if public_cli is not None else fallback


def _flow_payload(project_root: Path, settings: Settings) -> dict[str, Any]:
    memory = CodingMemoryService(
        project_root=project_root,
        max_turns=getattr(settings, "coding_flow_max_turns", 5),
        max_tasks=getattr(settings, "coding_flow_max_tasks", 20),
    )
    flow_id = memory.get_active_flow_id()
    if not flow_id:
        return {"active": False, "summary": None, "warnings": ["No active coding flow found."]}
    summary = _build_flow_summary_payload(memory, flow_id)
    if summary is None:
        return {"active": False, "summary": None, "warnings": [f"Flow not found: {flow_id}"]}
    return {"active": True, "summary": summary, "warnings": []}


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = ["# Unified Analyze Report", ""]
    meta = payload["meta"]
    lines.extend(
        [
            "## Overview",
            f"- Root: `{meta['project_root']}`",
            f"- Generated: {meta['generated_at']}",
            f"- Model: `{meta['model']}`",
            f"- Status: {meta['status']}",
            "",
        ]
    )

    index = payload["index"]
    lines.extend(
        [
            "## Index",
            f"- Index dir: `{index.get('index_dir', '')}`",
            f"- Total files: {index.get('total_files', 0)}",
            f"- Indexed files: {index.get('indexed_files', 0)}",
            f"- New chunks: {index.get('new_chunks', 0)}",
            "",
        ]
    )

    search = payload["search"]
    lines.append("## Search")
    if search.get("query"):
        lines.append(f"- Query: `{search['query']}`")
        for hit in search.get("results", []):
            lines.append(f"- `{hit.get('file_path')}`:{hit.get('start_line')}-{hit.get('end_line')} {hit.get('symbol_name')} ({hit.get('score')})")
    else:
        lines.append("- No query provided; search results omitted.")
    lines.append("")

    deps = payload["dependencies"]
    lines.extend(
        [
            "## Dependencies",
            f"- Languages: {', '.join(deps.get('languages', [])) or 'unknown'}",
            f"- Frameworks: {', '.join(deps.get('frameworks', [])) or 'none'}",
            f"- Package managers: {', '.join(deps.get('package_managers', [])) or 'unknown'}",
            f"- Runtime dependencies: {len(deps.get('runtime_dependencies', []))}",
            f"- Dev dependencies: {len(deps.get('dev_dependencies', []))}",
            f"- Module edges: {payload['graph'].get('module_edges', 0)}",
            f"- External edges: {payload['graph'].get('external_edges', 0)}",
            "",
        ]
    )

    findings = payload["findings"]
    lines.extend(
        [
            "## Findings",
            f"- Static findings: {len(findings.get('static', []))}",
            f"- LLM findings: {len(findings.get('llm', []))}",
            f"- Total findings: {len(findings.get('merged', []))}",
            f"- By severity: {json.dumps(findings.get('by_severity', {}), sort_keys=True)}",
            "",
        ]
    )
    for finding in findings.get("merged", [])[:50]:
        lines.append(
            f"- **{str(finding.get('severity', '')).upper()}** `{finding.get('rule_id')}` "
            f"{finding.get('file_path')}:{finding.get('line')}:{finding.get('column')} - {finding.get('message')}"
        )
    lines.append("")

    describe = payload["describe"]
    lines.extend(["## Repository Description", "", "### Architecture", describe.get("architecture_summary", ""), "", "### Technology", describe.get("tech_summary", ""), "", "### File Summaries"])
    for item in describe.get("descriptions", [])[:50]:
        lines.append(f"- `{item.get('file_path')}` ({item.get('language')}) - {item.get('summary')}")
    if not describe.get("descriptions"):
        lines.append("- none")
    lines.append("")

    structure = payload["structure"]
    lines.extend(["## Structure", f"- Project root: `{structure.get('project_root', meta['project_root'])}`"])
    if structure.get("language_counts"):
        lines.append(f"- Language counts: {json.dumps(structure.get('language_counts'), sort_keys=True)}")
    lines.append("")

    security = payload["security"]
    lines.extend(
        [
            "## Security",
            f"- Source: {security.get('source', 'osv')}",
            f"- Status: {security.get('status', 'unknown')}",
            f"- Packages scanned: {len(security.get('scanned_packages', []))}",
            f"- Vulnerabilities: {len(security.get('vulnerabilities', []))}",
            "",
        ]
    )

    flow = payload["flow"]
    lines.append("## Flow")
    if flow.get("active") and flow.get("summary"):
        summary = flow["summary"]
        lines.append(f"- Flow: `{summary.get('flow_id')}`")
        lines.append(f"- Objective: {summary.get('objective')}")
        for task in summary.get("open_tasks", [])[:20]:
            lines.append(f"- [ ] {task}")
    else:
        lines.append("- No active coding flow found.")
    lines.append("")

    lines.append("## Warnings")
    warnings = payload.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


@app.command()
@guard_root
def analyze(
    path: str,
    query: str | None = typer.Option(None, "--query"),
    k: int | None = typer.Option(None, "--k"),
    model: str | None = typer.Option(None, "--model"),
    llm_max_files: int = typer.Option(10, "--llm-max-files"),
    summary_max_files: int = typer.Option(12, "--summary-max-files"),
    include_tests: bool = typer.Option(True, "--include-tests/--no-include-tests"),
    online: bool = typer.Option(True, "--online/--offline"),
    osv_timeout_seconds: int = typer.Option(10, "--osv-timeout-seconds"),
    security_scope: str = typer.Option("all", "--security-scope"),
    report_profile: str = typer.Option("standard", "--report-profile"),
    detail_line_target: int = typer.Option(350, "--detail-line-target"),
    security_lens: str = typer.Option("defensive-red-team", "--security-lens"),
    output_format: str = typer.Option("all", "--output-format"),
    fail_on: str = typer.Option("none", "--fail-on"),
    as_json: bool = typer.Option(False, "--json"),
    with_llm: bool = typer.Option(True, "--with-llm/--no-llm", hidden=True),
    full_structure: bool = typer.Option(True, "--full-structure/--no-full-structure", hidden=True),
) -> None:
    _ = (with_llm, full_structure)
    if output_format not in {"json", "markdown", "html", "all"}:
        raise typer.BadParameter("--output-format must be one of: json, markdown, html, all")
    if fail_on not in {"none", "warning", "error"}:
        raise typer.BadParameter("--fail-on must be one of: none, warning, error")
    if security_scope not in {"all", "runtime", "dev"}:
        raise typer.BadParameter("--security-scope must be one of: all, runtime, dev")
    if report_profile not in {"standard", "deep"}:
        raise typer.BadParameter("--report-profile must be standard|deep")
    if security_lens not in {"defensive-red-team", "architecture", "compliance"}:
        raise typer.BadParameter("--security-lens must be defensive-red-team|architecture|compliance")

    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    settings = _public_attr("Settings", Settings)()
    _require_llm_settings(settings)
    final_model = _resolve_model(settings, model)
    resolved_k = k or settings.default_top_k
    sink = build_output_sink(command_name="analyze", json_mode=as_json, output_file=None, console=console)
    warnings: list[str] = []

    try:
        dependency_service = _public_attr("build_dependency_service", build_dependency_service)()
        deps_report = dependency_service.analyze(str(root))

        index_dir = default_index_dir(root)
        index_service = _public_attr("build_index_service", build_index_service)(settings)
        try:
            index_result = index_service.index(root, index_dir=index_dir, rebuild=False, vectors=True)
        except TypeError:
            index_result = index_service.index(root, index_dir=index_dir, rebuild=False)
        if index_result.get("vector_error"):
            raise RuntimeError(f"Embedding/vector index failed: {index_result['vector_error']}")

        search_payload: dict[str, Any] = {"query": query or "", "k": resolved_k, "results": [], "warnings": []}
        if query:
            hits = _public_attr("build_search_service", build_search_service)(settings).search(index_dir=index_dir, query=query, k=resolved_k)
            search_payload["results"] = _to_list_of_dicts(hits)

        static_findings = _public_attr("build_analyze_service", build_analyze_service)().analyze(str(root))
        llm_service = _build_llm_findings_service(settings, final_model)
        llm_findings = llm_service.analyze(str(root), static_findings=static_findings, max_files=llm_max_files)
        merged_findings = _merge_findings(static_findings, llm_findings)

        describe_builder = _public_attr("build_describe_service", build_describe_service)
        try:
            describe_service = describe_builder(
                settings,
                model_override=final_model,
                use_llm=True,
                include_tests=include_tests,
            )
        except TypeError:
            describe_service = describe_builder(
                settings,
                model_override=final_model,
                use_llm=True,
            )
        describe_report = describe_service.describe(
            root,
            max_files=summary_max_files,
            include_functions=True,
            use_llm=True,
        )

        structure_report = _public_attr("StructureService", StructureService)(include_tests=include_tests).analyze_project(str(root))

        try:
            inventory = dependency_service.collect_inventory(str(root))
        except Exception as exc:
            inventory = []
            warnings.append(f"collect_inventory failed ({type(exc).__name__}): {exc}")
        security_report = VulnerabilityService().scan_dependencies(
            inventory,
            online=online,
            timeout_seconds=osv_timeout_seconds,
            scope=security_scope,
        )
        warnings.extend(getattr(security_report, "warnings", []) or [])

        flow = _flow_payload(root, settings)
        warnings.extend(flow.get("warnings", []))

        deps_payload = _to_dict(deps_report)
        payload = {
            "meta": {
                "project_root": str(root),
                "generated_at": _utc_now(),
                "model": final_model,
                "status": "ok",
                "output_format": output_format,
                "report_profile": report_profile,
                "detail_line_target": detail_line_target,
                "security_lens": security_lens,
            },
            "index": index_result,
            "search": search_payload,
            "dependencies": deps_payload,
            "graph": {
                "project_root": deps_payload.get("project_root", str(root)),
                "module_edges": len(deps_payload.get("module_edges", []) or []),
                "external_edges": len(deps_payload.get("dependency_edges", []) or []),
            },
            "findings": {
                "static": _to_list_of_dicts(static_findings),
                "llm": _to_list_of_dicts(llm_findings),
                "merged": _to_list_of_dicts(merged_findings),
                "by_severity": _severity_counts(merged_findings),
            },
            "describe": _to_dict(describe_report),
            "structure": _to_dict(structure_report),
            "security": security_report.to_dict(),
            "flow": flow,
            "warnings": warnings,
        }
        markdown = _render_markdown(payload)
        html_payload = {
            **payload,
            "findings": payload["findings"]["merged"],
            "summarization": payload["describe"],
            "tech": {
                "languages": deps_payload.get("languages", []),
                "file_count": index_result.get("total_files", 0),
                "chain_profile": report_profile,
                "chain_config": "",
            },
            "project_structure_analysis": {
                "line_count": 0,
                "analysis_lines": [],
            },
        }
        html = render_analyze_html(html_payload, markdown)

        out_json, out_md, out_html = _resolve_analyze_artifact_paths(root)
        out_dot = root / ".mana" / "analyze.dot"
        out_graphml = root / ".mana" / "analyze.graphml"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        if output_format in {"json", "all"}:
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if output_format in {"markdown", "all"}:
            out_md.write_text(markdown, encoding="utf-8")
        if output_format in {"html", "all"}:
            out_html.write_text(html, encoding="utf-8")
        out_dot.write_text(deps_report.to_dot(), encoding="utf-8")
        out_graphml.write_text(deps_report.to_graphml(), encoding="utf-8")

        if as_json:
            sink.emit_json(payload)
        else:
            sink.emit_text(markdown)

        if fail_on == "warning" and any(getattr(f, "severity", "") in {"warning", "error"} for f in merged_findings):
            raise typer.Exit(code=1)
        if fail_on == "error" and any(getattr(f, "severity", "") == "error" for f in merged_findings):
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        if as_json:
            sink.emit_json({"error": str(exc), "type": type(exc).__name__})
        else:
            sink.emit_error(f"Analyze failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)
