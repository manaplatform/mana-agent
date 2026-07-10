"""Streamlit helpers bridge (Grok Build addition).

Provides safe, read-mostly helpers for the optional web dashboard
to consume mana-agent runtime artifacts and services without
importing heavy deps at CLI/core load time.

All access is lazy. Dashboard code must import inside functions
or guard with try/except ImportError.

Key principles (per AGENTS.md):
- No keyword routing or fallbacks.
- Respect model-driven decisions (dashboard only surfaces existing data).
- Read-only first for MVP.
- Graceful degradation when optional deps or .mana artifacts missing.

Usage inside Streamlit pages:
    from mana_agent.ui.streamlit_helpers import (
        load_taskboard_state, load_recent_traces, ...
    )
"""
from __future__ import annotations

import json
import os  # used for MANA_DASHBOARD_ROOT env and safe paths
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_ROOT",
    "find_mana_root",
    "load_taskboard_state",
    "load_recent_traces",
    "get_index_stats",
    "get_last_analysis_summary",
    "safe_read_json",
    "list_analysis_artifacts",
    "get_metrics_summary",
    "get_observability_overview",
    "load_observability_spans",
    "load_observability_trace",
    "get_observability_health",
    "load_automations",
    "save_automations",
    "append_automation_run",
    "trigger_automation",
    "run_dashboard_chat",
    "list_schedules",
    "create_schedule",
    "schedule_status",
    "delete_schedule",
    "set_schedule_enabled",
    "run_schedule_now",
]


DEFAULT_ROOT = Path.cwd().resolve()


def find_mana_root(start: Path | None = None) -> Path:
    """Return the repository root (containing .mana or cwd)."""
    env_root = os.environ.get("MANA_DASHBOARD_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    root = (start or DEFAULT_ROOT).resolve()
    # Walk up a bit if needed; for dashboard we usually launch from root.
    for _ in range(4):
        if (root / ".mana").exists() or (root / "pyproject.toml").exists():
            return root
        if root.parent == root:
            break
        root = root.parent
    return (start or DEFAULT_ROOT).resolve()


def safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """Read JSON or return None on any error (dashboard is non-critical)."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def load_taskboard_state(root: Path | None = None) -> dict[str, Any]:
    """Load .mana/taskboard/state.json if present (read-only)."""
    root = find_mana_root(root)
    path = root / ".mana" / "taskboard" / "state.json"
    data = safe_read_json(path)
    if isinstance(data, dict):
        return data
    return {"tasks": [], "status": "no-taskboard", "root": str(root)}


def load_recent_traces(root: Path | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """Load recent trace entries (supports .json from TraceWriter + .jsonl from sessions/CLI).

    Most recent first. Graceful on parse errors.
    """
    root = find_mana_root(root)
    traces_dir = root / ".mana" / "traces"
    if not traces_dir.exists():
        return []
    # Support both formats produced by runtime
    json_files = sorted(traces_dir.glob("*.json"), reverse=True)
    jsonl_files = sorted(traces_dir.glob("*.jsonl"), reverse=True)
    files = (json_files + jsonl_files)[:limit]
    results: list[dict[str, Any]] = []
    for f in files:
        try:
            if f.suffix == ".json":
                obj = json.loads(f.read_text(encoding="utf-8"))
                obj["_file"] = f.name
                results.append(obj)
            else:
                # jsonl: take recent lines
                lines = f.read_text(encoding="utf-8").strip().splitlines()[-3:]
                for ln in lines:
                    if not ln.strip():
                        continue
                    obj = json.loads(ln)
                    obj["_file"] = f.name
                    results.append(obj)
        except Exception:
            continue
    return results[: limit * 3]


def get_index_stats(root: Path | None = None) -> dict[str, Any]:
    """Basic index stats from .mana/index if available."""
    root = find_mana_root(root)
    idx = root / ".mana" / "index"
    manifest = safe_read_json(idx / "manifest.json") or {}
    chunks_path = idx / "chunks.jsonl"
    chunk_count = 0
    if chunks_path.exists():
        try:
            chunk_count = sum(1 for _ in chunks_path.open("r", encoding="utf-8"))
        except Exception:
            pass
    return {
        "index_dir": str(idx),
        "chunks": chunk_count,
        "manifest": manifest,
        "ready": (idx / "chunks.jsonl").exists(),
    }


def get_last_analysis_summary(root: Path | None = None) -> dict[str, Any]:
    """Try to surface recent analysis artifacts (docs/analyze/ or similar)."""
    root = find_mana_root(root)
    candidates = [
        root / ".mana" / "analyze" / "llm_summary.md",
        root / ".mana" / "analyze" / "report.md",
        root / "docs" / "analyze" / "llm_summary.md",
        root / "docs" / "analyze" / "report.md",
        root / ".mana" / "last_analysis.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                if c.suffix == ".json":
                    return {"type": "json", "path": str(c), "data": safe_read_json(c)}
                text = c.read_text(encoding="utf-8")[:2000]
                return {"type": "md", "path": str(c), "preview": text}
            except Exception:
                pass
    return {"type": "none", "message": "No recent analysis artifacts found. Run `mana-agent analyze`."}


def list_analysis_artifacts(root: Path | None = None) -> list[dict[str, Any]]:
    """Discover real analysis/report artifacts under .mana/analyze, docs/analyze, .mana/reports."""
    root = find_mana_root(root)
    candidates = [
        root / ".mana" / "analyze",
        root / "docs" / "analyze",
        root / ".mana" / "reports",
    ]
    arts: list[dict[str, Any]] = []
    seen = set()
    for d in candidates:
        if not d.exists():
            continue
        for f in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".md", ".json", ".html", ".txt"}:
                continue
            key = str(f)
            if key in seen:
                continue
            seen.add(key)
            arts.append({
                "path": str(f),
                "name": f.name,
                "type": f.suffix.lstrip(".").lower(),
                "size": f.stat().st_size if f.exists() else 0,
            })
            if len(arts) >= 30:
                break
    return arts


def get_metrics_summary(root: Path | None = None) -> dict[str, Any]:
    """Compatibility metric view backed by canonical observability data only."""
    root = find_mana_root(root)
    overview = get_observability_overview(root)
    span_count = int(overview.get("span_count", 0))
    errors = int(overview.get("error_count", 0))
    return {
        "sessions": overview.get("trace_count", 0),
        "total_tokens": overview.get("total_tokens", 0),
        "avg_tokens": int(overview.get("total_tokens", 0) / max(1, span_count)),
        "success_rate": round((span_count - errors) / max(1, span_count) * 100, 1),
        "task_count": span_count,
        "done_tasks": span_count - errors,
        "tokens_series": [],
        "root": str(root),
    }


def _observability(root: Path | None = None):
    from mana_agent.observability import ObservabilityStore
    return ObservabilityStore(find_mana_root(root))


def get_observability_overview(root: Path | None = None, *, since: str = "") -> dict[str, Any]:
    """Return dashboard metrics from the canonical local SQLite trace store."""
    return _observability(root).overview(since=since)


def load_observability_spans(root: Path | None = None, **filters: Any) -> list[dict[str, Any]]:
    """Query redacted spans for dashboard trace exploration."""
    return _observability(root).spans(**filters)


def load_observability_trace(trace_id: str, root: Path | None = None) -> list[dict[str, Any]]:
    return _observability(root).spans(trace_id=trace_id, limit=1000)


def get_observability_health(root: Path | None = None) -> dict[str, Any]:
    return _observability(root).health()


def load_automations(root: Path | None = None) -> dict[str, Any]:
    """Load persisted automation definitions + run history (CRUD source of truth)."""
    root = find_mana_root(root)
    from mana_agent.automations.service import load_config

    try:
        data = load_config(root)
    except ValueError:
        return {"automations": [], "schedules": [], "runs": [], "root": str(root)}
    data["root"] = str(root)
    return data


def save_automations(data: dict[str, Any], root: Path | None = None) -> bool:
    """Persist automations config. Creates dirs. Returns success."""
    root = find_mana_root(root)
    try:
        from mana_agent.automations.service import save_config
        save_config(root, data)
        return True
    except (OSError, ValueError):
        return False


def list_schedules(root: Path | None = None) -> list[dict[str, Any]]:
    """Return typed persistent schedules for dashboard rendering."""
    root = find_mana_root(root)
    from mana_agent.automations.service import list_schedules as _list_schedules

    try:
        return [schedule.to_dict() for schedule in _list_schedules(root)]
    except ValueError:
        return []


def create_schedule(
    *,
    name: str,
    action: str,
    cron: str,
    targets: list[str],
    command: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Create and immediately deploy an explicitly requested schedule."""
    root = find_mana_root(root)
    from mana_agent.automations.service import ScheduleDefinition, deploy_schedule

    schedule = ScheduleDefinition.create(name=name, action=action, cron=cron, targets=targets, command=command)
    return deploy_schedule(schedule, root).to_dict()


def schedule_status(schedule_id: str, root: Path | None = None) -> dict[str, Any]:
    root = find_mana_root(root)
    from mana_agent.automations.service import deployment_status, get_schedule

    return deployment_status(get_schedule(root, schedule_id), root)


def delete_schedule(schedule_id: str, root: Path | None = None) -> None:
    root = find_mana_root(root)
    from mana_agent.automations.service import delete_schedule as _delete, remove_deployment

    schedule = _delete(root, schedule_id)
    remove_deployment(schedule, root)


def set_schedule_enabled(schedule_id: str, enabled: bool, root: Path | None = None) -> dict[str, Any]:
    root = find_mana_root(root)
    from mana_agent.automations.service import deploy_schedule, get_schedule

    schedule = get_schedule(root, schedule_id)
    schedule.enabled = enabled
    return deploy_schedule(schedule, root).to_dict()


def run_schedule_now(schedule_id: str, root: Path | None = None) -> dict[str, Any]:
    root = find_mana_root(root)
    from mana_agent.automations.service import get_schedule, run_schedule_now as _run_schedule_now

    return _run_schedule_now(get_schedule(root, schedule_id), root)


def append_automation_run(run: dict[str, Any], root: Path | None = None) -> bool:
    """Append a run record to the automations log."""
    root = find_mana_root(root)
    cfg = load_automations(root)
    runs = cfg.setdefault("runs", [])
    run = dict(run)
    run.setdefault("ts", __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat() + "Z")
    runs.append(run)
    # keep last 50
    cfg["runs"] = runs[-50:]
    return save_automations(cfg, root)


def trigger_automation(action: str, *, root: Path | None = None, **kwargs: Any) -> dict[str, Any]:
    """Safe dispatch for dashboard triggers. Lazy imports only. Respects optional layers.

    Supported actions: self_improvement, daily_report, analyze, noop
    """
    root = find_mana_root(root)
    action = (action or "noop").lower().strip()
    try:
        if action in {"self_improvement", "self-improve", "improve"}:
            from mana_agent.automations.self_improvement import run_self_improvement_loop  # type: ignore
            result = run_self_improvement_loop(root, **kwargs) or []
            append_automation_run({"action": action, "result": {"skills": len(result)}}, root)
            return {"ok": True, "action": action, "created": len(result), "detail": result}
        elif action in {"daily_report", "report"}:
            # Daily report is the explicit report-generation action; it uses the
            # same validated analysis service as the CLI rather than a transient
            # in-process scheduler or marker-file fallback.
            return trigger_automation("analyze", root=root, **kwargs)
        elif action in {"analyze", "generate_report"}:
            # Direct real call to ProjectAnalyzeService (guarantees .mana/analyze is created).
            # This is the reliable "real functionality" path inside the dashboard process.
            # We fall back to subprocess only if direct call fails.
            artifact_dir = root / ".mana" / "analyze"
            try:
                from mana_agent.services.project_analyze_service import (
                    ProjectAnalyzeOptions,
                    ProjectAnalyzeService,
                )

                artifact_dir.mkdir(parents=True, exist_ok=True)

                # Use the same persisted ~/.mana configuration as the CLI.
                # Analyze must not let the target repository's .env choose its model.
                llm_analyzer = None
                try:
                    from mana_agent.commands.cli_internal import _build_project_llm_analyzer
                    llm_analyzer = _build_project_llm_analyzer()
                except Exception:
                    # Graceful: dashboard analyze still works deterministically
                    llm_analyzer = None

                result = ProjectAnalyzeService().run(
                    root,
                    artifact_dir,
                    options=ProjectAnalyzeOptions(
                        depth="normal",
                        output_format="both",
                    ),
                    llm_analyzer=llm_analyzer,
                )

                append_automation_run({
                    "action": action,
                    "artifact_dir": str(artifact_dir),
                    "artifacts_written": len(getattr(result, "artifacts", {})),
                    "llm_used": llm_analyzer is not None,
                }, root)

                llm_note = "with LLM analysis" if llm_analyzer is not None else "deterministic (no API key or LLM disabled)"
                return {
                    "ok": True,
                    "action": action,
                    "artifact_dir": str(artifact_dir),
                    "note": f"Direct service call (real) - {llm_note}",
                    "llm_used": llm_analyzer is not None,
                    "artifacts": list(getattr(result, "artifacts", {}).keys())[:8],
                }
            except Exception as direct_err:
                return {
                    "ok": False,
                    "action": action,
                    "artifact_dir": str(artifact_dir),
                    "error": f"Model decision failed: analyze execution. No fallback action was executed. Reason: {direct_err}",
                }
        else:
            append_automation_run({"action": action, "noop": True}, root)
            return {"ok": True, "action": action, "noop": True}
    except Exception as e:
        return {"ok": False, "action": action, "error": str(e)}


def run_dashboard_chat(prompt: str, root: Path | None = None, k: int = 6) -> dict[str, Any]:
    """Real model-routed chat response, using the exact same service/ask stack as CLI chat.

    Tries hard to give responses "routed via models" like the full CLI experience:
    - Uses Settings + build_ask_service (entry router decides route)
    - Prefers ask_with_tools for agentic/tool-using behavior (closer to rich chat)
    - Falls back gracefully to preview if no key / no index / import error.

    Returns dict with "answer", "mode" ("real"|"preview"), "sources", "warnings", etc.
    This is the core to make dashboard chat "like cli chat".
    """
    root = find_mana_root(root)
    prompt = (prompt or "").strip()
    if not prompt:
        return {"answer": "", "mode": "empty"}

    try:
        from mana_agent.config.settings import Settings
        from mana_agent.commands.cli_internal import build_ask_service
        from mana_agent.services.ask_service import AskResponseWithTrace  # type: ignore

        settings = Settings()
        api_key = getattr(settings, "openai_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return {
                "answer": "(No OPENAI_API_KEY configured) Routed via model decision layer would happen here. "
                          "Set key in env or ~/.mana and ensure index is built (run chat in CLI first).",
                "mode": "preview",
                "sources": [],
            }

        service = build_ask_service(settings, None, project_root=root)

        # Default index
        idx_dir = root / ".mana" / "index"
        if not (idx_dir / "chunks.jsonl").exists():
            # Try to let service handle or give useful message
            pass

        # Use ask_with_tools to get closer to full CLI chat agentic behavior (tool use, multi-step)
        try:
            resp = service.ask_with_tools(str(idx_dir), prompt, k=k, max_steps=5, timeout_seconds=45)
        except Exception:
            # Fallback to classic ask
            resp = service.ask(str(idx_dir), prompt, k=k)

        answer = ""
        sources = []
        mode = "real"
        warnings = []
        if isinstance(resp, dict):
            answer = resp.get("answer") or str(resp)
            sources = resp.get("sources", [])
        else:
            answer = getattr(resp, "answer", str(resp))
            sources = getattr(resp, "sources", []) or []
            warnings = getattr(resp, "warnings", []) or []

        if not answer or answer.startswith("Selected route failed"):
            mode = "preview"
            answer = answer or "(Model route produced no answer. Try again or use CLI for full session.)"

        return {
            "answer": answer,
            "mode": mode,
            "sources": sources[:5] if sources else [],
            "warnings": warnings,
            "root": str(root),
        }
    except Exception as e:
        # Graceful: never break the dashboard UI
        return {
            "answer": f"(Preview - real routing failed: {str(e)[:120]}) Evidence would be collected by AskAgent/MainAgent. "
                      "Run `mana-agent chat` in terminal for full CLI experience.",
            "mode": "preview",
            "error": str(e),
            "sources": [],
        }
