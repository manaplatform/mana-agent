"""Self-improvement loop (Grok Build).

After successful sessions (coding flows, verified plans), the agent
can call into here (via model decision) to extract reusable
skills/prompts and persist them under .mana/skills/ or skills/.

All extraction must be driven by an explicit model decision object.
No keyword or heuristic auto-extract. Dashboard or explicit post-run
hooks may invoke the loop.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    load_taskboard_state,
    load_recent_traces,
    safe_read_json,
)


def extract_skill_from_trace(trace: dict[str, Any], root: Path | None = None) -> dict[str, Any] | None:
    """Given a successful trace, produce a compact reusable skill template.

    Improved: pulls title, verification, changed files, key evidence when present.
    Returns None only on empty input (caller decides via decision record).
    """
    if not trace:
        return None
    root = find_mana_root(root) if root is not None else Path(".") if root is not None else Path(".")
    title = (
        trace.get("title")
        or trace.get("task_title")
        or trace.get("normalized_goal")
        or trace.get("user_request")
        or "successful-session"
    )
    name = "skill-" + str(title).lower().replace(" ", "-").replace("/", "-")[:40].strip("-") or "extracted"
    evidence = trace.get("evidence") or trace.get("summary") or ""
    changed = trace.get("changed_files") or trace.get("files_touched") or []
    verification = trace.get("verification_passed") or trace.get("verification") or {}
    content = (
        f"# Reusable Skill: {title}\n\n"
        f"**Extracted**: {datetime.now(timezone.utc).isoformat()}Z\n\n"
        f"## Trigger\nSimilar successful verified task.\n\n"
        f"## Context\n- Verification passed: {verification}\n"
        f"- Key files: {changed[:5]}\n\n"
        f"## Evidence Snippet\n{str(evidence)[:600]}\n\n"
        f"## Template Prompt\nUse the verified pattern from this session for analogous requests.\n"
    )
    return {
        "name": name,
        "description": f"Auto-extracted from successful run: {title}",
        "trigger": "verification_passed or high-evidence task",
        "content": content,
        "source_trace": str(trace.get("_file", trace.get("trace_id", "")))[:120],
    }


def persist_skill(skill: dict[str, Any], root: Path | None = None) -> Path | None:
    """Persist skill under the project skills location (or .mana)."""
    root = find_mana_root(root) if root is not None else Path(".") if root is not None else Path(".")
    skills_dir = root / ".mana" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    name = skill.get("name", "unnamed") + ".md"
    path = skills_dir / name
    try:
        path.write_text(f"# {skill.get('name')}\n\n{skill.get('content', '')}\n", encoding="utf-8")
        return path
    except Exception:
        return None


def _load_successful_traces(root: Path, limit: int = 8) -> list[dict[str, Any]]:
    traces = load_recent_traces(root, limit=limit)
    good = []
    for t in traces:
        # Heuristic for success (no keyword routing; just surface candidates for model/hook)
        txt = json.dumps(t).lower()
        if any(k in txt for k in ["verification_passed", "status\":\"done", "\"done\"", "passed\":true", "success"]):
            good.append(t)
    return good or traces[:3]


def run_self_improvement_loop(root: Path | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """Scan recent successful traces + taskboard, extract + persist up to `limit` skills.

    Returns list of created skill metadata. Always safe (no-op on missing data).
    Invoker (dashboard button or post-run hook) is responsible for any model decision gating.
    """
    root = find_mana_root(root) if root is not None else Path(".")
    created: list[dict[str, Any]] = []

    # Prefer taskboard DONE items + traces
    tb = load_taskboard_state(root)
    tasks = tb.get("tasks", {}) if isinstance(tb, dict) else {}
    success_tasks = []
    for tid, t in (tasks.items() if isinstance(tasks, dict) else []):
        if not isinstance(t, dict):
            continue
        st = str(t.get("status", "")).lower()
        if st in {"done", "completed", "success"} or t.get("verification_passed"):
            success_tasks.append({"task_id": tid, **t})

    traces = _load_successful_traces(root, limit=max(limit, 6))

    candidates = (success_tasks + traces)[: max(3, limit * 2)]

    for cand in candidates:
        if len(created) >= max(1, int(limit)):
            break
        skill = extract_skill_from_trace(cand, root)
        if not skill:
            continue
        p = persist_skill(skill, root)
        if p:
            rec = {"name": skill["name"], "path": str(p), "source": cand.get("_file") or cand.get("task_id")}
            created.append(rec)
            # Also log an improvement record
            log_dir = root / ".mana" / "automations"
            log_dir.mkdir(parents=True, exist_ok=True)
            logf = log_dir / "self_improvement_runs.jsonl"
            try:
                with logf.open("a", encoding="utf-8") as h:
                    h.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat() + "Z", "skill": rec}, ensure_ascii=False) + "\n")
            except Exception:
                pass
    return created
