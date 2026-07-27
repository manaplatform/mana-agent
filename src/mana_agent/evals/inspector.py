from __future__ import annotations

import json
from typing import Any

from .storage import EvalStorage


def inspect_run(storage: EvalStorage, run_id: str, *, event_type: str | None = None) -> dict[str, Any]:
    run = storage.load_run(run_id, allow_incomplete=True)
    events: list[dict[str, Any]] = []
    path = storage.run_dir(run_id) / "events.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event_type and event.get("event_type") != event_type:
                continue
            events.append(event)
    return {"run": run.model_dump(mode="json"), "timeline": events, "incomplete": not (storage.run_dir(run_id) / ".complete").exists(), "patch_path": str(storage.run_dir(run_id) / "final.patch")}


def inspect_text(payload: dict[str, Any]) -> str:
    run = payload["run"]
    outcome = run.get("outcome") or {}
    lines = [
        f"Run: {run['run_id']}",
        f"Task / variant: {run['task_id']} / {run['variant_id']}",
        f"Status: {run['status']}{' (incomplete)' if payload['incomplete'] else ''}",
        f"Score: {outcome.get('normalized_score', 'n/a')}",
        f"Success: {outcome.get('task_success', False)}",
        f"Latency: {run.get('latency_seconds', 0):.3f}s",
        f"Cost: {run.get('calculated_cost') if run.get('calculated_cost') is not None else 'unknown'}",
        f"Patch: {payload['patch_path']}",
        "",
        "Timeline:",
    ]
    for event in payload["timeline"]:
        lines.append(f"  {event['sequence']:04d} {event['event_type']} {event.get('created_at', '')}")
    if run.get("errors"):
        lines.extend(["", "Errors:", *(f"  - {item}" for item in run["errors"])])
    return "\n".join(lines)
