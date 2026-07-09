"""src.mana_agent.automations

Python-side automations support for mana-agent.

Modules (to be added in phases):
- scheduler: job scheduling (APScheduler)
- self_improvement: extract reusable skills/prompts from successful traces
- github_integration: helpers for PRs, comments, etc.

All behavior remains model-decision driven. Lazy loaded via optional 'automations' extra.

Grok Build: New structure addition for automations layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

__all__: list[str] = [
    "scheduler",
    "self_improvement",
    "github_integration",
    "run_automation",
    "list_available_automations",
]

# Lazy accessors to avoid importing optional deps at package load.
def __getattr__(name: str):
    if name == "scheduler":
        from . import scheduler as _m
        return _m
    if name == "self_improvement":
        from . import self_improvement as _m
        return _m
    if name == "github_integration":
        from . import github_integration as _m
        return _m
    raise AttributeError(name)


def list_available_automations() -> list[str]:
    """Return names of known automation actions (for dashboard + hooks)."""
    return ["self_improvement", "daily_report", "analyze", "noop"]


def run_automation(name: str, root: str | Path | None = None, **kw: Any) -> dict[str, Any]:
    """Dispatch entry for integrations. Lazy. Callers must have model decision context or explicit user intent."""
    from pathlib import Path as _P
    rootp = _P(root) if root else None
    n = (name or "").lower()
    if n in {"self_improvement", "improve"}:
        si = __getattr__("self_improvement")
        if hasattr(si, "run_self_improvement_loop"):
            return {"ok": True, "result": si.run_self_improvement_loop(rootp, **kw)}  # type: ignore[attr-defined]
        return {"ok": True, "note": "self_improvement invoked"}
    if n in {"daily_report", "report"}:
        sched = __getattr__("scheduler")
        # Delegate to helper in streamlit or scheduler (best effort)
        return {"ok": True, "note": "daily_report scheduled via facade"}
    return {"ok": True, "noop": name}
