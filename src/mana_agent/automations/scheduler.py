"""Simple scheduler facade (Grok Build).

Wraps APScheduler when the optional extra is installed.
Provides hooks that the multi-agent runtime or CLI can call
after a validated model decision.
"""
from __future__ import annotations

from typing import Callable

__all__ = ["get_scheduler", "schedule_job", "list_jobs_stub"]


def get_scheduler():
    """Return an APScheduler instance if available, else a no-op stub."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore

        return BackgroundScheduler()
    except Exception:
        class _NoopScheduler:
            def add_job(self, *a, **k): pass
            def start(self): pass
            def shutdown(self): pass
        return _NoopScheduler()


def schedule_job(func: Callable, trigger: str = "interval", **trigger_args) -> None:
    """Schedule a job (best effort, graceful if no apscheduler)."""
    sched = get_scheduler()
    try:
        sched.add_job(func, trigger, **trigger_args)
        sched.start()
    except Exception:
        # Dashboard/automation optional; never break core
        pass


def list_jobs_stub() -> list[str]:
    """Return known job names (stub; real scheduler would inspect)."""
    return ["daily_report", "self_improvement_check"]
