"""Compatibility boundary for the retired in-process scheduler.

Persistent automations are deployed through :mod:`mana_agent.automations.service`
to OS cron or a managed GitHub Actions workflow.  A background scheduler would
stop with the dashboard/CLI process, so it is intentionally not a fallback.
"""
from __future__ import annotations

from typing import Callable

__all__ = ["get_scheduler", "schedule_job", "list_jobs_stub"]


def get_scheduler() -> None:
    raise RuntimeError(
        "In-process scheduling was retired. Create automations through model-driven chat."
    )


def schedule_job(func: Callable[..., object], trigger: str = "cron", **trigger_args: object) -> None:
    _ = (func, trigger, trigger_args)
    get_scheduler()


def list_jobs_stub() -> list[str]:
    """Retained for import compatibility; persistent jobs live in config.json."""
    return []
