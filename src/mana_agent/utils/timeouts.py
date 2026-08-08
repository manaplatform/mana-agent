"""Shared timeout normalization for agent / coding turns.

Historically several call sites used ``min(max(timeout, 60), 600)``, which
silently capped long SWE-bench / full-auto runs at 10 minutes even when the
CLI passed a larger ``--agent-timeout-seconds`` / runner ``--timeout``.
"""

from __future__ import annotations

# Libraries that require a concrete int for "no timeout" get a long but finite
# ceiling (7 days). Operators still own the outer process wall clock.
UNLIMITED_AGENT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60


def normalize_agent_timeout_seconds(
    value: int | float | str | None,
    *,
    floor: int = 60,
    default: int = 600,
    unlimited_seconds: int = UNLIMITED_AGENT_TIMEOUT_SECONDS,
) -> int:
    """Normalize an agent wall-clock timeout.

    * ``None`` / empty → ``default``
    * ``<= 0`` → effectively unlimited (``unlimited_seconds``)
    * positive → ``max(floor, value)`` with **no artificial 600s ceiling**
    """
    if value is None:
        return int(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return int(default)
        value = int(text)
    raw = int(value)
    if raw <= 0:
        return max(1, int(unlimited_seconds))
    return max(int(floor), raw)


__all__ = [
    "UNLIMITED_AGENT_TIMEOUT_SECONDS",
    "normalize_agent_timeout_seconds",
]
