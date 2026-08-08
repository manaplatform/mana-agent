"""Agent timeout normalization must not hard-cap at 600s."""

from __future__ import annotations

from mana_agent.utils.timeouts import (
    UNLIMITED_AGENT_TIMEOUT_SECONDS,
    normalize_agent_timeout_seconds,
)


def test_normalize_honors_large_explicit_timeout() -> None:
    assert normalize_agent_timeout_seconds(999999999, floor=60) == 999999999
    assert normalize_agent_timeout_seconds(570, floor=60) == 570


def test_normalize_floor_and_default() -> None:
    assert normalize_agent_timeout_seconds(10, floor=60) == 60
    assert normalize_agent_timeout_seconds(None, default=600) == 600


def test_normalize_zero_means_unlimited() -> None:
    assert normalize_agent_timeout_seconds(0) == UNLIMITED_AGENT_TIMEOUT_SECONDS
    assert normalize_agent_timeout_seconds(-1) == UNLIMITED_AGENT_TIMEOUT_SECONDS
