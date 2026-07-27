"""Secure, provider-neutral desktop automation."""

from mana_agent.integrations.computer_control.config import ComputerControlSettings
from mana_agent.integrations.computer_control.models import (
    ComputerAction,
    ComputerActionResult,
    ExecutionRisk,
    ExecutionState,
    SupportedPlatform,
)
from mana_agent.integrations.computer_control.service import ComputerControlService

__all__ = [
    "ComputerAction",
    "ComputerActionResult",
    "ComputerControlService",
    "ComputerControlSettings",
    "ExecutionRisk",
    "ExecutionState",
    "SupportedPlatform",
]

