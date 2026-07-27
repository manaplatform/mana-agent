"""Capability group declarations used by providers and adapters."""

from mana_agent.integrations.computer_control.capabilities.applications import APPLICATION_OPERATIONS
from mana_agent.integrations.computer_control.capabilities.browser import BROWSER_OPERATIONS
from mana_agent.integrations.computer_control.capabilities.calendar import CALENDAR_OPERATIONS
from mana_agent.integrations.computer_control.capabilities.clipboard import CLIPBOARD_OPERATIONS
from mana_agent.integrations.computer_control.capabilities.filesystem import FILESYSTEM_OPERATIONS
from mana_agent.integrations.computer_control.capabilities.music import MEDIA_OPERATIONS
from mana_agent.integrations.computer_control.capabilities.notes import NOTES_OPERATIONS
from mana_agent.integrations.computer_control.capabilities.notifications import NOTIFICATION_OPERATIONS
from mana_agent.integrations.computer_control.capabilities.screenshots import SCREENSHOT_OPERATIONS
from mana_agent.integrations.computer_control.capabilities.system import SYSTEM_OPERATIONS

__all__ = [
    "APPLICATION_OPERATIONS", "BROWSER_OPERATIONS", "CALENDAR_OPERATIONS",
    "CLIPBOARD_OPERATIONS", "FILESYSTEM_OPERATIONS", "MEDIA_OPERATIONS",
    "NOTES_OPERATIONS", "NOTIFICATION_OPERATIONS", "SCREENSHOT_OPERATIONS",
    "SYSTEM_OPERATIONS",
]

