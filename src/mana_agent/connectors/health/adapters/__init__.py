"""Connector-specific health adapters."""

from .gmail import GmailHealthAdapter, discover_gmail_adapters
from .telegram import TelegramHealthAdapter, discover_telegram_adapter

__all__ = [
    "GmailHealthAdapter",
    "TelegramHealthAdapter",
    "discover_gmail_adapters",
    "discover_telegram_adapter",
]
