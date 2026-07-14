"""Gateway layer for unified access to Mana-Agent multi-agent chat and runtime.

All frontends (TUI chat, Telegram, Dashboard/API) should connect through
the gateway to reach agents. The gateway centralizes construction of
AskService / ChatService / CodingAgent stacks and session management.

See chat_gateway.py for the main implementation.
"""

from .chat_gateway import AgentChatGateway, RichChatContext

__all__ = ["AgentChatGateway", "RichChatContext"]
