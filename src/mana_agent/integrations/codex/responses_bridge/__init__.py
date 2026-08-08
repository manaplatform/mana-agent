"""Provider-neutral OpenAI Responses → Chat Completions compatibility bridge.

Used by the Codex app-server integration when the selected Mana provider
exposes Chat Completions but not the Responses API (for example NVIDIA NIM).
"""

from mana_agent.integrations.codex.responses_bridge.lifecycle import (
    ResponsesBridgeHandle,
    ResponsesBridgeManager,
)
from mana_agent.integrations.codex.responses_bridge.models import (
    BridgeUpstreamConfig,
    ResponsesBridgeError,
)

__all__ = [
    "BridgeUpstreamConfig",
    "ResponsesBridgeError",
    "ResponsesBridgeHandle",
    "ResponsesBridgeManager",
]
