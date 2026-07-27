"""ACP stdio server lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.protocols.common.exceptions import OptionalProtocolDependencyError


def acp_sdk_info() -> dict[str, str | bool]:
    try:
        import acp
        from importlib.metadata import version
    except ImportError:
        return {"installed": False, "protocol_version": "1", "sdk_version": ""}
    return {
        "installed": True,
        "protocol_version": str(acp.PROTOCOL_VERSION),
        "sdk_version": version("agent-client-protocol"),
    }


async def _serve(gateway: AgentChatGateway) -> None:
    try:
        from acp import run_agent
    except ImportError as exc:
        raise OptionalProtocolDependencyError.for_protocol("acp") from exc
    from .agent import ManaAcpAgent

    settings = gateway.settings
    allowed = tuple(
        Path(item.strip()).expanduser().resolve()
        for item in str(getattr(settings, "mana_acp_allowed_roots", "") or "").split(",")
        if item.strip()
    )
    agent = ManaAcpAgent(
        gateway,
        allowed_roots=allowed,
        mcp_forwarding=bool(getattr(settings, "mana_acp_mcp_forwarding", True)),
        session_load=bool(getattr(settings, "mana_acp_session_load", True)),
    )
    try:
        # ACP SDK 0.11 gates session/close behind its unstable-router switch
        # even though the negotiated wire protocol remains v1.
        await run_agent(agent, use_unstable_protocol=True)
    finally:
        await agent.shutdown()


def run_acp_stdio(root: str | Path) -> None:
    """Run ACP over stdio; stdout is owned exclusively by the SDK transport."""
    gateway = AgentChatGateway(Path(root).expanduser().resolve())
    if not bool(getattr(gateway.settings, "mana_acp_enabled", True)):
        raise ValueError("ACP support is disabled in Mana-Agent configuration.")
    asyncio.run(_serve(gateway))
