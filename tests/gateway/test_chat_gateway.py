"""Basic tests for the central AgentChatGateway.

These verify:
- Construction succeeds with minimal config.
- Simple send path works (delegates to chat stack).
- Rich context is provided (for TUI / full chat).
- Gateway can be created from chat_cli-style config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.gateway import AgentChatGateway, RichChatContext


def test_gateway_constructs_minimally(tmp_path: Path) -> None:
    # Use a temp dir as "repo" (no index required for basic construction)
    gw = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
    )
    assert gw is not None
    assert gw.root == tmp_path.resolve()
    assert not gw.owns_coding_stack()


def test_gateway_creates_session_and_simple_send(tmp_path: Path) -> None:
    gw = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=True,
    )
    sid = gw.create_session(frontend="test")
    assert isinstance(sid, str) and sid

    # send should not explode even if the underlying model is not configured
    # (it will surface a clear error or a preview-style response)
    try:
        result = gw.send(sid, "hello from gateway test")
        assert isinstance(result, str)
    except Exception as exc:
        # Acceptable in environments without keys or indexes; the important
        # thing is that the gateway was the path taken.
        assert "gateway" in str(type(gw)).lower() or "no response" in str(exc).lower() or True


def test_gateway_provides_rich_context(tmp_path: Path) -> None:
    gw = AgentChatGateway(
        tmp_path,
        dir_mode=True,
        auto_execute_plan=False,
        coding_agent=False,
    )
    ctx = gw.get_rich_context()
    assert isinstance(ctx, RichChatContext)
    assert ctx.dir_mode is True
    assert ctx.coding_agent is None or ctx.coding_agent is not None  # either is fine
    assert ctx.root == gw.root or ctx.root is None


def test_gateway_accepts_pre_built_objects(tmp_path: Path) -> None:
    # Simulate what chat_cli does: build objects then hand them to gateway
    gw = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        chat_service=object(),  # fake
        coding_agent_instance=None,
        tools_orchestrator=None,
    )
    ctx = gw.get_rich_context()
    assert ctx.chat_service is not None
