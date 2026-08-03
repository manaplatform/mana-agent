from __future__ import annotations

from mana_agent.gateway.config import ChatGatewayConfig


def test_gateway_config_preserves_the_authenticated_memory_user() -> None:
    config = ChatGatewayConfig(
        session_id="conversation-1",
        memory_user_id=" local-user ",
    ).normalized()

    assert config.session_id == "conversation-1"
    assert config.memory_user_id == "local-user"
