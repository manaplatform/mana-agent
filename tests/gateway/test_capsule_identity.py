from __future__ import annotations

from mana_agent.gateway.config import ChatGatewayConfig


def test_gateway_config_preserves_the_authenticated_memory_user() -> None:
    config = ChatGatewayConfig(
        session_id="conversation-1",
        memory_user_id=" local-user ",
    ).normalized()

    assert config.session_id == "conversation-1"
    assert config.memory_user_id == "local-user"


def test_lane_token_budget_zero_means_unlimited() -> None:
    """Product policy: 0 is unlimited; only positive values cap the lane budget."""
    unlimited = ChatGatewayConfig(
        lane_session_token_budget=0,
        lane_global_token_budget=0,
    ).normalized()
    assert unlimited.lane_session_token_budget is None
    assert unlimited.lane_global_token_budget is None

    capped = ChatGatewayConfig(
        lane_session_token_budget=120_000,
        lane_global_token_budget=500_000,
    ).normalized()
    assert capped.lane_session_token_budget == 120_000
    assert capped.lane_global_token_budget == 500_000
