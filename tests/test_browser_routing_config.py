from __future__ import annotations

import json
from pathlib import Path

import pytest

from mana_agent.config.settings import Settings
from mana_agent.config import user_config
from mana_agent.multi_agent.routing.agent_decision import AGENT_DECISION_PROMPT, AGENT_DECISION_REVIEW_PROMPT, AgentDecision, AgentDecisionEngine, agent_tool_descriptions, verify_agent_decision
from mana_agent.multi_agent.runtime.prompts import BROWSER_AGENT_SYSTEM_PROMPT
from mana_agent.multi_agent.runtime.route_executor import available_tool_names


@pytest.fixture()
def isolated_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / ".mana"
    monkeypatch.setattr(user_config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(user_config, "CONFIG_FILE", config_dir / "config.toml")
    monkeypatch.setattr(user_config, "SECRETS_FILE", config_dir / "secrets.toml")
    monkeypatch.setattr(user_config, "MODEL_CACHE_FILE", config_dir / "model_cache.json")
    return config_dir


class _BrowserDecisionModel:
    def invoke(self, messages):  # noqa: ANN001
        payload = json.loads(messages[-1].content)
        tools = payload.get("tools") or payload.get("available_tools") or []
        assert any(tool["name"] == "browser_open" for tool in tools)
        return type(
            "Message",
            (),
            {
                "content": json.dumps(
                    {
                        "intent": "tool",
                        "confidence": 0.96,
                        "selected_tools": ["browser_open", "browser_inspect", "browser_click"],
                        "tool_inputs": {"browser_open": {"url": "https://example.test"}},
                        "repo_context_needed": False,
                        "web_search_needed": False,
                        "code_editing_needed": False,
                        "reasoning_summary": "The task requires an interactive website session.",
                    }
                )
            },
        )()


class _CorrectingBrowserDecisionModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):  # noqa: ANN001
        self.calls += 1
        decision = (
            {
                "intent": "web_research", "confidence": 0.8,
                "selected_tools": ["web_search"], "tool_inputs": {"web_search": {"query": "site"}},
                "repo_context_needed": False, "web_search_needed": True,
                "code_editing_needed": False, "reasoning_summary": "initial proposal",
            }
            if self.calls == 1
            else {
                "intent": "tool", "confidence": 0.99,
                "selected_tools": ["browser_open", "browser_inspect"],
                "tool_inputs": {"browser_open": {"url": "https://example.test"}},
                "repo_context_needed": False, "web_search_needed": False,
                "code_editing_needed": False, "reasoning_summary": "review corrected rendered-page inspection",
            }
        )
        return type("Message", (), {"content": json.dumps(decision)})()


def test_browser_tools_are_exposed_to_model_router() -> None:
    names = {item["name"] for item in agent_tool_descriptions()}
    assert {"browser_open", "browser_inspect", "browser_click"} <= names
    assert {"browser_open", "browser_inspect", "browser_check_links"} <= set(available_tool_names())


def test_model_can_select_multi_step_browser_tools() -> None:
    decision = AgentDecisionEngine(llm=_BrowserDecisionModel()).decide(
        user_request="Open the site, inspect it, and click the sign-up button"
    )
    assert decision.intent == "tool"
    assert decision.selected_tools == ["browser_open", "browser_inspect", "browser_click"]
    assert decision.verifier_passed is True


def test_model_review_corrects_web_search_proposal_for_rendered_page() -> None:
    model = _CorrectingBrowserDecisionModel()
    decision = AgentDecisionEngine(llm=model).decide(
        user_request="Check https://example.test and report visible controls and broken links"
    )
    assert model.calls == 2
    assert decision.intent == "tool"
    assert decision.selected_tools == ["browser_open", "browser_inspect"]
    assert decision.web_search_needed is False
    assert decision.verifier_passed is True


def test_browser_settings_load_from_user_config(isolated_user_config) -> None:  # noqa: ANN001
    user_config.save_user_config(
        {
            "MANA_BROWSER_ENABLED": False,
            "MANA_BROWSER_HEADLESS": False,
            "MANA_BROWSER_TIMEOUT_SECONDS": 45,
            "MANA_BROWSER_PERSIST_AUTH": True,
            "MANA_BROWSER_DOWNLOAD_MAX_MB": 25,
        },
        merge=True,
    )
    settings = Settings()
    assert settings.mana_browser_enabled is False
    assert settings.mana_browser_headless is False
    assert settings.mana_browser_timeout_seconds == 45
    assert settings.mana_browser_persist_auth is True
    assert settings.mana_browser_download_max_mb == 25


def test_disabled_browser_is_not_advertised_to_model(isolated_user_config) -> None:  # noqa: ANN001
    user_config.save_user_config({"MANA_BROWSER_ENABLED": False}, merge=True)
    names = {item["name"] for item in agent_tool_descriptions()}
    assert not any(name.startswith("browser_") for name in names)


def test_browser_prompts_explain_account_workflow_and_tool_sequence() -> None:
    assert 'intent="tool"' in AGENT_DECISION_PROMPT
    assert "account creation" in AGENT_DECISION_PROMPT
    assert "web_search cannot inspect a rendered target page" in AGENT_DECISION_PROMPT
    assert "broken links" in AGENT_DECISION_PROMPT
    assert "link checking require browser_* tools" in AGENT_DECISION_REVIEW_PROMPT
    assert "browser_open" in BROWSER_AGENT_SYSTEM_PROMPT
    assert "browser_inspect" in BROWSER_AGENT_SYSTEM_PROMPT
    assert "/approve-browser <token>" in BROWSER_AGENT_SYSTEM_PROMPT
    assert "browser_check_links" in BROWSER_AGENT_SYSTEM_PROMPT


def test_inconsistent_browser_decision_fails_validation_without_repository_execution() -> None:
    decision = AgentDecision(
        intent="edit",
        confidence=0.9,
        selected_tools=["browser_open", "browser_type"],
        repo_context_needed=True,
        code_editing_needed=True,
    )
    result = verify_agent_decision(decision, user_request="Create an account on https://example.test")
    assert result.passed is False
    assert "browser tools require intent=tool" in result.summary
    assert "must not request repository context" in result.summary
