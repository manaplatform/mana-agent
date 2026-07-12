from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError

from mana_agent.connectors.browser.approval import BrowserApprovalBinding, issue_approval
from mana_agent.connectors.browser.contracts import browser_tool_contracts
from mana_agent.connectors.browser.models import BrowserActionDecision, BrowserRisk
from mana_agent.connectors.browser.session import BrowserConnectorError, BrowserSession, BrowserSessionManager


def test_sensitive_decision_requires_confirmation() -> None:
    with pytest.raises(ValidationError):
        BrowserActionDecision(session_id="s1", action="click", risk=BrowserRisk.SENSITIVE, reason="submit", confirmation_required=False)


def test_missing_session_stops_without_default() -> None:
    manager = BrowserSessionManager()
    with pytest.raises(BrowserConnectorError, match="no default session"):
        manager.session("")
    assert manager._sessions == {}


def test_exact_action_approval_rejects_changed_binding_and_expiry() -> None:
    original = BrowserApprovalBinding(session_id="s", page_version=2, origin="https://example.test", action="click", target="e2-1", arguments={"value": None})
    approval = issue_approval(original)
    changed = original.model_copy(update={"target": "e2-2"})
    assert approval.valid_for(original)
    assert not approval.valid_for(changed)
    expired = approval.model_copy(update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)})
    assert not expired.valid_for(original)


def test_pending_confirmation_requires_explicit_interactive_promotion(tmp_path) -> None:  # noqa: ANN001
    manager = BrowserSessionManager()
    session = BrowserSession("model-session", None, None, None, tmp_path)
    binding = BrowserApprovalBinding(session_id="model-session", page_version=2, origin="https://example.test", action="click", target="e2-1", arguments={"value": None})
    challenge = issue_approval(binding)
    session.pending_approvals[challenge.token] = challenge
    manager._sessions[session.session_id] = session
    assert challenge.token not in session.approvals
    manager.approve(None, challenge.token)
    assert session.approvals[challenge.token].valid_for(binding)


def test_contracts_are_strict_and_complete() -> None:
    contracts = {item.name: item for item in browser_tool_contracts()}
    expected = {"browser_open", "browser_inspect", "browser_click", "browser_type", "browser_select", "browser_scroll", "browser_wait", "browser_screenshot", "browser_upload", "browser_download", "browser_check_links", "browser_back", "browser_tabs", "browser_switch_tab", "browser_close"}
    assert expected == set(contracts)
    assert all(item.input_schema["additionalProperties"] is False for item in contracts.values())
    assert all(any("model decision" in rule for rule in item.safety_rules) for item in contracts.values())


def test_playwright_status_is_structured_when_optional_dependency_missing(monkeypatch) -> None:  # noqa: ANN001
    status = BrowserSessionManager.status()
    assert {"ok", "package_installed", "chromium_installed"} <= set(status)
