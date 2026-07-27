from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

from fastapi.testclient import TestClient

from mana_agent.api.app import create_app
from mana_agent.github_autopilot.config import GitHubAutopilotSettings
from mana_agent.github_autopilot.models import JobState
from mana_agent.github_autopilot.router import route_event
from mana_agent.github_autopilot.security import sanitize_event_context
from mana_agent.github_autopilot.service import GitHubAutopilotService
from mana_agent.github_autopilot.signatures import verify_signature
from mana_agent.github_autopilot.state import GitHubAutopilotStore


def settings(**updates: object) -> GitHubAutopilotSettings:
    return GitHubAutopilotSettings(enabled=True, app_id="1", webhook_secret="test-secret", public_webhook_url="https://example.test/integrations/github/webhooks", allowed_repositories=frozenset({"acme/widget"}), **updates)


def payload() -> dict[str, object]:
    return {
        "action": "labeled",
        "installation": {"id": 44},
        "repository": {"id": 55, "full_name": "acme/widget", "default_branch": "main"},
        "sender": {"login": "maintainer", "type": "User"},
        "issue": {"number": 12, "state": "open", "title": "Repair widget"},
        "label": {"name": "mana-fix"},
    }


def test_signature_uses_raw_body_and_rejects_invalid_values() -> None:
    body = b'{"hello":"world"}'
    signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert verify_signature(body, signature, "test-secret") is True
    assert verify_signature(body + b"\n", signature, "test-secret") is False
    assert verify_signature(body, "sha256=short", "test-secret") is False
    assert verify_signature(body, None, "test-secret") is False


def test_router_requires_exact_issue_label_and_enabled_repository() -> None:
    decision = route_event("issues", payload(), settings(), "delivery-1")
    assert decision.execute is True
    assert decision.safe_to_continue is True
    changed = payload()
    changed["label"] = {"name": "mana-fix-later"}
    rejected = route_event("issues", changed, settings(), "delivery-2")
    assert rejected.execute is False
    assert rejected.reason == "issue_label_conditions_not_met"


def test_mentions_ignore_quoted_and_partial_matches() -> None:
    base = payload()
    base["action"] = "created"
    base["comment"] = {"id": 1, "body": "> @mana-agent please do this\nordinary text"}
    quoted = route_event("issue_comment", base, settings(), "d1")
    assert quoted.execute is False
    base["comment"] = {"id": 2, "body": "please ask @mana-agent-helper"}
    partial = route_event("issue_comment", base, settings(), "d2")
    assert partial.execute is False
    base["comment"] = {"id": 3, "body": "@mana-agent please investigate this exact failure"}
    explicit = route_event("issue_comment", base, settings(), "d3")
    assert explicit.execute is True


def test_unsupported_event_is_never_executable() -> None:
    decision = route_event("push", {**payload(), "action": "created"}, settings(), "delivery")
    assert decision.supported is False
    assert decision.execute is False
    assert decision.safe_to_continue is False


def test_issue_close_and_authorized_cancel_mention_route_to_cancellation() -> None:
    closed_payload = payload()
    closed_payload["action"] = "closed"
    closed = route_event("issues", closed_payload, settings(), "closed")
    assert closed.execute is True
    assert closed.trigger == "cancellation"

    comment_payload = payload()
    comment_payload["action"] = "created"
    comment_payload["comment"] = {"id": 9, "body": "@mana-agent cancel this task"}
    cancel = route_event("issue_comment", comment_payload, settings(), "cancel")
    assert cancel.execute is True
    assert cancel.trigger == "cancellation"


def test_secret_alert_context_is_redacted_before_storage() -> None:
    sanitized = sanitize_event_context({"secret": "github_pat_abcdefghijklmnopqrstuvwxyz", "location": {"token": "plain"}, "description": "password=hunter2"}, secret_alert=True)
    blob = json.dumps(sanitized)
    assert "hunter2" not in blob
    assert "github_pat" not in blob
    assert "plain" not in blob


def test_delivery_idempotency_and_durable_job(tmp_path: Path) -> None:
    store = GitHubAutopilotStore(tmp_path / "github")
    service = GitHubAutopilotService(settings(), store=store)
    import asyncio

    first = asyncio.run(service.accept("delivery-1", "issues", payload()))
    second = asyncio.run(service.accept("delivery-1", "issues", payload()))
    assert first == second
    assert len(store.list_jobs()) == 1
    assert store.get_job(first.job_id).state == JobState.QUEUED
    assert store.delivery_path("delivery-1").is_file()


def test_review_feedback_restores_original_pr_session_and_queues_after_running_job(tmp_path: Path) -> None:
    store = GitHubAutopilotStore(tmp_path / "github")
    service = GitHubAutopilotService(settings(), store=store)
    import asyncio

    original_receipt = asyncio.run(service.accept("issue-delivery", "issues", payload()))
    original = store.get_job(original_receipt.job_id)
    original.pull_request_number = 77
    original.state = JobState.RUNNING
    original.result = {"thread_id": "thread-original"}
    store.save_job(original)

    review_payload = payload()
    review_payload.update({"action": "submitted", "pull_request": {"number": 77, "base": {"ref": "main"}, "head": {"sha": "abc123"}}, "review": {"id": 900, "state": "changes_requested", "body": "Please cover the edge case."}})
    receipt = asyncio.run(service.accept("review-delivery", "pull_request_review", review_payload))
    followup = store.get_job(receipt.job_id)

    assert followup.job_id != original.job_id
    assert followup.session_id == original.session_id
    assert followup.pull_request_number == 77
    assert followup.result["thread_id"] == "thread-original"
    assert followup.route_decision.trigger == "review_changes"
    assert followup.state == JobState.QUEUED


def test_api_rejects_bad_signature_and_persists_ignored_delivery(tmp_path: Path) -> None:
    store = GitHubAutopilotStore(tmp_path / "github")
    service = GitHubAutopilotService(settings(), store=store)
    app = create_app(github_autopilot=service, telegram_config=type("Cfg", (), {"enabled": False, "effective_transport": "polling"})())
    # The legacy Starlette/httpx test transport normalizes JSON-like byte content
    # with default separators before sending it; sign those exact transmitted bytes.
    raw = json.dumps(payload()).encode()
    headers = {"X-Hub-Signature-256": "sha256=bad", "X-GitHub-Delivery": "api-1", "X-GitHub-Event": "push"}
    with TestClient(app) as client:
        assert client.post("/integrations/github/webhooks", content=raw, headers=headers).status_code == 401
        headers["X-Hub-Signature-256"] = "sha256=" + hmac.new(b"test-secret", raw, hashlib.sha256).hexdigest()
        response = client.post("/integrations/github/webhooks", content=raw, headers=headers)
    assert response.status_code == 202
    assert response.json()["result"] == "ignored"
    assert store.get_delivery("api-1") is not None


class PermissionClient:
    def permission(self, _repository: str, _actor: str, _token: str) -> str:
        return "read"


def test_authorization_below_write_stops_without_codex(tmp_path: Path) -> None:
    called = False

    def codex(*_args: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("Codex must not run")

    service = GitHubAutopilotService(settings(), store=GitHubAutopilotStore(tmp_path / "github"), client=PermissionClient(), codex_factory=codex)  # type: ignore[arg-type]
    import asyncio

    receipt = asyncio.run(service.accept("delivery-auth", "issues", payload()))
    job = service.store.get_job(receipt.job_id)
    try:
        service._authorize(job, "installation-token")
    except Exception as exc:
        assert getattr(exc, "kind", "") == "unauthorized_sender"
    else:
        raise AssertionError("authorization should fail")
    assert called is False
