from __future__ import annotations

import hashlib
import re
from typing import Any

from .config import GitHubAutopilotSettings
from .models import RouteDecision


def _decision(delivery_id: str, *, supported: bool, execute: bool, safe: bool, reason: str, trigger: str = "", subject_type: str = "", subject_number: int | None = None, human: bool = True) -> RouteDecision:
    fingerprint = hashlib.sha256(f"{delivery_id}:{trigger}:{subject_type}:{subject_number}:{reason}".encode()).hexdigest()[:20]
    return RouteDecision(decision_id=f"ghroute_{fingerprint}", supported=supported, execute=execute, safe_to_continue=safe, trigger=trigger, subject_type=subject_type, subject_number=subject_number, requires_human_authorization=human, reason=reason)


def _repository_allowed(payload: dict[str, Any], settings: GitHubAutopilotSettings) -> bool:
    full_name = str(payload.get("repository", {}).get("full_name") or "").lower()
    owner = full_name.partition("/")[0]
    if settings.allowed_repositories and full_name not in settings.allowed_repositories:
        return False
    if settings.allowed_organizations and owner not in settings.allowed_organizations:
        return False
    return True


def _explicit_invocation(body: str, invocation: str) -> bool:
    clean_lines = [line for line in body.splitlines() if not line.lstrip().startswith(">")]
    clean = "\n".join(clean_lines)
    name = invocation.strip()
    if not name:
        return False
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", clean, re.IGNORECASE) is not None


def _cancel_invocation(body: str, invocation: str) -> bool:
    clean = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith(">"))
    return _explicit_invocation(clean, invocation) and re.search(r"(?<![\w-])cancel(?![\w-])", clean, re.IGNORECASE) is not None


def route_event(event_name: str, payload: dict[str, Any], settings: GitHubAutopilotSettings, delivery_id: str) -> RouteDecision:
    """Make the validated, deterministic pre-model execution decision."""
    action = str(payload.get("action") or "")
    if not settings.enabled:
        return _decision(delivery_id, supported=False, execute=False, safe=False, reason="autopilot_disabled")
    if not _repository_allowed(payload, settings):
        return _decision(delivery_id, supported=True, execute=False, safe=False, reason="repository_not_enabled")

    issue = payload.get("issue") or {}
    pr = payload.get("pull_request") or {}
    alert = payload.get("alert") or {}
    sender = payload.get("sender") or {}
    sender_type = str(sender.get("type") or "")
    sender_login = str(sender.get("login") or "")
    if sender_type.lower() == "bot" and not settings.allow_bots:
        return _decision(delivery_id, supported=True, execute=False, safe=False, reason="bot_sender_not_allowed")
    if sender_login.lower().endswith("[bot]") and not settings.allow_bots:
        return _decision(delivery_id, supported=True, execute=False, safe=False, reason="bot_sender_not_allowed")

    if event_name == "issues" and action == "labeled":
        number = issue.get("number")
        valid = issue.get("state") == "open" and not issue.get("pull_request") and str((payload.get("label") or {}).get("name") or "") == settings.fix_label
        return _decision(delivery_id, supported=True, execute=valid, safe=valid, reason="issue_label_actionable" if valid else "issue_label_conditions_not_met", trigger="issue_label", subject_type="issue", subject_number=number)

    if event_name == "issues" and action in {"closed", "unlabeled"}:
        valid = action == "closed" or str((payload.get("label") or {}).get("name") or "") == settings.fix_label
        return _decision(delivery_id, supported=True, execute=valid, safe=valid, reason="task_cancellation" if valid else "unrelated_label_removed", trigger="cancellation", subject_type="issue", subject_number=issue.get("number"))

    if event_name == "issue_comment" and action == "created":
        body = str((payload.get("comment") or {}).get("body") or "")
        valid = _explicit_invocation(body, settings.invocation_name)
        kind = "pull_request" if issue.get("pull_request") else "issue"
        if valid and _cancel_invocation(body, settings.invocation_name):
            return _decision(delivery_id, supported=True, execute=True, safe=True, reason="authorized_cancel_requested", trigger="cancellation", subject_type=kind, subject_number=issue.get("number"))
        return _decision(delivery_id, supported=True, execute=valid, safe=valid, reason="explicit_mention" if valid else "invocation_not_found", trigger="mention", subject_type=kind, subject_number=issue.get("number"))

    if event_name == "workflow_run" and action == "completed":
        run = payload.get("workflow_run") or {}
        branch = str(run.get("head_branch") or "").lower()
        workflow = str(run.get("name") or "").lower()
        allowed = run.get("conclusion") in {"failure", "timed_out", "startup_failure"}
        allowed = allowed and (not settings.allowed_branches or branch in settings.allowed_branches)
        allowed = allowed and (not settings.allowed_workflows or workflow in settings.allowed_workflows)
        allowed = allowed and not branch.startswith("mana/")
        return _decision(delivery_id, supported=True, execute=allowed, safe=allowed, reason="failed_workflow_actionable" if allowed else "workflow_conditions_not_met", trigger="workflow_failure", subject_type="workflow_run", subject_number=run.get("id"), human=False)

    if event_name == "pull_request_review" and action == "submitted":
        review = payload.get("review") or {}
        valid = str(review.get("state") or "").lower() == "changes_requested"
        return _decision(delivery_id, supported=True, execute=valid, safe=valid, reason="changes_requested" if valid else "review_not_actionable", trigger="review_changes", subject_type="pull_request", subject_number=pr.get("number"))

    if event_name == "pull_request_review_comment" and action == "created":
        body = str((payload.get("comment") or {}).get("body") or "")
        valid = _explicit_invocation(body, settings.invocation_name)
        return _decision(delivery_id, supported=True, execute=valid, safe=valid, reason="explicit_review_mention" if valid else "invocation_not_found", trigger="review_comment", subject_type="pull_request", subject_number=pr.get("number"))

    if event_name == "dependabot_alert" and action in {"created", "reopened", "reintroduced"}:
        valid = settings.security_events_enabled and str(alert.get("state") or "open") == "open"
        return _decision(delivery_id, supported=True, execute=valid, safe=valid, reason="dependabot_alert_actionable" if valid else "security_event_not_enabled", trigger="dependabot", subject_type="dependabot_alert", subject_number=alert.get("number"), human=False)

    if event_name in {"code_scanning_alert", "secret_scanning_alert"} and action in {"created", "reopened", "appeared_in_branch"}:
        valid = settings.security_events_enabled and alert.get("number") is not None
        return _decision(delivery_id, supported=True, execute=valid, safe=valid, reason="security_alert_actionable" if valid else "security_event_not_enabled", trigger=event_name, subject_type=event_name, subject_number=alert.get("number"), human=False)

    if event_name in {"dependabot_alert", "code_scanning_alert", "secret_scanning_alert"} and action in {"dismissed", "fixed", "resolved"}:
        valid = settings.security_events_enabled and alert.get("number") is not None
        return _decision(delivery_id, supported=True, execute=valid, safe=valid, reason="task_cancellation" if valid else "security_event_not_enabled", trigger="cancellation", subject_type=event_name, subject_number=alert.get("number"), human=False)

    return _decision(delivery_id, supported=False, execute=False, safe=False, reason="unsupported_event")
