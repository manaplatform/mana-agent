"""Normalize a completed Codex turn into a coding task result."""

from __future__ import annotations

from typing import Any

from mana_agent.coding.models import CodingTask, CodingTaskResult, WorkspaceContext


def parse_codex_result(
    *,
    task: CodingTask,
    workspace: WorkspaceContext,
    worker_id: str,
    thread_id: str,
    turn_id: str,
    notifications: list[dict[str, Any]],
    changed_files: list[str],
) -> CodingTaskResult:
    commands: list[str] = []
    tests: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    summary = ""
    usage: dict[str, int] | None = None
    status = "completed"
    test_failures: list[str] = []

    for notification in notifications:
        method = str(notification.get("method") or "")
        params = notification.get("params")
        payload = params if isinstance(params, dict) else {}
        item = payload.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type") or "")
            command = str(item.get("command") or "").strip()
            if command and command not in commands:
                commands.append(command)
                if _is_test_command(command):
                    tests.append(command)
                    exit_code = item.get("exitCode")
                    command_status = str(item.get("status") or "").lower()
                    if (isinstance(exit_code, int) and exit_code != 0) or command_status in {"failed", "error"}:
                        test_failures.append(command)
            if item_type in {"agentMessage", "agent_message"}:
                text = str(item.get("text") or item.get("message") or "").strip()
                if text:
                    summary = text
        if method == "warning":
            message = str(payload.get("message") or "").strip()
            if message:
                warnings.append(message)
        if method in {"turn/failed", "error"}:
            status = "failed"
            errors.append(_format_turn_failure(payload))
        if method == "turn/cancelled":
            status = "cancelled"
        if method == "turn/completed":
            raw_usage = payload.get("usage")
            turn = payload.get("turn")
            if not isinstance(raw_usage, dict) and isinstance(turn, dict):
                raw_usage = turn.get("usage")
            if isinstance(raw_usage, dict):
                usage = {
                    str(key): int(value)
                    for key, value in raw_usage.items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
            turn_status = str(turn.get("status") or "").lower() if isinstance(turn, dict) else ""
            if turn_status in {"interrupted", "cancelled"}:
                status = "cancelled"
            elif turn_status in {"failed", "error"}:
                status = "failed"
                errors.append("Codex turn completed with a failed status")

    if test_failures:
        warnings.append("Test command failed: " + ", ".join(test_failures))
    tests_passed = bool(tests) and not test_failures and status == "completed" and not errors
    return CodingTaskResult(
        task_id=task.task_id,
        worker_id=worker_id,
        backend="codex",
        status=status,  # type: ignore[arg-type]
        summary=summary or ("Codex task completed." if status == "completed" else "Codex task did not complete."),
        changed_files=changed_files,
        commands_run=commands,
        tests_run=tests,
        tests_passed=tests_passed,
        warnings=warnings,
        errors=errors,
        branch_name=workspace.branch_name,
        token_usage=usage,
        thread_id=thread_id,
        turn_id=turn_id,
    )


def _is_test_command(command: str) -> bool:
    executable = command.strip().split(maxsplit=1)[0] if command.strip() else ""
    return executable in {"pytest", "tox", "nox", "npm", "pnpm", "yarn", "cargo", "go", "mvn", "gradle"}


def _format_turn_failure(payload: dict[str, Any]) -> str:
    """Prefer structured provider diagnostics over vague reconnect messages."""
    message = payload.get("message") or payload.get("error") or "Codex turn failed"
    if isinstance(message, dict):
        additional = message.get("additionalDetails") or message.get("details")
        info = message.get("codexErrorInfo") or message.get("error")
        banner = str(message.get("message") or message.get("error") or "").strip()
        # Prefer the upstream diagnostic when Codex only reported a reconnect banner.
        if additional and (
            "reconnecting" in banner.lower()
            or (isinstance(info, dict) and "responseStreamDisconnected" in info)
        ):
            text = str(additional).strip()
        elif additional:
            text = f"{banner} | {additional}".strip(" |") if banner else str(additional)
        else:
            text = banner or "Codex turn failed"
        if isinstance(info, dict):
            kind = info.get("failure_kind") or info.get("type")
            if kind and f"failure_kind={kind}" not in text:
                text = f"{text} | failure_kind={kind}"
            if info.get("retryable") is False and "retryable=false" not in text:
                text = f"{text} | retryable=false"
        # When Codex reported a stream disconnect that was actually a non-retryable
        # upstream rejection, stamp retryable=false even if the banner was dropped.
        if (
            isinstance(info, dict)
            and "responseStreamDisconnected" in info
            and any(
                marker in text.lower()
                for marker in (
                    "http 400",
                    "http 401",
                    "http 403",
                    "http 404",
                    "http 410",
                    "http 422",
                    "rejected the request",
                )
            )
            and "retryable=false" not in text
        ):
            text = f"{text} | retryable=false | attempts=1"
        return _rewrite_reconnect_banner(text)
    text = str(message).strip() or "Codex turn failed"
    additional = payload.get("additionalDetails") or payload.get("details")
    if additional and str(additional) not in text:
        text = f"{text} | {additional}"
    return _rewrite_reconnect_banner(text)


def _rewrite_reconnect_banner(text: str) -> str:
    """Rewrite Codex reconnect banners when a non-retryable provider diagnostic is present."""
    lowered = text.lower()
    if "reconnecting" not in lowered:
        return text
    non_retryable_markers = (
        "http 400",
        "http 401",
        "http 403",
        "http 404",
        "http 410",
        "http 422",
        "invalid_request",
        "rejected the request",
        "model has been retired",
        "authentication failed",
        "retryable=false",
    )
    if any(marker in lowered for marker in non_retryable_markers):
        cleaned = text
        for token in ("Reconnecting... 1/5", "Reconnecting... ", "Reconnecting"):
            cleaned = cleaned.replace(token, "")
        cleaned = " ".join(cleaned.split()).strip(" |")
        if "not retrying" not in cleaned.lower() and "retryable=false" not in cleaned.lower():
            cleaned = f"{cleaned} | retryable=false | attempts=1"
        return cleaned
    return text


__all__ = ["parse_codex_result"]
