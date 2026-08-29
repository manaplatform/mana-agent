"""Normalize a completed Codex turn into a coding task result.

Success and failure are determined from structured notifications and repository
evidence (changed files, mutation item types, command exit codes). Agent message
text is never the correctness signal and is not used as the user-facing answer
for failed write turns.
"""

from __future__ import annotations

from mana_agent.coding.models import (
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
    compute_duration_breakdown,
)
from mana_agent.integrations.codex.text_cleanup import (
    looks_like_freeform_tool_garbage,
    sanitize_assistant_visible_text,
)


# Codex item types that indicate a repository mutation was attempted.
_MUTATION_ITEM_TYPES = frozenset(
    {
        "fileChange",
        "file_change",
        "applyPatch",
        "apply_patch",
        "patchApplication",
        "patch_application",
    }
)

_ASSISTANT_ITEM_TYPES = frozenset({"agentMessage", "agent_message"})


def parse_codex_result(
    *,
    task: CodingTask,
    workspace: WorkspaceContext,
    worker_id: str,
    thread_id: str,
    turn_id: str,
    notifications: list[dict[str, Any]],
    changed_files: list[str],
    task_created_at: Any = None,
    scheduled_at: Any = None,
    worker_claimed_at: Any = None,
    provider_started_at: Any = None,
    provider_completed_at: Any = None,
    task_completed_at: Any = None,
    duration_breakdown: dict[str, int] | None = None,
) -> CodingTaskResult:
    commands: list[str] = []
    tests: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    # Internal last agent message (may be sanitized). Not used as the coding
    # answer for failed write turns — see terminal_summary.
    agent_message_text = ""
    usage: dict[str, int] | None = None
    status = "completed"
    test_failures: list[str] = []
    mutation_attempted = False

    parsed_http_status: int | None = None
    parsed_original_error: str = ""

    for notification in notifications:
        method = str(notification.get("method") or "")
        params = notification.get("params")
        payload = params if isinstance(params, dict) else {}
        item = payload.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type") or "")
            if item_type in {"systemError", "system_error", "error"}:
                status = "failed"
                item_err = str(item.get("message") or item.get("error") or item.get("text") or "systemError")
                if item_err not in errors:
                    errors.append(item_err)
                if not parsed_original_error:
                    parsed_original_error = item_err
            if item_type in _MUTATION_ITEM_TYPES:
                mutation_attempted = True
            command = str(item.get("command") or "").strip()
            if command and command not in commands:
                commands.append(command)
                if _is_test_command(command):
                    tests.append(command)
                    exit_code = item.get("exitCode")
                    command_status = str(item.get("status") or "").lower()
                    if (isinstance(exit_code, int) and exit_code != 0) or command_status in {
                        "failed",
                        "error",
                    }:
                        test_failures.append(command)
            if item_type in _ASSISTANT_ITEM_TYPES:
                text = str(item.get("text") or item.get("message") or "").strip()
                if text:
                    # Keep for plan-mode terminal notes / internal warnings only.
                    if looks_like_freeform_tool_garbage(text):
                        warnings.append(
                            "assistant_freeform_tool_text_redacted: model emitted "
                            "protocol soup instead of structured tools"
                        )
                        agent_message_text = sanitize_assistant_visible_text(text)
                    else:
                        agent_message_text = sanitize_assistant_visible_text(text)
        if method == "warning":
            message = str(payload.get("message") or "").strip()
            if message:
                warnings.append(message)
                # Explicit capability degradation when Codex lacks model metadata.
                if "fallback metadata" in message.lower() or "model metadata" in message.lower():
                    warnings.append("codex_model_metadata_fallback")
        if method in {"turn/failed", "error", "systemError"}:
            status = "failed"
            err = _format_turn_failure(payload)
            parsed_original_error = str(payload.get("message") or payload.get("error") or err)
            raw_status = payload.get("http_status") or payload.get("status_code")
            if raw_status is None and isinstance(payload.get("error"), dict):
                raw_status = payload["error"].get("http_status") or payload["error"].get("status_code")
            if isinstance(raw_status, int):
                parsed_http_status = raw_status
            elif isinstance(raw_status, str) and raw_status.isdigit():
                parsed_http_status = int(raw_status)
            elif "400" in err:
                parsed_http_status = 400
            elif "401" in err:
                parsed_http_status = 401
            elif "403" in err:
                parsed_http_status = 403
            elif "404" in err:
                parsed_http_status = 404
            elif "410" in err:
                parsed_http_status = 410
            elif "429" in err:
                parsed_http_status = 429

            err_code = str(payload.get("error_code") or "")
            if not err_code:
                lowered = err.lower()
                if parsed_http_status == 400 or "400" in lowered or "invalid_request" in lowered:
                    if (
                        "server-tool" in lowered
                        or "server tool" in lowered
                        or "host tool" in lowered
                        or "tool" in lowered
                        or "function" in lowered
                    ):
                        err_code = "CODING_PROVIDER_TOOL_PROTOCOL_ERROR"
                    else:
                        err_code = "CODING_PROVIDER_BAD_REQUEST"
                elif parsed_http_status == 401 or "401" in lowered or "unauthorized" in lowered:
                    err_code = "CODING_PROVIDER_AUTH_ERROR"
                elif parsed_http_status == 403 or "403" in lowered or "permission" in lowered:
                    err_code = "CODING_PROVIDER_PERMISSION_ERROR"
                elif parsed_http_status == 404 or "404" in lowered or "not found" in lowered:
                    err_code = "CODING_PROVIDER_MODEL_NOT_FOUND"
                elif parsed_http_status == 410 or "410" in lowered or "retired" in lowered:
                    err_code = "CODING_PROVIDER_MODEL_RETIRED"
                elif parsed_http_status == 429 or "429" in lowered or "rate limit" in lowered:
                    err_code = "CODING_PROVIDER_RATE_LIMIT"
                elif payload.get("reason") == "timeout" or "timed out" in lowered or "timeout" in lowered:
                    err_code = "CODING_PROVIDER_TIMEOUT"
                else:
                    err_code = "CODING_AGENT_FAILED"

            if err_code and err_code not in err:
                errors.append(f"{err_code}: {err}")
            else:
                errors.append(err)
        if method == "turn/cancelled":
            status = "cancelled"
            reason = str(payload.get("reason") or payload.get("error_code") or "USER_INTERRUPTED")
            warnings.append(f"interrupted:{reason}")
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

    # Write-required turns that finish "successfully" with no repository diff are
    # not complete. Mutation success is repository state, not agentMessage prose.
    if (
        status == "completed"
        and task.requires_repository_write
        and not [path for path in changed_files if str(path).strip()]
    ):
        status = "failed"
        if mutation_attempted:
            errors.append("mutation_required_but_no_changed_files")
        else:
            errors.append("mutation_required_but_no_mutation_tool_attempted")

    # Summary stored on the result is evidence-oriented. Terminal user answers are
    # built separately by terminal_summary (failed write → no draft).
    if status == "failed" and task.requires_repository_write:
        summary = ""
    elif status == "completed":
        summary = agent_message_text or "Codex task completed."
    elif status == "cancelled":
        summary = "Codex task cancelled."
    else:
        summary = ""

    interruption_reason = ""
    for w in warnings:
        if w.startswith("interrupted:"):
            interruption_reason = w.split(":", 1)[1]
            break
    if not interruption_reason:
        for e in errors:
            for candidate_code in (
                "CODING_PROVIDER_TOOL_PROTOCOL_ERROR",
                "CODING_PROVIDER_BAD_REQUEST",
                "CODING_PROVIDER_AUTH_ERROR",
                "CODING_PROVIDER_PERMISSION_ERROR",
                "CODING_PROVIDER_MODEL_NOT_FOUND",
                "CODING_PROVIDER_MODEL_RETIRED",
                "CODING_PROVIDER_RATE_LIMIT",
                "CODING_PROVIDER_PROTOCOL_ERROR",
                "CODING_CAPABILITY_ERROR",
                "CODING_PROVIDER_TIMEOUT",
                "CODING_TIMEOUT",
                "MODEL_INTERRUPTED",
                "USER_INTERRUPTED",
                "DEADLINE_EXPIRED",
                "PROVIDER_TIMEOUT",
                "LEASE_LOST_DURING_EXECUTION",
                "CODING_AGENT_FAILED",
            ):
                if candidate_code in e:
                    interruption_reason = candidate_code
                    break
            if interruption_reason:
                break

    calculated_breakdown = duration_breakdown or compute_duration_breakdown(
        task_created_at=task_created_at or getattr(task, "task_created_at", None),
        scheduled_at=scheduled_at,
        worker_claimed_at=worker_claimed_at,
        provider_started_at=provider_started_at,
        provider_completed_at=provider_completed_at,
        task_completed_at=task_completed_at,
    )
    codex_meta = {
        "interruption_reason": interruption_reason,
        "mutation_attempted": mutation_attempted,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "http_status": parsed_http_status,
        "original_error": parsed_original_error,
        "error_code": interruption_reason or "",
        "task_created_at": str(task_created_at) if task_created_at else None,
        "scheduled_at": str(scheduled_at) if scheduled_at else None,
        "worker_claimed_at": str(worker_claimed_at) if worker_claimed_at else None,
        "provider_started_at": str(provider_started_at) if provider_started_at else None,
        "provider_completed_at": str(provider_completed_at) if provider_completed_at else None,
        "task_completed_at": str(task_completed_at) if task_completed_at else None,
        "duration_breakdown": calculated_breakdown,
    }

    tests_passed = bool(tests) and not test_failures and status == "completed" and not errors
    return CodingTaskResult(
        task_id=task.task_id,
        worker_id=worker_id,
        backend="codex",
        status=status,  # type: ignore[arg-type]
        summary=summary,
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
        task_created_at=task_created_at or getattr(task, "task_created_at", None),
        scheduled_at=scheduled_at,
        worker_claimed_at=worker_claimed_at,
        provider_started_at=provider_started_at,
        provider_completed_at=provider_completed_at,
        task_completed_at=task_completed_at,
        duration_breakdown=calculated_breakdown,
        codex_metadata=codex_meta,
    )


def _is_test_command(command: str) -> bool:
    executable = command.strip().split(maxsplit=1)[0] if command.strip() else ""
    return executable in {
        "pytest",
        "tox",
        "nox",
        "npm",
        "pnpm",
        "yarn",
        "cargo",
        "go",
        "mvn",
        "gradle",
    }


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
