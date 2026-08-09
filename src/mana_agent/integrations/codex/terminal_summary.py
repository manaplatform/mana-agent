"""Build user-facing coding answers from structured execution evidence.

The final coding answer must not be assembled from arbitrary agentMessage text.
It is derived from validated terminal state: status, mutation evidence, changed
files, commands, tests, and typed error codes.
"""

from __future__ import annotations

from typing import Any

from mana_agent.coding.models import CodingTaskResult

# Deterministic failure copy keyed by structured terminal reason.
_MUTATION_FAILURE_MESSAGES: dict[str, str] = {
    "mutation_required_but_no_mutation_tool_attempted": (
        "Codex did not complete the requested repository mutation. "
        "No mutation tool was executed and no files were changed."
    ),
    "mutation_required_but_no_changed_files": (
        "Codex reported a mutation attempt but no repository files were changed."
    ),
}


def terminal_reason_from_result(result: CodingTaskResult) -> str:
    """Map a CodingTaskResult to a stable terminal reason code."""
    if result.status == "completed":
        return "completed"
    if result.status == "cancelled":
        return "codex_cancelled"
    for err in result.errors:
        text = str(err or "").strip()
        for code in _MUTATION_FAILURE_MESSAGES:
            if text == code or text.startswith(f"{code}:"):
                return code
    return "codex_failed"


def build_coding_terminal_answer(
    result: CodingTaskResult,
    *,
    requires_repository_write: bool,
    terminal_reason: str | None = None,
) -> str:
    """Return exactly one user-facing terminal summary for a coding turn."""
    reason = str(terminal_reason or terminal_reason_from_result(result)).strip()

    if result.status == "cancelled":
        return "Codex coding turn was cancelled."

    if result.status == "failed":
        if reason in _MUTATION_FAILURE_MESSAGES:
            return _MUTATION_FAILURE_MESSAGES[reason]
        # Prefer structured error codes / short diagnostics — never a model draft.
        for err in result.errors:
            text = str(err or "").strip()
            if not text:
                continue
            if text in _MUTATION_FAILURE_MESSAGES:
                return _MUTATION_FAILURE_MESSAGES[text]
            # Provider / protocol diagnostics are already structured.
            if _looks_like_structured_error(text):
                return f"Codex coding turn failed: {text}"
        return "Codex coding turn failed."

    # Completed
    if requires_repository_write:
        return _successful_write_summary(result)
    return _successful_plan_summary(result)


def _successful_write_summary(result: CodingTaskResult) -> str:
    lines: list[str] = []
    changed = [str(path).strip() for path in result.changed_files if str(path).strip()]
    if changed:
        lines.append("Codex completed the repository mutation.")
        lines.append("Changed files:")
        for path in changed[:40]:
            lines.append(f"- {path}")
        if len(changed) > 40:
            lines.append(f"- … and {len(changed) - 40} more")
    else:
        lines.append("Codex completed the coding turn.")

    tests = [str(t).strip() for t in result.tests_run if str(t).strip()]
    if tests:
        status = "passed" if result.tests_passed else "reported failures"
        lines.append(f"Verification ({status}): " + ", ".join(tests[:8]))

    commands = [str(c).strip() for c in result.commands_run if str(c).strip()]
    if commands and not tests:
        lines.append("Commands run: " + ", ".join(commands[:8]))

    warnings = [
        str(w).strip()
        for w in result.warnings
        if str(w).strip() and not str(w).startswith("assistant_")
    ]
    if warnings:
        lines.append("Warnings: " + "; ".join(warnings[:5]))

    # Optional concise model summary only when it is not protocol garbage and
    # does not dominate the structured evidence. Callers already strip free-form
    # tool soup before storing summary on completed write turns with mutations.
    model_summary = str(result.summary or "").strip()
    if model_summary and _is_safe_optional_summary(model_summary, requires_write=True):
        lines.append(model_summary[:500])

    return "\n".join(lines).strip()


def _successful_plan_summary(result: CodingTaskResult) -> str:
    model_summary = str(result.summary or "").strip()
    if model_summary and _is_safe_optional_summary(model_summary, requires_write=False):
        return model_summary
    return "Codex completed the planning turn."


def _looks_like_structured_error(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "http ",
        "failure_kind=",
        "retryable=",
        "upstream_",
        "codex ",
        "rejected",
        "invalid_request",
        "bridge",
        "timeout",
        "unavailable",
        "configuration",
        "compatibility",
    )
    if any(marker in lowered for marker in markers):
        return True
    # Short code-like strings without multi-paragraph chatter.
    if len(text) <= 400 and "\n\n" not in text:
        return True
    return False


def _is_safe_optional_summary(text: str, *, requires_write: bool) -> bool:
    """Allow a short terminal note when it is clearly not a mid-turn draft dump."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    # Long multi-paragraph chatter is never the terminal answer body.
    if len(cleaned) > 1200:
        return False
    if cleaned.count("\n") > 24:
        return False
    lowered = cleaned.lower()
    # Structured redaction / free-form tool markers mean the model draft is junk.
    if "redacted" in lowered and "structured tools" in lowered:
        return False
    if "prior model output was invalid" in lowered:
        return False
    # For write success, prefer evidence lines; only keep short closing notes.
    if requires_write and len(cleaned) > 600:
        return False
    return True


def attach_terminal_fields(
    payload: dict[str, Any],
    result: CodingTaskResult,
    *,
    requires_repository_write: bool,
) -> dict[str, Any]:
    """Mutate a result payload dict with terminal reason + evidence-based answer."""
    reason = terminal_reason_from_result(result)
    payload["auto_execute_terminal_reason"] = reason
    payload["answer"] = build_coding_terminal_answer(
        result,
        requires_repository_write=requires_repository_write,
        terminal_reason=reason,
    )
    return payload


__all__ = [
    "attach_terminal_fields",
    "build_coding_terminal_answer",
    "terminal_reason_from_result",
]
