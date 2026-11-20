"""Report observed verification receipts, independently of task status."""

from __future__ import annotations

import shlex
from typing import Any

from mana_agent.coding.models import ISOLATED_CHECK_TOOL


def isolated_verification_ids(task: Any) -> list[str]:
    """Return the host-owned receipt IDs adopted by a wiring verification result."""
    from mana_agent.human_inbox.models import canonical_digest

    provenance = getattr(task, "verification_provenance", None) or {}
    ids = provenance.get("isolated_operation_ids") or []
    results = getattr(task, "verification_results", ())
    if (getattr(task, "integration_role", "") != "wiring" or not isinstance(ids, list) or not ids or not results or not results[-1].passed
            or results[-1].verified_by_agent_id != "isolated_verifier"
            or not all(isinstance(identity, str) and identity for identity in ids)
            or results[-1].verification_id != provenance.get("verification_id")
            or canonical_digest(ids) != results[-1].verification_id):
        return []
    return list(ids)


def verification_evidence(receipts: list[dict[str, Any]], selected_commands: list[str]) -> tuple[list[str], bool]:
    observed: dict[str, bool] = {}
    selected = {command.strip() for command in selected_commands if command.strip()}
    for receipt in receipts:
        tool = receipt.get("tool")
        result = receipt.get("result")
        if not isinstance(result, dict):
            continue
        if tool == ISOLATED_CHECK_TOOL:
            argv = result.get("argv")
            if isinstance(argv, list) and argv and result.get("expected") != "fail":
                label = shlex.join(argv)
                cwd = result.get("cwd", ".")
                if cwd != ".":
                    label = f"{label} (in {cwd})"
                observed[label] = result.get("passed") is True and result.get("exit_code") == 0
        elif tool == "verify_project":
            for check in result.get("checks") or []:
                if not isinstance(check, dict) or check.get("status") not in {"passed", "failed"}:
                    continue
                command = check.get("command")
                label = shlex.join(command) if isinstance(command, list) else str(command or check.get("name") or "")
                if label:
                    observed[label] = check["status"] == "passed"
        elif tool in {"run_command", "run_script_once", "shell"}:
            args = receipt.get("arguments") or {}
            command = str(args.get("command") or args.get("script") or args.get("cmd") or "").strip()
            if command in selected and isinstance(result.get("returncode"), int):
                observed[command] = result.get("ok") is True and result["returncode"] == 0
    return list(observed), bool(observed) and all(observed.values()) and selected.issubset(observed)
