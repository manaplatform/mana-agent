"""Report observed verification receipts, independently of task status."""

from __future__ import annotations

import shlex
from typing import Any


def verification_evidence(receipts: list[dict[str, Any]], selected_commands: list[str]) -> tuple[list[str], bool]:
    observed: dict[str, bool] = {}
    selected = {command.strip() for command in selected_commands if command.strip()}
    for receipt in receipts:
        tool = receipt.get("tool")
        result = receipt.get("result")
        if not isinstance(result, dict):
            continue
        if tool == "verify_project":
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
