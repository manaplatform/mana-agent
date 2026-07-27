"""Observable verification providers; completion alone is never success."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from .models import VerificationRule


class BuiltinVerifier:
    def verify(self, rule: VerificationRule, context: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if rule.type == "file.exists":
            path = _render(str(rule.arguments.get("path", "")), context.get("inputs", {}))
            target = Path(path).expanduser()
            passed = target.is_file()
            return passed, {"type": rule.type, "path": str(target), "exists": passed}
        if rule.type == "browser.url_matches":
            expected = str(rule.arguments.get("pattern", ""))
            actual = str(context.get("browser_url", ""))
            passed = fnmatch.fnmatch(actual, expected)
            return passed, {"type": rule.type, "expected": expected, "actual": actual}
        if rule.type == "ui.text_visible":
            expected = str(rule.arguments.get("text", ""))
            visible = list(context.get("visible_text", []))
            passed = expected in visible
            return passed, {"type": rule.type, "expected": expected, "visible": passed}
        if rule.type == "command.status":
            expected = int(rule.arguments.get("status", 0))
            actual = context.get("command_status")
            return actual == expected, {"type": rule.type, "expected": expected, "actual": actual}
        return False, {"type": rule.type, "error": "verification provider unavailable"}


def _render(value: str, inputs: dict[str, Any]) -> str:
    for key, item in inputs.items():
        value = value.replace(f"{{{{ {key} }}}}", str(item))
    return value
