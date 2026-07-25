from __future__ import annotations

import re

from mana_agent.multi_agent.core.errors import ToolPermissionError
from mana_agent.multi_agent.core.types import AgentRole, QueueJobType

READ_ONLY_ROLES = {AgentRole.RESEARCH, AgentRole.PLANNER, AgentRole.CODING, AgentRole.VERIFIER}
WRITE_JOB_TYPES = {QueueJobType.APPLY_PATCH}
DANGEROUS_SHELL_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\brm\s+-rf\s+\*",
    r"\bsudo\s+rm\b",
    r"\bcat\s+\.env\b",
    r"(^|\s)printenv(\s|$)",
    r"(^|\s)env(\s|$)",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-fd\b",
    r"\bgit\s+push\b.*\s--force(?:-with-lease)?\b",
    r"\bgit\s+rebase\s+--(?:abort|skip)\b",
    r"\bcurl\b.*(secret|token|credential)",
]
ALLOWED_SHELL_PREFIXES = (
    "python -m compileall",
    "pytest",
    "ruff check",
    "mypy",
    "git status --short",
    "git status",
    "git branch",
    "git remote",
    "git fetch",
    "git switch",
    "git checkout",
    "git add",
    "git diff --stat",
    "git diff",
    "git commit",
    "git push",
    "git pull",
    "git log",
    "git rev-parse",
    "git merge",
    "git rebase",
    "git reset",
    "git restore",
    "git tag",
)


def assert_shell_allowed(command: str) -> None:
    text = str(command or "").strip()
    for pattern in DANGEROUS_SHELL_PATTERNS:
        if re.search(pattern, text, re.I):
            raise ToolPermissionError(f"dangerous shell command blocked: {text}")
    if text.startswith(ALLOWED_SHELL_PREFIXES):
        return
    raise ToolPermissionError(f"shell command requires policy approval: {text}")
