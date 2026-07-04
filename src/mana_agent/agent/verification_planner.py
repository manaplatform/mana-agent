from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


VerificationProfile = Literal["task_verification", "project_verification"]


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    verification_profile: VerificationProfile
    reason: str
    commands: tuple[str, ...]
    skip_full_pytest_reason: str = ""

    def trace_row(self) -> dict[str, object]:
        return {
            "layer": "verification_planner",
            "decision": self.verification_profile,
            "reason": self.reason,
            "commands": list(self.commands),
            "skip_full_pytest_reason": self.skip_full_pytest_reason,
        }


def plan_verification(*, changed_files: Sequence[str], core_agent_change: bool = False) -> VerificationDecision:
    files = [str(path).replace("\\", "/").lstrip("./") for path in changed_files if str(path).strip()]
    docs_only = bool(files) and all(path.lower().endswith((".md", ".txt", ".rst")) for path in files)
    if docs_only and not core_agent_change:
        target = files[0]
        return VerificationDecision(
            verification_profile="task_verification",
            reason="Only documentation changed; no source-code behavior changed.",
            commands=("git status --short", f"git diff -- {target}"),
            skip_full_pytest_reason="README-only documentation change" if target.lower() == "readme.md" else "docs-only documentation change",
        )
    return VerificationDecision(
        verification_profile="project_verification",
        reason="Core agent behavior or source files changed.",
        commands=("pytest -q",),
    )


__all__ = ["VerificationDecision", "VerificationProfile", "plan_verification"]
