from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    commands: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def default_verification_plan(*, mode: str, request: str) -> VerificationPlan:
    """Return compact verification guidance for the current mode."""
    text = f"{mode} {request}".lower()
    if mode == "verify":
        return VerificationPlan(notes=("Run the requested checks and summarize exact results.",))
    if any(token in text for token in ("pytest", "python", ".py", "django", "fastapi")):
        return VerificationPlan(commands=("pytest -q", "python -m compileall src"))
    if any(token in text for token in ("package.json", "npm", "node", "typescript", "react", "next")):
        return VerificationPlan(commands=("npm test",), notes=("Use lint/typecheck scripts when tests are absent.",))
    return VerificationPlan(notes=("Run the most relevant focused test, smoke check, or syntax check after edits.",))


def render_verification_rules(plan: VerificationPlan) -> str:
    lines = [
        "Verification Rules",
        "- Verify after repository changes.",
        "- Prefer focused checks that cover the changed behavior.",
        "- Report exact commands and outcomes; do not claim checks that were not run.",
    ]
    if plan.commands:
        lines.append(f"- candidate_commands: {', '.join(plan.commands)}")
    if plan.notes:
        lines.append(f"- notes: {' '.join(plan.notes)}")
    return "\n".join(lines)

