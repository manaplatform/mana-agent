from __future__ import annotations


_MODE_RULES: dict[str, str] = {
    "answer_only": "Chat mode: answer directly when repository evidence is already sufficient; inspect the repo only when needed.",
    "plan_only": "Plan mode: produce an actionable implementation plan; ask only when a missing detail blocks a safe plan.",
    "edit": "Edit mode: inspect relevant files, patch code, verify changed-file evidence, and summarize the result.",
    "review": "Review mode: prioritize bugs, regressions, risks, and missing tests before summaries.",
    "verify": "Verify mode: run or identify the relevant checks and report exact outcomes.",
    "analyze": "Analyze mode: produce evidence-backed project analysis without replacing existing artifacts unless requested.",
}


def render_mode_rules(mode: str) -> str:
    normalized = str(mode or "answer_only").strip().lower()
    rule = _MODE_RULES.get(normalized, _MODE_RULES["answer_only"])
    return f"Mode Rules\n- selected_mode: {normalized}\n- {rule}"

