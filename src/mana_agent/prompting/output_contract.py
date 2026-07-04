from __future__ import annotations


_CONTRACTS: dict[str, tuple[str, ...]] = {
    "plan_only": (
        "Return a practical plan with goal, requirements, implementation steps, verification, and done criteria.",
        "Do not claim repository edits were made.",
    ),
    "analyze": (
        "Return a concise analysis with summary, evidence-backed findings, risks, and recommended next steps.",
        "Do not mutate repository files in analysis mode.",
    ),
    "edit": (
        "Return what changed, files changed, verification result, and remaining risks or blockers.",
        "Do not say fixed, done, or verified unless the patch and command evidence support it.",
    ),
    "verify": (
        "Return the command run, pass/fail/could-not-run result, important output, and conclusion.",
        "Do not claim success without command evidence.",
    ),
    "review": (
        "Lead with findings ordered by severity and grounded in file and line references.",
        "Keep summaries secondary to bugs, regressions, risks, and missing tests.",
    ),
}


def render_output_contract(mode: str) -> str:
    normalized = str(mode or "answer_only").strip().lower()
    rules = _CONTRACTS.get(
        normalized,
        (
            "Answer directly and usefully.",
            "Never claim changes, verification, or fixes that did not happen.",
        ),
    )
    lines = [
        "Output Contract",
        "- Final responses must be useful and honest.",
    ]
    lines.extend(f"- {rule}" for rule in rules)
    return "\n".join(lines)
