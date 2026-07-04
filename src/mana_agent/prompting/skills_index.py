from __future__ import annotations

from pathlib import Path

from mana_agent.skills.manager import DEFAULT_SKILL_NAMES, SkillManager


def render_compact_skills_index(request: str, *, repo_root: str | Path | None = None, limit: int = 6) -> str:
    manager = SkillManager(project_root=repo_root)
    names = manager.match_skill_names(request, limit=max(1, limit))
    lines = ["Matched Skills"]
    if not names:
        lines.append("- none matched")
        return "\n".join(lines)
    for name in names:
        item = next((entry for entry in manager.build_index() if entry.name == name), None)
        if item is None:
            lines.append(f"- {name}\n  description: unavailable\n  trigger: unavailable")
            continue
        lines.append(f"- {item.name}\n  description: {item.description}\n  trigger: {item.trigger}")
    return "\n".join(lines)


def render_matched_skill_context(request: str, *, repo_root: str | Path | None = None, limit: int = 3) -> str:
    manager = SkillManager(project_root=repo_root)
    names = manager.match_skill_names(request, limit=max(1, limit))
    if not names:
        return ""
    sections = ["Matched Skill Content", "- Loaded only after task/trigger matching via read_skill()."]
    for name in names:
        content = manager.read_skill(name)
        if content.startswith("Error:"):
            sections.append(f"## {name}\n{content}")
        else:
            sections.append(f"## {name}\n{content.strip()}")
    return "\n\n".join(sections)


def render_stable_skills_index(*, repo_root: str | Path | None = None, limit: int = 24) -> str:
    """Render a compact, request-independent skill index for the stable prompt."""
    manager = SkillManager(project_root=repo_root)
    items = manager.build_index()[: max(1, limit)]
    if not items:
        items = [
            item
            for item in manager.build_index()
            if item.name in DEFAULT_SKILL_NAMES[: max(1, limit)]
        ]

    lines = [
        "Available skills:",
        "- Stable index only. Full SKILL.md bodies belong in ephemeral context after read_skill(skill_name).",
    ]
    for item in items[: max(1, limit)]:
        lines.append(f"- {item.name}\n  description: {item.description}\n  trigger: {item.trigger}")
    return "\n".join(lines)
