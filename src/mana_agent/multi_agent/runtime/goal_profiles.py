from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Pattern


PathPriorityFn = Callable[[str, Path], int]
GoalMatcher = Callable[[str], bool]


@dataclass(frozen=True)
class GoalProfile:
    """Goal-specific candidate discovery policy."""

    id: str
    description: str
    goal_matcher: GoalMatcher
    discovery_globs: tuple[str, ...]
    include_patterns: tuple[Pattern[str], ...] = field(default_factory=tuple)
    exclude_patterns: tuple[Pattern[str], ...] = field(default_factory=tuple)
    content_matchers: tuple[Pattern[str], ...] = field(default_factory=tuple)
    priority_fn: PathPriorityFn | None = None

    def matches_goal(self, goal: str) -> bool:
        return bool(self.goal_matcher(str(goal or "")))

    def priority(self, path: str, repo_root: Path) -> int:
        if self.priority_fn is None:
            return 50
        return int(self.priority_fn(str(path or ""), Path(repo_root)))

    def is_excluded(self, path: str) -> bool:
        text = _clean_path(path)
        return any(pattern.search(text) for pattern in self.exclude_patterns)

    def is_included_by_name(self, path: str) -> bool:
        text = _clean_path(path)
        return any(pattern.search(text) for pattern in self.include_patterns)

    def is_relevant(self, path: str, repo_root: Path) -> bool:
        text = _clean_path(path)
        if not text or self.is_excluded(text):
            return False
        if self.is_included_by_name(text):
            return True
        if not self.content_matchers:
            return True
        content = _read_text(Path(repo_root) / text)
        return any(pattern.search(content) for pattern in self.content_matchers)


def _clean_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("./")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _model_docs_goal_matcher(goal: str) -> bool:
    lowered = str(goal or "").lower()
    compact = re.sub(r"[\s_/-]+", "", lowered)
    if "docsmodels.md" in compact or "docsmodelsmd" in compact:
        return True
    if "models.md" in lowered and "doc" in lowered:
        return True
    if "docs/models.md" in lowered and "model" in lowered:
        return True
    return bool(
        "model" in lowered
        and any(token in lowered for token in ("doc", "document", "documentation", "update"))
    )


def _model_docs_priority(path: str, repo_root: Path) -> int:
    _ = repo_root
    text = _clean_path(path)
    parts = set(Path(text).parts)
    name = Path(text).name.lower()
    if name == "__init__.py" or name == "apply_patch.py":
        return 99
    if text == "project_structure_analysis.json" or text.endswith("/project_structure_analysis.json"):
        return 99
    if re.fullmatch(r"src/(?:.*/)?models\.py", text) or re.fullmatch(r"src/(?:.*/)?[\w.-]*_models\.py", text):
        return 1
    if text == "docs/models.md" or name in {"readme.md", "readme.rst"}:
        return 4
    if parts & {"tests", "test"} and text.endswith(".py"):
        return 99
    if parts & {"admin", "serializers", "frontend", "front", "cli", "commands"}:
        return 99
    if "/migrations/" in f"/{text}/" and text.endswith(".py"):
        return 99
    return 9


class ModelDocsGoalProfile(GoalProfile):
    def __init__(self) -> None:
        super().__init__(
            id="model_docs",
            description="Discover model/schema sources for docs/models.md updates.",
            goal_matcher=_model_docs_goal_matcher,
            discovery_globs=(
                "src/**/models.py",
                "src/**/*_models.py",
                "src/**/*.py",
                "docs/models.md",
            ),
            include_patterns=(
                re.compile(r"(^|/)docs/models\.md$"),
                re.compile(r"^src/(?:.*/)?models\.py$"),
                re.compile(r"^src/(?:.*/)?[\w.-]*_models\.py$"),
            ),
            exclude_patterns=(
                re.compile(r"(^|/)(__init__\.py|apply_patch\.py)$"),
                re.compile(r"(^|/)project_structure_analysis\.json$"),
                re.compile(r"(^|/)package-lock\.json$"),
                re.compile(r"(^|)(\.mana|node_modules|venv|\.venv|env|site-packages|dist-packages)(/|$)"),
                re.compile(r"(^|)(build|dist|tests|test)(/|$)"),
                re.compile(r"(^|/)build/lib/"),
                re.compile(r"(^|/)src/(?:.*/)?(commands|cli|tools|utils?|helpers?|frontend|front|admin|serializers)(/|$)"),
            ),
            content_matchers=(
                re.compile(r"\bclass\s+\w+\s*\([^)]*(?:BaseModel|TypedDict|Enum|models\.Model)[^)]*\)"),
                re.compile(r"@dataclass\b"),
                re.compile(r"\bclass\s+\w+\s*\([^)]*dataclass[^)]*\)"),
            ),
            priority_fn=_model_docs_priority,
        )

    def is_relevant(self, path: str, repo_root: Path) -> bool:
        text = _clean_path(path)
        if not text or self.is_excluded(text) or self.priority(text, repo_root) >= 99:
            return False
        if text == "docs/models.md":
            return True
        if not text.startswith("src/") or not text.endswith(".py"):
            return False
        content = _read_text(Path(repo_root) / text)
        if text.endswith("/coding_agent_models.py"):
            return bool(
                re.search(r"\bclass\s+\w+\s*\([^)]*(?:BaseModel|TypedDict|Enum|models\.Model)[^)]*\)", content)
                or re.search(r"@dataclass\b", content)
            )
        if text.endswith("/models.py") or text.endswith("_models.py"):
            return True
        if text.endswith(".py"):
            if re.search(r"\bclass\s+\w+\s*\([^)]*(?:BaseModel|TypedDict|Enum|models\.Model)[^)]*\)", content):
                return True
            if re.search(r"@dataclass\b", content):
                return True
        return False


BUILTIN_GOAL_PROFILES: tuple[GoalProfile, ...] = (ModelDocsGoalProfile(),)


def active_goal_profile(goal: str, profiles: tuple[GoalProfile, ...] = BUILTIN_GOAL_PROFILES) -> GoalProfile | None:
    for profile in profiles:
        if profile.matches_goal(goal):
            return profile
    return None
