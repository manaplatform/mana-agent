from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable


DEFAULT_SKILL_NAMES = (
    "cli",
    "planning",
    "coding-agent",
    "django",
    "vue",
    "fastapi",
    "nestjs",
    "nextjs",
    "reactjs",
    "laravel",
    "telegram-bot",
    "celery",
    "testing",
    "git",
    "security",
)

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cli": ("cli", "command", "typer", "rich", "prompt", "banner", "terminal", "flag"),
    "planning": ("plan", "planning", "implementation plan", "question", "approval"),
    "coding-agent": ("agent", "tool", "patch", "apply_patch", "code edit", "write_file"),
    "django": ("django", "model", "migration", "admin", "serializer", "view"),
    "vue": ("vue", "frontend", "component", "table", "filter"),
    "fastapi": ("fastapi", "uvicorn", "pydantic", "starlette", "endpoint", "router"),
    "nestjs": ("nestjs", "@nestjs", "controller", "provider", "module", "dependency injection"),
    "nextjs": ("next", "nextjs", "next.js", "ssr", "app router", "routing", "server components"),
    "reactjs": ("react", "reactjs", "jsx", "tsx", "component", "hooks"),
    "laravel": ("laravel", "php", "artisan", "eloquent", "blade", "middleware", "controller"),
    "telegram-bot": ("telegram", "bot", "handler", "inline button", "message"),
    "celery": ("celery", "background job", "queue", "task", "scheduled"),
    "testing": ("test", "pytest", "verify", "verification"),
    "git": ("git", "commit", "branch", "status", "diff"),
    "security": ("security", "permission", "auth", "role", "access", "secret"),
}

DEFAULT_SKILL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fastapi": ("fastapi", "uvicorn", "pydantic", "starlette", "endpoint", "router"),
    "nestjs": ("nestjs", "@nestjs", "controller", "provider", "module", "dependency injection"),
    "nextjs": ("next", "nextjs", "next.js", "ssr", "app router", "routing", "server components"),
    "reactjs": ("react", "reactjs", "jsx", "tsx", "component", "hooks"),
    "laravel": ("laravel", "php", "artisan", "eloquent", "blade", "middleware", "controller"),
}


def build_default_skill_registry_text(text: str, skill_names: Iterable[str]) -> str:
    """Return manager.py text with missing built-in skill names/keywords added.

    This deterministic path is for simple registry edits where stale LLM line
    numbers are risky. It edits by stable markers and preserves surrounding file
    content.
    """
    requested = [_normalize_name(name) for name in skill_names]
    requested = [name for name in requested if name]
    if not requested:
        return text

    updated = text
    if "DEFAULT_SKILL_NAMES = (" not in updated:
        raise ValueError("DEFAULT_SKILL_NAMES marker not found")
    block_start = updated.index("DEFAULT_SKILL_NAMES = (")
    block_end = updated.index(")", block_start)
    block = updated[block_start:block_end]
    missing_names = [name for name in requested if f'"{name}"' not in block]
    if missing_names:
        insertion = "".join(f'    "{name}",\n' for name in sorted(missing_names))
        marker = '    "telegram-bot",\n'
        if marker in block:
            block = block.replace(marker, insertion + marker, 1)
        else:
            block = block + insertion
        updated = updated[:block_start] + block + updated[block_end:]

    if "_KEYWORDS: dict[str, tuple[str, ...]] = {" not in updated:
        raise ValueError("_KEYWORDS marker not found")
    keyword_start = updated.index("_KEYWORDS: dict[str, tuple[str, ...]] = {")
    keyword_end = updated.index("\n}", keyword_start)
    keyword_block = updated[keyword_start:keyword_end]
    missing_keywords = [
        name for name in requested
        if name in DEFAULT_SKILL_KEYWORDS and f'    "{name}":' not in keyword_block
    ]
    if missing_keywords:
        insertion = "".join(
            f'    "{name}": {DEFAULT_SKILL_KEYWORDS[name]!r},\n'
            for name in sorted(missing_keywords)
        ).replace("'", '"')
        marker = '    "telegram-bot":'
        marker_index = keyword_block.find(marker)
        if marker_index >= 0:
            keyword_block = keyword_block[:marker_index] + insertion + keyword_block[marker_index:]
        else:
            keyword_block = keyword_block + "\n" + insertion.rstrip("\n")
        updated = updated[:keyword_start] + keyword_block + updated[keyword_end:]

    return updated


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    source: str
    path: Path | None
    content: str


def _normalize_name(name: str) -> str:
    return str(name or "").strip().lower().removesuffix(".md")


def detect_skill_names(text: str, explicit: Iterable[str] | None = None) -> list[str]:
    """Return skill names that match task text plus explicit skill requests."""
    selected: list[str] = []
    for raw in explicit or []:
        name = _normalize_name(raw)
        if name and name not in selected:
            selected.append(name)
    haystack = str(text or "").lower()
    for name, keywords in _KEYWORDS.items():
        if name in selected:
            continue
        if any(keyword in haystack for keyword in keywords):
            selected.append(name)
    return selected


class SkillManager:
    """Load project, global, and built-in skills in priority order."""

    def __init__(self, project_root: str | Path | None = None, home: str | Path | None = None) -> None:
        self.project_root = Path(project_root or Path.cwd()).expanduser().resolve()
        self.home = Path(home or Path.home()).expanduser().resolve()

    @property
    def project_skills_dir(self) -> Path:
        return self.project_root / "skills"

    @property
    def global_skills_dir(self) -> Path:
        return self.home / ".mana" / "skills"

    def _builtin_content(self, name: str) -> str | None:
        try:
            return (
                resources.files("mana_agent.default_skills")
                .joinpath(f"{name}.md")
                .read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ModuleNotFoundError):
            return None

    def _read_file_skill(self, directory: Path, name: str, source: str) -> Skill | None:
        path = directory / f"{name}.md"
        if not path.exists():
            return None
        return Skill(name=name, source=source, path=path, content=path.read_text(encoding="utf-8"))

    def get(self, name: str) -> Skill | None:
        normalized = _normalize_name(name)
        if not normalized:
            return None
        return (
            self._read_file_skill(self.project_skills_dir, normalized, "project")
            or self._read_file_skill(self.global_skills_dir, normalized, "global")
            or self._builtin_skill(normalized)
        )

    def _builtin_skill(self, name: str) -> Skill | None:
        content = self._builtin_content(name)
        if content is None:
            return None
        return Skill(name=name, source="built-in", path=None, content=content)

    def load_for_task(self, text: str, explicit: Iterable[str] | None = None) -> list[Skill]:
        skills: list[Skill] = []
        missing: list[str] = []
        for name in detect_skill_names(text, explicit):
            skill = self.get(name)
            if skill is None:
                missing.append(name)
            else:
                skills.append(skill)
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"Unknown skill(s): {names}")
        return skills

    def list_by_source(self) -> dict[str, list[str]]:
        return {
            "Project Root Skills": self._list_dir(self.project_skills_dir),
            "Global Skills": self._list_dir(self.global_skills_dir),
            "Built-in Skills": list(DEFAULT_SKILL_NAMES),
        }

    def _list_dir(self, directory: Path) -> list[str]:
        if not directory.exists():
            return []
        return sorted(_normalize_name(path.name) for path in directory.glob("*.md") if path.is_file())

    def init_project_skills(self, *, force: bool = False) -> list[Path]:
        self.project_skills_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name in DEFAULT_SKILL_NAMES:
            content = self._builtin_content(name)
            if content is None:
                continue
            target = self.project_skills_dir / f"{name}.md"
            if target.exists() and not force:
                continue
            target.write_text(content, encoding="utf-8")
            written.append(target)
        return written
