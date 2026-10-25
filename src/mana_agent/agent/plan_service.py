"""Plan Mode service: safe repository exploration, dependency analysis, and structured plan generation."""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Sequence

from mana_agent.agent.mode import ExecutionMode, StructuredPlan
from mana_agent.tools.repository import repo_search
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path

logger = logging.getLogger(__name__)

_KEYWORD_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_\-\.]{2,}\b")
_IGNORE_TERMS = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "from", "into", "when",
        "implement", "create", "build", "update", "mode", "plan", "agent",
        "mana", "make", "allow", "should", "code", "file", "support",
    }
)


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", str(text or "").lower()).strip()
    slug = re.sub(r"[-\s]+", "-", cleaned)
    return slug[:60] or "task-plan"


class PlanService:
    """Explores codebase context in a read-only manner and produces structured plans."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()

    def explore_context(
        self,
        task: str,
        *,
        candidate_files: Sequence[str] = (),
        max_files: int = 8,
    ) -> dict[str, Any]:
        """Perform read-only exploration of the codebase to discover affected files and dependencies."""
        discovered_files: list[str] = []
        dependencies: list[str] = []
        file_snippets: dict[str, str] = {}

        # 1. Add any explicitly provided candidate files
        for f in candidate_files:
            rel = str(f).strip().replace("\\", "/").lstrip("./")
            target = self.repo_root / rel
            if target.is_file() and rel not in discovered_files:
                discovered_files.append(rel)

        # 2. Extract search terms from task
        terms = [
            t.lower() for t in _KEYWORD_RE.findall(task)
            if t.lower() not in _IGNORE_TERMS and len(t) > 3
        ]

        # 3. Search repo for relevant files if fewer than max_files
        if len(discovered_files) < max_files and terms:
            for term in terms[:3]:
                search_res = repo_search(self.repo_root, query=term, limit=5)
                if search_res.get("ok"):
                    for hit in search_res.get("matches", []):
                        path = str(hit.get("file") or "").strip()
                        if path and path not in discovered_files:
                            discovered_files.append(path)
                            if len(discovered_files) >= max_files:
                                break

        # 4. Discover dependencies and imports from discovered Python files
        for rel_path in discovered_files[:max_files]:
            full_path = self.repo_root / rel_path
            if not full_path.is_file():
                continue

            # Read bounded snippet (safe read-only)
            try:
                snippet = full_path.read_text(encoding="utf-8", errors="replace")[:10000]
                file_snippets[rel_path] = snippet[:600]
            except Exception:
                continue

            # If Python file, parse imports for dependencies
            if rel_path.endswith(".py"):
                try:
                    tree = ast.parse(snippet)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                dep_name = alias.name.split(".")[0]
                                if dep_name not in dependencies and dep_name != "mana_agent":
                                    dependencies.append(dep_name)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            if node.module.startswith("mana_agent."):
                                mod_sub = node.module.split(".")[1]
                                dep_sub = f"mana_agent.{mod_sub}"
                                if dep_sub not in dependencies:
                                    dependencies.append(dep_sub)
                            else:
                                dep_name = node.module.split(".")[0]
                                if dep_name not in dependencies:
                                    dependencies.append(dep_name)
                except Exception:
                    pass

        # 5. Check pyproject.toml / requirements.txt for top-level dependencies
        pyproject = self.repo_root / "pyproject.toml"
        if pyproject.is_file():
            try:
                txt = pyproject.read_text(encoding="utf-8", errors="replace")
                for match in re.finditer(r'["\']([a-zA-Z0-9_\-\.]+)(?:[>=<~!].*)?["\']', txt):
                    dep_name = match.group(1).strip()
                    if dep_name and dep_name not in {"project", "name", "dummy"} and dep_name not in dependencies:
                        dependencies.append(dep_name)
            except Exception:
                pass

        reqs = self.repo_root / "requirements.txt"
        if reqs.is_file():
            try:
                txt = reqs.read_text(encoding="utf-8", errors="replace")
                for line in txt.splitlines():
                    dep = line.strip().split(">=")[0].split("==")[0].split("<=")[0].strip()
                    if dep and not dep.startswith("#") and dep not in dependencies:
                        dependencies.append(dep)
            except Exception:
                pass

        return {
            "discovered_files": discovered_files[:max_files],
            "dependencies": dependencies[:12],
            "file_snippets": file_snippets,
            "terms": terms,
        }

    def generate_plan(
        self,
        task: str,
        *,
        candidate_files: Sequence[str] = (),
        skills: Sequence[Any] = (),
        context: dict[str, Any] | None = None,
    ) -> StructuredPlan:
        """Create a decision-complete, structured plan for the task."""
        task_clean = " ".join(str(task or "").strip().split())
        if not task_clean:
            raise ValueError("Task is required to generate a plan.")

        # Explore context if not provided
        exploration = context or self.explore_context(task_clean, candidate_files=candidate_files)
        affected = list(exploration.get("discovered_files", []))
        deps = list(exploration.get("dependencies", []))

        # Build objectives
        objectives = [
            f"Implement: {task_clean}",
            "Preserve backwards compatibility with existing contracts and public CLI/API interfaces.",
            "Enforce strict read-only guarantees in Plan Mode and explicit approval gating before Act Mode.",
        ]

        # Build context summary
        context_parts = [
            f"Repository root: `{self.repo_root}`",
        ]
        if skills:
            skill_names = [getattr(s, "name", str(s)) for s in skills]
            context_parts.append(f"Loaded skills: {', '.join(skill_names)}")
        if affected:
            context_parts.append(f"Discovered relevant files: {', '.join(f'`{f}`' for f in affected[:6])}")
        context_summary = "\n".join(context_parts)

        # Build implementation steps
        steps = [
            f"Inspect current implementation and dependencies for `{task_clean}`.",
            "Design modular, typed additions adhering to repository patterns and AGENTS.md rules.",
            "Implement required components with strict validation and error handling.",
            "Update related commands, registries, or routers to wire new functionality.",
            "Update documentation and CHANGELOG.md per Rule 47.",
            "Run targeted unit tests and verify all acceptance criteria.",
        ]

        # Potential risks
        risks = [
            "Ensure existing public APIs and CLI parameters maintain backwards compatibility.",
            "Ensure Plan Mode cannot mutate any repository files or execute destructive commands.",
            "Ensure mode transitions cleanly preserve task context and require explicit user approval.",
        ]

        # Validation strategies
        validation = [
            "pytest tests/ -k 'test_' -v (targeted unit tests for the affected component)",
            "Verify read-only restrictions are strictly enforced in Plan Mode",
            "Verify transition to Act Mode requires explicit approval and retains context",
        ]

        # Assumptions
        assumptions = [
            "Existing public commands and APIs remain compatible.",
            "Plan Mode operates read-only; Act Mode executes authorized changes.",
            "User approval is required before applying the plan in Act Mode.",
        ]

        plan_id = f"plan-{uuid.uuid4().hex[:8]}"

        return StructuredPlan(
            task_id=plan_id,
            goal=task_clean,
            objectives=objectives,
            context_summary=context_summary,
            affected_files=affected,
            dependencies=deps,
            implementation_steps=steps,
            potential_risks=risks,
            validation_strategies=validation,
            assumptions=assumptions,
            approved=False,
        )

    def save_plan(
        self,
        plan: StructuredPlan,
        output_path: str | Path | None = None,
    ) -> Path:
        """Save structured plan as a markdown file in .mana/plans/ or specified path."""
        if output_path:
            target = Path(output_path).expanduser().resolve()
        else:
            repo_id = repository_id_for_path(self.repo_root)
            target = repository_dir(repo_id) / "plans" / f"{_slugify(plan.goal)}.md"

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(plan.to_markdown(), encoding="utf-8")
        return target
