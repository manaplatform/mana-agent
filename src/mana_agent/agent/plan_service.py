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
from mana_agent.tools.repository import list_files, repo_search
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path

logger = logging.getLogger(__name__)

_KEYWORD_RE = re.compile(r"\b[A-Za-z0-9_\-\.]{2,}\b")

_IGNORE_TERMS = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "from", "into", "when",
        "implement", "create", "build", "update", "mode", "plan", "agent",
        "mana", "make", "allow", "should", "code", "file", "support",
        "dont", "not", "only", "show", "seperate", "separate",
        "also", "too", "need", "first", "please", "sample", "then", "just",
        "like", "which", "will", "would", "could", "have", "been", "where",
        "what", "how", "why", "are", "were", "was", "can", "may", "its",
        "their", "them", "some", "such", "than", "each", "does", "done",
    }
)

_NOISE_PATH_SUBSTRINGS = (
    ".github/",
    ".git/",
    ".mana/",
    ".mana_logs/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".egg-info",
    "__pycache__",
    "venv/",
    ".venv",
    "node_modules/",
    "build/",
    "dist/",
    "CHANGELOG.md",
    "README.md",
    "AGENTS.md",
)


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", str(text or "").lower()).strip()
    slug = re.sub(r"[-\s]+", "-", cleaned)
    return slug[:60] or "task-plan"


def _is_noise_path(path_str: str, allow_docs_or_ci: bool = False) -> bool:
    normalized = path_str.replace("\\", "/")
    if not allow_docs_or_ci:
        for noise in _NOISE_PATH_SUBSTRINGS:
            if noise in normalized or normalized.endswith(noise):
                return True
        if normalized.endswith(".md"):
            return True
    return False


def _invoke_llm(llm: Any, prompt_or_messages: Any) -> str:
    try:
        if hasattr(llm, "invoke"):
            res = llm.invoke(prompt_or_messages)
        else:
            res = llm(prompt_or_messages)
        return str(getattr(res, "content", res) or "")
    except Exception as exc:
        logger.debug("PlanService LLM invocation failed: %s", exc)
        return ""


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    candidate = match.group(1).strip() if match else raw
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None


class PlanService:
    """Explores codebase context in a read-only manner and produces structured plans."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        llm: Any = None,
        progress_callback: Any = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.llm = llm
        self.progress_callback = progress_callback

    def _emit_progress(
        self,
        *,
        event_id: str,
        event_type: str,
        title: str,
        status: str,
        tool_name: str | None = None,
        detail: str = "",
    ) -> None:
        """Publish safe, compact plan progress without making callbacks authoritative."""
        if not callable(self.progress_callback):
            return
        event: dict[str, Any] = {
            "event_id": event_id,
            "event_type": event_type,
            "title": title,
            "status": status,
            "metadata": {"phase": "plan", "tool_name": tool_name} if tool_name else {"phase": "plan"},
        }
        if tool_name:
            event["tool_name"] = tool_name
        if detail:
            event["detail"] = detail
        try:
            self.progress_callback(event)
        except Exception:
            logger.debug("Plan progress callback failed", exc_info=True)

    def _query_llm_for_search_terms(self, task: str) -> tuple[list[str], list[str]]:
        if not self.llm:
            return [], []
        system_prompt = (
            "You are an expert codebase exploration assistant. Analyze the user's software task and return "
            "targeted search queries and file patterns to locate the affected source files and tests in the repository.\n"
            "Respond ONLY with a valid JSON object in this exact format:\n"
            "{\n"
            '  "queries": ["symbol_or_phrase_1", "symbol_or_phrase_2"],\n'
            '  "patterns": ["glob_keyword_1", "glob_keyword_2"]\n'
            "}"
        )
        user_prompt = f"Task: {task}"
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        except Exception:
            messages = f"{system_prompt}\n\n{user_prompt}"

        raw = _invoke_llm(self.llm, messages)
        data = _extract_json_payload(raw)
        if isinstance(data, dict):
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
            patterns = [str(p).strip() for p in data.get("patterns", []) if str(p).strip()]
            return queries[:6], patterns[:6]
        return [], []

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
        file_summaries: dict[str, Any] = {}

        task_clean = " ".join(str(task or "").strip().split())
        task_lower = task_clean.lower()
        allow_docs_or_ci = any(
            t in task_lower
            for t in ("github", "workflow", "action", "changelog", "readme", "release")
        )

        # 1. Add any explicitly provided candidate files
        for f in candidate_files:
            rel = str(f).strip().replace("\\", "/").lstrip("./")
            target = self.repo_root / rel
            if target.is_file() and rel not in discovered_files:
                discovered_files.append(rel)

        # 2. Extract search terms, compound phrases, and file glob patterns
        words = [
            w.lower()
            for w in re.findall(r"[a-zA-Z0-9_\-]+", task_clean)
            if w.lower() not in _IGNORE_TERMS and len(w) > 2
        ]

        raw_tokens = re.findall(r"[a-zA-Z0-9_\-]+", task_clean)
        phrases: list[str] = []
        for i in range(len(raw_tokens) - 1):
            w1 = raw_tokens[i].lower()
            w2 = raw_tokens[i + 1].lower()
            if w1 not in _IGNORE_TERMS and w2 not in _IGNORE_TERMS and len(w1) > 2 and len(w2) > 2:
                phrases.append(f"{w1} {w2}")
                phrases.append(f"{w1}_{w2}")

        search_terms_event = f"plan-search-terms-{uuid.uuid4().hex}"
        self._emit_progress(
            event_id=search_terms_event,
            event_type="agent.planning",
            title="Finding relevant repository search terms",
            status="running",
        )
        llm_queries, llm_patterns = self._query_llm_for_search_terms(task_clean)
        self._emit_progress(
            event_id=search_terms_event,
            event_type="agent.planning",
            title="Repository search terms ready",
            status="completed",
        )
        combined_queries = list(dict.fromkeys(llm_queries + phrases + words))
        glob_patterns = list(dict.fromkeys(llm_patterns + words))

        # 3. Query repository for candidate files with scoring
        scores: dict[str, int] = {}
        line_matches: dict[str, list[str]] = {}

        # Prioritize candidate files
        for f in discovered_files:
            scores[f] = 100

        # Glob search
        for pat in glob_patterns[:10]:
            event_id = f"plan-list-files-{uuid.uuid4().hex}"
            self._emit_progress(
                event_id=event_id,
                event_type="tool.started",
                title=f"Searching filenames for {pat!r}",
                status="running",
                tool_name="list_files",
            )
            res = list_files(self.repo_root, glob=f"**/*{pat}*.py", limit=15)
            self._emit_progress(
                event_id=event_id,
                event_type="tool.completed" if res.get("ok", True) else "tool.failed",
                title=f"Filename search for {pat!r}",
                status="completed" if res.get("ok", True) else "failed",
                tool_name="list_files",
                detail=f"Found {len(res.get('files', []))} candidate files",
            )
            for path in res.get("files", []):
                if _is_noise_path(path, allow_docs_or_ci):
                    continue
                scores[path] = scores.get(path, 0) + 5

        # Code content search
        for q in combined_queries[:12]:
            event_id = f"plan-repo-search-{uuid.uuid4().hex}"
            self._emit_progress(
                event_id=event_id,
                event_type="tool.started",
                title=f"Searching repository for {q!r}",
                status="running",
                tool_name="repo_search",
            )
            res = repo_search(self.repo_root, query=q, limit=10)
            self._emit_progress(
                event_id=event_id,
                event_type="tool.completed" if res.get("ok", True) else "tool.failed",
                title=f"Repository search for {q!r}",
                status="completed" if res.get("ok", True) else "failed",
                tool_name="repo_search",
                detail=f"Found {len(res.get('matches', []))} matches",
            )
            for match in res.get("matches", []):
                path = str(match.get("file") or "").strip()
                if not path or _is_noise_path(path, allow_docs_or_ci):
                    continue
                score_increment = 3 if path.startswith("src/") else (2 if path.startswith("tests/") else 1)
                scores[path] = scores.get(path, 0) + score_increment
                text = str(match.get("text") or "").strip()
                if text:
                    line_matches.setdefault(path, []).append(text[:120])

        # Apply structural path scoring
        for path in list(scores.keys()):
            if path in discovered_files:
                continue
            if path.startswith("src/"):
                scores[path] += 4
            elif path.startswith("tests/"):
                scores[path] += 2
            bname = Path(path).stem.lower()
            if any(w in bname for w in words):
                scores[path] += 5

        ranked = sorted(scores.items(), key=lambda item: -item[1])
        for path, _score in ranked:
            if path not in discovered_files:
                discovered_files.append(path)
                if len(discovered_files) >= max_files:
                    break

        # 4. Inspect discovered files (AST parsing of classes, functions, dependencies)
        for rel_path in discovered_files[:max_files]:
            event_id = f"plan-inspect-{uuid.uuid4().hex}"
            self._emit_progress(
                event_id=event_id,
                event_type="tool.started",
                title=f"Inspecting {rel_path}",
                status="running",
                tool_name="read_file",
            )
            full_path = self.repo_root / rel_path
            if not full_path.is_file():
                self._emit_progress(
                    event_id=event_id,
                    event_type="tool.failed",
                    title=f"Could not inspect {rel_path}",
                    status="failed",
                    tool_name="read_file",
                )
                continue

            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")[:20000]
                file_snippets[rel_path] = content[:1000]
            except Exception:
                self._emit_progress(
                    event_id=event_id,
                    event_type="tool.failed",
                    title=f"Could not inspect {rel_path}",
                    status="failed",
                    tool_name="read_file",
                )
                continue

            classes: list[str] = []
            functions: list[str] = []

            if rel_path.endswith(".py"):
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            classes.append(node.name)
                        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            functions.append(node.name)
                        elif isinstance(node, ast.Import):
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

            file_summaries[rel_path] = {
                "classes": classes[:8],
                "functions": functions[:12],
                "snippets": line_matches.get(rel_path, [])[:4],
            }
            self._emit_progress(
                event_id=event_id,
                event_type="tool.completed",
                title=f"Inspected {rel_path}",
                status="completed",
                tool_name="read_file",
            )

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
            "file_summaries": file_summaries,
            "terms": words[:10],
        }

    def _query_llm_for_proposal_plan(
        self,
        task: str,
        *,
        affected_files: list[str],
        dependencies: list[str],
        file_summaries: dict[str, Any],
        skills: Sequence[Any] = (),
    ) -> StructuredPlan | None:
        if not self.llm:
            return None

        context_lines: list[str] = []
        for f in affected_files[:6]:
            info = file_summaries.get(f, {})
            classes = info.get("classes", [])
            functions = info.get("functions", [])
            snippets = info.get("snippets", [])
            summary_parts = [f"- File: `{f}`"]
            if classes:
                summary_parts.append(f"  Classes: {', '.join(classes[:4])}")
            if functions:
                summary_parts.append(f"  Functions: {', '.join(functions[:6])}")
            if snippets:
                snippet_text = "\n    ".join(snippets[:3])
                summary_parts.append(f"  Key Lines:\n    {snippet_text}")
            context_lines.append("\n".join(summary_parts))

        context_block = "\n".join(context_lines)
        skill_block = ", ".join(getattr(s, "name", str(s)) for s in skills) if skills else "none"

        system_prompt = (
            "You are an autonomous repository engineering planner. Based on the user's task and the inspected codebase context, "
            "generate a comprehensive, concrete proposal implementation plan.\n"
            "Your implementation steps must refer directly to the specific files, classes, and components discovered.\n"
            "Respond ONLY with a valid JSON object in this format:\n"
            "{\n"
            '  "objectives": ["Concrete objective 1", ...],\n'
            '  "context_summary": "Detailed technical summary of discovered components and current architecture.",\n'
            '  "affected_files": ["exact/file1.py", "exact/file2.py"],\n'
            '  "dependencies": ["dep1", ...],\n'
            '  "implementation_steps": ["1. In `file1.py`: [specific changes]", "2. In `file2.py`: [specific changes]", ...],\n'
            '  "potential_risks": ["Specific risk 1 and mitigation", ...],\n'
            '  "validation_strategies": ["pytest tests/test_... -v", "Manual check: ..."],\n'
            '  "assumptions": ["Assumption 1", ...]\n'
            "}"
        )
        user_prompt = (
            f"User Task: {task}\n"
            f"Loaded Skills: {skill_block}\n\n"
            f"Inspected Repository Context:\n{context_block}\n"
        )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        except Exception:
            messages = f"{system_prompt}\n\n{user_prompt}"

        raw = _invoke_llm(self.llm, messages)
        data = _extract_json_payload(raw)
        if isinstance(data, dict) and data.get("objectives") and data.get("implementation_steps"):
            plan_id = f"plan-{uuid.uuid4().hex[:8]}"
            return StructuredPlan(
                task_id=plan_id,
                goal=task,
                objectives=list(data.get("objectives", [])),
                context_summary=str(data.get("context_summary") or "").strip(),
                affected_files=list(data.get("affected_files", affected_files)) or affected_files,
                dependencies=list(data.get("dependencies", dependencies)) or dependencies,
                implementation_steps=list(data.get("implementation_steps", [])),
                potential_risks=list(data.get("potential_risks", [])),
                validation_strategies=list(data.get("validation_strategies", [])),
                assumptions=list(data.get("assumptions", [])),
                approved=False,
            )
        return None

    def _synthesize_grounded_proposal_plan(
        self,
        task: str,
        *,
        affected_files: list[str],
        dependencies: list[str],
        file_summaries: dict[str, Any],
        skills: Sequence[Any] = (),
    ) -> StructuredPlan:
        objectives = [
            f"Implement: {task}",
            "Preserve backwards compatibility with existing contracts and public CLI/API interfaces.",
            "Enforce strict read-only guarantees in Plan Mode and explicit approval gating before Act Mode.",
        ]

        context_parts = [
            f"Repository root: `{self.repo_root}`",
        ]
        if skills:
            skill_names = [getattr(s, "name", str(s)) for s in skills]
            context_parts.append(f"Loaded skills: {', '.join(skill_names)}")
        if affected_files:
            file_items = []
            for f in affected_files[:6]:
                info = file_summaries.get(f, {})
                cls_list = info.get("classes", [])
                cls_info = f" (classes: {', '.join(f'`{c}`' for c in cls_list[:3])})" if cls_list else ""
                file_items.append(f"- `{f}`{cls_info}")
            context_parts.append("Discovered relevant components and source files:\n" + "\n".join(file_items))
        else:
            context_parts.append("- Repository context explored; ready for implementation.")
        context_summary = "\n\n".join(context_parts)

        steps: list[str] = []
        if affected_files:
            primary_file = affected_files[0]
            primary_info = file_summaries.get(primary_file, {})
            primary_cls = primary_info.get("classes", [])
            cls_mention = f" ({', '.join(f'`{c}`' for c in primary_cls[:2])})" if primary_cls else ""
            steps.append(f"Inspect existing implementation and state flow in `{primary_file}`{cls_mention}.")
            steps.append(f"Update `{primary_file}` to implement `{task}` while preserving architectural patterns.")
            for other_file in affected_files[1:4]:
                other_info = file_summaries.get(other_file, {})
                other_cls = other_info.get("classes", [])
                cls_str = f" ({', '.join(f'`{c}`' for c in other_cls[:2])})" if other_cls else ""
                steps.append(f"Update integration points in `{other_file}`{cls_str} to align with `{primary_file}`.")
            steps.append("Update related configurations, registries, or command wiring if necessary.")
            steps.append("Update documentation and CHANGELOG.md per Rule 47.")
        else:
            steps = [
                f"Inspect current implementation and dependencies for `{task}`.",
                "Design modular, typed additions adhering to repository patterns and AGENTS.md rules.",
                "Implement required components with strict validation and error handling.",
                "Update related commands, registries, or routers to wire new functionality.",
                "Update documentation and CHANGELOG.md per Rule 47.",
                "Run targeted unit tests and verify all acceptance criteria.",
            ]

        validation: list[str] = []
        test_files = [f for f in affected_files if "test" in f]
        if not test_files:
            for f in affected_files:
                base = Path(f).stem
                res = list_files(self.repo_root, glob=f"tests/**/test_*{base}*.py", limit=3)
                for tf in res.get("files", []):
                    if tf not in test_files:
                        test_files.append(tf)
        if test_files:
            for tf in test_files[:2]:
                validation.append(f"pytest {tf} -v")
        else:
            validation.append("pytest tests/ -k 'test_' -v (targeted unit tests for the affected component)")
        validation.append("Verify read-only restrictions are strictly enforced in Plan Mode")
        validation.append("Verify transition to Act Mode requires explicit approval and retains context")

        risks = [
            "Ensure existing public APIs and CLI parameters maintain backwards compatibility.",
            "Ensure Plan Mode cannot mutate any repository files or execute destructive commands.",
            "Ensure mode transitions cleanly preserve task context and require explicit user approval.",
        ]

        assumptions = [
            "Existing public commands and APIs remain compatible.",
            "Plan Mode operates read-only; Act Mode executes authorized changes.",
            "User approval is required before applying the plan in Act Mode.",
        ]

        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        return StructuredPlan(
            task_id=plan_id,
            goal=task,
            objectives=objectives,
            context_summary=context_summary,
            affected_files=affected_files,
            dependencies=dependencies,
            implementation_steps=steps,
            potential_risks=risks,
            validation_strategies=validation,
            assumptions=assumptions,
            approved=False,
        )

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

        exploration = context or self.explore_context(task_clean, candidate_files=candidate_files)
        affected = list(exploration.get("discovered_files", []))
        deps = list(exploration.get("dependencies", []))
        file_summaries = dict(exploration.get("file_summaries", {}))
        synthesis_event_id = f"plan-synthesis-{uuid.uuid4().hex}"
        self._emit_progress(
            event_id=synthesis_event_id,
            event_type="agent.planning",
            title="Drafting implementation plan",
            status="running",
        )

        # Try model-directed proposal plan generation if LLM is provided
        if self.llm is not None:
            model_plan = self._query_llm_for_proposal_plan(
                task_clean,
                affected_files=affected,
                dependencies=deps,
                file_summaries=file_summaries,
                skills=skills,
            )
            if model_plan is not None:
                self._emit_progress(
                    event_id=synthesis_event_id,
                    event_type="agent.planning",
                    title="Implementation plan drafted",
                    status="completed",
                )
                return model_plan

        # Grounded proposal plan synthesis
        plan = self._synthesize_grounded_proposal_plan(
            task_clean,
            affected_files=affected,
            dependencies=deps,
            file_summaries=file_summaries,
            skills=skills,
        )
        self._emit_progress(
            event_id=synthesis_event_id,
            event_type="agent.planning",
            title="Implementation plan drafted",
            status="completed",
        )
        return plan

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
