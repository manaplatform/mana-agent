from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field


MutationTool = Literal["write_file", "apply_patch", "create_file"]
RegisteredMutationTool = Literal[
    "write_file",
    "create_file",
    "edit_file",
    "multi_edit_file",
    "apply_patch",
    "apply_patch_batch",
    "delete_file",
]
EditType = Literal["docs_update", "code_change", "config_change", "test_change", "unknown"]

REGISTERED_MUTATION_TOOLS = {
    "write_file",
    "create_file",
    "edit_file",
    "multi_edit_file",
    "apply_patch",
    "apply_patch_batch",
    "delete_file",
}


ARCHITECTURE_SOURCE_DIRS = (
    "src/mana_agent/commands/",
    "src/mana_agent/llm/",
    "src/mana_agent/agent/",
    "src/mana_agent/services/",
    "src/mana_agent/tools/",
    "src/mana_agent/prompting/",
    "src/mana_agent/skills/",
    "src/mana_agent/vector_store/",
    "src/mana_agent/ui/",
    "src/mana_agent/renderers/",
)

ARCHITECTURE_INTENDED_CHANGES = (
    "CLI entry and interactive mode",
    "Chat/coding-agent orchestration",
    "Work queue and decision lifecycle",
    "Tool worker process",
    "Tool manager and mutation tools",
    "Repository search/read/write tools",
    "Analyze service and generated artifacts",
    "Indexing and FAISS vector store",
    "Skills / progressive skill loading",
    "UI/rendering layer",
    "Verification and safety gates",
)


class MutationPlan(BaseModel):
    plan_id: str = ""
    target_files: list[str] = Field(default_factory=list)
    user_goal: str = ""
    edit_type: EditType = "unknown"
    required_evidence_files: list[str] = Field(default_factory=list)
    evidence_files_read: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    intended_changes: list[str] = Field(default_factory=list)
    patch_strategy: str = ""
    mutation_tool: MutationTool = "apply_patch"
    allowed_to_mutate: bool = False
    blocked_reason: str | None = None
    quality_checks: list[str] = Field(default_factory=list)

    def model_post_init(self, _ctx: Any) -> None:
        if not self.plan_id:
            payload = "|".join(
                [
                    self.user_goal,
                    ",".join(self.target_files),
                    self.edit_type,
                    ",".join(self.required_evidence_files),
                    self.patch_strategy,
                ]
            )
            self.plan_id = "mp_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


class MutationCommand(BaseModel):
    plan_id: str
    tool_name: RegisteredMutationTool
    tool_args: dict[str, Any]
    target_files: list[str]
    reason: str


def normalize_repo_path(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./")


def is_architecture_docs_update(request: str, target_files: Sequence[str]) -> bool:
    text = str(request or "").lower()
    targets = " ".join(str(item).lower() for item in target_files)
    return (
        "architecture" in text
        and "src" in text
        and any("08-architecture.md" in item or "architecture.md" in item for item in (text, targets))
    )


def representative_architecture_sources(repo_root: Path) -> list[str]:
    out: list[str] = []
    for dirname in ARCHITECTURE_SOURCE_DIRS:
        root = repo_root / dirname
        if not root.exists():
            continue
        candidates = sorted(
            path for path in root.rglob("*.py")
            if path.is_file() and "__pycache__" not in path.parts
        )
        non_init = [path for path in candidates if path.name != "__init__.py"]
        candidates = non_init or candidates
        if candidates:
            out.append(candidates[0].resolve().relative_to(repo_root.resolve()).as_posix())
    return out


def build_mutation_plan(
    *,
    repo_root: Path,
    user_goal: str,
    target_files: Sequence[str],
    evidence_files_read: Sequence[str],
) -> MutationPlan:
    targets = [normalize_repo_path(path) for path in target_files if normalize_repo_path(path)]
    read = sorted(dict.fromkeys(normalize_repo_path(path) for path in evidence_files_read if normalize_repo_path(path)))
    arch_update = is_architecture_docs_update(user_goal, targets)
    required = list(targets)
    intended: list[str]
    checks: list[str]
    if arch_update:
        required.extend(representative_architecture_sources(repo_root))
        intended = list(ARCHITECTURE_INTENDED_CHANGES)
        checks = [
            "file contains real src/mana_agent/... paths",
            "file contains source-backed architecture sections",
            "file does not only append a request note",
            "file is meaningfully rewritten or expanded",
            "diff includes architecture content, not only metadata",
            "no duplicate heading noise",
            "no accidental pyproject/changelog dump in final answer",
        ]
        edit_type: EditType = "docs_update"
        strategy = (
            "Patch docs/08-architecture.md from current target content using the source-read architecture evidence; "
            "rewrite or expand concrete architecture sections instead of appending a request note."
        )
    else:
        intended = [f"Implement the requested change in {path}" for path in targets] or ["Implement the requested repository change"]
        checks = ["changed files match the mutation plan targets", "relevant verification passes"]
        edit_type = "unknown"
        strategy = "Patch the listed target files using current file content and gathered source evidence."

    plan = MutationPlan(
        target_files=targets,
        user_goal=str(user_goal or "").strip(),
        edit_type=edit_type,
        required_evidence_files=sorted(dict.fromkeys(required)),
        evidence_files_read=read,
        evidence_summary=_evidence_summary(read, arch_update=arch_update),
        intended_changes=intended,
        patch_strategy=strategy,
        mutation_tool="apply_patch",
        allowed_to_mutate=True,
        blocked_reason=None,
        quality_checks=checks,
    )
    errors = validate_mutation_plan(plan, repo_root=repo_root)
    if errors:
        return plan.model_copy(update={"allowed_to_mutate": False, "blocked_reason": "; ".join(errors)})
    return plan


def validate_mutation_plan(plan: MutationPlan | dict[str, Any], *, repo_root: Path) -> list[str]:
    if not isinstance(plan, MutationPlan):
        try:
            plan = MutationPlan.model_validate(plan)
        except Exception as exc:
            return [f"invalid mutation plan: {exc}"]
    errors: list[str] = []
    targets = [normalize_repo_path(path) for path in plan.target_files]
    read = {normalize_repo_path(path) for path in plan.evidence_files_read}
    if not plan.allowed_to_mutate:
        errors.append("allowed_to_mutate is false")
    if plan.blocked_reason:
        errors.append(f"blocked_reason is set: {plan.blocked_reason}")
    if not targets:
        errors.append("no target files")
    create_targets: set[str] = set()
    for target in targets:
        exists = (repo_root / target).exists()
        create_intent = plan.mutation_tool == "create_file" or re.search(r"\b(create|add|generate)\b", plan.user_goal, re.I)
        if create_intent and not exists:
            create_targets.add(target)
        if not exists and not create_intent:
            errors.append(f"target file does not exist: {target}")
        if exists and target not in read:
            errors.append(f"current target file was not read: {target}")
    missing = [
        path for path in plan.required_evidence_files
        if normalize_repo_path(path) not in read and normalize_repo_path(path) not in create_targets
    ]
    if missing:
        errors.append(f"required evidence files not read: {missing[:8]}")
    if not plan.evidence_summary.strip():
        errors.append("evidence summary is empty")
    specific_changes = [item for item in plan.intended_changes if len(str(item).strip().split()) >= 3]
    if len(specific_changes) < max(1, min(3, len(plan.intended_changes))):
        errors.append("intended changes are not specific")
    generic_strategy = plan.patch_strategy.strip().lower() in {"", "update file", "make changes", "patch file"}
    if generic_strategy or len(plan.patch_strategy.strip().split()) < 8:
        errors.append("patch strategy is generic")
    return errors


def _creation_allowed(plan: MutationPlan, target: str, repo_root: Path) -> bool:
    if (repo_root / target).exists():
        return False
    return plan.mutation_tool == "create_file" or bool(re.search(r"\b(create|add|generate)\b", plan.user_goal, re.I))


def compile_mutation_command(
    *,
    repo_root: Path,
    plan: MutationPlan,
    current_files: dict[str, str],
    synthesized_content: str | None = None,
    patch: str | None = None,
) -> MutationCommand:
    targets = [normalize_repo_path(path) for path in plan.target_files if normalize_repo_path(path)]
    target = targets[0] if targets else ""
    strategy = str(plan.patch_strategy or "").lower()
    if patch and ("patch" in strategy or "diff" in strategy or plan.mutation_tool == "apply_patch"):
        return MutationCommand(
            plan_id=plan.plan_id,
            tool_name="apply_patch",
            tool_args={"patch": patch},
            target_files=targets,
            reason="compiled from patch/diff mutation strategy",
        )
    if target and _creation_allowed(plan, target, repo_root):
        return MutationCommand(
            plan_id=plan.plan_id,
            tool_name="create_file",
            tool_args={"path": target, "content": synthesized_content or ""},
            target_files=targets,
            reason="compiled for allowed target-file creation",
        )
    if plan.edit_type == "docs_update" or "full rewrite" in strategy or "rewrite" in strategy or "docs" in strategy:
        return MutationCommand(
            plan_id=plan.plan_id,
            tool_name="write_file",
            tool_args={"path": target, "content": synthesized_content or "", "force": True},
            target_files=targets,
            reason="compiled for full-file/docs update",
        )
    if patch:
        return MutationCommand(
            plan_id=plan.plan_id,
            tool_name="apply_patch",
            tool_args={"patch": patch},
            target_files=targets,
            reason="compiled from supplied patch",
        )
    return MutationCommand(
        plan_id=plan.plan_id,
        tool_name="write_file",
        tool_args={"path": target, "content": synthesized_content or "", "force": True},
        target_files=targets,
        reason="compiled for full-file write",
    )


def validate_mutation_command(command: MutationCommand | dict[str, Any]) -> list[str]:
    if not isinstance(command, MutationCommand):
        try:
            command = MutationCommand.model_validate(command)
        except Exception as exc:
            return [f"invalid mutation command: {exc}"]
    errors: list[str] = []
    if command.tool_name not in REGISTERED_MUTATION_TOOLS:
        errors.append("unregistered mutation tool")
    if command.tool_name in {"write_file", "create_file"}:
        if not command.tool_args.get("path"):
            errors.append("missing path")
        if not command.tool_args.get("content"):
            errors.append("missing content")
    if command.tool_name == "apply_patch":
        if not command.tool_args.get("patch"):
            errors.append("missing patch")
    if command.tool_name == "apply_patch_batch":
        if not command.tool_args.get("patches"):
            errors.append("missing patches")
    if command.tool_name == "delete_file" and not command.tool_args.get("path"):
        errors.append("missing path")
    if not command.plan_id:
        errors.append("missing mutation plan id")
    return errors


def mutation_trace_has_plan(trace: Sequence[dict[str, Any]], plan_id: str) -> bool:
    wanted = str(plan_id or "").strip()
    if not wanted:
        return False
    for row in trace:
        if not isinstance(row, dict):
            continue
        tool = str(row.get("tool_name") or row.get("tool") or row.get("name") or "").strip().lower()
        if tool not in REGISTERED_MUTATION_TOOLS:
            continue
        if str(row.get("mutation_plan_id") or row.get("plan_id") or "").strip() != wanted:
            continue
        if str(row.get("status") or "").strip().lower() != "ok":
            continue
        changed = {normalize_repo_path(path) for path in row.get("changed_files") or row.get("files_changed") or []}
        targets = {normalize_repo_path(path) for path in row.get("target_files") or []}
        if targets and changed and not changed.issubset(targets):
            continue
        if changed or not targets:
            return True
    return False


def changed_files_match_plan(changed_files: Sequence[str], plan: MutationPlan | None) -> bool:
    if plan is None:
        return False
    targets = {normalize_repo_path(path) for path in plan.target_files}
    changed = {normalize_repo_path(path) for path in changed_files}
    return bool(targets and changed and targets.issubset(changed))


def _evidence_summary(read: Sequence[str], *, arch_update: bool) -> str:
    source_reads = [path for path in read if path.startswith("src/mana_agent/")]
    if arch_update:
        return "Architecture evidence read from " + ", ".join(source_reads[:12])
    return "Evidence read from " + ", ".join(list(read)[:12])
