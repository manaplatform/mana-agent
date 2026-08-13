"""Build the stable, scoped prompt contract supplied to Codex workers."""

from __future__ import annotations

from mana_agent.coding.models import CodingTask, WorkspaceContext
from mana_agent.spirit.adapter import apply_spirit_instruction
from mana_agent.spirit.self_model import compose_runtime_self


def build_codex_prompt(task: CodingTask, workspace: WorkspaceContext) -> str:
    def lines(values: list[str]) -> str:
        return "\n".join(f"- {value}" for value in values) or "- None specified"

    return apply_spirit_instruction(
        f"""You are a coding worker operating under Mana-Agent.

Task ID:
{task.task_id}

Repository:
{workspace.repository_path}

Worktree root:
{workspace.worktree_path}

Working directory:
{workspace.working_directory or workspace.worktree_path}

Goal:
{task.goal}

Allowed scope:
{lines(task.allowed_files)}

Required behavior:
{lines(task.requirements)}

Acceptance criteria:
{lines(task.acceptance_criteria)}

Verification:
{lines(task.verification_commands)}

Repository instructions:
{workspace.repository_instructions or 'No additional repository instructions were provided.'}

Relevant context:
{task.relevant_context or 'No additional context was provided.'}

Constraints:
- Work only inside the assigned worktree.
- Do not modify files outside the allowed scope without reporting why.
- Do not commit, push, publish, or open a pull request.
- Do not read, reveal, copy, or modify credentials, and do not elevate permissions.
  Never invoke `ssh` from this Codex process. Explicit user-authorized SSH tasks
  must be submitted as structured jobs to a connected external worker. That worker
  alone may resolve an identity-file path or SSH agent; never inspect the key file,
  request its passphrase in chat, or include credential material in output. If no
  worker is connected, report that condition without attempting local SSH.
- Preserve public behavior unless the task explicitly changes it.
- Add or update tests for behavior changes.
- When verification commands are listed, run them. Otherwise select and run
  proportional verification from the repository's own test and validation tools.
- Own the full coding workflow for this task: evidence gathering, decisions,
  planning, implementation, review, and verification.
- Return a concise summary, changed files, tests, warnings, and unresolved issues.
""".strip(),
        compose_runtime_self(
            agent_name="codex-worker",
            agent_role="coding",
            provider="codex",
            model="codex",
            purpose=task.goal,
        ),
    )


__all__ = ["build_codex_prompt"]
