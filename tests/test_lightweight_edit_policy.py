from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mana_agent.agent.verification_planner import plan_verification, verify_documentation_changes
from mana_agent.multi_agent.runtime.agent_work_queue import execute_registered_mutation_command
from mana_agent.multi_agent.runtime.edit_scope import budget_for_scope, resolve_repo_path, select_scope
from mana_agent.multi_agent.runtime.mutation_plan import MutationCommand, build_mutation_plan
from mana_agent.multi_agent.runtime.tool_worker_process import ToolRunResponse


def test_case_safe_path_resolution_contract(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")

    exact = resolve_repo_path(tmp_path, "README.md")
    folded = resolve_repo_path(tmp_path, "./readme.md")
    missing = resolve_repo_path(tmp_path, "missing.md")
    traversal = resolve_repo_path(tmp_path, "../README.md")

    assert (exact.resolved_path, exact.method) == ("README.md", "exact")
    assert (folded.resolved_path, folded.method) == ("README.md", "case_insensitive")
    assert missing.method == "missing" and not missing.ok
    assert traversal.method == "rejected" and traversal.reason == "path_outside_repository"


def test_case_safe_path_resolution_reports_ambiguity(tmp_path: Path) -> None:
    upper = tmp_path / "Guide.md"
    lower = tmp_path / "guide.md"
    upper.write_text("upper\n", encoding="utf-8")
    lower.write_text("lower\n", encoding="utf-8")
    if len({path.name for path in tmp_path.iterdir() if path.name.casefold() == "guide.md"}) < 2:
        pytest.skip("filesystem is case-insensitive")

    result = resolve_repo_path(tmp_path, "GUIDE.md")

    assert result.method == "ambiguous"
    assert set(result.matches) == {"Guide.md", "guide.md"}


def test_localized_readme_plan_has_zero_search_budget_and_no_architecture_evidence(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "telegram.md").write_text("# Telegram\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "runtime.py").write_text("value = 1\n", encoding="utf-8")

    plan = build_mutation_plan(
        repo_root=tmp_path,
        user_goal="add a Telegram description and documentation link to readme.md",
        target_files=["README.md", "docs/telegram.md"],
        evidence_files_read=["README.md", "docs/telegram.md"],
    )

    assert plan.allowed_to_mutate is True
    assert plan.task_scope == "localized_change"
    assert plan.scope_budget["max_initial_searches"] == 0
    assert plan.scope_budget["architecture_evidence_allowed"] is False
    assert plan.required_evidence_files == ["README.md", "docs/telegram.md"]
    assert not any(path.startswith("src/") for path in plan.required_evidence_files)
    assert plan.intended_changes == [
        "Implement the requested change in README.md",
        "Implement the requested change in docs/telegram.md",
    ]


def test_scope_budgets_are_centralized_and_bounded() -> None:
    scope = select_scope(resolved_targets=["README.md", "docs/telegram.md"], model_scope="multi_file")
    budget = budget_for_scope(scope)

    assert scope == "localized_change"
    assert budget.max_initial_searches == 0
    assert budget.mutation_plan_limit == 1
    assert budget.patch_retry_limit == 1


def test_patch_precondition_failure_rereads_target_and_retries_once(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    original = "# Demo\n\nTelegram is available.\n"
    target.write_text(original, encoding="utf-8")
    expected_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
    # Simulate an external edit after the mutation command was generated.
    target.write_text("# Demo\n\nTelegram is now available.\n", encoding="utf-8")
    command = MutationCommand(
        plan_id="mp_retry",
        tool_name="apply_patch",
        tool_args={
            "patch": (
                "*** Begin Patch\n"
                "*** Update File: README.md\n"
                "@@\n"
                "-Telegram is available.\n"
                "+Telegram is available through the integration.\n"
                "*** End Patch\n"
            ),
            "content_hashes": {"README.md": expected_hash},
        },
        target_files=["README.md"],
        reason="test stale content precondition",
    )

    result = execute_registered_mutation_command(repo_root=tmp_path, command=command)

    assert result.ok is True
    assert result.files_changed == ["README.md"]
    assert "Telegram is available through the integration." in target.read_text(encoding="utf-8")
    assert [row.get("patch_retry_count") for row in result.trace if row.get("patch_retry_count")] == [1]
    assert [row.get("path") for row in result.trace if row.get("tool_name") == "read_file"] == ["README.md"]


def test_patch_precondition_accepts_unchanged_crlf_file(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    normalized = "# Demo\n\n## Documentation\n"
    target.write_bytes(normalized.replace("\n", "\r\n").encode("utf-8"))
    command = MutationCommand(
        plan_id="mp_crlf",
        tool_name="apply_patch",
        tool_args={
            "patch": (
                "*** Begin Patch\n"
                "*** Update File: README.md\n"
                "@@\n"
                "-## Documentation\n"
                "+## Windows documentation\n"
                "*** End Patch\n"
            ),
            "content_hashes": {"README.md": hashlib.sha256(normalized.encode("utf-8")).hexdigest()},
        },
        target_files=["README.md"],
        reason="test platform-independent text precondition",
    )

    result = execute_registered_mutation_command(repo_root=tmp_path, command=command)

    assert result.ok is True
    assert result.files_changed == ["README.md"]
    assert "## Windows documentation" in target.read_text(encoding="utf-8")
    patch_rows = [row for row in result.trace if row.get("tool_name") == "apply_patch"]
    assert len(patch_rows) == 1
    assert patch_rows[0]["status"] == "ok"
    assert not patch_rows[0].get("patch_retry_count")


def test_patch_retry_failure_is_precise_and_does_not_discover(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("# Completely different\n", encoding="utf-8")
    command = MutationCommand(
        plan_id="mp_retry_fail",
        tool_name="apply_patch",
        tool_args={
            "patch": (
                "*** Begin Patch\n*** Update File: README.md\n@@\n"
                "-Synthetic heading never observed\n+Real heading\n*** End Patch\n"
            )
        },
        target_files=["README.md"],
        reason="test unsafe recovery",
    )

    result = execute_registered_mutation_command(repo_root=tmp_path, command=command)

    assert result.ok is False
    assert result.error.startswith("patch_context_not_found")
    assert sum(1 for row in result.trace if row.get("patch_retry_count") == 1) == 1
    assert all(row.get("tool_name") not in {"repo_search", "list_files"} for row in result.trace)


def test_reapplying_same_patch_is_an_idempotent_noop(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("# Demo\n\n## Telegram\n", encoding="utf-8")
    command = MutationCommand(
        plan_id="mp_idempotent",
        tool_name="apply_patch",
        tool_args={
            "patch": (
                "*** Begin Patch\n*** Update File: README.md\n@@\n"
                "-## Messaging\n+## Telegram\n*** End Patch\n"
            )
        },
        target_files=["README.md"],
        reason="idempotency regression",
    )

    result = execute_registered_mutation_command(repo_root=tmp_path, command=command)

    assert result.ok is True
    assert result.files_changed == []
    assert target.read_text(encoding="utf-8").count("## Telegram") == 1
    assert any(row.get("no_op_reason") == "requested patch content is already present" for row in result.trace)


def test_documentation_verification_checks_links_duplicates_and_skips_project_tests(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "telegram.md").write_text("# Telegram\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Demo\n\n## Telegram\n\nSee [Telegram docs](docs/telegram.md).\n",
        encoding="utf-8",
    )

    decision = plan_verification(changed_files=["README.md", "docs/telegram.md"])
    result = verify_documentation_changes(
        repo_root=tmp_path,
        changed_files=["README.md", "docs/telegram.md"],
    )

    assert decision.verification_profile == "documentation_verification"
    assert decision.commands == ("verify_changed_artifacts",)
    assert result.ok is True
    assert any("pytest" in reason for reason in result.skipped_checks)
    assert {check["name"] for check in result.checks} == {"duplicate_headings", "local_links", "content_reread"}


def test_localized_python_change_uses_code_verification() -> None:
    decision = plan_verification(changed_files=["src/mana_agent/example.py"])

    assert decision.verification_profile == "task_verification"
    assert decision.verification_class == "code"
    assert decision.commands[0] == "python -m py_compile src/mana_agent/example.py"
    assert decision.verification_profile != "documentation_verification"


def test_two_file_documentation_edit_has_no_search_or_model_backed_verification(tmp_path: Path) -> None:
    from mana_agent.multi_agent.runtime.agent_work_queue import QueueManager

    (tmp_path / "README.md").write_text("# Demo\n\n## Documentation\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "telegram.md").write_text("# Telegram\n", encoding="utf-8")

    class _Worker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.requests.append(request)
            if request.tool_name == "read_file":
                rel = str(request.tool_args["path"])
                return ToolRunResponse(
                    answer=(tmp_path / rel).read_text(encoding="utf-8"),
                    trace=[{"tool_name": "read_file", "status": "ok", "path": rel}],
                )
            if "MutationCommand" in request.question:
                return ToolRunResponse(
                    answer=json.dumps(
                        {
                            "tool_name": "apply_patch",
                            "tool_args": {
                                "patch": (
                                    "*** Begin Patch\n*** Update File: README.md\n@@\n-## Documentation\n"
                                    "+## Telegram\n+\n+Telegram integration overview covering setup, operation, and safe channel usage for repository automation workflows.\n+\n+## Documentation\n"
                                    "+- [Telegram](docs/telegram.md)\n*** Update File: docs/telegram.md\n@@\n"
                                    "-# Telegram\n+# Telegram\n+\n+Integration usage, configuration, operational guidance, troubleshooting, and maintenance details for the Telegram channel and its repository automation workflows.\n*** End Patch\n"
                                )
                            },
                        }
                    ),
                    trace=[],
                )
            raise AssertionError(f"unexpected model-backed tool call: {request.tool_name} {request.question}")

    worker = _Worker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="Update README.md and docs/telegram.md with a Telegram overview and documentation link",
        target_files=["README.md", "docs/telegram.md"],
        requires_edit=True,
    )

    tool_names = [str(request.tool_name or "") for request in worker.requests]
    read_paths = [str(request.tool_args["path"]) for request in worker.requests if request.tool_name == "read_file"]
    assert result.run_status == "completed", (result.terminal_reason, result.trace, result.answer)
    assert tool_names.count("repo_search") == 0
    assert read_paths == ["README.md", "docs/telegram.md"]
    assert sum("MutationCommand" in request.question for request in worker.requests) == 1, [request.question for request in worker.requests]
    assert all("Verify the changes" not in request.question for request in worker.requests)
    assert any(row.get("tool_name") == "verify_changed_artifacts" for row in result.trace)
    assert result.planner_decisions[0]["selected_task_scope"] == "localized_change"
    assert result.planner_decisions[0]["scope_budget"]["max_initial_searches"] == 0
