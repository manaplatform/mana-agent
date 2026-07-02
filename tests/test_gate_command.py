"""Regression tests for the coding-agent gate command authority.

These cover the worker-autonomy bug where workers could advance gates, mark
todos done, or "succeed" without ever running an allowed concrete tool.
"""

from __future__ import annotations

from pathlib import Path

from mana_agent.llm.gate_command import (
    can_run_final_report,
    can_run_verify,
    reconcile_gate_pointer,
)
from mana_agent.llm.tools_manager import RunStateStore


def _store(tmp_path: Path, run_id: str, goal: str = "update docs/models.md") -> RunStateStore:
    store = RunStateStore(repo_root=tmp_path, run_id=run_id)
    store.ensure(goal=goal, flow_id="")
    store.ensure_todo_ledger(goal=goal)
    return store


# --------------------------------------------------------------------------- #
# Test A: gate pointer reconciliation
# --------------------------------------------------------------------------- #


def test_A_resume_reconciles_to_plan_patch(tmp_path: Path) -> None:
    store = _store(tmp_path, "reconcile-A")
    state = store.read_json("state.json", {})
    # Simulate the failed run: completed discovery/read/classify but the pointer
    # is stranded on locate_candidates.
    state["completed_gates"] = ["locate_candidates", "read_candidates", "classify_evidence"]
    state["pending_gates"] = ["plan_patch", "apply_changes", "verify_changes", "final_report"]
    state["current_gate"] = "locate_candidates"
    state["current_phase"] = "DISCOVERY"
    store.write_json("state.json", state)

    reconciled = store.reconcile_gate_pointer()

    assert reconciled["current_gate"] == "plan_patch"
    assert reconciled["current_phase"] == "PATCHING"


def test_A_pure_reconcile_never_resumes_completed_locate(tmp_path: Path) -> None:
    # pending_files == 0 and locate completed -> never resume into locate.
    gate, phase = reconcile_gate_pointer(
        completed_gates=["locate_candidates", "read_candidates", "classify_evidence"],
        pending_gates=["locate_candidates", "plan_patch", "apply_changes"],
        pending_files=0,
    )
    assert gate == "plan_patch"
    assert phase == "PATCHING"


# --------------------------------------------------------------------------- #
# Test B: apply_changes blocks read_file before execution
# --------------------------------------------------------------------------- #


def test_B_apply_changes_blocks_read_file(tmp_path: Path) -> None:
    store = _store(tmp_path, "preflight-B")
    command = store.build_gate_command("apply_changes", goal="update docs/models.md")

    assert set(command.allowed_tools) == {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}

    decision = store.preflight_tool(command, tool_name="read_file")

    assert decision.allowed is False
    assert decision.is_policy_violation is True
    assert decision.outcome == "policy_violation"
    # A blocked tool is neither success nor no_progress.
    assert decision.outcome not in {"ok", "no_progress"}


def test_B_apply_changes_allows_mutation(tmp_path: Path) -> None:
    store = _store(tmp_path, "preflight-B2")
    command = store.build_gate_command("apply_changes", goal="update docs/models.md")
    decision = store.preflight_tool(command, tool_name="write_file")
    assert decision.allowed is True
    assert decision.outcome == "ok"


# --------------------------------------------------------------------------- #
# Test C: verify_changes forbids toolsmanager_request
# --------------------------------------------------------------------------- #


def test_C_verify_forbids_toolsmanager_request(tmp_path: Path) -> None:
    store = _store(tmp_path, "verify-C")
    command = store.build_gate_command("verify_changes", goal="update docs/models.md")

    decision = store.preflight_tool(command, tool_name="toolsmanager_request")
    assert decision.allowed is False
    assert decision.is_policy_violation is True

    # Even a recorded proof claiming verification via the wrapper is rejected.
    proof = {
        "tool_name": "toolsmanager_request",
        "verification_command": "pytest",
        "verification_result": "passed",
    }
    result = store.validate_gate_proof("verify_changes", proof)
    assert result.valid is False


# --------------------------------------------------------------------------- #
# Test D: wrapper without concrete inner tool is no_progress, not success
# --------------------------------------------------------------------------- #


def test_D_wrapper_without_inner_tool_is_no_progress(tmp_path: Path) -> None:
    store = _store(tmp_path, "wrapper-D")
    command = store.build_gate_command("locate_candidates", goal="update docs/models.md")

    decision = store.preflight_tool(command, tool_name="toolsmanager_request", inner_tool="")
    assert decision.allowed is False
    assert decision.outcome == "no_progress"

    proof = {"tool_name": "toolsmanager_request", "answer": "here is some text"}
    assert store.validate_gate_proof("locate_candidates", proof).valid is False


def test_D_wrapper_with_allowed_inner_tool_passes(tmp_path: Path) -> None:
    store = _store(tmp_path, "wrapper-D2")
    command = store.build_gate_command("locate_candidates", goal="update docs/models.md")
    decision = store.preflight_tool(command, tool_name="toolsmanager_request", inner_tool="repo_search")
    assert decision.allowed is True
    assert decision.concrete_tool == "repo_search"


# --------------------------------------------------------------------------- #
# Test E: attempt_count never exceeds max_attempts
# --------------------------------------------------------------------------- #


def test_E_attempt_count_never_exceeds_max(tmp_path: Path) -> None:
    store = _store(tmp_path, "attempts-E")
    todo = store.current_todo_for_gate("apply_changes", goal="update docs/models.md")
    assert todo.max_attempts == 3

    current = todo
    for _ in range(10):
        current = store.mark_todo_worker_failure(current, reason="policy_violation")
        assert current.attempt_count <= current.max_attempts

    assert current.attempt_count == 3
    assert current.status == "blocked"


# --------------------------------------------------------------------------- #
# Test F: empty modified_files cannot complete apply_changes nor final_report
# --------------------------------------------------------------------------- #


def test_F_empty_modified_files_blocks_completion(tmp_path: Path) -> None:
    store = _store(tmp_path, "empty-F")

    proof = {"tool_name": "apply_patch", "modified_files": [], "successful_patches": 0}
    assert store.validate_gate_proof("apply_changes", proof).valid is False

    # The todo path agrees: cannot confirm with empty modified files.
    todo = store.current_todo_for_gate("apply_changes", goal="update docs/models.md")
    worker_done = store.mark_todo_worker_done(todo, tool_name="apply_patch", files_changed=[])
    result = store.confirm_or_reject_todo(worker_done, files_changed=[])
    assert result.status == "failed"
    assert result.agent_confirmed is False

    # final_report cannot run while verify is not done.
    assert can_run_verify(successful_patches=0, modified_files=[]) is False
    assert can_run_final_report(verify_completed=False) is False


# --------------------------------------------------------------------------- #
# Test G: duplicate fingerprint allows one retry_once, then blocks
# --------------------------------------------------------------------------- #


def test_G_duplicate_after_no_progress_retry_once_then_block(tmp_path: Path) -> None:
    store = _store(tmp_path, "dupe-G")
    fingerprint = store.gate_tool_fingerprint(
        tool_name="repo_search",
        args={"query": "models"},
        gate="locate_candidates",
        target_file="",
    )

    # Never seen yet -> allow.
    assert store.duplicate_decision(fingerprint) == "allow"

    # First no_progress recorded -> exactly one retry permitted.
    store.record_tool_call(
        gate="locate_candidates",
        tool_name="repo_search",
        normalized_args={"query": "models"},
        fingerprint=fingerprint,
        status="no_progress",
    )
    assert store.duplicate_decision(fingerprint) == "retry_once"

    # Second no_progress -> block duplicate.
    store.record_tool_call(
        gate="locate_candidates",
        tool_name="repo_search",
        normalized_args={"query": "models"},
        fingerprint=fingerprint,
        status="no_progress",
    )
    assert store.duplicate_decision(fingerprint) == "block_duplicate"


def test_G_successful_fingerprint_is_skipped(tmp_path: Path) -> None:
    store = _store(tmp_path, "dupe-G2")
    fingerprint = store.gate_tool_fingerprint(
        tool_name="read_file",
        args={"path": "docs/models.md"},
        gate="read_candidates",
        target_file="docs/models.md",
    )
    store.record_tool_call(
        gate="read_candidates",
        tool_name="read_file",
        normalized_args={"path": "docs/models.md"},
        fingerprint=fingerprint,
        status="ok",
        files_read=["docs/models.md"],
    )
    assert store.duplicate_decision(fingerprint) == "skip_completed"
