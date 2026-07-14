from __future__ import annotations

import json
from pathlib import Path

import pytest

from mana_agent.multi_agent.communication.message_bus import MessageBus
from mana_agent.multi_agent.core.types import MessageType
from mana_agent.multi_agent.runtime.agent_work_queue import QueueManager
from mana_agent.multi_agent.runtime.delegation import DelegationRequest
from mana_agent.multi_agent.runtime.evidence_ledger import EvidenceLedger
from mana_agent.multi_agent.runtime.execution_scope import (
    ExecutionScopeDecisionError,
    approve_scope_escalation,
    validate_execution_scope,
)
from mana_agent.multi_agent.runtime.tool_worker_process import ToolRunResponse


def _documentation_link_scope() -> dict[str, object]:
    return {
        "decision_id": "scope_doc_link",
        "task_type": "edit",
        "scope_level": 0,
        "complexity": "trivial",
        "risk": "low",
        "explicit_target_files": ["readme.md"],
        "related_files": ["docs/17-telegram-connector.md"],
        "required_evidence": ["README documentation section", "canonical connector-document path"],
        "allowed_tool_families": ["read", "mutation", "verification"],
        "search_scope": "none",
        "max_search_operations": 0,
        "max_unique_file_reads": 2,
        "mutation_strategy": "single_patch",
        "verification_strategy": "artifact",
        "verification_commands": [],
        "delegated_agents": [],
        "stop_conditions": [
            "README contains the requested link",
            "the local link resolves",
            "no blocking correction is outstanding",
        ],
        "confidence": 0.99,
        "escalation_reason": "",
        "unresolved_questions": [],
        "out_of_bounds": ["all files except README.md", "repository-wide search", "full pytest"],
    }


def test_exact_documentation_link_task_uses_minimal_motion(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\n## Documentation\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "17-telegram-connector.md").write_text("# Telegram connector\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "unrelated.py").write_text("raise AssertionError('must not read')\n", encoding="utf-8")

    class Worker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.requests.append(request)
            assert "MutationCommand" in request.question
            return ToolRunResponse(
                answer=json.dumps(
                    {
                        "tool_name": "apply_patch",
                        "tool_args": {
                            "patch": (
                                "*** Begin Patch\n*** Update File: README.md\n@@\n"
                                " ## Documentation\n"
                                "+\n+- [Telegram connector](docs/17-telegram-connector.md)\n"
                                "*** End Patch\n"
                            )
                        },
                    }
                ),
                trace=[],
            )

    worker = Worker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="17-telegram-connector.md add this to readme.md",
        requires_edit=True,
        target_files=["README.md"],
        execution_scope=_documentation_link_scope(),
    )

    assert result.run_status == "completed", (result.terminal_reason, result.answer, result.trace)
    assert result.changed_files == ["README.md"]
    assert len(worker.requests) == 1
    assert not (tmp_path / "docs" / "18-telegram.md").exists()
    assert "[Telegram connector](docs/17-telegram-connector.md)" in (tmp_path / "README.md").read_text(encoding="utf-8")
    tool_names = [str(row.get("tool_name") or "") for row in result.trace]
    assert "repo_batch_read" in tool_names
    assert "repo_search" not in tool_names
    assert "verify_project" not in tool_names
    assert "verify_changed_artifacts" in tool_names
    reads = next(row for row in result.trace if row.get("tool_name") == "repo_batch_read")
    assert reads["paths"] == ["README.md", "docs/17-telegram-connector.md"]
    assert all("unrelated" not in path for path in reads["paths"])
    decision = result.planner_decisions[0]
    assert decision["search_tools_called"] == []
    assert decision["mutation_tools_attempted"] == ["apply_patch"]
    assert decision["run_metrics"]["delegation_model_calls"] == 1
    assert decision["run_metrics"]["unique_file_reads"] == 2
    assert decision["run_metrics"]["verification_commands"] == 1
    assert decision["skip_full_pytest_reason"] == "README-only documentation change"


def test_evidence_ledger_canonicalizes_case_and_full_read_covers_chunks(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("one\ntwo\nthree\n", encoding="utf-8")
    ledger = EvidenceLedger(tmp_path)

    full, first_hit = ledger.read_file("./readme.md", purpose="parent")
    chunk, second_hit = ledger.read_file("README.md", start_line=2, end_line=2, purpose="child")

    assert first_hit is False
    assert second_hit is True
    assert chunk.evidence_id == full.evidence_id
    assert ledger.metrics.unique_file_reads == 1
    assert ledger.metrics.cache_hits == 1


def test_mutation_invalidates_only_changed_evidence(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 1\n", encoding="utf-8")
    ledger = EvidenceLedger(tmp_path)
    first_a, _ = ledger.read_file("a.py")
    first_b, _ = ledger.read_file("b.py")

    (tmp_path / "a.py").write_text("a = 2\n", encoding="utf-8")
    ledger.invalidate(["a.py"])
    second_a, a_hit = ledger.read_file("a.py")
    second_b, b_hit = ledger.read_file("b.py")

    assert a_hit is False and second_a.evidence_id != first_a.evidence_id
    assert b_hit is True and second_b.evidence_id == first_b.evidence_id


def test_invalid_scope_stops_before_tools(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    invalid = _documentation_link_scope()
    invalid["max_search_operations"] = 1

    with pytest.raises(ExecutionScopeDecisionError, match="No action executed"):
        validate_execution_scope(invalid, repo_root=tmp_path)


def test_dynamic_delegation_contains_only_bounded_task_context() -> None:
    request = DelegationRequest(
        user_goal="update one file",
        delegated_objective="produce one patch",
        known_repository_facts=["target exists"],
        canonical_target_paths=["src/app.py"],
        evidence_references=["ev_123"],
        unresolved_questions=[],
        allowed_tools=["apply_patch"],
        max_tool_calls=1,
        max_tokens=2000,
        out_of_bounds=["tests outside test_app.py"],
        expected_result="one mutation command",
        success_conditions=["target changed"],
        stop_conditions=["target changed"],
        parent_agent_id="agent_parent",
        task_id="task_child",
        root_task_id="task_root",
    )

    payload = json.loads(request.ephemeral_prompt())
    assert payload["max_tool_calls"] == 1
    assert payload["evidence_references"] == ["ev_123"]
    assert payload["parent_agent_id"] == "agent_parent"
    assert "repository_history" not in payload


def test_typed_agent_messages_are_bounded_and_carry_evidence(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path, max_messages_per_task=1)
    message = bus.send(
        task_id="task_1",
        root_task_id="root_1",
        from_agent_id="coding_1",
        to_agent_id="search_1",
        message_type=MessageType.SCOPE_ESCALATION,
        content="Need one symbol owner.",
        evidence_references=["ev_1"],
        confidence=0.7,
        requested_action="search the selected module only",
    )

    assert message.root_task_id == "root_1"
    assert message.evidence_references == ["ev_1"]
    assert message.requested_action == "search the selected module only"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        bus.send(
            task_id="task_1",
            from_agent_id="search_1",
            to_agent_id="coding_1",
            message_type=MessageType.EVIDENCE_FOUND,
            content="owner found",
        )


def test_child_scope_expansion_requires_one_parent_approved_level(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    current = validate_execution_scope(
        {
            **_documentation_link_scope(),
            "related_files": [],
            "max_unique_file_reads": 1,
        },
        repo_root=tmp_path,
    )
    request = {
        "current_decision_id": current.decision_id,
        "requested_level": 1,
        "requested_by_agent_id": "search_child",
        "missing_evidence": ["link destination ownership"],
        "evidence_references": ["ev_readme"],
        "reason": "the named destination did not resolve in the selected directory",
        "requested_search_operations": 1,
        "requested_unique_file_reads": 3,
    }

    escalated = approve_scope_escalation(current, request, approved_by_agent_id="parent")

    assert int(escalated.scope_level) == 1
    assert escalated.search_scope == "bounded"
    assert escalated.max_search_operations == 1
    with pytest.raises(ExecutionScopeDecisionError, match="exactly one level"):
        approve_scope_escalation(
            current,
            {**request, "requested_level": 2},
            approved_by_agent_id="parent",
        )


def test_cross_cutting_scope_retains_broad_discovery_and_full_verification(tmp_path: Path) -> None:
    decision = validate_execution_scope(
        {
            "decision_id": "scope_refactor",
            "task_type": "edit",
            "scope_level": 3,
            "complexity": "large",
            "risk": "high",
            "explicit_target_files": [],
            "related_files": [],
            "required_evidence": ["all public call sites", "affected tests"],
            "allowed_tool_families": ["search", "read", "symbols", "mutation", "verification", "agents"],
            "search_scope": "repository",
            "max_search_operations": 4,
            "max_unique_file_reads": 24,
            "mutation_strategy": "multi_file_patch",
            "verification_strategy": "full",
            "verification_commands": [["python", "-m", "pytest", "-q"]],
            "delegated_agents": ["search", "coding", "reviewer", "verifier"],
            "stop_conditions": ["all selected call sites migrated", "full verification passes"],
            "confidence": 0.82,
            "escalation_reason": "public API refactor crosses packages and call sites",
            "unresolved_questions": ["complete caller set"],
            "out_of_bounds": ["generated and vendor files"],
        },
        repo_root=tmp_path,
    )

    assert int(decision.scope_level) == 3
    assert decision.search_scope == "repository"
    assert decision.verification_strategy == "full"
    assert "reviewer" in decision.delegated_agents


def test_explicit_source_edit_runs_only_model_selected_nearest_check(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

    class Worker:
        def __init__(self) -> None:
            self.calls = 0

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.calls += 1
            return ToolRunResponse(
                answer=json.dumps(
                    {
                        "tool_name": "apply_patch",
                        "tool_args": {
                            "patch": (
                                "*** Begin Patch\n*** Update File: app.py\n@@\n"
                                "-VALUE = 1\n+VALUE = 2\n*** End Patch\n"
                            )
                        },
                    }
                ),
                trace=[],
            )

    scope = {
        "decision_id": "scope_source",
        "task_type": "edit",
        "scope_level": 0,
        "complexity": "small",
        "risk": "low",
        "explicit_target_files": ["app.py"],
        "related_files": [],
        "required_evidence": ["current assignment"],
        "allowed_tool_families": ["read", "mutation", "verification"],
        "search_scope": "none",
        "max_search_operations": 0,
        "max_unique_file_reads": 1,
        "mutation_strategy": "single_patch",
        "verification_strategy": "targeted",
        "verification_commands": [[str(Path(__import__("sys").executable)), "-m", "py_compile", "app.py"]],
        "delegated_agents": [],
        "stop_conditions": ["assignment updated", "selected syntax check passes"],
        "confidence": 0.98,
        "escalation_reason": "",
        "unresolved_questions": [],
        "out_of_bounds": ["all other files", "full pytest"],
    }
    worker = Worker()

    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="set app.py VALUE to 2",
        requires_edit=True,
        target_files=["app.py"],
        execution_scope=scope,
    )

    assert result.run_status == "completed", result.answer
    assert worker.calls == 1
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    verification = next(row for row in result.trace if row.get("tool_name") == "verify_selected_commands")
    assert verification["status"] == "ok"
    assert len(verification["checks"]) == 1
    assert all(row.get("tool_name") != "repo_search" for row in result.trace)
