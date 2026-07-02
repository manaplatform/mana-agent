from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.services.coding_memory_service import CodingMemoryService


@pytest.fixture
def memory_service(tmp_path: Path) -> CodingMemoryService:
    return CodingMemoryService(project_root=tmp_path, max_turns=5, max_tasks=20)


def test_extract_helpers_cover_constraints_acceptance_tasks_and_decisions() -> None:
    request = (
        "- only edit src/\n"
        "- do not touch migrations\n"
        "- success means tests pass\n"
        "- should include docstrings\n"
    )
    constraints = CodingMemoryService._extract_constraints(request)
    acceptance = CodingMemoryService._extract_acceptance(request)

    assert "only edit src/" in constraints
    assert "do not touch migrations" in constraints
    assert "success means tests pass" in acceptance
    assert "should include docstrings" in acceptance

    answer = (
        "Decision: Keep SQLite schema unchanged\n"
        "- [x] Added command wiring\n"
        "- [ ] Add regression tests\n"
    )
    done, open_tasks = CodingMemoryService._extract_tasks(answer)
    assert done == ["Added command wiring"]
    assert open_tasks == ["Add regression tests"]

    decisions = CodingMemoryService._extract_decisions(
        answer,
        warnings=[
            "mutation_failed_no_changes after patch mismatch",
            "patch-only loop detected; stopping retries",
            "mutation_failed_no_changes after patch mismatch",
        ],
    )
    decision_titles = [item["decision"] for item in decisions]
    assert "Keep SQLite schema unchanged" in decision_titles
    assert decision_titles.count("Stop after failed mutation") == 1
    assert "Stop patch-only retries" in decision_titles


def test_record_turn_and_build_flow_context_capture_expected_summary(memory_service: CodingMemoryService) -> None:
    flow_id = memory_service.ensure_flow(
        flow_id=None,
        request=(
            "Refactor coding flow persistence\n"
            "- only edit src/\n"
            "- success means no lint regressions\n"
        ),
    )

    memory_service.record_turn(
        flow_id=flow_id,
        user_request="Refactor coding flow persistence",
        effective_prompt="system prompt",
        agent_answer=(
            "Decision: Use warning-based patch fallback tracking\n"
            "- [x] Persist checklist snapshot\n"
            "- [ ] Add memory service regression tests\n"
        ),
        changed_files=["src/mana_agent/services/coding_memory_service.py"],
        warnings=["mutation_failed_no_changes after patch mismatch"],
        static_findings=[{"rule": "missing-docstring", "path": "src/mana_agent/services/index_service.py"}],
        checklist={
            "objective": "Flow persistence hardening",
            "steps": [
                {"status": "done", "title": "Persist checklist snapshot"},
                {"status": "blocked", "title": "Add regression tests"},
            ],
        },
        transitions=[
            {"from_phase": "discover", "to_phase": "edit", "reason": "schema confirmed"},
            {"from_phase": "edit", "to_phase": "blocked", "reason": "awaiting test updates"},
        ],
    )

    summary = memory_service.get_flow_summary(flow_id)
    assert summary is not None
    assert summary.objective.startswith("Refactor coding flow persistence")
    assert "only edit src/" in summary.constraints
    assert "success means no lint regressions" in summary.acceptance
    assert summary.open_tasks == ["Add memory service regression tests"]
    assert "src/mana_agent/services/coding_memory_service.py" in summary.last_changed_files
    assert any("missing-docstring" in item for item in summary.unresolved_static_findings)
    assert summary.last_blocked_reason == "awaiting test updates"
    assert isinstance(summary.checklist, dict)
    assert isinstance(summary.transitions, list)

    decision_titles = [item["decision"] for item in summary.recent_decisions]
    assert "Use warning-based patch fallback tracking" in decision_titles
    assert "Stop after failed mutation" in decision_titles

    context = memory_service.build_flow_context(flow_id, repo_delta_paths=["README.md"])
    assert f"Flow ID: {flow_id}" in context
    assert "Locked constraints:" in context
    assert "Open tasks:" in context
    assert "Current checklist:" in context
    assert "Last blocked reason: awaiting test updates" in context
    assert "Current repository delta paths:" in context
    assert "- README.md" in context


def test_patch_failure_and_conflict_heuristics(memory_service: CodingMemoryService) -> None:
    flow_id = memory_service.ensure_flow(
        flow_id=None,
        request="Refactor parser prompt plumbing and update checklist handling",
    )
    memory_service.record_turn(
        flow_id=flow_id,
        user_request="Refactor parser prompt plumbing",
        effective_prompt="system prompt",
        agent_answer="- [ ] Follow-up",
        changed_files=[],
        warnings=["patch-only loop detected; forcing fallback"],
        static_findings=[],
    )
    assert memory_service.has_prior_patch_failures(flow_id) is True

    flow_without_failures = memory_service.ensure_flow(
        flow_id="flow-no-prior-patch-failure",
        request="Improve report output rendering",
    )
    memory_service.record_turn(
        flow_id=flow_without_failures,
        user_request="Improve report output rendering",
        effective_prompt="system prompt",
        agent_answer="- [ ] Follow-up",
        changed_files=[],
        warnings=["normal warning without patch retry signals"],
        static_findings=[],
    )
    assert memory_service.has_prior_patch_failures(flow_without_failures) is False

    assert (
        memory_service.is_conflicting_request(
            flow_id,
            "Add billing invoice endpoint and checkout workflow support",
        )
        is True
    )
    assert memory_service.is_conflicting_request(flow_id, "implement plan.") is False
    assert memory_service.is_conflicting_request(flow_id, "Can you summarize current progress?") is False


def test_build_flow_context_returns_empty_for_unknown_flow(memory_service: CodingMemoryService) -> None:
    assert memory_service.build_flow_context("missing-flow", repo_delta_paths=["src/app.py"]) == ""
