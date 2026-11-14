from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mana_agent.builtin_skills.skill_creator import (
    ExperienceEvaluator,
    ExperienceRecord,
    ExperienceWorkshopHook,
    ProposalStorage,
    SkillCreator,
    SkillDraft,
    WorkshopConfig,
    WorkshopPaths,
)
from mana_agent.commands.cli import app
from mana_agent.services.execution_event_hub import reset_execution_event_hub_for_tests


def config() -> WorkshopConfig:
    return WorkshopConfig()


@pytest.fixture
def paths(tmp_path: Path) -> WorkshopPaths:
    return WorkshopPaths(tmp_path / "skills", tmp_path / "proposals", tmp_path / "quarantine")


@pytest.fixture
def storage(paths: WorkshopPaths) -> ProposalStorage:
    return ProposalStorage(paths=paths, config=config())


def experience(**updates) -> ExperienceRecord:
    values = {
        "session_id": "session_20260716_0012",
        "task_id": "task_20260716_0004",
        "summary": "Implement a transaction-safe approval workflow.",
        "result": "Approval is protected against concurrent duplicates.",
        "workflow_steps": ["Locate mutation boundary", "Add lock", "Verify concurrency"],
        "decisions": [{"decision": "use atomic locking", "reason": "prevent races"}],
        "tool_calls": [{"tool": "apply_patch", "status": "ok"}],
        "changed_files": ["src/orders/service.py", "tests/test_orders.py"],
        "verification_commands": ["pytest tests/test_orders.py"],
        "verification_results": [{"command": "pytest tests/test_orders.py", "passed": True, "tests_passed": 7}],
        "verification_passed": True,
        "successful_runs": 1,
        "user_accepted": True,
        "reusable_trigger_present": True,
        "deterministic_verification": True,
        "repository_specificity": "low",
    }
    values.update(updates)
    return ExperienceRecord(**values)


def draft(**updates) -> SkillDraft:
    values = {
        "name": "transaction-safe-approval",
        "display_name": "Transaction-Safe Approval",
        "description": "Safely implement approval mutations with locking and duplicate prevention.",
        "triggers": ["Implement approval under concurrent updates."],
        "required_tools": ["repo_search", "repo_read", "apply_patch", "run_tests"],
        "required_permissions": ["repository_read", "repository_write", "command_execution"],
        "risk_level": "medium",
        "risk_reasons": ["Changes transaction-sensitive logic."],
        "purpose": "Prevent duplicate state transitions under concurrency.",
        "when_to_use": ["A state transition can be attempted concurrently."],
        "when_not_to_use": ["The operation is read-only."],
        "preconditions": ["The persistence layer supports atomic transactions and row locking."],
        "procedure": [
            "Locate the mutation boundary and mutable state checks.",
            "Wrap the read-and-write sequence in an atomic transaction and lock the row before validating state.",
            "Add focused tests for sequential duplicates and concurrent attempts.",
        ],
        "safety_constraints": ["Keep external side effects outside a retried transaction."],
        "verification": ["Relevant tests pass.", "Concurrent attempts cannot both succeed."],
        "failure_recovery": ["Roll back the focused change and retain the failing concurrency test."],
    }
    values.update(updates)
    return SkillDraft(**values)


def create(storage: ProposalStorage, **experience_updates):
    creator = SkillCreator(config=config(), storage=storage)
    return creator.create_from_draft(experience(**experience_updates), draft())


def test_eligible_successful_task_creates_complete_pending_proposal(storage: ProposalStorage) -> None:
    result = create(storage)
    assert result.decision.eligible is True
    assert result.proposal is not None and result.proposal.status == "pending_review"
    assert result.path is not None
    assert {item.name for item in result.path.iterdir()} == {"proposal.yaml", "SKILL.md", "evidence.json", "validation.json", "README.md"}
    assert "# Safety constraints" in (result.path / "SKILL.md").read_text(encoding="utf-8")
    assert not any(storage.paths.skills.iterdir())


def test_simple_or_failed_verification_is_ineligible() -> None:
    evaluator = ExperienceEvaluator(config())
    simple = experience(workflow_steps=["answer"], changed_files=[], verification_passed=False, verification_results=[])
    decision = evaluator.evaluate(simple)
    assert decision.eligible is False
    assert any("multi-step" in reason for reason in decision.reasons)
    assert any("verification" in reason.lower() for reason in decision.reasons)


def test_confidence_weights_corrections_and_thresholds() -> None:
    evaluator = ExperienceEvaluator(config())
    accepted = evaluator.evaluate(experience())
    corrected = evaluator.evaluate(experience(unresolved_corrections=True))
    attention = evaluator.evaluate(experience(user_accepted=False))
    assert accepted.confidence == 0.85
    assert corrected.confidence == 0.55 and corrected.eligible is False
    assert attention.status == "needs_attention" and attention.confidence == 0.65


def test_secret_evidence_is_redacted_before_write(storage: ProposalStorage) -> None:
    result = create(storage, decisions=[{"decision": "use token", "api_key": "sk-super-secret-value"}])
    text = (result.path / "evidence.json").read_text(encoding="utf-8")
    assert "super-secret" not in text
    assert "[REDACTED]" in text


def test_secret_evidence_is_redacted_before_model_generation(storage: ProposalStorage) -> None:
    seen: list[ExperienceRecord] = []

    class CapturingGenerator:
        def generate(self, recorded, decision, **_native_options):
            seen.append(recorded)
            return draft()

    creator = SkillCreator(config=config(), storage=storage)
    result = creator.create(
        experience(decisions=[{"decision": "authenticate", "access_token": "private-token-value"}]),
        CapturingGenerator(),
    )
    assert result.path is not None
    assert seen[0].decisions[0]["access_token"] == "[REDACTED]"


def test_invalid_names_permissions_and_recursive_creator_are_rejected() -> None:
    with pytest.raises(ValidationError):
        draft(name="../escape")
    with pytest.raises(ValidationError):
        draft(name="skill-creator")
    with pytest.raises(ValidationError):
        draft(required_permissions=["root_access"])


def test_duplicate_pending_proposal_merges_evidence(storage: ProposalStorage) -> None:
    first = create(storage)
    second_experience = experience(session_id="session_2", task_id="task_2", successful_runs=2)
    second = SkillCreator(config=config(), storage=storage).create_from_draft(second_experience, draft())
    assert second.merged_into == first.proposal.proposal_id
    _path, manifest, evidence_row, _report, _markdown = storage.load(first.proposal.proposal_id)
    assert manifest.successful_runs == 3
    assert evidence_row.session_ids == ["session_20260716_0012", "session_2"]


def test_duplicate_active_skill_records_evidence_without_new_proposal(storage: ProposalStorage) -> None:
    active = storage.paths.skills / "transaction-safe-approval"
    active.mkdir()
    (active / "SKILL.md").write_text("transaction safe approval concurrent locking", encoding="utf-8")
    result = create(storage)
    assert result.merged_into == "active:transaction-safe-approval"
    assert not list(storage.paths.proposals.glob("skill_proposal_*"))
    assert list((storage.paths.proposals / "duplicate-evidence" / "transaction-safe-approval").glob("*.json"))


def test_concurrent_generation_produces_one_proposal(storage: ProposalStorage) -> None:
    def worker():
        return SkillCreator(config=config(), storage=storage).create_from_draft(experience(), draft())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: worker(), range(2)))
    assert len(list(storage.paths.proposals.glob("skill_proposal_*"))) == 1
    assert sum(result.path is not None for result in results) == 1
    assert sum(result.merged_into is not None for result in results) == 1


def test_install_revalidates_preserves_provenance_and_wont_overwrite(storage: ProposalStorage) -> None:
    result = create(storage)
    target = storage.install(result.proposal.proposal_id, approved=True)
    assert (target / "SKILL.md").exists()
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["proposal_id"] == result.proposal.proposal_id
    assert (target / "versions" / "1.0.0" / "provenance.json").exists()
    assert storage.load(result.proposal.proposal_id)[1].status == "installed"
    with pytest.raises(ValueError):
        storage.install(result.proposal.proposal_id, approved=True)


def test_install_requires_explicit_approval(storage: ProposalStorage) -> None:
    result = create(storage)
    with pytest.raises(PermissionError):
        storage.install(result.proposal.proposal_id, approved=False)


def test_install_cannot_overwrite_active_skill(storage: ProposalStorage) -> None:
    result = create(storage)
    active = storage.paths.skills / result.proposal.name
    active.mkdir()
    (active / "SKILL.md").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        storage.install(result.proposal.proposal_id, approved=True)


def test_edit_revalidates_and_resets_status(storage: ProposalStorage) -> None:
    result = create(storage)
    updated = SkillCreator(config=config(), storage=storage).edit(
        result.proposal.proposal_id,
        draft=draft(description="Updated reusable transaction procedure."),
    )
    assert updated.status == "needs_attention"
    assert storage.load(updated.proposal_id)[2].task_ids == ["task_20260716_0004"]


def test_reject_retains_metadata_and_quarantine_is_never_active(storage: ProposalStorage) -> None:
    rejected = create(storage)
    storage.reject(rejected.proposal.proposal_id, "too broad")
    assert storage.load(rejected.proposal.proposal_id)[1].status == "rejected"
    other = SkillCreator(config=config(), storage=storage).create_from_draft(
        experience(session_id="s-other", task_id="t-other"),
        draft(name="other-safe-workflow", display_name="Other Safe Workflow"),
    )
    path = storage.quarantine(other.proposal.proposal_id, "manual safety review")
    assert path.parent == storage.paths.quarantine
    assert not (storage.paths.skills / "other-safe-workflow").exists()


def test_recursive_source_is_ineligible() -> None:
    decision = ExperienceEvaluator(config()).evaluate(experience(source_component="skill_creator"))
    assert decision.eligible is False
    assert any("Recursive" in reason for reason in decision.reasons)


def test_invalid_generated_content_is_quarantined(storage: ProposalStorage) -> None:
    unsafe = draft(description="Use api_key=sk-this-is-secret-material when running.")
    result = SkillCreator(config=config(), storage=storage).create_from_draft(experience(), unsafe)
    assert result.proposal.status == "quarantined"
    assert result.path.parent == storage.paths.quarantine
    assert not (storage.paths.skills / unsafe.name).exists()


def test_workshop_failure_does_not_change_original_task_success(storage: ProposalStorage) -> None:
    class BrokenGenerator:
        def generate(self, experience, decision, **_native_options):
            raise RuntimeError("model unavailable")
    hook = ExperienceWorkshopHook(SkillCreator(config=config(), storage=storage))
    result = hook.run(experience(), generator=BrokenGenerator(), original_task_succeeded=True)
    assert result.original_task_succeeded is True
    assert "model unavailable" in result.warning


def test_configuration_disabled_behavior(storage: ProposalStorage) -> None:
    disabled = WorkshopConfig(enabled=False)
    result = SkillCreator(config=disabled, storage=storage).create_from_draft(experience(), draft())
    assert result.decision.eligible is False
    assert not list(storage.paths.proposals.glob("skill_proposal_*"))


def test_cli_list_show_reject_and_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_SKILLS_ROOT", str(tmp_path / "skills"))
    monkeypatch.setenv("MANA_SKILL_PROPOSALS_ROOT", str(tmp_path / "proposals"))
    monkeypatch.setenv("MANA_SKILL_QUARANTINE_ROOT", str(tmp_path / "quarantine"))
    cli_storage = ProposalStorage(config=WorkshopConfig.load())
    created = SkillCreator(config=WorkshopConfig.load(), storage=cli_storage).create_from_draft(experience(), draft())
    runner = CliRunner()
    listed = runner.invoke(app, ["skill", "proposals", "--status", "pending_review", "--min-confidence", "0.8"])
    assert listed.exit_code == 0
    assert created.proposal.proposal_id in listed.stdout
    shown = runner.invoke(app, ["skill", "proposal", "show", created.proposal.proposal_id])
    assert shown.exit_code == 0 and "# Procedure" in shown.stdout
    rejected = runner.invoke(app, ["skill", "proposal", "reject", created.proposal.proposal_id, "--reason", "too broad"])
    assert rejected.exit_code == 0
    assert cli_storage.load(created.proposal.proposal_id)[1].status == "rejected"


def test_cli_install_and_edit_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_SKILLS_ROOT", str(tmp_path / "skills"))
    monkeypatch.setenv("MANA_SKILL_PROPOSALS_ROOT", str(tmp_path / "proposals"))
    monkeypatch.setenv("MANA_SKILL_QUARANTINE_ROOT", str(tmp_path / "quarantine"))
    cli_storage = ProposalStorage(config=WorkshopConfig.load())
    created = SkillCreator(config=WorkshopConfig.load(), storage=cli_storage).create_from_draft(experience(), draft())
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(draft(description="Edited description.").model_dump_json(), encoding="utf-8")
    runner = CliRunner()
    edited = runner.invoke(app, ["skill", "proposal", "edit", created.proposal.proposal_id, "--draft-file", str(draft_path)])
    assert edited.exit_code == 0
    installed = runner.invoke(app, ["skill", "proposal", "install", created.proposal.proposal_id])
    assert installed.exit_code == 0
    assert (tmp_path / "skills" / "transaction-safe-approval" / "SKILL.md").exists()


def test_structured_lifecycle_events_are_published(storage: ProposalStorage) -> None:
    hub = reset_execution_event_hub_for_tests()
    events: list[dict] = []
    unsubscribe = hub.subscribe_all(events.append)
    try:
        created = create(storage)
        storage.reject(created.proposal.proposal_id, "not reusable")
    finally:
        unsubscribe()
    raw_types = {str(item.get("metadata", {}).get("raw_kind") or "") for item in events}
    assert "skill_proposal_rejected" in raw_types
