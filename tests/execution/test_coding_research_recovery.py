"""Durable research boundaries must not replay unknown coding effects."""
from __future__ import annotations

from copy import deepcopy

import pytest

from mana_agent.coding.research import CodingResearchError, CodingResearchReconciliationRequired
from mana_agent.execution.continuation_decision import ContinuationDecision
from mana_agent.execution_supervisor.errors import StaleLeaseError


def test_saved_search_reservation_is_consumed_without_repeating_network(coding_research_case):
    c = coding_research_case
    c.model.responses = [c.decision(research_plan=c.plan), c.decision()]
    runtime = c.runtime()
    runtime.decide(goal=c.goal, result=c.result)
    key = next(iter(runtime.state["queries"]))
    runtime.state["queries"][key].update(status="reserved", results=[])
    runtime._save("simulate_crash_after_search_reservation")
    c.model.responses = [c.decision(research_plan=c.plan), c.decision("BLOCKED")]
    resumed = c.runtime()
    resumed.decide(goal=c.goal, result=c.result)
    assert len(c.router.calls) == 1
    assert resumed.state["searches_used"] == 1
    assert resumed.state["queries"][key]["status"] == "uncertain"


def test_recovery_cannot_enlarge_saved_limits(coding_research_case):
    c = coding_research_case
    c.settings.mana_coding_research_max_searches = 1
    c.settings.mana_coding_research_max_rounds = 1
    c.model.responses = [c.decision()]
    c.runtime().decide(goal=c.goal, result=c.result)
    c.settings.mana_coding_research_max_searches = 12
    c.settings.mana_coding_research_max_rounds = 4
    resumed = c.runtime()
    assert resumed.state["max_searches"] == resumed.state["max_rounds"] == 1


def test_completed_search_results_survive_new_runtime(coding_research_case):
    c = coding_research_case
    c.model.responses = [c.decision(research_plan=c.plan), c.decision()]
    first = c.runtime()
    first.decide(goal=c.goal, result=c.result)
    c.model.responses = [c.decision(research_plan=c.plan), c.decision()]
    second = c.runtime()
    second.decide(goal=c.goal, result=c.result)
    assert len(c.router.calls) == 1
    assert c.board.get_task(c.item.task_id).coding_research_state["searches_used"] == 1
    checkpoint = c.supervisor.store.get_checkpoint(c.supervisor.get_task(c.item.task_id).checkpoint_id)
    assert checkpoint.resume_payload["coding_research"]["searches_used"] == 1


def test_started_delivery_requires_reconciliation(coding_research_case):
    c = coding_research_case
    runtime = c.runtime()
    decision = ContinuationDecision.model_validate(c.decision())
    ticket = runtime.prepare_delivery(decision, c.result)["continuation_id"]
    assert runtime.begin_delivery(ticket) is None
    resumed = c.runtime()
    with pytest.raises(CodingResearchReconciliationRequired):
        resumed.begin_delivery(ticket)
    with pytest.raises(CodingResearchReconciliationRequired):
        resumed.prepare_delivery(decision, c.result)
    assert not c.router.calls


def test_delivered_coding_result_reuses_receipt_without_reexecuting(coding_research_case):
    c = coding_research_case
    runtime = c.runtime()
    ticket = runtime.prepare_delivery(ContinuationDecision.model_validate(c.decision()), c.result)["continuation_id"]
    runtime.begin_delivery(ticket)
    result = {"status": "completed", "changed_files": ["app.py"], "commands_run": ["authorized command"], "thread_id": "original-thread"}
    runtime.acknowledge_delivery(ticket, result)
    reused = c.runtime().begin_delivery(ticket)
    assert reused["changed_files"] == ["app.py"]
    assert reused["thread_id"] == "original-thread"
    assert c.supervisor.get_task(c.item.task_id).state.value == "running"


def test_batch_suspension_reuses_same_continuation_ticket(coding_research_case):
    c = coding_research_case
    runtime = c.runtime()
    ticket = runtime.prepare_delivery(ContinuationDecision.model_validate(c.decision()), c.result)["continuation_id"]
    runtime.begin_delivery(ticket)
    runtime.acknowledge_delivery(ticket, {"status": "waiting", "pending_operation": {"operation_id": "batch-operation"}})
    resumed = c.runtime()
    assert resumed.state["delivery"]["status"] == "suspended"
    assert resumed.begin_delivery(ticket) is None
    assert resumed.state["delivery"]["id"] == ticket


def test_pending_model_request_is_exactly_restored(coding_research_case):
    c = coding_research_case
    from mana_agent.batch.policy import BatchPending, PendingBatchOperation
    c.model.responses = [BatchPending(PendingBatchOperation(operation_id="main-decision", batch_id="batch", task_id=c.item.task_id, phase="research"))]
    runtime = c.runtime()
    with pytest.raises(BatchPending):
        runtime.decide(goal=c.goal, result=c.result)
    first_request = c.model.calls[0][-1].content
    c.model.responses = [c.decision()]
    c.runtime().decide(goal=c.goal, result=c.result)
    assert c.model.calls[1][-1].content == first_request
    assert c.supervisor.get_task(c.item.task_id).provider_metadata["coding_research"]["decisions_used"] == 1


def test_cancel_during_search_discards_late_evidence(coding_research_case):
    c = coding_research_case
    original = c.router.run
    def cancel_during_request(**kwargs):
        result = original(**kwargs)
        c.supervisor.cancel(c.item.task_id, reason="user cancelled")
        return result
    c.router.run = cancel_during_request
    c.model.responses = [c.decision(research_plan=c.plan)]
    runtime = c.runtime()
    with pytest.raises((CodingResearchError, StaleLeaseError)):
        runtime.decide(goal=c.goal, result=c.result)
    saved = c.supervisor.get_task(c.item.task_id).provider_metadata["coding_research"]
    assert all(not row["results"] for row in saved["queries"].values())
    assert not saved["findings"]


def test_stale_attempt_cannot_publish_or_deliver(coding_research_case):
    c = coding_research_case
    runtime = c.runtime()
    c.supervisor.store.update_task(c.item.task_id, lambda task: setattr(task, "attempt_id", "replacement-attempt"))
    with pytest.raises(StaleLeaseError):
        runtime.prepare_delivery(ContinuationDecision.model_validate(c.decision()), c.result)
    assert not c.router.calls


def test_research_preserves_completed_children_and_task_identity(coding_research_case):
    c = coding_research_case
    child = c.board.create_child_task(c.item.task_id, title="Completed local inspection", user_request="Inspect initializer")
    c.board.mark_completed(child.task_id, evidence="Local source inspected")
    before = deepcopy(c.board.get_task(child.task_id))
    c.model.responses = [c.decision(research_plan=c.plan), c.decision("BLOCKED")]
    c.runtime().decide(goal=c.goal, result=c.result)
    after = c.board.get_task(child.task_id)
    assert after == before
    assert after.parent_task_id == c.item.task_id
    assert len(c.board.tasks) == 2


def test_research_cannot_switch_backends(coding_research_case):
    c = coding_research_case
    runtime = c.runtime(backend="internal")
    runtime._save("selected_internal")
    with pytest.raises(CodingResearchError, match="backend changed"):
        c.runtime(backend="external")


def test_matching_result_escrow_recovers_started_delivery(coding_research_case):
    from mana_agent.execution_supervisor.models import EscrowResult
    c = coding_research_case
    runtime = c.runtime()
    ticket = runtime.prepare_delivery(ContinuationDecision.model_validate(c.decision()), c.result)["continuation_id"]
    runtime.begin_delivery(ticket)
    receipt = EscrowResult(task_id=c.item.task_id, attempt_id=c.task.attempt_id, attempt_generation=c.task.attempt_generation,
        payload={"status": "completed", "changed_files": ["app.py"], "continuation_context": {"coding_continuation_id": ticket}})
    c.supervisor.store.save_result(receipt)
    resumed = c.runtime()
    assert resumed.begin_delivery(ticket)["changed_files"] == ["app.py"]
    assert resumed.state["delivery"]["result_escrow_reference"] == receipt.result_id
