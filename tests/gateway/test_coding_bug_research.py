"""Production coding adapter delivery, steering and Activity regression coverage."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mana_agent.chat.activity import project_activity
from mana_agent.chat.execution_trace import ExecutionTrace, classify_activity_category
from mana_agent.execution.autonomous_controller import AutonomousExecutionController
from mana_agent.execution.continuation_evaluator import ContinuationEvaluator
from mana_agent.gateway.config import ChatGatewayConfig
from mana_agent.gateway.turn_engine import process_chat_turn
from mana_agent.multi_agent.routing.agent_decision import AgentDecision


def run_turn(case, agent, **overrides):
    controller = AutonomousExecutionController(supervisor=case.supervisor, taskboard=case.board,
                                               evaluator=ContinuationEvaluator(case.model))
    original = controller.coding_research_runtime
    def runtime(**kwargs):
        return original(**kwargs, router=case.router)
    controller.coding_research_runtime = runtime
    args = dict(root=agent.working_directory, text=case.goal, chat_service=SimpleNamespace(), ask_service=SimpleNamespace(),
        coding_agent=agent, config=ChatGatewayConfig().normalized(), session_state={},
        gateway_task_id=case.item.task_id, supervisor_attempt_id=case.task.attempt_id, supervisor_lease_token=case.lease,
        autonomous_controller=controller, continuation_evaluator=controller.evaluator,
        coding_research_authorizer=lambda tool: None, coding_research_settings=case.settings,
        agent_decision=AgentDecision(intent="edit", code_editing_needed=True, selected_tools=["apply_patch"],
            confidence=1.0, verifier_passed=True, reasoning_summary="Resolve the coding bug", flow_action="none"))
    args.update(overrides)
    return process_chat_turn(**args)


@pytest.mark.parametrize("backend", ["internal", "external"])
def test_research_returns_to_same_task_backend_and_thread(coding_research_case, tmp_path, backend):
    c = coding_research_case
    class Agent:
        supports_gateway_task_identity = True
        active_engine_mode = backend
        working_directory = tmp_path
        resume_thread_id = "original-thread"
        def __init__(self):
            self.calls = []
        def generate(self, request, **kwargs):
            self.calls.append((request, kwargs))
            if len(self.calls) == 1:
                return {**c.result, "backend": backend, "thread_id": "original-thread", "continuation_context": {"commands_run": ["already completed"]}}
            assert "Main Model continuation" in request
            assert "migration" in request
            assert kwargs["gateway_task_id"] == c.item.task_id
            assert kwargs["coding_resume_context"]["commands_run"] == ["already completed"]
            return {"status": "completed", "answer": "Applied researched fix", "changed_files": ["app.py"], "thread_id": "original-thread"}
        generate_auto_execute = generate
    agent = Agent()
    c.model.responses = [c.decision(research_plan=c.plan), c.decision(research_recommendation={
        "summary": "Inspect the migration guide and match widget 2.0", "source_urls": ["https://docs.example.org/migration"], "applicability": "uncertain"}),
        c.decision("FINALIZE")]
    result = run_turn(c, agent)
    assert len(agent.calls) == 2
    assert agent.calls[1][1]["coding_continuation_id"].startswith("research-")
    assert len(c.router.calls) == 1
    assert agent.resume_thread_id == "original-thread"
    assert result.used_coding_agent


def test_invalid_main_decision_stops_without_another_coding_attempt(coding_research_case, tmp_path):
    c = coding_research_case
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return c.result
    agent = SimpleNamespace(working_directory=tmp_path, supports_gateway_task_identity=True, generate=generate)
    c.model.responses = [{"intent": "research"}]
    result = run_turn(c, agent)
    assert len(calls) == 1
    assert not c.router.calls
    assert result.error_code == "CODING_RESEARCH_DECISION_INVALID"
    assert result.retry_possible is False
    assert result.payload["pending_required_work"]


def test_live_steering_is_consumed_before_coding_resumes(coding_research_case, tmp_path):
    c = coding_research_case
    state = {}
    calls = []
    def generate(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            state["pending_steering"] = [{"status": "pending", "kind": "guidance", "text": "Preserve the existing public API", "request_id": "steer-test"}]
            return c.result
        assert "Preserve the existing public API" in request
        return {"status": "completed", "answer": "Implemented", "changed_files": ["app.py"]}
    agent = SimpleNamespace(working_directory=tmp_path, supports_gateway_task_identity=True, generate=generate)
    c.model.responses = [c.decision(remaining_work="Continue local debugging"), c.decision("FINALIZE")]
    run_turn(c, agent, session_state=state)
    assert len(calls) == 2
    assert state["pending_steering"][0]["status"] == "applied"


def test_research_activity_is_public_and_stays_in_coding_box():
    kind = "investigation.coding_research.finding"
    projected = project_activity({"type": kind, "title": "Sourced recommendation", "task_id": "task", "execution_id": "turn",
                                  "agent_id": "main", "backend": "external", "operation_id": "research-op",
                                  "status": "success", "metadata": {"phase": "coding", "evidence_status": "unverified"}})
    assert projected is not None
    assert classify_activity_category(kind, metadata={"backend": "external"}) == "coding"
    trace = ExecutionTrace("turn")
    trace.apply_event(projected)
    assert len(trace.coding_events) == 1
    assert not trace.main_events


def test_inspection_uses_only_observed_sources_and_no_mutation_tools(coding_research_case):
    c = coding_research_case
    inspected = []
    def inspect(url):
        inspected.append(url)
        return {"ok": True, "url": url, "text": "Version 2 migration guide"}
    inspect_plan = {"objective": "Inspect version applicability", "inspect_urls": ["https://docs.example.org/migration"]}
    c.model.responses = [c.decision(research_plan=c.plan), c.decision(research_plan=inspect_plan), c.decision()]
    runtime = c.runtime(inspector=inspect)
    runtime.decide(goal=c.goal, result=c.result)
    assert inspected == ["https://docs.example.org/migration"]
    assert next(iter(runtime.state["inspections"].values()))["evidence_kind"] == "source_page"


def test_unknown_source_cannot_be_opened(coding_research_case):
    from mana_agent.coding.research import CodingResearchError
    c = coding_research_case
    c.model.responses = [c.decision(research_plan={"objective": "Read unseen source", "inspect_urls": ["https://unseen.example.org/doc"]})]
    with pytest.raises(CodingResearchError, match="observed"):
        c.runtime(inspector=lambda _: pytest.fail("must not open an unseen source")).decide(goal=c.goal, result=c.result)


def test_main_model_wait_uses_durable_human_input_callback(coding_research_case, tmp_path):
    c = coding_research_case
    agent = SimpleNamespace(working_directory=tmp_path, supports_gateway_task_identity=True,
                            generate=lambda *args, **kwargs: c.result)
    c.model.responses = [c.decision("WAIT_FOR_USER", user_message="Which supported widget release must remain compatible?")]
    requests = []
    def wait(question, decision_id):
        requests.append((question, decision_id))
        return "durable-inbox-item"
    result = run_turn(c, agent, coding_research_waiter=wait)
    assert len(requests) == 1
    assert result.payload["permission_request_id"] == "durable-inbox-item"
    assert result.payload["status"] == "waiting"
    assert result.payload["goal_satisfied"] is False


def test_parent_task_tool_budget_remains_enforced(coding_research_case):
    from mana_agent.coding.research import CodingResearchError
    c = coding_research_case
    c.supervisor.store.update_task(c.item.task_id, lambda task: setattr(task, "autonomous_tool_calls", 1))
    c.model.responses = [c.decision(research_plan=c.plan)]
    with pytest.raises(CodingResearchError, match="Parent task tool budget"):
        c.runtime(max_tool_calls=1).decide(goal=c.goal, result=c.result)
    assert not c.router.calls


def test_researched_fix_still_runs_the_existing_integration_gate(coding_research_case, tmp_path, monkeypatch):
    c = coding_research_case
    calls, gate_calls = [], []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return c.result if len(calls) == 1 else {"status": "completed", "answer": "Applied fix", "changed_files": ["app.py"]}
    def gate(**kwargs):
        gate_calls.append(kwargs)
        return SimpleNamespace(passed=False, error_code="INCOMPLETE_FEATURE_WIRING", result=kwargs["core_result"])
    monkeypatch.setattr("mana_agent.gateway.turn_engine.FeatureIntegrationCoordinator", lambda **_: SimpleNamespace(run=gate))
    agent = SimpleNamespace(working_directory=tmp_path, supports_gateway_task_identity=True, generate=generate)
    c.model.responses = [c.decision(research_plan=c.plan), c.decision(), c.decision("FINALIZE")]
    result = run_turn(c, agent)
    assert len(calls) == 2 and len(gate_calls) == 1
    assert result.error_code == "INCOMPLETE_FEATURE_WIRING"
    assert gate_calls[0]["gateway_task_id"] == c.item.task_id
