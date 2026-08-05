from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mana_agent.gateway import (
    AgentChatGateway,
    EntryRouteRegistry,
    EntryRouter,
    RouteAvailability,
    RouteRegistration,
)
from mana_agent.gateway.entry_routing import (
    ENTRY_ROUTER_PROMPT,
    EntryRouteContext,
    EntryRoutingDecision,
    EntryRoutingError,
    EntryRoutingOutput,
)
from mana_agent.gateway.entry_routing import gmail_route_availability
from mana_agent.gateway.checkpoint_resume import (
    CHECKPOINT_RESUME_PROMPT,
    CheckpointResumeDecision,
    CheckpointResumeError,
)
from mana_agent.gateway.followup_classifier import (
    FollowupClassification,
    FollowupClassificationError,
)
from mana_agent.gateway.lanes import LaneId, LaneTaskState
from mana_agent.memory import MemoryContent
from mana_agent.multi_agent.routing.agent_decision import AgentDecision
from mana_agent.workspaces.service import WorkspaceService


class _RouteModel:
    def __init__(self, *routes: str) -> None:
        self.routes = list(routes)
        self.payloads: list[dict[str, Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        if messages[0].content == CHECKPOINT_RESUME_PROMPT:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "decision_id": "test-start-fresh",
                        "action": "start_fresh",
                        "task_id": "",
                        "checkpoint_id": "",
                        "same_work": False,
                        "fresh_data_required": True,
                        "checkpoint_still_valid": False,
                        "side_effects_safe_to_repeat": False,
                        "safe_to_continue": True,
                        "reason": "Gmail data must be fetched again.",
                    }
                )
            )
        self.payloads.append(json.loads(messages[-1].content))
        route = self.routes.pop(0) if self.routes else "conversation"
        source_by_route = {
            "conversation": ["none"], "unsupported": ["none"], "coding": ["repository"],
            "mcp": ["mcp"],
            "gmail": ["gmail"], "calendar": ["calendar"], "browser": ["browser"],
            "search": ["search"], "github": ["github"], "repository": ["repository"],
            "memory": ["memory"], "automation": ["repository"], "api": ["api"],
            "canvas": ["canvas"],
            "artifact": ["artifact"],
            "capability_error": ["gmail"],
        }
        return SimpleNamespace(
            content=json.dumps(
                {
                    "route": route,
                    "confidence": 0.98,
                    "reason": f"selected {route}",
                    "required_sources": source_by_route.get(route, ["none"]),
                    "target_urls": ["https://example.com"] if route == "browser" else [],
                    "requires_live_data": route in {"browser", "search", "github", "mcp"},
                    "reason_code": "TEST_ROUTE",
                    "error_code": "GMAIL_NOT_AVAILABLE" if route == "capability_error" else "",
                    "reuse_active_route": len(self.payloads) > 1,
                    "artifact_family": "pdf" if route == "artifact" else "",
                    "automation_operation": "create" if route == "automation" else "",
                    "mcp_request": {"provider_id": "kaggle"} if route == "mcp" else None,
                }
            )
        )


class _AskAgent:
    def __init__(self, response: Any | None = None) -> None:
        self.response = response or SimpleNamespace(
            answer="Latest Gmail: Subject: hello",
            sources=[],
            warnings=[],
            trace=[{"tool_name": "email_search", "status": "ok"}],
        )
        self.calls: list[dict[str, Any]] = []
        self.project_roots: list[Path] = []

    def run(self, **kwargs: Any) -> Any:
        if hasattr(self, "project_root"):
            self.project_roots.append(Path(self.project_root).resolve())
        self.calls.append(kwargs)
        return self.response


class _AskService:
    def __init__(self, ask_agent: _AskAgent | None = None) -> None:
        self.ask_agent = ask_agent or _AskAgent()
        self.qna_chain = SimpleNamespace(llm=None, chat=lambda question: "chat")
        self.entry_router = SimpleNamespace(llm=None)


class _ChatService:
    def __init__(self, ask_service: _AskService) -> None:
        self._ask_service = ask_service
        self.conversation_calls: list[str] = []

    def ask_conversation(self, question: str) -> str:
        self.conversation_calls.append(question)
        return "ordinary conversation"

    def ask(self, question: str, **kwargs: Any) -> Any:
        return SimpleNamespace(answer="repository answer", sources=[], warnings=[], trace=[])


class _CodingAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.session_id = "bootstrap-session"

    def generate(self, request: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(request)
        return {"answer": "coding route", "changed_files": [], "warnings": []}

    generate_auto_execute = generate
    generate_dir_mode = generate

    def get_active_flow_id(self) -> None:
        return None

    def flow_summary(self, flow_id: str) -> None:
        return None

    def reset_flow(self, flow_id: str) -> str:
        return flow_id

    def _tool_policy_for_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"allowed_tools": ["read_file"]}


def _registry(
    gmail: RouteAvailability | None = None,
    mcp: RouteAvailability | None = None,
) -> EntryRouteRegistry:
    registry = EntryRouteRegistry()
    for name, description in (
        ("multi_task", "compound task orchestration"),
        ("conversation", "ordinary conversation"),
        ("coding", "Codex coding"),
        ("mcp", "MCP provider operations"),
        ("gmail", "Gmail inbox"),
        ("calendar", "calendar"),
        ("browser", "browser inspection"),
        ("search", "public search"),
        ("github", "GitHub inspection"),
        ("repository", "repository inspection"),
        ("memory", "memory retrieval"),
        ("automation", "automation"),
        ("api", "external API manager"),
        ("canvas", "Live Canvas"),
        ("artifact", "artifact operations"),
        ("unsupported", "safe stop"),
        ("capability_error", "missing capability"),
    ):
        availability = (
            gmail
            if name == "gmail" and gmail is not None
            else mcp
            if name == "mcp" and mcp is not None
            else RouteAvailability(False, reason="No MCP provider is configured.")
            if name == "mcp"
            else RouteAvailability(True)
        )
        registry.register(
            RouteRegistration(
                name,  # type: ignore[arg-type]
                description,
                lambda value=availability: value,
            )
        )
    return registry


def _gateway(
    root: Path,
    model: _RouteModel,
    *,
    gmail: RouteAvailability | None = None,
    mcp: RouteAvailability | None = None,
    ask_agent: _AskAgent | None = None,
    coding_agent: _CodingAgent | None = None,
) -> tuple[AgentChatGateway, _ChatService, _AskAgent]:
    agent = ask_agent or _AskAgent()
    ask_service = _AskService(agent)
    chat_service = _ChatService(ask_service)
    registry = _registry(gmail, mcp)
    gateway = AgentChatGateway(
        root,
        coding_agent=coding_agent is not None,
        coding_agent_instance=coding_agent,
        agent_tools=True,
        chat_service=chat_service,
        entry_route_registry=registry,
        entry_router=EntryRouter(llm=model, registry=registry),
    )
    return gateway, chat_service, agent


def test_latest_gmail_routes_to_connector_and_preserves_identifiers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, chat, ask_agent = _gateway(tmp_path, _RouteModel("gmail"))
    session_id = gateway.create_session(frontend="test")
    result = gateway.process_turn(session_id, "Check my latest Gmail", turn_id="turn_exact")

    assert result.mode == "route-gmail"
    assert not chat.conversation_calls
    assert ask_agent.calls[0]["flow_id"] == session_id
    assert ask_agent.calls[0]["run_id"] == "turn_exact"
    assert ask_agent.calls[0]["tool_policy"]["capability_discovery_required"] is True
    assert "capability_search" in ask_agent.calls[0]["system_prompt"]
    assert result.payload["session_id"] == session_id
    assert result.payload["conversation_id"] == session_id
    assert result.payload["turn_id"] == "turn_exact"


def test_configured_kaggle_mcp_route_uses_only_the_model_selected_provider(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _chat, ask_agent = _gateway(
        tmp_path,
        _RouteModel("mcp"),
        mcp=RouteAvailability(
            True,
            details={"providers": [{"id": "kaggle", "namespace": "mcp.kaggle"}]},
        ),
    )

    result = gateway.process_turn(
        gateway.create_session(frontend="test"),
        "Use Kaggle MCP to upload the submission and submit it to the competition.",
    )

    assert result.mode == "route-mcp"
    assert result.payload["provider_id"] == "kaggle"
    assert ask_agent.calls[0]["required_mcp_server"] == "kaggle"
    assert ask_agent.calls[0]["tool_policy"]["mcp_provider_only"] == "kaggle"
    assert "require_initial_tool_call" not in ask_agent.calls[0]["tool_policy"]
    assert ask_agent.calls[0]["transactional_parent_task_id"] == result.payload["lane_task_id"]
    assert "Do not send an empty object" in ask_agent.calls[0]["system_prompt"]


def test_failed_kaggle_mcp_tool_marks_the_route_failed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    failed_agent = _AskAgent(
        SimpleNamespace(
            answer="The upload was not performed.",
            sources=[],
            warnings=[],
            trace=[
                {
                    "tool_name": "mcp__kaggle__start_competition_submission_upload",
                    "status": "error",
                    "output_preview": "no registered transactional action adapter",
                }
            ],
        )
    )
    gateway, _chat, _ask_agent = _gateway(
        tmp_path,
        _RouteModel("mcp"),
        mcp=RouteAvailability(
            True,
            details={"providers": [{"id": "kaggle", "namespace": "mcp.kaggle"}]},
        ),
        ask_agent=failed_agent,
    )

    result = gateway.process_turn(
        gateway.create_session(frontend="test"),
        "Use Kaggle MCP to upload the submission.",
    )

    assert result.mode == "route-mcp-error"
    assert result.error == "mcp_tool_execution_failed"
    assert result.payload["failed_tool"] == "mcp__kaggle__start_competition_submission_upload"


def test_kaggle_mcp_action_approval_keeps_the_route_waiting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    approval_agent = _AskAgent(
        SimpleNamespace(
            answer="Approval is required.",
            sources=[],
            warnings=[],
            trace=[
                {
                    "tool_name": "mcp__kaggle__authorize",
                    "status": "error",
                    "output_preview": json.dumps(
                        {
                            "ok": False,
                            "error_code": "approval_required",
                            "permission_request_id": "inbox_kaggle_authorize",
                            "inbox_item_id": "inbox_kaggle_authorize",
                            "action_id": "act_kaggle_authorize",
                        }
                    ),
                }
            ],
        )
    )
    gateway, _chat, _ask_agent = _gateway(
        tmp_path,
        _RouteModel("mcp"),
        mcp=RouteAvailability(
            True,
            details={"providers": [{"id": "kaggle", "namespace": "mcp.kaggle"}]},
        ),
        ask_agent=approval_agent,
    )

    result = gateway.process_turn(
        gateway.create_session(frontend="test"),
        "Use Kaggle MCP to authorize access.",
    )

    assert result.mode == "route-mcp-awaiting-approval"
    assert result.payload["confirmation_request_id"] == "inbox_kaggle_authorize"
    assert result.payload["inbox_item_id"] == "inbox_kaggle_authorize"


def test_unsupported_route_bypasses_checkpoint_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))

    def _checkpoint_must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unsupported routes must not enter checkpoint recovery")

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.CheckpointResumeDecider.decide",
        _checkpoint_must_not_run,
    )
    gateway, _chat, _ask_agent = _gateway(tmp_path, _RouteModel("unsupported"))

    result = gateway.process_turn(
        gateway.create_session(frontend="test"),
        "Perform an unavailable external action.",
    )

    assert result.mode == "route-unsupported"
    assert result.error == "unsupported_route"


def test_failed_followup_classification_stops_before_recovery_or_new_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _chat, _ask_agent = _gateway(tmp_path, _RouteModel("conversation"))
    session_id = gateway.create_session(frontend="test")
    prior = gateway._lane_coordinator.reserve(
        normalized_intent="previous gateway task",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(prior)
    gateway._lane_coordinator.finish(
        prior.execution.task_id,
        state=LaneTaskState.FAILED,
        error="prior worker failed",
    )

    def fail_classification(*_args: Any, **_kwargs: Any) -> Any:
        raise FollowupClassificationError(
            "Model decision failed: followup_classification. No recovery or new task was started."
        )

    new_reservations: list[Any] = []
    original_reserve = gateway._lane_coordinator.reserve

    def record_reserve(*args: Any, **kwargs: Any) -> Any:
        new_reservations.append((args, kwargs))
        return original_reserve(*args, **kwargs)

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        fail_classification,
    )
    monkeypatch.setattr(gateway._lane_coordinator, "reserve", record_reserve)

    result = gateway.process_turn(session_id, "Continue the previous gateway task.")

    assert result.mode == "followup-classification-error"
    assert result.error == "followup_classification_invalid"
    assert new_reservations == []


def test_checkpoint_resume_context_budget_block_stops_without_lane_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _chat, _ask_agent = _gateway(tmp_path, _RouteModel("conversation"))
    session_id = gateway.create_session(frontend="test")
    prior = gateway._lane_coordinator.reserve(
        normalized_intent="previous gateway task",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(prior)
    gateway._lane_coordinator.finish(
        prior.execution.task_id,
        state=LaneTaskState.FAILED,
        error="prior worker failed",
    )
    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *_args, **_kwargs: FollowupClassification(
            decision_id="followup-decision",
            category="resume_request",
            related_task_id=prior.execution.task_id,
            safe_to_continue=True,
            reason="The same durable task was selected.",
        ),
    )

    def block_checkpoint_resume(*_args: Any, **_kwargs: Any) -> Any:
        raise CheckpointResumeError(
            "Model decision failed: checkpoint_resume. No task was resumed or started. "
            "Reason: Context budget blocked: context_limit_deficit:510. "
            "No provider call was executed.",
            code="context_budget_blocked",
        )

    new_reservations: list[Any] = []
    original_reserve = gateway._lane_coordinator.reserve

    def record_reserve(*args: Any, **kwargs: Any) -> Any:
        new_reservations.append((args, kwargs))
        return original_reserve(*args, **kwargs)

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.CheckpointResumeDecider.decide",
        block_checkpoint_resume,
    )
    monkeypatch.setattr(gateway._lane_coordinator, "reserve", record_reserve)

    result = gateway.process_turn(session_id, "Continue the previous gateway task.")

    assert result.mode == "checkpoint-resume-budget-blocked"
    assert result.error == "context_budget_blocked"
    assert "Gateway lane coordination failed" not in result.answer
    assert "No task was resumed or started" in result.answer
    assert new_reservations == []


@pytest.mark.parametrize(
    ("action", "with_checkpoint"),
    (("resume_checkpoint", True), ("retry_task", False)),
)
def test_gateway_recovery_handoff_uses_validated_recovery_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    with_checkpoint: bool,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _chat, _ask_agent = _gateway(tmp_path, _RouteModel("conversation"))
    session_id = gateway.create_session(frontend="test")
    prior = gateway._lane_coordinator.reserve(
        normalized_intent="previous gateway task",
        lane_id=LaneId.RESEARCH,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(prior)
    checkpoint_id = (
        gateway._lane_coordinator.checkpoint(prior.execution.task_id, boundary="before-retry")
        if with_checkpoint
        else ""
    )
    gateway._lane_coordinator.finish(
        prior.execution.task_id,
        state=LaneTaskState.FAILED,
        error="prior worker failed",
    )
    monkeypatch.setattr(gateway, "_recall_task_capsules", lambda **_kwargs: "")
    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *_args, **_kwargs: FollowupClassification(
            decision_id="followup-decision",
            category="resume_request" if with_checkpoint else "retry_request",
            related_task_id=prior.execution.task_id,
            safe_to_continue=True,
            reason="The same durable task was selected.",
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.CheckpointResumeDecider.decide",
        lambda *_args, **_kwargs: CheckpointResumeDecision(
            decision_id=f"{action}-decision",
            action=action,  # type: ignore[arg-type]
            task_id=prior.execution.task_id,
            checkpoint_id=checkpoint_id,
            same_work=True,
            fresh_data_required=False,
            checkpoint_still_valid=with_checkpoint,
            side_effects_safe_to_repeat=True,
            safe_to_continue=True,
            reason="The validated recovery decision is safe.",
        ),
    )
    recovery_calls: list[Any] = []
    method_name = "resume_checkpoint" if with_checkpoint else "retry_task"
    original_recovery = getattr(gateway._lane_coordinator, method_name)

    def record_recovery(task_id: str, *, decision: Any, session_id: str) -> Any:
        recovery_calls.append((task_id, decision, session_id))
        return original_recovery(task_id, decision=decision, session_id=session_id)

    monkeypatch.setattr(gateway._lane_coordinator, method_name, record_recovery)

    result = gateway.process_turn(session_id, "Continue the previous gateway task.")

    assert result.error == ""
    assert recovery_calls[0][0] == prior.execution.task_id
    assert recovery_calls[0][1].decision_id == f"{action}-decision"
    assert recovery_calls[0][1].action.value == (
        "resume_checkpoint" if with_checkpoint else "retry"
    )
    assert recovery_calls[0][2] == session_id


def test_uploaded_spreadsheet_routes_to_artifact_lane_without_coding_agent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    upload = tmp_path / "uploads" / "test.xls"
    upload.parent.mkdir()
    upload.write_bytes(b"worksheet")
    coding = _CodingAgent()
    gateway, _chat, ask = _gateway(tmp_path, _RouteModel("artifact"), coding_agent=coding)
    ask.project_root = tmp_path

    result = gateway.process_turn(
        gateway.create_session(frontend="test"),
        "test.xls\nin cell under age add average of age.",
        attachments=[{"path": str(upload), "mime_type": "application/vnd.ms-excel"}],
    )

    assert result.payload["route"] == "artifact"
    assert result.payload["lane_id"] == "artifact"
    assert result.payload["routing_evidence"]["artifact_families"] == ["spreadsheet"]
    assert coding.calls == []
    assert ask.calls[0]["index_dir"] is not None
    assert Path(ask.calls[0]["index_dir"]).name == ".artifact-index"


def test_pdf_creation_without_attachment_uses_isolated_concrete_index_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _, ask = _gateway(tmp_path, _RouteModel("artifact"))
    ask.project_root = tmp_path

    result = gateway.process_turn(
        gateway.create_session(frontend="test"),
        "Create a PDF from the completed research.",
    )

    assert result.mode == "route-artifact"
    assert ask.calls[0]["index_dir"] is not None
    assert Path(ask.calls[0]["index_dir"]).name == ".artifact-index"
    assert Path(ask.calls[0]["index_dir"]).parent.name.startswith("turn_")
    assert ask.project_roots == [tmp_path.resolve()]
    assert ask.calls[0]["tool_policy"]["skill_root"] == str(tmp_path.resolve())
    assert ask.calls[0]["tool_policy"]["allowed_tools"][0] == "read_skill"
    assert "read_skill(skill_name='pdf-create')" in ask.calls[0]["question"]
    assert "Mana-Agent launch directory" in ask.calls[0]["question"]


def test_missing_gmail_configuration_returns_truthful_setup_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    unavailable = RouteAvailability(
        False,
        configured=False,
        authorized=False,
        reason="No enabled Gmail account is configured.",
        setup_action="Run `mana-agent connector email add --provider gmail ...`.",
    )

    def _checkpoint_must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("capability errors must not enter checkpoint recovery")

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.CheckpointResumeDecider.decide",
        _checkpoint_must_not_run,
    )
    gateway, chat, ask_agent = _gateway(tmp_path, _RouteModel("capability_error"), gmail=unavailable)
    result = gateway.process_turn(gateway.create_session(frontend="test"), "Check Gmail")

    assert result.mode == "route-capability-error"
    assert "gmail" in result.answer.lower()
    assert not chat.conversation_calls
    assert not ask_agent.calls


def test_gateway_route_registry_has_executors_for_every_available_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    ask_service = _AskService()
    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
        chat_service=_ChatService(ask_service),
    )
    executor_contracts = {
        "conversation": "_execute_entry_route",
        "multi_task": "process_turn",
        "coding": "_execute_entry_route",
        "mcp": "_execute_mcp_route",
        "remote_execution": "_execute_entry_route",
        "server": "_execute_entry_route",
        "artifact": "_execute_artifact_route",
        "media": "_execute_media_route",
        "command": "process_turn",
        "gmail": "_execute_gmail_route",
        "computer": "_execute_computer_route",
        "browser": "_execute_required_sources",
        "search": "_execute_required_sources",
        "github": "_execute_required_sources",
        "repository": "_execute_entry_route",
        "memory": "_execute_memory_route",
        "automation": "_execute_automation_route",
        "api": "_execute_api_route",
        "canvas": "_execute_canvas_route",
        "unsupported": "_execute_entry_route",
        "capability_error": "_execute_entry_route",
    }
    snapshot = {row["name"]: row["availability"] for row in gateway._entry_route_registry.snapshot()}

    assert {name for name, availability in snapshot.items() if availability["available"]} <= set(
        executor_contracts
    )
    assert all(callable(getattr(gateway, executor)) for executor in executor_contracts.values())
    assert snapshot["calendar"]["available"] is False
    assert snapshot["calendar"]["reason"] == "No calendar connector is registered."


def test_capability_error_cannot_claim_enabled_search_is_unavailable() -> None:
    class _IncorrectCapabilityModel:
        def invoke(self, _messages: list[Any]) -> Any:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "route": "capability_error",
                        "confidence": 0.98,
                        "reason": "search is unavailable",
                        "required_sources": ["search"],
                        "target_urls": [],
                        "requires_live_data": True,
                        "reason_code": "SEARCH_UNAVAILABLE",
                        "error_code": "SEARCH_NOT_AVAILABLE",
                        "reuse_active_route": False,
                    }
                )
            )

    router = EntryRouter(llm=_IncorrectCapabilityModel(), registry=_registry())

    with pytest.raises(EntryRoutingError, match=r"declared available source\(s\) unavailable: search"):
        router.route(
            user_prompt="Find current public information about Mana-Agent.",
            context=EntryRouteContext(session_id="session", conversation_id="session", turn_id="turn"),
        )


def test_search_route_availability_requires_a_configured_provider(tmp_path: Path, monkeypatch) -> None:
    from mana_agent.search.config import SearchConfig

    gateway, _chat, _ask_agent = _gateway(tmp_path, _RouteModel("conversation"))
    monkeypatch.setattr(
        SearchConfig,
        "from_env",
        classmethod(lambda cls: SearchConfig(enable_web=True, web_provider="tavily")),
    )

    availability = gateway._search_route_availability()

    assert availability.available is False
    assert availability.configured is True
    assert availability.authorized is False
    assert "credentials" in availability.reason.lower()


def test_gmail_availability_reads_live_account_and_credential_registry(monkeypatch) -> None:
    from mana_agent.connectors.email.models import (
        EmailAccount,
        EmailAddress,
        EmailPermission,
    )

    account = EmailAccount(
        id="gmail-live",
        provider="gmail",
        address=EmailAddress(address="me@example.com"),
        granted_permissions={EmailPermission.READ},
        secret_ref="credential-ref",
    )
    monkeypatch.setattr(
        "mana_agent.connectors.email.config.load_accounts",
        lambda: [account],
    )
    monkeypatch.setattr(
        "mana_agent.connectors.email.auth.credential_store.CredentialStore.get",
        lambda self, reference: {"token": "present"},
    )

    availability = gmail_route_availability()

    assert availability.available is True
    assert availability.configured is True
    assert availability.authorized is True
    assert availability.details["account_id"] == "gmail-live"


def test_gmail_provider_authorization_details_are_not_replaced(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    provider_error = (
        "email_authorization_failed provider=gmail provider_status=403 "
        "reconnect_required=true missing_scope=email.read"
    )
    ask_agent = _AskAgent(
        SimpleNamespace(answer=provider_error, sources=[], warnings=[], trace=[])
    )
    gateway, chat, _ = _gateway(tmp_path, _RouteModel("gmail"), ask_agent=ask_agent)
    result = gateway.process_turn(gateway.create_session(frontend="test"), "Read latest Gmail")

    assert result.answer == provider_error
    assert result.mode == "route-gmail"
    assert not chat.conversation_calls


def test_conversation_and_coding_use_their_selected_routes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    model = _RouteModel("conversation", "coding")
    coding = _CodingAgent()
    gateway, chat, _ = _gateway(tmp_path, model, coding_agent=coding)
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.handle_small_direct_edit",
        lambda *args, **kwargs: SimpleNamespace(handled=False),
    )
    session_id = gateway.create_session(frontend="test")

    conversation = gateway.process_turn(session_id, "Hello, how are you?")
    coding_result = gateway.process_turn(session_id, "Change the parser implementation")

    assert conversation.mode == "route-conversation"
    assert conversation.payload["lane_id"] == "research"
    assert len(chat.conversation_calls) == 1
    assert coding_result.used_coding_agent is True
    assert coding_result.payload["lane_id"] == "coding"
    assert coding.calls
    assert coding.session_id == session_id
    assert coding_result.payload["session_id"] == session_id


def test_entry_router_exposes_authorized_ssh_requests_to_coding_workflow() -> None:
    assert "remote_execution" in ENTRY_ROUTER_PROMPT
    assert "Never select coding for" in ENTRY_ROUTER_PROMPT


def test_entry_router_assigns_deferred_gmail_actions_only_to_automation() -> None:
    assert "Do not select gmail when the requested mailbox action is deferred" in ENTRY_ROUTER_PROMPT
    assert "select automation instead and do not select gmail as a preliminary action" in ENTRY_ROUTER_PROMPT
    assert "At 12:52, check my Gmail" in ENTRY_ROUTER_PROMPT
    assert "listing is not a prerequisite for creation" in ENTRY_ROUTER_PROMPT


def test_entry_router_validates_canvas_as_an_explicit_model_route() -> None:
    router = EntryRouter(llm=_RouteModel("canvas"), registry=_registry())
    decision = router.route(
        user_prompt="Open an interactive project planning canvas.",
        context=EntryRouteContext(
            session_id="session", conversation_id="session", turn_id="turn",
        ),
    )
    assert decision.route == "canvas"
    assert decision.required_sources == ("canvas",)


def test_entry_router_validates_api_as_an_explicit_model_route() -> None:
    router = EntryRouter(llm=_RouteModel("api"), registry=_registry())
    decision = router.route(
        user_prompt="Use the saved CRM API to fetch contact 123.",
        context=EntryRouteContext(
            session_id="session", conversation_id="session", turn_id="turn",
        ),
    )
    assert decision.route == "api"
    assert decision.required_sources == ("api",)
    assert "never expose a raw unrestricted HTTP tool" in ENTRY_ROUTER_PROMPT


def test_entry_router_validates_private_memory_task_selection() -> None:
    router = EntryRouter(llm=_RouteModel("memory"), registry=_registry())
    context = EntryRouteContext(
        session_id="session",
        conversation_id="session",
        turn_id="turn",
        memory_capsules_enabled=True,
        memory_task_candidates=(
            {
                "task_id": "task-offered",
                "normalized_intent": "inspect the gateway",
                "state": "completed",
            },
        ),
    )
    payload = {
        "route": "memory",
        "confidence": 0.99,
        "reason": "Retrieve the selected task's private result.",
        "required_sources": ["memory"],
        "memory_task_id": "task-offered",
    }

    decision = router._validate(payload, context=context)

    assert decision.memory_task_id == "task-offered"
    with pytest.raises(EntryRoutingError, match="not offered"):
        router._validate({**payload, "memory_task_id": "task-foreign"}, context=context)
    with pytest.raises(EntryRoutingError, match="requires a selected task ID"):
        router._validate({**payload, "memory_task_id": ""}, context=context)


def test_gateway_memory_route_reads_only_the_selected_private_capsules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _chat, _ask_agent = _gateway(tmp_path, _RouteModel("conversation"))
    calls: list[Any] = []

    class CapsuleMemory:
        user_id = "authenticated-user"
        config = SimpleNamespace(capsules=SimpleNamespace(enabled=True))

        def __init__(self) -> None:
            self.capsules = self

        def query_capsules(self, request: Any, *, correlation_id: str = "") -> list[Any]:
            calls.append((request, correlation_id))
            return [
                SimpleNamespace(
                    capsule_id="capsule-1",
                    revision=2,
                    summary="Gateway audit evidence",
                    content={"note": "private task result"},
                )
            ]

    gateway._stack.memory_service = CapsuleMemory()
    context = EntryRouteContext(
        session_id="session",
        conversation_id="session",
        turn_id="turn-memory",
        memory_capsules_enabled=True,
        memory_task_candidates=(
            {"task_id": "task-offered", "normalized_intent": "audit", "state": "completed"},
        ),
    )
    decision = EntryRoutingDecision(
        route="memory",
        confidence=0.99,
        reason="Read the selected task memory.",
        required_sources=("memory",),
        memory_task_id="task-offered",
    )

    result = gateway._execute_memory_route(decision=decision, context=context, query="gateway")

    assert result.mode == "route-memory"
    assert result.payload["memory_task_id"] == "task-offered"
    assert result.payload["memory_record_count"] == 1
    assert "capsule-1" in result.answer
    assert calls[0][0].principal.task_id == "task-offered"
    assert calls[0][0].task_context.session_id == "session"
    assert calls[0][1] == "turn-memory"


def test_gateway_memory_route_rejects_unoffered_task_before_private_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _chat, _ask_agent = _gateway(tmp_path, _RouteModel("conversation"))

    class CapsuleMemory:
        user_id = "authenticated-user"
        config = SimpleNamespace(capsules=SimpleNamespace(enabled=True))

        def __init__(self) -> None:
            self.capsules = self
            self.reads = 0

        def query_capsules(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            self.reads += 1
            return []

    memory = CapsuleMemory()
    gateway._stack.memory_service = memory
    result = gateway._execute_memory_route(
        decision=EntryRoutingDecision(
            route="memory",
            confidence=0.99,
            reason="Read a task memory.",
            required_sources=("memory",),
            memory_task_id="task-foreign",
        ),
        context=EntryRouteContext(
            session_id="session",
            conversation_id="session",
            turn_id="turn-memory",
            memory_capsules_enabled=True,
            memory_task_candidates=(
                {"task_id": "task-offered", "normalized_intent": "audit", "state": "completed"},
            ),
        ),
        query="gateway",
    )

    assert result.error == "memory_task_id_invalid"
    assert memory.reads == 0
    missing_task_result = gateway._execute_memory_route(
        decision=EntryRoutingDecision(
            route="memory",
            confidence=0.99,
            reason="Read a task memory.",
            required_sources=("memory",),
        ),
        context=EntryRouteContext(
            session_id="session",
            conversation_id="session",
            turn_id="turn-memory-missing",
            memory_capsules_enabled=True,
            memory_task_candidates=(
                {"task_id": "task-offered", "normalized_intent": "audit", "state": "completed"},
            ),
        ),
        query="gateway",
    )

    assert missing_task_result.error == "memory_task_id_invalid"
    assert memory.reads == 0


def test_gateway_memory_route_requires_an_authenticated_capsule_principal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _chat, _ask_agent = _gateway(tmp_path, _RouteModel("conversation"))

    class CapsuleMemory:
        user_id = ""
        config = SimpleNamespace(capsules=SimpleNamespace(enabled=True))

        def __init__(self) -> None:
            self.capsules = self
            self.reads = 0

        def query_capsules(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            self.reads += 1
            return []

    memory = CapsuleMemory()
    gateway._stack.memory_service = memory
    result = gateway._execute_memory_route(
        decision=EntryRoutingDecision(
            route="memory",
            confidence=0.99,
            reason="Read a task memory.",
            required_sources=("memory",),
            memory_task_id="task-offered",
        ),
        context=EntryRouteContext(
            session_id="session",
            conversation_id="session",
            turn_id="turn-memory",
            memory_capsules_enabled=True,
            memory_task_candidates=(
                {"task_id": "task-offered", "normalized_intent": "audit", "state": "completed"},
            ),
        ),
        query="gateway",
    )

    assert result.error == "memory_principal_unavailable"
    assert memory.reads == 0


def test_gateway_memory_route_uses_legacy_scoped_search_when_capsules_are_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _chat, _ask_agent = _gateway(tmp_path, _RouteModel("conversation"))

    class LegacyMemory:
        config = SimpleNamespace(capsules=SimpleNamespace(enabled=False))

        def __init__(self) -> None:
            self.requests: list[Any] = []

        def search_blocking(self, request: Any) -> list[Any]:
            self.requests.append(request)
            return [
                SimpleNamespace(content=MemoryContent("Scoped legacy evidence")),
            ]

    memory = LegacyMemory()
    gateway._stack.memory_service = memory
    result = gateway._execute_memory_route(
        decision=EntryRoutingDecision(
            route="memory",
            confidence=0.99,
            reason="Read scoped legacy memory.",
            required_sources=("memory",),
        ),
        context=EntryRouteContext(
            session_id="session",
            conversation_id="conversation",
            turn_id="turn-memory",
        ),
        query="gateway",
    )

    assert result.mode == "route-memory"
    assert result.answer == "Scoped legacy evidence"
    assert memory.requests[0].scope.session_id == "session"
    assert memory.requests[0].scope.conversation_id == "conversation"


def test_entry_router_requires_a_typed_automation_operation() -> None:
    router = EntryRouter(llm=SimpleNamespace(), registry=_registry())

    with pytest.raises(EntryRoutingError, match="automation route requires automation_operation"):
        router._validate({
            "route": "automation",
            "confidence": 0.99,
            "reason": "Create the requested schedule.",
            "required_sources": ["repository"],
        })

    decision = router._validate({
        "route": "automation",
        "confidence": 0.99,
        "reason": "Create the requested schedule.",
        "required_sources": ["repository"],
        "automation_operation": "create",
    })

    assert decision.automation_operation == "create"


def test_followup_gmail_reuses_one_session_and_supplies_previous_route(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    model = _RouteModel("gmail", "gmail")
    gateway, _, ask_agent = _gateway(tmp_path, model)
    session_id = gateway.create_session(frontend="test")

    gateway.process_turn(session_id, "Check latest Gmail")
    gateway.process_turn(session_id, "Open the first one")

    assert len(ask_agent.calls) == 2
    assert {call["flow_id"] for call in ask_agent.calls} == {session_id}
    assert model.payloads[1]["context"]["previous_route"] == "gmail"
    assert {row["session_id"] for row in gateway.session_messages(session_id)} == {session_id}
    sessions = WorkspaceService().store.list_sessions()
    assert [item.session_id for item in sessions if item.status == "active"] == [session_id]


def test_invalid_entry_decision_stops_without_connector_refusal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    model = _RouteModel("not-a-route")
    gateway, chat, ask_agent = _gateway(tmp_path, model)
    result = gateway.process_turn(gateway.create_session(frontend="test"), "Check Gmail")

    assert result.mode == "route-error"
    assert result.payload["route"] == "unsupported"
    assert "integration" not in result.answer.lower()
    assert not chat.conversation_calls
    assert not ask_agent.calls


def test_direct_url_is_a_browser_signal_and_executes_only_browser(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    model = _RouteModel("browser")
    gateway, chat, ask_agent = _gateway(tmp_path, model)
    result = gateway.process_turn(gateway.create_session(frontend="test"), "Review https://example.com/about")

    assert result.mode == "route-browser"
    assert model.payloads[0]["direct_url_signals"] == ["https://example.com/about"]
    assert "browser_open" in ask_agent.calls[0]["tool_policy"]["allowed_tools"]
    assert ask_agent.calls[0]["tool_policy"]["disable_external_search"] is True
    assert not chat.conversation_calls


def test_browser_capability_manifest_uses_live_runtime_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mana_agent.config.user_config.get_setting", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "mana_agent.connectors.browser.session.BrowserSessionManager.status",
        lambda: {"ok": True, "package_installed": True, "chromium_installed": True},
    )
    gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"))

    availability = gateway._browser_route_availability()

    assert availability.available is True
    assert availability.details["chromium_installed"] is True


def test_router_rejects_missing_required_sources_instead_of_guessing(tmp_path: Path) -> None:
    registry = _registry()
    model = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(
            content='{"route":"search","confidence":0.9,"reason":"needs current information"}'
        )
    )
    router = EntryRouter(llm=model, registry=registry)

    try:
        router.route(
            user_prompt="Find current competitors",
            context=SimpleNamespace(to_dict=lambda: {"session_id": "s"}),
        )
    except EntryRoutingError as exc:
        assert "required_sources" in str(exc)
    else:
        raise AssertionError("invalid routing output must stop without selecting a source")


def test_router_repairs_url_less_browser_discovery_with_a_new_model_decision() -> None:
    registry = _registry()
    payloads = iter(
        [
            {
                "route": "browser",
                "confidence": 0.9,
                "reason": "research jobs",
                "required_sources": ["browser"],
                "target_urls": [],
                "requires_live_data": True,
            },
            {
                "route": "search",
                "confidence": 0.95,
                "reason": "open-ended job discovery needs search",
                "required_sources": ["search"],
                "target_urls": [],
                "requires_live_data": True,
            },
        ]
    )
    router = EntryRouter(
        llm=SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(content=json.dumps(next(payloads)))
        ),
        registry=registry,
    )

    decision = router.route(
        user_prompt="Find remote AI agent jobs",
        context=SimpleNamespace(to_dict=lambda: {"session_id": "s"}),
    )

    assert decision.route == "search"
    assert decision.required_sources == ("search",)


def test_router_prompt_requires_a_none_source_for_ping() -> None:
    """Tool-free model decisions must still satisfy the explicit source contract."""
    assert 'required_sources is required for every decision and must never be omitted or empty' in ENTRY_ROUTER_PROMPT
    assert '“ping” -> conversation, ["none"].' in ENTRY_ROUTER_PROMPT


def test_ping_uses_a_valid_tool_free_model_decision(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, chat, _ = _gateway(tmp_path, _RouteModel("conversation"))

    result = gateway.process_turn(gateway.create_session(frontend="test"), "ping")

    assert result.mode == "route-conversation"
    assert result.answer == "ordinary conversation"
    assert len(chat.conversation_calls) == 1
def test_router_exposes_exact_required_source_contract_to_model() -> None:
    registry = _registry()
    captured: dict[str, Any] = {}

    def invoke(messages: list[Any]) -> Any:
        captured.update(json.loads(messages[-1].content))
        return SimpleNamespace(content=json.dumps({
            "route": "conversation",
            "confidence": 0.9,
            "reason": "ordinary discussion",
            "required_sources": ["none"],
            "target_urls": [],
        }))

    decision = EntryRouter(llm=SimpleNamespace(invoke=invoke), registry=registry).route(
        user_prompt="Hello",
        context=SimpleNamespace(to_dict=lambda: {"session_id": "s"}),
    )

    assert decision.required_sources == ("none",)
    assert captured["required_source_rules"]["command"] == [["none"]]
    assert captured["required_source_rules"]["multi_task"] == [["none"]]
    assert "command" not in captured["required_source_vocabulary"]
    assert "repository" in captured["required_source_vocabulary"]
    assert "server" in captured["required_source_vocabulary"]


def test_multi_task_route_is_registered_and_requires_no_parent_sources() -> None:
    registry = _registry()
    model = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content=json.dumps({
            "route": "multi_task",
            "confidence": 0.97,
            "reason": "The request has separately routed work.",
            "required_sources": ["none"],
            "target_urls": [],
        }))
    )

    decision = EntryRouter(llm=model, registry=registry).route(
        user_prompt="Inspect GitHub issues, then update the repository.",
        context=EntryRouteContext(session_id="s", conversation_id="s", turn_id="t"),
    )

    assert decision.route == "multi_task"
    assert decision.required_sources == ("none",)


def test_multi_task_route_rejects_claimed_child_source() -> None:
    registry = _registry()
    model = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content=json.dumps({
            "route": "multi_task",
            "confidence": 0.97,
            "reason": "Invalid parent source claim.",
            "required_sources": ["repository"],
            "target_urls": [],
        }))
    )

    with pytest.raises(EntryRoutingError, match="tool-free routes"):
        EntryRouter(llm=model, registry=registry).route(
            user_prompt="Inspect GitHub issues and update the repository.",
            context=EntryRouteContext(session_id="s", conversation_id="s", turn_id="t"),
        )


def test_atomic_child_excludes_recursive_orchestration_from_model_registry() -> None:
    registry = _registry()
    model = _RouteModel("search")

    decision = EntryRouter(llm=model, registry=registry).route(
        user_prompt="Research OpenClaw.",
        context=EntryRouteContext(
            session_id="s",
            conversation_id="s",
            turn_id="t:research_openclaw",
            conversation_summary="Research OpenClaw and create a PDF.",
            atomic_child=True,
            orchestration_parent_task_id="task-root",
        ),
    )

    assert decision.route == "search"
    assert "multi_task" not in {row["name"] for row in model.payloads[0]["routes"]}
    assert model.payloads[0]["routing_constraints"] == {
        "atomic_child": True,
        "disallowed_routes": ["multi_task"],
        "orchestration_parent_task_id": "task-root",
    }


def test_atomic_child_rejects_model_attempt_to_select_recursive_orchestration() -> None:
    registry = _registry()
    model = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content=json.dumps({
            "route": "multi_task",
            "confidence": 0.97,
            "reason": "Incorrectly reconsidered the parent compound request.",
            "required_sources": ["none"],
            "target_urls": [],
        }))
    )

    with pytest.raises(EntryRoutingError, match="atomic compound child"):
        EntryRouter(llm=model, registry=registry).route(
            user_prompt="Research OpenClaw.",
            context=EntryRouteContext(
                session_id="s",
                conversation_id="s",
                turn_id="t:research_openclaw",
                atomic_child=True,
                orchestration_parent_task_id="task-root",
            ),
        )


def test_artifact_creation_uses_model_selected_family_without_file_evidence() -> None:
    registry = _registry()
    model = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content=json.dumps({
            "route": "artifact",
            "confidence": 0.98,
            "reason": "Create a PDF artifact.",
            "required_sources": ["artifact"],
            "target_urls": [],
            "artifact_family": "pdf",
        }))
    )

    decision = EntryRouter(llm=model, registry=registry).route(
        user_prompt="Create a PDF from the supplied research.",
        context=EntryRouteContext(session_id="s", conversation_id="s", turn_id="t"),
    )

    assert decision.artifact_family == "pdf"


def test_router_unknown_source_error_identifies_invalid_model_value() -> None:
    registry = _registry()
    model = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content=json.dumps({
            "route": "conversation",
            "confidence": 0.9,
            "reason": "invalid model contract",
            "required_sources": ["command"],
            "target_urls": [],
        }))
    )

    try:
        EntryRouter(llm=model, registry=registry).route(
            user_prompt="Show my sessions",
            context=SimpleNamespace(to_dict=lambda: {"session_id": "s"}),
        )
    except EntryRoutingError as exc:
        assert "unknown source identifier(s): command" in str(exc)
        assert "Allowed values:" in str(exc)
    else:
        raise AssertionError("unknown source identifiers must fail closed")


def test_router_uses_structured_output_when_model_supports_it() -> None:
    registry = _registry()
    calls: list[tuple[object, str, bool]] = []

    class StructuredModel:
        def with_structured_output(self, schema, *, method: str, strict: bool):
            calls.append((schema, method, strict))
            return SimpleNamespace(
                invoke=lambda _messages: {
                    "route": "conversation",
                    "confidence": 0.9,
                    "reason": "No tool is needed.",
                    "required_sources": ["none"],
                }
            )

        def invoke(self, _messages):  # pragma: no cover - must not be used
            raise AssertionError("structured output was available")

    decision = EntryRouter(llm=StructuredModel(), registry=registry).route(
        user_prompt="Hello",
        context=EntryRouteContext(session_id="s", conversation_id="s", turn_id="t"),
    )

    assert decision.route == "conversation"
    assert calls[0][1:] == ("json_schema", True)


def test_server_request_structured_output_schema_is_closed() -> None:
    schema = EntryRoutingOutput.model_json_schema()

    server_request_schema = schema["$defs"]["EntryRoutingServerRequest"]
    server_decision_schema = schema["$defs"]["EntryRoutingServerDecision"]

    assert server_request_schema["additionalProperties"] is False
    assert server_decision_schema["properties"]["decision_id"]["minLength"] == 1
    assert server_decision_schema["additionalProperties"] is False
    assert "arguments_json" in server_decision_schema["properties"]
    assert "arguments" not in server_decision_schema["properties"]


def test_server_route_availability_exposes_exact_non_secret_contracts() -> None:
    gateway = object.__new__(AgentChatGateway)
    gateway.server_management_service = SimpleNamespace(
        list_servers=lambda: [
            SimpleNamespace(
                server_id="mana-agent-server-1",
                name="production",
                username="ubuntu",
                mode="managed_admin",
                provider="ssh",
                operating_system="ubuntu",
                architecture="x86_64",
                allowed_capabilities={"inspect", "package.write"},
            )
        ]
    )

    availability = gateway._server_route_availability()
    package_contract = next(
        contract
        for contract in availability.details["tool_contracts"]
        if contract["tool_name"] == "server_package_install"
    )

    assert availability.available is True
    assert availability.details["server_catalog"] == [
        {
            "server_id": "mana-agent-server-1",
            "name": "production",
            "login_user": "ubuntu",
            "mode": "managed_admin",
            "provider": "ssh",
            "operating_system": "ubuntu",
            "architecture": "x86_64",
            "allowed_capabilities": ["inspect", "package.write"],
        }
    ]
    assert package_contract == {
        "tool_name": "server_package_install",
        "action": "package",
        "required_capability": "package.write",
        "read_only": False,
        "consequential": True,
        "destructive": False,
        "arguments_json_example": (
            '{"manager":"auto|apt|dnf|yum|pacman|apk|zypper|brew",'
            '"packages":["nginx"]}'
        ),
    }
    shell_contract = next(
        contract
        for contract in availability.details["tool_contracts"]
        if contract["tool_name"] == "server_shell_execute"
    )
    assert shell_contract["arguments_json_example"] == (
        '{"argv":["mkdir","-p","mana-agent-test"]}'
    )
    assert "exactly from the selected entry" in ENTRY_ROUTER_PROMPT
    assert "route availability tool_contracts" in ENTRY_ROUTER_PROMPT
    assert "server catalog's login_user" in ENTRY_ROUTER_PROMPT
    assert "capability_error is only for a route-wide unavailable" in ENTRY_ROUTER_PROMPT


def test_server_route_decodes_closed_arguments_json() -> None:
    registry = EntryRouteRegistry()
    registry.register(
        RouteRegistration(
            "server",
            "enrolled server management",
            lambda: RouteAvailability(available=True),
        )
    )
    model = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content=json.dumps({
            "route": "server",
            "confidence": 0.98,
            "reason": "Install nginx on the enrolled server.",
            "required_sources": ["server"],
            "target_urls": [],
            "server_request": {
                "decision": {
                    "decision_id": "decision-1",
                    "server_id": "production-1",
                    "action": "package",
                    "tool_name": "server_package_install",
                    "arguments_json": json.dumps({"manager": "apt", "packages": ["nginx"]}),
                    "required_capability": "package.write",
                    "read_only": False,
                    "consequential": True,
                    "destructive": False,
                    "affected_resources": ["package:nginx"],
                    "recovery_plan": "Remove nginx if verification fails.",
                    "verification_commands": [["nginx", "-v"]],
                    "safe_to_continue": True,
                    "reason": "The user requested nginx installation.",
                }
            },
        }))
    )

    decision = EntryRouter(llm=model, registry=registry).route(
        user_prompt="install nginx.",
        context=EntryRouteContext(session_id="s", conversation_id="s", turn_id="t"),
    )

    assert decision.route == "server"
    assert decision.server_request["decision"]["arguments"] == {
        "manager": "apt",
        "packages": ["nginx"],
    }


def test_server_directory_listing_repairs_a_tool_contract_mismatch() -> None:
    registry = EntryRouteRegistry()
    registry.register(
        RouteRegistration(
            "server",
            "enrolled server management",
            lambda: RouteAvailability(available=True),
        )
    )
    calls: list[dict[str, Any]] = []
    invalid_decision = {
        "decision_id": "directory-list-1",
        "server_id": "mana-agent-server-1",
        "action": "inspect",
        "tool_name": "server_directory_list",
        "arguments_json": json.dumps({"path": "/home/ubuntu"}),
        "required_capability": "inspect",
        "read_only": True,
        "consequential": False,
        "destructive": False,
        "affected_resources": ["directory:/home/ubuntu"],
        "safe_to_continue": True,
        "reason": "List the requested directory.",
    }
    valid_decision = {
        **invalid_decision,
        "action": "file_read",
        "required_capability": "filesystem.read",
    }

    class Model:
        def invoke(self, messages: list[Any]) -> Any:
            calls.append(json.loads(messages[-1].content))
            decision = invalid_decision if len(calls) == 1 else valid_decision
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "route": "server",
                        "confidence": 0.98,
                        "reason": "Use the enrolled server directory-list tool.",
                        "required_sources": ["server"],
                        "server_request": {"decision": decision},
                    }
                )
            )

    decision = EntryRouter(llm=Model(), registry=registry).route(
        user_prompt="Connect to mana-agent-server-1 and list /home/ubuntu.",
        context=EntryRouteContext(session_id="s", conversation_id="s", turn_id="t"),
    )

    assert len(calls) == 2
    assert calls[1]["previous_invalid_decision"]["server_request"]["decision"] == invalid_decision
    assert "tool_contracts" in calls[1]["correction"]
    assert decision.server_request["decision"]["action"] == "file_read"
    assert decision.server_request["decision"]["required_capability"] == "filesystem.read"


def test_remote_routing_requires_direct_ssh_without_a_managed_worker() -> None:
    registry = EntryRouteRegistry()
    registry.register(
        RouteRegistration(
            "remote_execution",
            "SSH execution",
            lambda: RouteAvailability(
                available=True,
                details={"managed_worker_available": False, "direct_ssh_available": True},
            ),
        )
    )
    model = SimpleNamespace(
        invoke=lambda _messages: SimpleNamespace(content=json.dumps({
            "route": "remote_execution",
            "confidence": 0.9,
            "reason": "No managed worker is available.",
            "required_sources": ["remote_execution"],
            "target_urls": [],
            "remote_request": {
                "provider": "remote-ssh",
                "worker_id": "",
                "target": {"host": "example.test", "user": "root"},
                "authentication": {"mode": "agent"},
                "command": {"argv": ["true"]},
            },
        }))
    )

    decision = EntryRouter(llm=model, registry=registry).route(
        user_prompt="Run true over SSH.",
        context=EntryRouteContext(session_id="s", conversation_id="s", turn_id="t"),
    )

    assert decision.remote_request["provider"] == "remote-ssh"


def test_failed_required_browser_source_stops_multi_source_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    failing_browser = _AskAgent(SimpleNamespace(answer="", sources=[], warnings=[], trace=[]))
    gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"), ask_agent=failing_browser)
    decision = EntryRoutingDecision(
        route="browser",
        confidence=0.9,
        reason="page and search evidence are both required",
        required_sources=("browser", "search"),
        target_urls=("https://example.com",),
        requires_live_data=True,
        reason_code="SEO_AUDIT",
    )

    result = gateway._execute_required_sources(
        decision=decision,
        text="Inspect example.com",
        ask_service=gateway.get_ask_service(),
        callbacks=None,
    )

    assert result.error == "browser_execution_failed"
    assert result.payload["route_status"] == "failed"
    assert result.payload["executions"] == {
        "browser": {"status": "failed", "error": "browser returned no evidence"}
    }
    assert result.answer.endswith("browser returned no evidence. No alternative source was used.")
    assert len(failing_browser.calls) == 1


def test_required_search_source_uses_constrained_operation_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"))
    captured: dict[str, Any] = {}
    operation = AgentDecision(
        intent="web_research",
        confidence=0.9,
        selected_tools=["web_search"],
        tool_inputs={"web_search": {"query": "latest Mana-Agent release"}},
        web_search_needed=True,
        verifier_passed=True,
    )

    def decide_search_operation(**kwargs: Any) -> AgentDecision:
        captured.update(kwargs)
        return operation

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.decide_search_operation",
        decide_search_operation,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.run_web_research_answer",
        lambda **_kwargs: ("current search evidence", [], []),
    )
    decision = EntryRoutingDecision(
        route="search",
        confidence=0.9,
        reason="Current information is required.",
        required_sources=("search",),
        requires_live_data=True,
    )

    result = gateway._execute_required_sources(
        decision=decision,
        text="What is the latest Mana-Agent release?",
        ask_service=gateway.get_ask_service(),
        callbacks=None,
    )

    assert result.answer == "current search evidence"
    assert captured["required_tool"] == "web_search"
    assert captured["question"] == "What is the latest Mana-Agent release?"


def test_compound_child_search_uses_only_its_validated_child_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"))
    captured: dict[str, str] = {}

    def execute_required_sources(**kwargs: Any):
        captured["text"] = kwargs["text"]
        from mana_agent.gateway.turn_engine import ChatTurnResult

        return ChatTurnResult(answer="research complete", payload={"route": "search"})

    monkeypatch.setattr(gateway, "_execute_required_sources", execute_required_sources)
    child_prompt = "Research current public information about Hermes Agent."
    decision = EntryRoutingDecision(
        route="search",
        confidence=0.98,
        reason="Current public research is required.",
        required_sources=("search",),
    )

    gateway._execute_entry_route(
        decision=decision,
        context=EntryRouteContext(session_id="s", conversation_id="s", turn_id="t:research"),
        text=child_prompt,
        state={"messages": [{"role": "user", "content": "unrelated parent conversation"}]},
        ask_service=gateway.get_ask_service(),
        sink=None,
        options={"_isolated_child_prompt": True},
    )

    assert captured["text"] == child_prompt
    assert "unrelated parent conversation" not in captured["text"]


def test_session_close_new_history_and_stale_finalization(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"))
    first = gateway.create_session(frontend="test")
    gateway.process_turn(first, "Remember this")
    assert gateway.close_session(first) == first
    assert gateway.close_session(first) == first

    service = WorkspaceService()
    first_record = service.store.get_session(first)
    assert first_record.status == "closed"
    assert first_record.opened_at
    assert first_record.closed_at
    assert gateway.session_messages(first)

    second_gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"))
    second = second_gateway.create_session(frontend="test")
    assert second != first

    stale = service.create_session(tmp_path)
    stale.owner_pid = 999_999_999
    service.store.save_session(stale)
    finalized = service.finalize_stale_sessions(tmp_path)
    assert stale.session_id in {item.session_id for item in finalized}
    assert service.store.get_session(stale.session_id).status == "abandoned"


def test_new_closes_previous_and_opens_fresh_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway, _, _ = _gateway(tmp_path, _RouteModel("conversation"))
    first = gateway.create_session(frontend="test")
    second = gateway.start_new_conversation(first, frontend="test")

    service = WorkspaceService()
    assert second != first
    assert first not in {item.session_id for item in service.store.list_sessions()}
    assert service.store.get_session(second).status == "active"
