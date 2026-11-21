from types import SimpleNamespace

from mana_agent.chat_commands import CommandContext, CommandDispatcher, build_default_registry
from mana_agent.chat_commands.models import CommandInvocation, CommandResult


class RecordingGateway:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.mode_workflow = SimpleNamespace(active=lambda _: None, start=self.start_plan)

    def start_plan(self, session_id: str, request: str) -> CommandResult:
        assert session_id == "session-1"
        self.requests.append(request)
        return CommandResult(status="success", message=f"Implementation Plan: {request}",
                             data={"plan": {"goal": request}})

    def process_turn(self, session_id: str, request: str) -> SimpleNamespace:
        assert session_id == "session-1"
        self.requests.append(request)
        return SimpleNamespace(answer="Plan generated", error=None)


def test_plan_command_accepts_unmatched_quote_in_natural_language() -> None:
    gateway = RecordingGateway()
    result = CommandDispatcher(build_default_registry()).dispatch(
        "/plan Fix the user's issue in the planning workflow",
        CommandContext(
            frontend="cli",
            session_id="session-1",
            capabilities={"gateway"},
            gateway=gateway,
        ),
    )

    assert result is not None
    assert result.status == "success"
    assert result.data["plan"]["goal"] == "Fix the user's issue in the planning workflow"
    assert "Implementation Plan: Fix the user's issue in the planning workflow" in result.message


def test_plan_command_still_accepts_quoted_request() -> None:
    gateway = RecordingGateway()
    result = CommandDispatcher(build_default_registry()).dispatch(
        '/plan "Improve the planning workflow"',
        CommandContext(
            frontend="cli",
            session_id="session-1",
            capabilities={"gateway"},
            gateway=gateway,
        ),
    )

    assert result is not None
    assert result.status == "success"
    assert result.data["plan"]["goal"] == "Improve the planning workflow"
    assert "Implementation Plan: Improve the planning workflow" in result.message


def test_plan_command_forwards_a_long_multiline_request() -> None:
    gateway = RecordingGateway()
    requirements = " ".join(f"requirement-{index}" for index in range(657))
    request = r'Keep "existing flows" in C:\repo\settings.json.' + f"\n\n- {requirements}"

    result = CommandDispatcher(build_default_registry()).dispatch(
        f"/plan\n{request}",
        CommandContext(
            frontend="tui",
            session_id="session-1",
            capabilities={"gateway"},
            gateway=gateway,
        ),
    )

    assert result is not None and result.status == "success"
    assert gateway.requests == [request]


def test_structured_commands_still_reject_unmatched_quotes() -> None:
    result = CommandDispatcher(build_default_registry()).dispatch(
        "/mode 'plan",
        CommandContext(frontend="cli", capabilities={"gateway"}),
    )

    assert result is not None
    assert result.status == "error"
    assert result.message == "Invalid command arguments: No closing quotation"


def test_structured_commands_still_reject_too_many_arguments() -> None:
    arguments = " ".join(["review-id"] * 65)
    result = CommandDispatcher(build_default_registry()).dispatch(
        f"/approve {arguments}",
        CommandContext(frontend="cli", capabilities={"gateway"}),
    )

    assert result is not None and result.status == "error"
    assert result.message.startswith("Invalid /approve arguments:")


def test_analyze_command_accepts_unmatched_quote_in_natural_language() -> None:
    gateway = RecordingGateway()
    result = CommandDispatcher(build_default_registry()).dispatch(
        "/analyze Check the user's requested behavior",
        CommandContext(
            frontend="cli",
            session_id="session-1",
            capabilities={"gateway"},
            gateway=gateway,
        ),
    )

    assert result is not None
    assert result.status == "success"
    assert gateway.requests == [
        "Run the registered analyze workflow: Check the user's requested behavior"
    ]


def test_analyze_invocation_preserves_already_parsed_arguments() -> None:
    gateway = RecordingGateway()
    result = CommandDispatcher(build_default_registry()).dispatch_invocation(
        CommandInvocation(name="analyze", arguments=["Inspect", "the user's requested behavior"]),
        CommandContext(
            frontend="cli",
            session_id="session-1",
            capabilities={"gateway"},
            gateway=gateway,
        ),
    )

    assert result.status == "success"
    assert gateway.requests == [
        "Run the registered analyze workflow: Inspect the user's requested behavior"
    ]


def test_activity_shows_recent_events_without_backward_page_navigation(monkeypatch) -> None:
    calls = []
    recent = [{"type": "tool.started", "status": "running", "title": "Reading implementation", "sequence": 12}]
    def history(**kwargs):
        calls.append(kwargs)
        return recent
    monkeypatch.setattr("mana_agent.services.execution_event_hub.get_execution_event_hub",
                        lambda: SimpleNamespace(history=history))
    monkeypatch.setattr("mana_agent.config.user_config.get_setting", lambda name, default=None: default)
    dispatcher = CommandDispatcher(build_default_registry())
    context = CommandContext(frontend="cli", session_id="session-1", repository_id="repo-1", capabilities={"chat"})

    result = dispatcher.dispatch("/activity", context)
    assert result is not None and result.status == "success"
    assert result.data["events"] == recent
    assert "Reading implementation" in result.message
    assert "Earlier:" not in result.message
    assert "before_sequence" not in calls[0]

    retired = dispatcher.dispatch("/activity history 12", context)
    assert retired is not None and retired.status == "error"
    assert retired.message == "Usage: /activity [minimal|normal|verbose]"
    assert len(calls) == 1
