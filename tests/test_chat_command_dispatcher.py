from types import SimpleNamespace

from mana_agent.chat_commands import CommandContext, CommandDispatcher, build_default_registry


class RecordingGateway:
    def __init__(self) -> None:
        self.requests: list[str] = []

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


def test_structured_commands_still_reject_unmatched_quotes() -> None:
    result = CommandDispatcher(build_default_registry()).dispatch(
        "/mode 'plan",
        CommandContext(frontend="cli", capabilities={"gateway"}),
    )

    assert result is not None
    assert result.status == "error"
    assert result.message == "Invalid command arguments: No closing quotation"


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
