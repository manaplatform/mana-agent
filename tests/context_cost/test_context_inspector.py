from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from mana_agent.commands.ui_helpers import _classify_direct_command, _render_context_inspector
from mana_agent.context_cost.inspector import ContextInspector, MAX_SEGMENTS, MAX_SNAPSHOTS
from mana_agent.context_cost.models import ContextSegment

SCOPE = ("session", "workspace", "repository")


def test_order_accounting_redaction_and_removed_candidates():
    inspector = ContextInspector(SCOPE)
    secret = "sk-secret /private/path [bold red]"
    system = ContextSegment("system", secret, 19, source_id=secret, metadata={"reason": secret})
    memory = ContextSegment("memory", secret, 9, source_id=secret)
    schema = ContextSegment("schema", {"secret": secret}, 13, source_id=secret)
    candidates = [system, memory, schema]
    estimate = SimpleNamespace(components={"system:0": 11, "schema:1": 7, "safety_margin": 3}, input_tokens=21)
    inspector.capture("private-call", candidates, [system, schema], scope=SCOPE, boundary="compatible_chat", blocked=False, estimate=estimate, context_window=100, model_identity=secret, task_id=secret)
    snapshot = inspector.history(scope=SCOPE)[0]
    assert [s.category for s in snapshot.segments] == ["system", "memory", "schema"]
    assert [s.token_estimate for s in snapshot.segments] == [11, None, 7]
    assert snapshot.segments[1].status == "removed before admission"
    assert snapshot.admitted_segment_tokens == 18
    assert dict(snapshot.category_totals) == {"system": 11, "schema": 7}
    assert snapshot.accounting_allowance == 3
    assert secret not in repr(asdict(snapshot))
    assert "private-call" not in repr(asdict(snapshot))
    assert snapshot.model_ref and snapshot.task_ref and snapshot.model_ref != snapshot.task_ref
    assert candidates == [system, memory, schema] and system.content == secret
    inspector.finish("private-call", lifecycle="completed", provider_input_tokens=25)
    assert inspector.history(scope=SCOPE)[0].provider_input_tokens == 25


def test_scope_reset_lifecycle_and_bounded_parallel_history():
    inspector = ContextInspector(SCOPE)
    segment = ContextSegment("history", "private", 2)
    def capture(i):
        inspector.capture(str(i), [segment] * (MAX_SEGMENTS + 1), [segment] * (MAX_SEGMENTS + 1), scope=SCOPE, boundary="admission", blocked=False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(capture, range(MAX_SNAPSHOTS + 10)))
    snapshots = inspector.history(scope=SCOPE)
    assert len(snapshots) == MAX_SNAPSHOTS
    assert all(len(item.segments) == MAX_SEGMENTS and item.omitted_segments == 1 for item in snapshots)
    assert all(item.admitted_segment_tokens == 2 * (MAX_SEGMENTS + 1) for item in snapshots)
    assert inspector.history(scope=("other", "workspace", "repository")) == ()
    assert inspector.history(scope=("session", "", "repository")) == ()
    inspector.reset(("other", "workspace", "repository"))
    inspector.capture("stale", [segment], [segment], scope=SCOPE, boundary="admission", blocked=False)
    assert inspector.history(scope=("other", "workspace", "repository")) == ()


def test_old_admission_cannot_reappear_after_returning_to_same_scope():
    inspector = ContextInspector(SCOPE)
    scope, epoch = inspector.stamp()
    segment = ContextSegment("user", "private", 1)
    inspector.reset(("other", "workspace", "repository"))
    inspector.reset(SCOPE)
    inspector.capture("old", [segment], [segment], scope=scope, epoch=epoch, boundary="admission", blocked=False)
    assert inspector.history(scope=SCOPE) == ()


def test_blocked_and_released_states_are_not_submitted():
    inspector = ContextInspector(SCOPE)
    segment = ContextSegment("user", "secret", 5)
    inspector.capture("blocked", [segment], [segment], scope=SCOPE, boundary="codex_outer", blocked=True)
    snapshot = inspector.history(scope=SCOPE)[0]
    assert snapshot.lifecycle == "blocked"
    assert snapshot.segments[0].status == "blocked; not submitted"
    assert snapshot.accounting_input_estimate is None
    inspector.capture("failed", [segment], [segment], scope=SCOPE, boundary="ask_agent", blocked=False)
    inspector.finish("failed", lifecycle="released")
    assert inspector.history(scope=SCOPE)[0].lifecycle == "released"


def test_cli_displays_only_scoped_safe_metadata():
    inspector = ContextInspector(SCOPE)
    segment = ContextSegment("system", "supersecret", 5, source_id="/private/path")
    inspector.capture("internal", [segment], [segment], scope=SCOPE, boundary="codex_outer", blocked=False)
    state = SimpleNamespace(context_cost_governor=SimpleNamespace(context_inspector=inspector), session_id=SCOPE[0], workspace_id=SCOPE[1], repository_id=SCOPE[2])
    output = StringIO()
    console = Console(file=output, width=160, color_system=None)
    assert _classify_direct_command("/context history") == "/context"
    _render_context_inspector(console, state, "/context last")
    text = output.getvalue()
    assert "NOT_OBSERVABLE" in text and "codex_outer" in text
    assert "supersecret" not in text and "/private/path" not in text
    output.seek(0); output.truncate(0)
    state.session_id = "other"
    _render_context_inspector(console, state, "/context last")
    assert "not found" in output.getvalue()


def test_registered_local_context_command_is_scoped_and_metadata_only():
    from mana_agent.chat_commands import CommandDispatcher, build_default_registry
    from mana_agent.chat_commands.models import CommandContext

    inspector = ContextInspector(SCOPE)
    secret = "supersecret /private/path"
    segment = ContextSegment("system", secret, 5, source_id=secret)
    inspector.capture("private-call-id", [segment], [segment], scope=SCOPE, boundary="codex_outer", blocked=False)
    snapshot = inspector.history(scope=SCOPE)[0]
    gateway = SimpleNamespace(get_stack=lambda: SimpleNamespace(context_cost_governor=SimpleNamespace(context_inspector=inspector)))
    dispatcher = CommandDispatcher(build_default_registry())
    context = CommandContext(frontend="tui", session_id=SCOPE[0], workspace_id=SCOPE[1], repository_id=SCOPE[2], capabilities={"gateway", "chat"}, gateway=gateway)

    assert "/context" in dispatcher.dispatch("/help", context).message
    for command in ("/context", "/context last", f"/context show {snapshot.invocation_id}"):
        result = dispatcher.dispatch(command, context)
        assert result.status == "success" and "NOT_OBSERVABLE" in result.message
        assert secret not in result.message and "private-call-id" not in result.message
    assert snapshot.invocation_id in dispatcher.dispatch("/context history", context).message
    assert "not found" in dispatcher.dispatch("/context show nonexistent", context).message
    assert dispatcher.dispatch("/context show", context).status == "error"
    assert dispatcher.dispatch('/context show "two words"', context).status == "error"
    context.session_id = "other"
    assert "not found" in dispatcher.dispatch("/context last", context).message
    context.session_id = SCOPE[0]
    context.workspace_id = ""
    assert "No context snapshots" in dispatcher.dispatch("/context history", context).message
    context.frontend = "api"
    assert "intentionally unavailable" in dispatcher.dispatch("/context", context).message
    context.frontend = "tui"
    context.gateway = None
    assert dispatcher.dispatch("/context", context).status == "error"
    assert "Unsupported command" in dispatcher.dispatch("/not-a-command", context).message


def test_governor_admission_and_usage_are_observed_without_mutating_segments(tmp_path, monkeypatch):
    from mana_agent.context_cost.governor import ContextCostGovernor

    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    governor = ContextCostGovernor(
        session_id=SCOPE[0], workspace_id=SCOPE[1], repository_id=SCOPE[2],
        settings=SimpleNamespace(mana_context_cost_log_enabled=False),
    )
    segments = [ContextSegment("user", "request", 3, protected=True)]
    call_id, decision = governor.before_model_call(
        segments, model="unknown-test", expected_output_tokens=20,
        historical_prediction_enabled=False,
    )
    assert decision.allowed
    assert decision.segments == tuple(segments)
    snapshot = governor.context_inspector.history(scope=SCOPE)[0]
    assert snapshot.lifecycle == "admitted"
    assert snapshot.segments[0].status == "admitted; submission not observed"
    governor.record_model_call(call_id, usage={"input_tokens": 12, "output_tokens": 3}, model="unknown-test")
    snapshot = governor.context_inspector.history(scope=SCOPE)[0]
    assert snapshot.lifecycle == "completed" and snapshot.provider_input_tokens == 12
    assert segments[0].content == "request"
    second_call, _ = governor.before_model_call(
        segments, model="unknown-test", expected_output_tokens=20,
        historical_prediction_enabled=False,
    )
    governor.record_model_call(second_call, usage={"output_tokens": 3}, model="unknown-test")
    assert governor.context_inspector.history(scope=SCOPE)[0].provider_input_tokens is None
