"""CLI chat planning shares the durable workflow and approval commands."""
from types import SimpleNamespace

from mana_agent.chat_commands import CommandContext, CommandDispatcher, build_default_registry
from mana_agent.chat_commands.models import CommandResult
from mana_agent.human_inbox.models import ResponseOperation


def dispatcher_for(workflow, **gateway_methods):
    gateway = SimpleNamespace(mode_workflow=workflow, **gateway_methods)
    dispatcher = CommandDispatcher(build_default_registry())
    context = CommandContext(session_id="session", frontend="cli", gateway=gateway, capabilities={"gateway"})
    return lambda text: dispatcher.dispatch(text, context)


def test_plan_enters_explicit_mode_without_making_up_a_goal():
    selected = []
    dispatch = dispatcher_for(SimpleNamespace(active=lambda _: None),
                              set_session_execution_mode=lambda *args: selected.append(args))
    result = dispatch("/plan")
    assert result.status == "input_required"
    assert selected == [("session", "plan")]


def test_plan_status_reuses_pending_clarification_without_another_model_call():
    pending = object()
    dispatch = dispatcher_for(SimpleNamespace(
        active=lambda _: pending,
        describe=lambda task: CommandResult(status="input_required", message="Which behavior?",
                                            next_prompt={"kind": "mode.clarification", "item": task is pending}),
    ))
    result = dispatch("/plan")
    assert result.next_prompt == {"kind": "mode.clarification", "item": True}


def test_approval_aliases_require_and_forward_the_exact_reviewed_item():
    responses = []
    workflow = SimpleNamespace(active=lambda _: object(), respond=lambda *args, **kw: (
        responses.append((args, kw)) or CommandResult(status="success", message="Approved")
    ))
    dispatch = dispatcher_for(workflow)
    assert dispatch("/act").status == "error"
    assert dispatch("/approve").status == "error"
    assert responses == []
    for command in ["/approve", "/accept", "/act"]:
        assert dispatch(f"{command} inbox-review-2").status == "success"
    assert responses == [(("session", "inbox-review-2", ResponseOperation.APPROVE), {"frontend": "cli"})] * 3


def test_model_failure_remains_visible_at_the_entry_point():
    def start(*_args):
        raise RuntimeError("Invalid or unavailable ModeDecision model decision")

    dispatch = dispatcher_for(SimpleNamespace(active=lambda _: None, start=start))
    result = dispatch("/plan Update the service")
    assert result.status == "error"
    assert "ModeDecision" in result.message


def test_tui_returns_the_selected_clarification():
    import asyncio

    from textual.app import App
    from textual.widgets import Select
    from mana_agent.tui.mode_prompt import ModePrompt

    prompt = {
        "kind": "mode.clarification",
        "questions": [{"field_id": "behavior", "prompt": "What should happen?",
                       "choices": {"keep": "Keep the value", "reject": "Reject the value"},
                       "recommended_choice": "keep", "allow_free_form": True}],
    }
    answers = []

    async def run():
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ModePrompt(prompt), answers.append)
            await pilot.pause()
            app.screen.query_one("#answer-0", Select).value = "reject"
            await pilot.click("#submit")
            await pilot.pause()

    asyncio.run(run())
    assert answers == [("answer", {"behavior": "reject"})]
