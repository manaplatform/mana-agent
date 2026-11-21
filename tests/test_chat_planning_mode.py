"""CLI chat planning shares the durable workflow and approval commands."""
from types import SimpleNamespace

import pytest

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
    def start(*_args, **_kwargs):
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
        async with app.run_test(size=(32, 12)) as pilot:
            app.push_screen(ModePrompt(prompt), answers.append)
            await pilot.pause()
            app.screen.query_one("#answer-0", Select).value = "reject"
            await pilot.click("#submit")
            await pilot.pause()

    asyncio.run(run())
    assert answers == [("answer", {"behavior": "reject"})]


@pytest.mark.parametrize("action", ["approve", "reject", "revise", "later"])
def test_tui_plan_buttons_stay_visible_when_the_terminal_narrows(wait_for_tui_widgets, action):
    import asyncio

    from textual.app import App
    from textual.containers import VerticalScroll
    from textual.widgets import Button, Input

    from mana_agent.tui.mode_prompt import ModePrompt

    prompt = {"kind": "mode.approval", "revision": 2, "preview": "One step to review.\n\n" * 40}
    responses = []

    async def run():
        app = App()
        async with app.run_test(size=(100, 32)) as pilot:
            screen = ModePrompt(prompt)
            app.push_screen(screen, responses.append)
            await wait_for_tui_widgets(pilot, "Button", count=4)

            dialog = screen.query_one("#mode-dialog")
            body = screen.query_one("#mode-content", VerticalScroll)
            for size in ((100, 32), (32, 12)):
                await pilot.resize_terminal(*size)
                await pilot.pause()
                assert body.max_scroll_y > 0
                body.scroll_end(animate=False)
                await pilot.pause()
                assert body.scroll_y > 0
                for button in screen.query(Button):
                    assert screen.region.contains_region(button.region)
                    assert dialog.content_region.contains_region(button.region)
                    assert button.region.y >= body.region.bottom

            button = screen.query_one(f"#{action}", Button)
            if action == "revise":
                button.active_effect_duration = 0
                assert await pilot.click(button)
                await pilot.pause()
                revision = screen.query_one("#revision-request", Input)
                assert app.focused is revision
                revision.value = "Change step two."
            if action in {"approve", "revise"}:
                button.focus()
                await pilot.pause()
                await pilot.press("enter")
            else:
                assert await pilot.click(button)
            await pilot.pause()

    asyncio.run(run())
    expected = None if action == "later" else (action, "Change step two." if action == "revise" else None)
    assert responses == [expected]


def test_tui_mode_dialogs_keep_review_content_out_of_the_transcript(tmp_path, wait_for_tui_widgets):
    import asyncio
    from threading import Event

    from textual.widgets import Input, Markdown

    from mana_agent.chat.events import AssistantMessageEvent, UserMessageEvent
    from mana_agent.chat.history import ChatHistory
    from mana_agent.tui.app import ManaChatApp
    from mana_agent.tui.widgets.message_input import MessageInput

    clarification = {
        "kind": "mode.clarification", "task_id": "task", "inbox_id": "question",
        "questions": [{"field_id": "behavior", "prompt": "Which behavior?", "choices": {}, "allow_free_form": True}],
    }
    review = {
        "kind": "mode.approval", "task_id": "task", "inbox_id": "review", "revision": 2,
        "preview": "Implementation plan body",
    }
    replies = []
    answered, approved = Event(), Event()
    history = ChatHistory()

    class Gateway:
        def __init__(self):
            self.mode_workflow = self
            self.prompt = clarification

        def create_session(self, **_kwargs):
            return "tui-mode-dialog-test"

        def pending_prompt(self, session_id, task_id, inbox_id):
            if self.prompt and (session_id, task_id, inbox_id) == (
                "tui-mode-dialog-test", self.prompt["task_id"], self.prompt["inbox_id"],
            ):
                return self.prompt
            return None

        def dispatch_command(self, text, **_kwargs):
            assert text in ("/plan Handle blank input", "/plan")
            return CommandResult(status="input_required", message="Duplicate question body", next_prompt=self.prompt)

        def respond(self, session_id, item_id, operation, *, frontend, answer, comment):
            replies.append((session_id, item_id, operation, frontend, answer, comment))
            if operation == ResponseOperation.ANSWER:
                self.prompt = review
                answered.set()
            else:
                self.prompt = None
                approved.set()
            return CommandResult(status="success", message="Duplicate response receipt")

    app = ManaChatApp(history=history, repo_root=tmp_path, gateway=Gateway())

    async def run():
        async with app.run_test(size=(58, 20)) as pilot:
            composer = app.query_one(MessageInput)
            composer.value = "/plan Handle blank input"
            await pilot.press("enter")
            await wait_for_tui_widgets(pilot, "#submit")
            assert await pilot.click("#later")
            await wait_for_tui_widgets(pilot, "#chat-input")
            assert replies == []
            composer.value = "/plan"
            await pilot.press("enter")
            await wait_for_tui_widgets(pilot, "#submit")
            app.screen.query_one("#custom-0", Input).value = "Keep the value."
            assert await pilot.click("#submit")
            assert await asyncio.to_thread(answered.wait, 5)

            app._handle_hub_session_event({
                "type": "mode.input_ready", "task_id": "task", "metadata": {"inbox_id": "review"},
            })
            await wait_for_tui_widgets(pilot, "#approve")
            assert len(app.screen.query(Markdown)) == 1
            assert await pilot.click("#approve")
            assert await asyncio.to_thread(approved.wait, 5)

            turn_id = next(event.turn_id for event in history.get_events() if isinstance(event, UserMessageEvent))
            app._handle_hub_session_event({
                "type": "coding.progress", "execution_id": turn_id, "title": "Applying planned changes",
            })
            await wait_for_tui_widgets(pilot, "ExecutionPanel")
            panel = app.chat_log.get_execution_panel(turn_id)
            assert any(event["title"] == "Applying planned changes" for event in panel.events)
            assert not [event for event in history.get_events() if isinstance(event, AssistantMessageEvent)]

    asyncio.run(run())
    assert replies == [
        ("tui-mode-dialog-test", "question", ResponseOperation.ANSWER, "tui", {"behavior": "Keep the value."}, ""),
        ("tui-mode-dialog-test", "review", ResponseOperation.APPROVE, "tui", None, ""),
    ]


def test_tui_mode_replay_shows_status_and_final_replies_without_review_content(wait_for_tui_widgets):
    import asyncio

    from mana_agent.chat.events import AssistantMessageEvent
    from mana_agent.chat.history import ChatHistory
    from mana_agent.tui.app import ManaChatApp

    messages = [
        {"content": "Full plan", "metadata": {"mode_prompt_kind": "mode.approval"}},
        {"content": "Full clarification", "metadata": {"mode_prompt_kind": "mode.clarification"}},
        {"content": "Approval receipt", "metadata": {"mode_response_operation": "approve"}},
        {"content": "Status plus full plan", "metadata": {"mode_status_message": "Plan is executing."}},
        {"content": "Verification failed."},
        {"content": "Implementation complete."},
    ]
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def run():
        async with app.run_test() as pilot:
            app._replace_timeline([{"role": "assistant", "turn_id": "plan-turn", **message} for message in messages])
            await wait_for_tui_widgets(pilot, "AssistantReply", count=3)
            assert [event.content for event in history.get_events() if isinstance(event, AssistantMessageEvent)] == [
                "Plan is executing.", "Verification failed.", "Implementation complete.",
            ]

    asyncio.run(run())
