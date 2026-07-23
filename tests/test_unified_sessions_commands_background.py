from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


from mana_agent.background.manager import BackgroundProcessManager
from mana_agent.background.store import ProcessStore
from mana_agent.chat_commands import CommandContext, CommandDispatcher, build_default_registry
from mana_agent.chat_commands.models import CommandInvocation
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRouteRegistry, EntryRouter, RouteAvailability, RouteRegistration
from mana_agent.sessions import SessionService
from mana_agent.sessions.migration import DashboardConversationMigration


class _IdleGateway:
    def status(self, _session_id: str) -> str:
        return "ready"

    def cancel(self, _session_id: str) -> bool:
        return False


def test_destructive_replacement_deletes_record_history_and_keeps_workspace(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "mana-home"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("MANA_HOME", str(home))
    service = SessionService()
    old = service.create(repo, frontend="test")
    service.history.append(old.session_id, role="user", content="private old text", turn_id="turn-1")
    workspace_path = home / "workspaces" / old.workspace_id / "workspace.json"
    repository_path = home / "repositories" / old.primary_repository_id / "repository.json"

    new = service.replace(old.session_id, gateway=_IdleGateway(), frontend="test")

    assert new.session_id != old.session_id
    assert not (home / "sessions" / old.session_id).exists()
    assert old.session_id not in {row.session_id for row in service.workspaces.store.list_sessions()}
    assert workspace_path.exists() and repository_path.exists()
    assert (home / "runtime" / "session-tombstones" / f"{old.session_id}.json").exists()


def test_switch_reopens_and_restores_exact_chronological_history(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    service = SessionService()
    record = service.create(repo)
    service.history.append(record.session_id, role="user", content="one", turn_id="1", created_at="2024-01-01T00:00:00+00:00")
    service.history.append(record.session_id, role="assistant", content="two", turn_id="1", created_at="2024-01-01T00:00:01+00:00")
    service.workspaces.close_session(record.session_id)

    activation = service.bind(record.session_id, frontend="tui", workspace_id=record.workspace_id)

    assert activation.session.status == "active"
    assert [item["content"] for item in activation.messages] == ["one", "two"]


def test_shared_registry_is_identical_and_requires_delete_confirmation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    sessions = SessionService()
    session = sessions.create(repo)
    registry = build_default_registry()
    dispatcher = CommandDispatcher(registry)
    base = dict(
        session_id=session.session_id, workspace_id=session.workspace_id,
        repository_id=session.primary_repository_id, capabilities={"chat", "sessions"}, sessions=sessions,
    )
    results = [
        dispatcher.dispatch("/sessions list", CommandContext(frontend=frontend, **base))
        for frontend in ("cli", "tui", "dashboard", "api", "telegram")
    ]
    assert all(result and result.data == results[0].data for result in results)
    pending = dispatcher.dispatch(
        f"/sessions delete {session.session_id}", CommandContext(frontend="cli", **base)
    )
    assert pending and pending.status == "confirmation_required"
    assert sessions.workspaces.store.get_session(session.session_id)


def test_natural_language_uses_model_resolver_not_keyword_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    sessions = SessionService()
    session = sessions.create(repo)

    class Resolver:
        def resolve_command(self, text, *, commands, context):  # noqa: ANN001
            assert text == "Show my sessions."
            return CommandInvocation(name="sessions", arguments=["list"])

    dispatcher = CommandDispatcher(build_default_registry(), natural_language_resolver=Resolver())
    context = CommandContext(
        frontend="cli", session_id=session.session_id, workspace_id=session.workspace_id,
        repository_id=session.primary_repository_id, capabilities={"chat", "sessions"}, sessions=sessions,
    )
    assert dispatcher.dispatch("Show my sessions.", context).status == "success"
    assert CommandDispatcher(build_default_registry()).dispatch("Show my sessions.", context) is None


def test_gateway_entry_model_resolves_natural_language_to_registered_command() -> None:
    registry = EntryRouteRegistry()
    registry.register(RouteRegistration("command", "shared commands", lambda: RouteAvailability(True), ("sessions",)))

    class Model:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(content=json.dumps({
                "route": "command", "confidence": 0.99, "reason": "list chats",
                "required_sources": ["none"], "target_urls": [],
                "command_name": "sessions", "command_arguments": ["list"],
            }))

    decision = EntryRouter(llm=Model(), registry=registry).route(
        user_prompt="Show my sessions.",
        context=EntryRouteContext(session_id="session_x", conversation_id="session_x", turn_id="turn_x"),
    )
    assert decision.route == "command"
    assert decision.command_name == "sessions"
    assert decision.command_arguments == ("list",)


def test_background_store_survives_manager_restart_and_recovers_stale(monkeypatch, tmp_path: Path) -> None:
    store = ProcessStore(tmp_path / "runtime")
    manager = BackgroundProcessManager(store)
    monkeypatch.setattr("mana_agent.background.manager._identity", lambda pid: "identity")
    monkeypatch.setattr("mana_agent.background.manager.subprocess.Popen", lambda *a, **k: SimpleNamespace(pid=54321))
    record = manager.start("connector.telegram", process_type="connector", singleton_key="telegram")
    assert BackgroundProcessManager(store).inspect(record.process_id).state == "running"
    monkeypatch.setattr("mana_agent.background.manager._identity", lambda pid: "different")
    recovered = BackgroundProcessManager(store).recover_stale(record.process_id)
    assert recovered[0].state == "stale"


def test_dashboard_migration_is_idempotent_and_preserves_order(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("MANA_HOME", str(home))
    service = SessionService()
    repository = service.workspaces.register_repository(repo)
    source = home / "repositories" / repository.repository_id / "dashboard" / "conversations" / "conv_old"
    source.mkdir(parents=True)
    (source / "meta.json").write_text(json.dumps({
        "conversation_id": "conv_old", "root": str(repo), "title": "Imported",
        "created_at": "2024-01-01T00:00:00+00:00", "updated_at": "2024-01-01T00:00:02+00:00",
    }), encoding="utf-8")
    (source / "messages.jsonl").write_text(
        json.dumps({"message_id": "one", "role": "user", "content": "first", "created_at": "2024-01-01T00:00:00+00:00"}) + "\n" +
        json.dumps({"message_id": "two", "role": "assistant", "content": "second", "created_at": "2024-01-01T00:00:01+00:00"}) + "\n",
        encoding="utf-8",
    )
    migration = DashboardConversationMigration(service.workspaces, service.history)
    first = migration.run()
    second = migration.run()
    assert len(first) == 1 and second == []
    assert [row.content for row in service.history.list(first[0])] == ["first", "second"]
