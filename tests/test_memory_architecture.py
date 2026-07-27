from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from mana_agent.config.session import ConfigurationDraft
from mana_agent.memory import (
    MemoryConfig,
    MemoryConfigurationError,
    MemoryContent,
    MemoryDependencyError,
    MemoryHealthStatus,
    MemoryScope,
    MemorySearchRequest,
    MemoryService,
    MemoryUpdateRequest,
    MemoryWriteRequest,
)
from mana_agent.memory.providers.mem0.mapper import response_to_record, scope_to_filters, scope_to_mem0


def test_internal_mode_is_default_and_reuses_existing_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = MemoryService(root=tmp_path)
    flow_id = service.ensure_flow(flow_id=None, request="Preserve existing memory")
    reopened = MemoryService(root=tmp_path)
    assert service.status() == {"mode": "internal", "provider": "mana"}
    assert reopened.get_flow_summary(flow_id).objective == "Preserve existing memory"


def test_internal_canonical_records_are_scope_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))

    async def run() -> None:
        service = MemoryService(root=tmp_path)
        first_scope = MemoryScope(user_id="one", session_id="session", repository_id="repo")
        second_scope = MemoryScope(user_id="two", session_id="session", repository_id="repo")
        record = await service.add(MemoryWriteRequest(MemoryContent("uses pytest"), first_scope))
        assert await service.get(record.id, second_scope) is None
        assert [item.id for item in await service.search(MemorySearchRequest("pytest", first_scope))] == [record.id]
        assert await service.search(MemorySearchRequest("pytest", second_scope)) == []
        await service.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("mode", "provider"),
    [("internal", "mem0"), ("external", "mana"), ("other", "mana")],
)
def test_invalid_mode_provider_combinations_stop(mode: str, provider: str) -> None:
    with pytest.raises(MemoryConfigurationError):
        MemoryConfig(mode=mode, provider=provider, api_key="test").validate()


def test_external_mode_requires_key_and_never_falls_back() -> None:
    with pytest.raises(MemoryConfigurationError, match="requires"):
        MemoryConfig(mode="external", provider="mem0").validate()
    with pytest.raises(MemoryConfigurationError, match="no fallback"):
        MemoryConfig(mode="external", provider="mem0", api_key="test", fallback_to_internal=True).validate()


def test_internal_mode_does_not_resolve_retained_external_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mana_agent.memory.config.MemorySecretStore.get",
        lambda *_args: (_ for _ in ()).throw(AssertionError("keyring should stay lazy")),
    )
    config = MemoryConfig.load(
        {
            "MANA_MEMORY_MODE": "internal",
            "MANA_MEMORY_PROVIDER": "mana",
            "MANA_MEMORY_SECRET_REF": "mem0:retained",
        }
    )
    assert config.mode == "internal"


def test_scope_mapping_keeps_dimensions_separate() -> None:
    scope = MemoryScope(
        user_id="u", agent_id="a", session_id="s", workspace_id="w",
        repository_id="r", conversation_id="c", task_id="t",
    )
    entities, metadata = scope_to_mem0(scope)
    assert entities == {"user_id": "u", "agent_id": "a", "run_id": "s", "app_id": "w"}
    assert metadata == {"mana_repository_id": "r", "mana_conversation_id": "c", "mana_task_id": "t"}
    assert scope_to_filters(scope) == {
        "AND": [
            *[{key: value} for key, value in entities.items()],
            {"metadata": metadata},
        ]
    }
    assert scope_to_filters(scope, {"mana_kind": "task"})["AND"][-1] == {
        "metadata": {**metadata, "mana_kind": "task"}
    }


def test_mem0_response_normalization_preserves_provider_id_and_metadata() -> None:
    record = response_to_record(
        {"id": "mem-1", "memory": "fact", "score": 0.75, "metadata": {"kind": "decision"}, "categories": ["work"]},
        MemoryScope(user_id="u"),
    )
    assert (record.id, record.content.text, record.score, record.provider) == ("mem-1", "fact", 0.75, "mem0")
    assert record.metadata == {"kind": "decision"}
    assert record.provider_metadata == {"categories": ["work"]}


def test_mem0_search_requires_positive_entity_scope(tmp_path: Path) -> None:
    async def run() -> None:
        service = MemoryService(
            root=tmp_path,
            config=MemoryConfig(mode="external", provider="mem0", api_key="test"),
        )
        with pytest.raises(MemoryConfigurationError, match="user, agent, session, or workspace"):
            await service.search(
                MemorySearchRequest(
                    query="task",
                    scope=MemoryScope(repository_id="repository-only"),
                    metadata={"mana_kind": "task"},
                )
            )

    asyncio.run(run())


def test_mem0_async_add_acknowledgement_is_a_pending_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class AsyncAckClient:
        def __init__(self, **_kwargs): pass
        def add(self, _text: str, **_kwargs):
            return {"event_id": "event-123", "message": "Memory processing started"}

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(MemoryClient=AsyncAckClient))

    async def run() -> None:
        service = MemoryService(
            root=tmp_path,
            config=MemoryConfig(mode="external", provider="mem0", api_key="test"),
        )
        record = await service.add(
            MemoryWriteRequest(
                MemoryContent("repository fact"),
                MemoryScope(repository_id="repo"),
            )
        )
        assert record.id == "event-123"
        assert record.provider_metadata["pending"] is True

    asyncio.run(run())


def test_mem0_client_is_lazy_reused_and_normalized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instances: list[object] = []

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "super-secret"
            instances.append(self)

        def add(self, text: str, **kwargs):
            assert "super-secret" not in repr(kwargs)
            return {"results": [{"id": "mem-1", "memory": text, "metadata": kwargs["metadata"]}]}

        def search(self, query: str, **kwargs):
            return {"results": [{"id": "mem-1", "memory": query, "score": 0.9}]}

        def get(self, memory_id: str):
            return {"id": memory_id, "memory": "old"}

        def update(self, memory_id: str, data: dict):
            return {"id": memory_id, "memory": data["text"], "metadata": data["metadata"]}

        def delete(self, memory_id: str): return {"id": memory_id}
        def delete_all(self, **kwargs): return kwargs

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(MemoryClient=FakeClient))

    async def run() -> None:
        service = MemoryService(root=tmp_path, config=MemoryConfig(mode="external", provider="mem0", api_key="super-secret"))
        assert instances == []
        scope = MemoryScope(user_id="u", session_id="s", repository_id="r")
        added = await service.add(MemoryWriteRequest(MemoryContent("fact"), scope, {"kind": "test"}))
        found = await service.search(MemorySearchRequest("fact", scope))
        updated = await service.update("mem-1", MemoryUpdateRequest(MemoryContent("new"), scope))
        assert added.id == found[0].id == updated.id == "mem-1"
        assert len(instances) == 1
        with pytest.raises(MemoryConfigurationError, match="no internal fallback"):
            service.ensure_flow(flow_id=None, request="must not write locally")
        await service.close()

    asyncio.run(run())


def test_external_runtime_operations_use_mem0_without_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writes: list[dict[str, object]] = []
    searches: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, **_kwargs): pass

        def add(self, text: str, **kwargs):
            writes.append({"text": text, **kwargs})
            return {"results": [{"id": f"mem-{len(writes)}", "memory": text}]}

        def search(self, _query: str, **kwargs):
            searches.append(kwargs)
            return {"results": []}

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(MemoryClient=FakeClient))
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = MemoryService(
        root=tmp_path,
        config=MemoryConfig(mode="external", provider="mem0", api_key="test"),
        workspace_id="workspace-1",
        repository_id="repository-1",
        session_id="session-1",
    )
    service.remember_repository_fact(f"Repository root: {tmp_path}")
    service.record_decision(
        agent_id="main",
        task_id="pending",
        decision_type="route",
        input_summary="chat command",
        memory_used=[],
        decision="create task",
        reason="validated model decision",
    )
    normalized, fingerprint = service.normalize_task(goal="chat command")
    service.register_task(
        task_id="task-1",
        normalized_goal=normalized,
        fingerprint=fingerprint,
        assigned_agent_id="main",
        repository_ids=["repository-1"],
    )
    bundle = service.build_bundle(
        agent_id="main",
        agent_role="main",
        task_id="task-1",
    )
    assert bundle.repository_ids == ["repository-1"]
    assert {row["metadata"]["mana_kind"] for row in writes} == {
        "repository_fact",
        "agent_decision",
        "task",
    }
    filters = searches[0]["filters"]
    assert set().union(*(clause.keys() for clause in filters["AND"])) <= {
        "user_id",
        "agent_id",
        "run_id",
        "app_id",
        "metadata",
    }
    assert filters["AND"][-1]["metadata"]["fingerprint"] == fingerprint
    assert not (tmp_path / "home" / "repositories" / "repository-1").exists()


def test_missing_mem0_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delitem(sys.modules, "mem0", raising=False)
    from mana_agent.memory.providers.mem0 import client as client_module

    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "mem0":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)

    async def run() -> None:
        service = MemoryService(root=tmp_path, config=MemoryConfig(mode="external", provider="mem0", api_key="test"))
        health = await service.healthcheck()
        assert health.status is MemoryHealthStatus.DEPENDENCY_ERROR
        assert "mana-agent[mem0]" in health.detail

    asyncio.run(run())


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (type("Unauthorized", (Exception,), {"status_code": 401})("bad key super-secret"), MemoryHealthStatus.AUTHENTICATION_ERROR),
        (TimeoutError("super-secret timed out"), MemoryHealthStatus.NETWORK_ERROR),
        (RuntimeError("provider broke with super-secret"), MemoryHealthStatus.PROVIDER_ERROR),
    ],
)
def test_mem0_health_normalizes_failures_without_secret(
    failure: Exception,
    expected: MemoryHealthStatus,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingClient:
        def __init__(self, **_kwargs): pass
        def get_all(self, **_kwargs): raise failure

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(MemoryClient=FailingClient))

    async def run() -> None:
        service = MemoryService(root=tmp_path, config=MemoryConfig(mode="external", provider="mem0", api_key="super-secret"))
        health = await service.healthcheck()
        assert health.status is expected
        assert "super-secret" not in health.detail
        assert "super-secret" not in caplog.text

    asyncio.run(run())


def test_mem0_key_becomes_keyring_reference_not_plain_config(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: dict[str, object] = {}
    monkeypatch.setattr("mana_agent.memory.config.MemorySecretStore.set", lambda _self, key, reference="": "mem0:ref" if key == "secret-value" else "")
    monkeypatch.setattr("mana_agent.config.session.save_effective_user_config", lambda values, merge=False: saved.update(values))
    monkeypatch.setattr("mana_agent.config.session.invalidate_model_cache", lambda: None)
    draft = ConfigurationDraft(
        original={},
        values={"MANA_MEMORY_MODE": "external", "MANA_MEMORY_PROVIDER": "mem0", "MEM0_API_KEY": "secret-value"},
    )
    draft.save()
    assert saved["MANA_MEMORY_SECRET_REF"] == "mem0:ref"
    assert "MEM0_API_KEY" not in saved
    assert "secret-value" not in repr(saved)


def test_low_level_config_writer_never_persists_mem0_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mana_agent.config import user_config

    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    user_config.save_effective_user_config(
        {"MANA_MEMORY_MODE": "external", "MANA_MEMORY_PROVIDER": "mem0", "MEM0_API_KEY": "never-write-this"},
        merge=False,
    )
    persisted = (tmp_path / "config.toml").read_text(encoding="utf-8") + (tmp_path / "secrets.toml").read_text(encoding="utf-8")
    assert "MEM0_API_KEY" not in persisted
    assert "never-write-this" not in persisted


def test_configuration_tui_conditionally_shows_external_fields() -> None:
    from textual.widgets import Select
    from mana_agent.tui.configuration_app import ManaConfigurationApp

    async def run() -> None:
        draft = ConfigurationDraft(
            original={"MANA_MEMORY_MODE": "internal", "MANA_MEMORY_PROVIDER": "mana"},
            values={"MANA_MEMORY_MODE": "internal", "MANA_MEMORY_PROVIDER": "mana"},
        )
        app = ManaConfigurationApp(draft=draft)
        async with app.run_test() as pilot:
            assert app.query_one("#mem0-api-key").display is False
            app.query_one("#memory-mode", Select).value = "external"
            await pilot.pause()
            assert app.query_one("#mem0-api-key").display is True

    asyncio.run(run())


def test_configuration_tui_survives_github_status_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    from mana_agent.tui.configuration_app import ManaConfigurationApp

    monkeypatch.setattr("mana_agent.tui.configuration_app.shutil.which", lambda _name: "/bin/gh")
    monkeypatch.setattr(
        "mana_agent.tui.configuration_app.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["gh", "auth", "status"], 5)
        ),
    )
    assert ManaConfigurationApp._github_cli_status() == "GitHub CLI status unavailable"
