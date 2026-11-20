from __future__ import annotations

import asyncio
import json
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from mana_agent.config.session import ConfigurationDraft
from mana_agent.memory import (
    MemoryConfig,
    MemoryConfigurationError,
    MemoryContent,
    MemoryHealthStatus,
    MemoryNotFoundError,
    MemoryProviderError,
    MemoryScope,
    MemorySearchRequest,
    MemoryService,
    MemoryUpdateRequest,
    MemoryWriteRequest,
)
from mana_agent.memory.providers.mem0.mapper import response_to_record, scope_to_filters, scope_to_mem0
from mana_agent.memory.providers.shared import supermemory_document_id, supermemory_namespace
from mana_agent.memory.providers.supermemory.backend import SupermemoryProvider


def test_internal_mode_is_default_and_reuses_existing_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = MemoryService(root=tmp_path)
    flow_id = service.ensure_flow(flow_id=None, request="Preserve existing memory")
    reopened = MemoryService(root=tmp_path)
    status = service.status()
    assert status["mode"] == "internal"
    assert status["provider"] == "mana"
    assert status["capabilities"]["evidence"] is True
    assert status["capabilities"]["coding_flow"] is True
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
    [("internal", "mem0"), ("internal", "supermemory"), ("external", "mana"), ("other", "mana")],
)
def test_invalid_mode_provider_combinations_stop(mode: str, provider: str) -> None:
    with pytest.raises(MemoryConfigurationError):
        MemoryConfig(mode=mode, provider=provider, api_key="test").validate()


def test_external_mode_requires_key_and_never_falls_back() -> None:
    with pytest.raises(MemoryConfigurationError, match="requires"):
        MemoryConfig(mode="external", provider="mem0").validate()
    with pytest.raises(MemoryConfigurationError, match="requires"):
        MemoryConfig(mode="external", provider="supermemory").validate()
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


def test_supermemory_config_prefers_secret_ref_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "env-key")
    monkeypatch.setattr("mana_agent.memory.config.MemorySecretStore.get", lambda *_args: "secret-store-key")
    config = MemoryConfig.load(
        {
            "MANA_MEMORY_MODE": "external",
            "MANA_MEMORY_PROVIDER": "supermemory",
            "MANA_MEMORY_SECRET_REF": "supermemory:ref",
        }
    )
    assert config.api_key == "secret-store-key"


def test_supermemory_explicit_key_overrides_stored_secret_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "env-key")
    monkeypatch.setattr("mana_agent.memory.config.MemorySecretStore.get", lambda *_args: "old-secret-key")
    config = MemoryConfig.load(
        {
            "MANA_MEMORY_MODE": "external",
            "MANA_MEMORY_PROVIDER": "supermemory",
            "MANA_MEMORY_SECRET_REF": "supermemory:ref",
            "SUPERMEMORY_API_KEY": "new-inline-key",
        }
    )
    assert config.api_key == "new-inline-key"


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


_SUPERMEMORY_SCOPE = MemoryScope(
    user_id="u-1", workspace_id="workspace 1", repository_id="repo", agent_id="main", session_id="s1",
)
_SUPERMEMORY_NAMESPACE = "mana:scope:e54cb13b86fe7a00e2aa775e"
_SUPERMEMORY_DOCUMENT_PATH = f"/ns/{_SUPERMEMORY_NAMESPACE}/document"
_SUPERMEMORY_METADATA = {"memory_kind": "task", "fingerprint": "abc123", "custom_id": "existing-hint"}
_SUPERMEMORY_CALLER_ID = "mana:task:4ce0e8ad51d44406bf9b918a:f8859e938990"
_SUPERMEMORY_TIMESTAMPS = {"createdAt": "2026-10-02T10:00:00Z", "updatedAt": "2026-10-03T10:00:00Z"}
_SUPERMEMORY_DOCUMENT_SYSTEM = {"status": "done", "dreamingStatus": "dreaming", **_SUPERMEMORY_TIMESTAMPS}
_SupermemoryHandler = Callable[[httpx.Request], httpx.Response]
_SupermemoryBackendFactory = Callable[[_SupermemoryHandler], SupermemoryProvider]


@pytest.fixture
def supermemory_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[_SupermemoryBackendFactory]:
    sdk = pytest.importorskip("supermemory", minversion="5.0.1")
    sdk_class = sdk.Supermemory
    with ExitStack() as stack:
        def create(handler: _SupermemoryHandler) -> SupermemoryProvider:
            def client_factory(**kwargs: Any) -> Any:
                assert kwargs["max_retries"] == 2
                assert kwargs["timeout"] == 15.0
                client = stack.enter_context(httpx.Client(transport=httpx.MockTransport(handler)))
                return sdk_class(**{**kwargs, "max_retries": 0}, http_client=client)

            monkeypatch.setattr(sdk, "Supermemory", client_factory)
            return SupermemoryProvider(MemoryConfig(
                mode="external", provider="supermemory", api_key="test", base_url="https://memory.example.test",
            ))

        yield create


def test_supermemory_namespace_and_document_id_preserve_legacy_values() -> None:
    assert supermemory_namespace(_SUPERMEMORY_SCOPE) == _SUPERMEMORY_NAMESPACE
    assert supermemory_document_id(
        scope=_SUPERMEMORY_SCOPE, content="fact", metadata=_SUPERMEMORY_METADATA,
    ) == _SUPERMEMORY_CALLER_ID


def test_supermemory_v5_add_and_hybrid_search(supermemory_backend: _SupermemoryBackendFactory) -> None:
    scope = _SUPERMEMORY_SCOPE
    scope_metadata = {"source": "mana-agent", **scope.as_dict(), "mana_repository_id": "repo"}
    search_metadata = {"reviewed": True, "priority": 3, "confidence": 0.75}
    result_fields = {"isLatest": True, "isInference": False, "similarity": 0.91, "metadata": search_metadata}
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.host == "memory.example.test"
        assert request.headers["Authorization"] == "Bearer test"
        assert not request.url.params
        requests.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == _SUPERMEMORY_DOCUMENT_PATH:
            assert body == {
                "id": _SUPERMEMORY_CALLER_ID,
                "content": "fact",
                "taskType": "memory",
                "dreaming": "dynamic",
                "metadata": {**scope_metadata, **_SUPERMEMORY_METADATA},
            }
            return httpx.Response(200, json={"id": "doc-1", "status": "queued"})

        assert request.url.path == f"/ns/{_SUPERMEMORY_NAMESPACE}/search"
        assert body == {
            "query": "find fact",
            "searchMode": "hybrid",
            "threshold": 0.6,
            "limit": 5,
            "include": {"documents": True},
            "filter": {
                "operator": "and",
                "operands": [
                    {"field": key, "operator": "eq", "value": value}
                    for key, value in {**scope_metadata, **search_metadata}.items()
                ],
            },
        }
        operands = {item["field"]: item["value"] for item in body["filter"]["operands"]}
        assert operands["reviewed"] is True
        assert type(operands["priority"]) is int
        assert type(operands["confidence"]) is float
        return httpx.Response(200, json={
            "searchTime": 1,
            "results": [
                {
                    **result_fields, "id": "mem-1", "memory": "fact", "system": _SUPERMEMORY_TIMESTAMPS,
                    "included": {"document": {
                        "id": "doc-1", "title": "Source", "type": "text", "metadata": {},
                        "system": _SUPERMEMORY_TIMESTAMPS,
                    }},
                },
                {
                    **result_fields, "id": "chunk-1", "chunk": "source passage",
                    "system": {"updatedAt": _SUPERMEMORY_TIMESTAMPS["updatedAt"]},
                },
            ],
        })

    backend = supermemory_backend(respond)

    async def run() -> None:
        add = MemoryWriteRequest(MemoryContent("fact"), scope, _SUPERMEMORY_METADATA)
        added = await backend.add(add)
        assert added.id == "doc-1"
        assert added.provider_metadata == {
            "status": "queued", "namespace": _SUPERMEMORY_NAMESPACE, "caller_id": _SUPERMEMORY_CALLER_ID,
        }
        assert (await backend.add(add)).id == added.id
        memory, chunk = await backend.search(MemorySearchRequest("find fact", scope, 5, search_metadata))
        assert (memory.id, memory.content.text, memory.score) == ("mem-1", "fact", 0.91)
        assert memory.metadata == search_metadata
        assert memory.created_at == datetime(2026, 10, 2, 10, tzinfo=timezone.utc)
        assert memory.updated_at == chunk.updated_at == datetime(2026, 10, 3, 10, tzinfo=timezone.utc)
        assert memory.provider_metadata["document"] == {"id": "doc-1", "title": "Source", "type": "text"}
        assert (chunk.id, chunk.content.text, chunk.created_at) == ("chunk-1", "source passage", None)
        assert "document" not in chunk.provider_metadata

    asyncio.run(run())
    assert requests == [_SUPERMEMORY_DOCUMENT_PATH] * 2 + [f"/ns/{_SUPERMEMORY_NAMESPACE}/search"]


def test_supermemory_v5_scoped_document_lifecycle(supermemory_backend: _SupermemoryBackendFactory) -> None:
    document_path = f"{_SUPERMEMORY_DOCUMENT_PATH}/doc-1"
    other_scope = MemoryScope(user_id="different-user")
    patches: list[dict[str, Any]] = []
    deleted: list[list[str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            assert request.url.path == _SUPERMEMORY_DOCUMENT_PATH
            body = json.loads(request.content)
            assert body == {"ids": ["doc-1"]}
            deleted.append(body["ids"])
            return httpx.Response(200, json={"count": 1, "errors": []})
        if request.url.path == f"/ns/{supermemory_namespace(other_scope)}/document/doc-1":
            assert request.method == "GET"
            return httpx.Response(404, json={"error": "Document not found"})
        assert request.url.path == document_path
        if request.method == "PATCH":
            patches.append(json.loads(request.content))
            status = "queued" if "content" in patches[-1] else "done"
            return httpx.Response(200, json={"id": "doc-1", "status": status})
        assert request.method == "GET"
        return httpx.Response(200, json={
            "id": "doc-1", "title": "Source", "type": "text", "content": "published", "metadata": {"category": "old"},
            "system": _SUPERMEMORY_DOCUMENT_SYSTEM,
        })

    backend = supermemory_backend(respond)

    async def run() -> None:
        scope = _SUPERMEMORY_SCOPE
        document = await backend.get("doc-1", scope)
        assert document is not None
        assert document.content.text == "published"
        assert document.created_at == datetime(2026, 10, 2, 10, tzinfo=timezone.utc)
        assert document.provider_metadata["status"] == "done"
        assert document.provider_metadata["dreaming_status"] == "dreaming"
        assert await backend.get("doc-1", other_scope) is None
        with pytest.raises(MemoryNotFoundError):
            await backend.update("doc-1", MemoryUpdateRequest(MemoryContent("new"), other_scope))
        with pytest.raises(MemoryNotFoundError):
            await backend.delete("doc-1", other_scope)
        assert await backend.update("doc-1", MemoryUpdateRequest(scope=scope)) == document
        assert patches == []
        metadata_only = await backend.update("doc-1", MemoryUpdateRequest(scope=scope, metadata={"category": "new"}))
        assert metadata_only.content.text == "published"
        assert metadata_only.metadata["category"] == "new"
        assert list(patches[0]) == ["metadata"]
        assert patches[0]["metadata"]["mana_repository_id"] == "repo"
        assert patches[0]["metadata"]["category"] == "new"

        updated = await backend.update("doc-1", MemoryUpdateRequest(MemoryContent("replacement"), scope))
        assert (updated.id, updated.content.text, updated.provider_metadata["status"]) == ("doc-1", "replacement", "queued")
        assert patches[1:] == [{"content": "replacement", "taskType": "memory", "dreaming": "dynamic"}]
        await backend.delete("doc-1", scope)

    asyncio.run(run())
    assert deleted == [["doc-1"]]


def test_supermemory_v5_clear_batches_documents_and_reports_partial_failure(
    supermemory_backend: _SupermemoryBackendFactory,
) -> None:
    requests: list[tuple[str, int]] = []
    first_batch = [f"doc-{index}" for index in range(1, 101)]

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.method == "POST":
            assert request.url.path == f"/ns/{_SUPERMEMORY_NAMESPACE}/list/documents"
            page = int(request.url.params["page"])
            assert request.url.params["limit"] == "100"
            assert request.url.params["sort"] == "createdAt"
            assert request.url.params["order"] == "asc"
            assert body["include"] == {"memoryCount": False}
            assert body["filter"]["operator"] == "and"
            assert {item["field"]: item["value"] for item in body["filter"]["operands"]} == {
                "source": "mana-agent", **_SUPERMEMORY_SCOPE.as_dict(), "mana_repository_id": "repo",
            }
            requests.append(("list", page))
            ids = first_batch if page == 1 else ["doc-100", "doc-101"]
            return httpx.Response(200, json={
                "documents": [
                    {"id": doc_id, "title": None, "type": "text", "metadata": {}, "system": _SUPERMEMORY_DOCUMENT_SYSTEM}
                    for doc_id in ids
                ],
                "memories": [], "chunks": [],
                "pagination": {"currentPage": page, "totalPages": 2, "totalItems": 102},
            })
        assert request.method == "DELETE"
        assert request.url.path == _SUPERMEMORY_DOCUMENT_PATH
        ids = body["ids"]
        requests.append(("delete", len(ids)))
        if ids == first_batch:
            return httpx.Response(200, json={"count": 99, "errors": [
                {"id": "doc-100", "error": "Document is still processing"},
            ]})
        assert ids == ["doc-101"]
        return httpx.Response(200, json={"count": 1, "errors": []})

    backend = supermemory_backend(respond)
    with pytest.raises(MemoryProviderError, match=r"deleted 100 of 101 documents.*still processing"):
        asyncio.run(backend.clear(_SUPERMEMORY_SCOPE))
    assert requests == [("list", 1), ("list", 2), ("delete", 100), ("delete", 1)]


def test_supermemory_v5_clear_requires_scope_and_handles_empty_namespace(
    supermemory_backend: _SupermemoryBackendFactory,
) -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.method)
        assert request.url.path == f"/ns/{_SUPERMEMORY_NAMESPACE}/list/documents"
        return httpx.Response(200, json={
            "documents": [], "chunks": [], "memories": [],
            "pagination": {"currentPage": 1, "totalPages": 0, "totalItems": 0},
        })

    backend = supermemory_backend(respond)
    with pytest.raises(MemoryProviderError, match="Refusing to clear"):
        asyncio.run(backend.clear(MemoryScope()))
    assert requests == []
    asyncio.run(backend.clear(_SUPERMEMORY_SCOPE))
    assert requests == ["POST"]


@pytest.mark.parametrize(("status_code", "health_status"), [
    (200, MemoryHealthStatus.HEALTHY),
    (401, MemoryHealthStatus.AUTHENTICATION_ERROR),
    (403, MemoryHealthStatus.PROVIDER_ERROR),
    (404, MemoryHealthStatus.PROVIDER_ERROR),
    (429, MemoryHealthStatus.NETWORK_ERROR),
    (503, MemoryHealthStatus.NETWORK_ERROR),
])
def test_supermemory_v5_healthcheck(
    supermemory_backend: _SupermemoryBackendFactory,
    status_code: int,
    health_status: MemoryHealthStatus,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path in {"/ns", "/namespaces"}
        assert dict(request.url.params) == {"limit": "1"}
        if status_code != 200:
            return httpx.Response(status_code, json={"error": "private provider details"})
        return httpx.Response(200, json={
            "namespaces": [], "pagination": {"currentPage": 1, "totalPages": 0, "totalItems": 0},
        })

    backend = supermemory_backend(respond)

    async def run() -> None:
        health = await backend.healthcheck()
        assert health.status is health_status
        assert "private provider details" not in health.detail
        await backend.close()

    asyncio.run(run())


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
        # Semantic AI writes stay on the external backend. Local coding-flow
        # continuity remains available as a system store.
        flow_id = service.ensure_flow(flow_id=None, request="local coding flow is system state")
        assert flow_id
        assert service.capabilities.coding_flow is True
        assert service.capabilities.semantic_search is True
        with pytest.raises(MemoryConfigurationError, match="unmapped memory operation|no internal AI-memory fallback"):
            _ = service.not_a_real_memory_operation
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
    with pytest.raises(MemoryConfigurationError, match="Broad legacy memory bundles are disabled"):
        service.build_bundle(
            agent_id="main",
            agent_role="main",
            task_id="task-1",
        )
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
    monkeypatch.setattr(
        "mana_agent.memory.config.MemorySecretStore.set",
        lambda _self, key, reference="", provider="mem0": "mem0:ref" if key == "secret-value" and provider == "mem0" else "",
    )
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


def test_supermemory_key_becomes_keyring_reference_not_plain_config(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: dict[str, object] = {}
    monkeypatch.setattr(
        "mana_agent.memory.config.MemorySecretStore.set",
        lambda _self, key, reference="", provider="mem0": "supermemory:ref" if key == "secret-value" and provider == "supermemory" else "",
    )
    monkeypatch.setattr("mana_agent.config.session.save_effective_user_config", lambda values, merge=False: saved.update(values))
    monkeypatch.setattr("mana_agent.config.session.invalidate_model_cache", lambda: None)
    draft = ConfigurationDraft(
        original={},
        values={"MANA_MEMORY_MODE": "external", "MANA_MEMORY_PROVIDER": "supermemory", "SUPERMEMORY_API_KEY": "secret-value"},
    )
    draft.save()
    assert saved["MANA_MEMORY_SECRET_REF"] == "supermemory:ref"
    assert "SUPERMEMORY_API_KEY" not in saved
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


def test_low_level_config_writer_never_persists_supermemory_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mana_agent.config import user_config

    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    user_config.save_effective_user_config(
        {"MANA_MEMORY_MODE": "external", "MANA_MEMORY_PROVIDER": "supermemory", "SUPERMEMORY_API_KEY": "never-write-this"},
        merge=False,
    )
    persisted = (tmp_path / "config.toml").read_text(encoding="utf-8") + (tmp_path / "secrets.toml").read_text(encoding="utf-8")
    assert "SUPERMEMORY_API_KEY" not in persisted
    assert "never-write-this" not in persisted


def test_memory_secret_store_uses_protected_mana_store_without_keyring_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mana_agent.config.user_config import save_effective_user_config
    from mana_agent.memory.config import MemorySecretStore

    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    monkeypatch.setattr(MemorySecretStore, "_recommended_keyring", staticmethod(lambda: None))
    store = MemorySecretStore()

    reference = store.set("memory-secret-value", provider="mem0")

    assert reference == "mana-secrets:MEM0_API_KEY"
    assert store.get(reference) == "memory-secret-value"
    config_path = tmp_path / "config.toml"
    assert not config_path.exists() or "memory-secret-value" not in config_path.read_text(encoding="utf-8")
    assert "memory-secret-value" in (tmp_path / "secrets.toml").read_text(encoding="utf-8")
    if sys.platform != "win32":
        assert (tmp_path / "secrets.toml").stat().st_mode & 0o077 == 0

    save_effective_user_config({"MANA_MEMORY_SECRET_REF": reference}, merge=False)
    assert store.get(reference) == "memory-secret-value"

    assert store.set("rotated-memory-secret", reference, provider="mem0") == reference
    assert store.get(reference) == "rotated-memory-secret"


def test_memory_secret_store_rejects_unknown_mana_secret_reference() -> None:
    from mana_agent.memory.config import MemorySecretStore

    with pytest.raises(MemoryConfigurationError, match="invalid"):
        MemorySecretStore().get("mana-secrets:UNREGISTERED_SECRET")


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
            assert app.query_one("#memory-api-key").display is False
            app.query_one("#memory-mode", Select).value = "external"
            await pilot.pause()
            assert app.query_one("#memory-api-key").display is True
            app.query_one("#memory-provider", Select).value = "supermemory"
            await pilot.pause()
            assert app.query_one("#mem0-org-id").display is False
            assert "Supermemory API key" in app.query_one("#memory-api-key").placeholder

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
