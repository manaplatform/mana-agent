from __future__ import annotations

import threading
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError

from mana_agent.api.app import create_app
from mana_agent.canvas.catalog import CatalogValidationError, validate_components
from mana_agent.canvas.config import CanvasConfig, LOCAL_CATALOG_PATH, MANA_CATALOG_ID
from mana_agent.canvas.generation import CanvasGenerationError, parse_generated_messages
from mana_agent.canvas.runtime_tools import build_canvas_langchain_tools
from mana_agent.canvas.models import (
    CanvasEventEnvelope,
    CanvasEventType,
    CanvasSource,
    RendererAction,
)
from mana_agent.canvas.reducer import CanvasStateError, reduce_canvas_event
from mana_agent.canvas.service import CanvasService, canvas_service_for_root
from mana_agent.canvas.store import CanvasStore
from mana_agent.canvas.transactional import CanvasActionAdapter
from mana_agent.config.settings import Settings
from mana_agent.config.user_config import save_effective_user_config
from mana_agent.services.conversation_service import ConversationService
from mana_agent.services.execution_event_hub import (
    ExecutionEventHub,
    reset_execution_event_hub_for_tests,
)
from mana_agent.transactional_actions.models import ActionState
from mana_agent.transactional_actions.runtime import default_action_gateway


def _components() -> list[dict]:
    return [
        {
            "id": "root",
            "component": "Column",
            "children": ["title", "priority", "approve"],
        },
        {"id": "title", "component": "Heading", "text": "Project plan", "level": 2},
        {
            "id": "priority",
            "component": "Select",
            "label": "Priority",
            "options": ["low", "high"],
            "value": {"path": "/priority"},
        },
        {
            "id": "approve",
            "component": "Button",
            "label": "Approve",
            "actions": [
                {"name": "plan.press", "context": {"priority": {"path": "/priority"}}}
            ],
        },
    ]


@pytest.fixture()
def canvas(tmp_path: Path) -> CanvasService:
    return CanvasService(
        config=CanvasConfig(),
        store=CanvasStore(tmp_path / "canvas"),
        event_hub=ExecutionEventHub(),
        repository_id="",
    )


def test_create_update_action_recover_delete_lifecycle(
    canvas: CanvasService, tmp_path: Path
) -> None:
    snapshot = canvas.create_surface(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="plan",
        owner={"agent_id": "main", "task_id": "task-1"},
        correlation_id="turn-1",
    )
    assert snapshot.catalog_id == MANA_CATALOG_ID
    snapshot = canvas.update_components(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="plan",
        components=_components(),
        correlation_id="turn-1",
    )
    snapshot = canvas.update_data(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="plan",
        value={"priority": "high"},
        correlation_id="turn-1",
    )
    assert snapshot.data_model == {"priority": "high"}

    delivered: list[str] = []
    canvas.register_action_handler(
        snapshot.owner, lambda action, _surface: delivered.append(action.name)
    )
    action = RendererAction(
        action_id="action-1",
        session_id="session_one",
        conversation_id="session_one",
        surface_id="plan",
        source_component_id="approve",
        name="plan.press",
        correlation_id="turn-2",
        context={"priority": "high"},
    )
    assert canvas.submit_action(action).status == "delivered"
    assert delivered == ["plan.press"]
    with pytest.raises(CanvasStateError, match="Replayed"):
        canvas.submit_action(action)

    reconstructed = CanvasService(
        config=CanvasConfig(),
        store=canvas.store,
        event_hub=ExecutionEventHub(),
        repository_id="",
    )
    recovered, delta = reconstructed.replay("session_one", "plan")
    assert recovered.components[-1].id == "approve"
    assert recovered.last_sequence == 3
    assert delta == []
    deleted = reconstructed.delete_surface(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="plan",
        correlation_id="turn-3",
    )
    assert deleted.deleted is True
    with pytest.raises(CanvasStateError, match="Deleted"):
        reconstructed.update_data(
            session_id="session_one",
            conversation_id="session_one",
            surface_id="plan",
            value={"priority": "low"},
            correlation_id="turn-3",
        )


def test_transactional_canvas_create_commits_after_snapshot_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    tools = {tool.name: tool for tool in build_canvas_langchain_tools(tmp_path)}
    arguments = {
        "source_decision_id": "decision-pet",
        "session_id": "session-pet",
        "conversation_id": "session-pet",
        "surface_id": "pet",
        "owner": {"agent_id": "main", "task_id": "task-pet"},
        "components": [
            {"id": "root", "component": "Column", "children": ["title"]},
            {"id": "title", "component": "Heading", "text": "Pixel pet"},
        ],
        "data_model": {"mood": "happy"},
    }
    adapter = CanvasActionAdapter(
        tool_name="canvas_create_surface",
        arguments=arguments,
        invoke=lambda: tools["canvas_create_surface"].invoke(arguments),
        parent_task_id="task-pet",
        actor="model_tool",
        originating_agent="ask_agent",
    )

    outcome = default_action_gateway(tmp_path, enable_human_inbox=False).execute(adapter)

    assert outcome.action.state is ActionState.COMMITTED
    assert outcome.action.verification and outcome.action.verification.complete
    snapshot = canvas_service_for_root(tmp_path).get_surface("session-pet", "pet")
    assert snapshot.components[0].id == "root"


def test_wait_for_action_resumes_only_matching_surface_and_name(
    canvas: CanvasService,
) -> None:
    canvas.create_surface(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="plan",
        owner={"node_id": "node-1", "workflow_id": "flow-1"},
        correlation_id="turn-1",
    )
    canvas.update_components(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="plan",
        components=_components(),
        correlation_id="turn-1",
    )
    result: list[str] = []
    thread = threading.Thread(
        target=lambda: result.append(
            canvas.wait_for_action(
                session_id="session_one",
                surface_id="plan",
                action_name="plan.press",
                timeout=2,
            ).action_id
        )
    )
    thread.start()
    canvas.submit_action(
        RendererAction(
            action_id="action-node",
            session_id="session_one",
            conversation_id="session_one",
            surface_id="plan",
            source_component_id="approve",
            name="plan.press",
            correlation_id="turn-2",
            context={"priority": "low"},
        )
    )
    thread.join(timeout=2)
    assert result == ["action-node"]


def test_catalog_rejects_unsupported_executable_and_invalid_tree() -> None:
    config = CanvasConfig()
    with pytest.raises(CatalogValidationError, match="Unsupported component"):
        validate_components(
            [{"id": "root", "component": "RawHtml", "html": "<b>x</b>"}],
            surface_id="x",
            config=config,
        )
    with pytest.raises(CatalogValidationError, match="Executable HTML"):
        validate_components(
            [
                {
                    "id": "root",
                    "component": "Markdown",
                    "text": "<script>alert(1)</script>",
                }
            ],
            surface_id="x",
            config=config,
        )
    with pytest.raises(CatalogValidationError, match="cycle"):
        validate_components(
            [{"id": "root", "component": "Column", "children": ["root"]}],
            surface_id="x",
            config=config,
        )


@pytest.mark.parametrize(
    ("component", "message"),
    [
        (
            {"id": "root", "component": "Text", "text": "x", "html": "x"},
            "Unsupported properties",
        ),
        (
            {
                "id": "root",
                "component": "TextField",
                "label": "x",
                "value": {"path": "relative"},
            },
            "absolute JSON Pointer",
        ),
        (
            {"id": "root", "component": "Text", "text": {"call": "window.alert"}},
            "function calls",
        ),
    ],
)
def test_catalog_rejects_undeclared_properties_invalid_bindings_and_functions(
    component: dict,
    message: str,
) -> None:
    with pytest.raises(CatalogValidationError, match=message):
        validate_components([component], surface_id="strict", config=CanvasConfig())


def test_side_effect_action_fails_closed_without_permission_broker(
    canvas: CanvasService,
) -> None:
    canvas.create_surface(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="danger",
        owner={"agent_id": "main"},
        correlation_id="turn-1",
    )
    canvas.update_components(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="danger",
        correlation_id="turn-1",
        components=[
            {
                "id": "root",
                "component": "Button",
                "label": "Delete",
                "actions": [
                    {
                        "name": "delete.press",
                        "context": {},
                        "side_effect": True,
                        "permission_scope": "artifact.delete",
                    }
                ],
            }
        ],
    )
    action = RendererAction(
        action_id="delete-1",
        session_id="session_one",
        conversation_id="session_one",
        surface_id="danger",
        source_component_id="root",
        name="delete.press",
        correlation_id="turn-2",
        context={},
    )
    with pytest.raises(CanvasStateError, match="permission broker"):
        canvas.submit_action(action)

    canvas.permission_authorizer = lambda _action, _snapshot, _scope: "permission-1"
    result = canvas.submit_action(action)
    assert result.status == "permission_required"
    with pytest.raises(CanvasStateError, match="permission verifier"):
        canvas.deliver_authorized_action(action, permission_request_id="permission-1")

    delivered: list[str] = []
    snapshot = canvas.get_surface("session_one", "danger")
    canvas.register_action_handler(
        snapshot.owner, lambda routed, _surface: delivered.append(routed.action_id)
    )
    canvas.permission_verifier = lambda _action, _snapshot, request_id: (
        request_id == "permission-1"
    )
    assert (
        canvas.deliver_authorized_action(
            action, permission_request_id="permission-1"
        ).status
        == "delivered"
    )
    assert delivered == ["delete-1"]


def test_reducer_enforces_sequence(canvas: CanvasService) -> None:
    snapshot = canvas.create_surface(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="plan",
        owner={"agent_id": "main"},
        correlation_id="turn-1",
    )
    event = CanvasEventEnvelope(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="plan",
        correlation_id="turn-1",
        sequence=4,
        source=CanvasSource.AGENT,
        event_type=CanvasEventType.DATA,
        payload={
            "version": "v0.9",
            "updateDataModel": {"surfaceId": "plan", "value": {}},
        },
    )
    with pytest.raises(CanvasStateError, match="Out-of-order"):
        reduce_canvas_event(snapshot, event, config=CanvasConfig())


def test_generation_boundary_has_one_bounded_correction() -> None:
    calls: list[list[str]] = []
    rows = parse_generated_messages(
        "not-json",
        config=CanvasConfig(validation_retry_limit=1),
        correct=lambda _raw, errors: (
            calls.append(errors)
            or '[{"version":"v0.9","deleteSurface":{"surfaceId":"x"}}]'
        ),
    )
    assert rows[0]["deleteSurface"]["surfaceId"] == "x"
    assert len(calls) == 1
    with pytest.raises(CanvasGenerationError):
        parse_generated_messages(
            "not-json", config=CanvasConfig(validation_retry_limit=0)
        )


def test_settings_reject_invalid_canvas_configuration() -> None:
    with pytest.raises(ValueError, match="default protocol"):
        Settings(
            MANA_CANVAS_PROTOCOL_VERSIONS="v0.9",
            MANA_CANVAS_DEFAULT_PROTOCOL_VERSION="v1.0",
        )


def test_canvas_configuration_is_loaded_from_config_toml_not_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    monkeypatch.setenv("MANA_CANVAS_ENABLED", "false")
    monkeypatch.setenv(
        "MANA_CANVAS_ALLOWED_CATALOGS", "https://env.invalid/catalog.json"
    )
    save_effective_user_config(
        {
            "MANA_CANVAS_ENABLED": True,
            "MANA_CANVAS_ALLOWED_CATALOGS": MANA_CATALOG_ID,
            "MANA_CANVAS_ALLOW_LOCALHOST": True,
        },
        merge=False,
    )

    settings = Settings()

    assert settings.mana_canvas_enabled is True
    assert settings.mana_canvas_allowed_catalogs == MANA_CATALOG_ID
    config_text = (tmp_path / "mana" / "config.toml").read_text(encoding="utf-8")
    assert "MANA_CANVAS_ALLOW_LOCALHOST = true" in config_text
    assert "env.invalid" not in config_text


def test_canvas_supports_only_explicit_loopback_http_catalogs_and_resources(
    tmp_path: Path,
) -> None:
    local_catalog = f"http://localhost:8765{LOCAL_CATALOG_PATH}"
    config = CanvasConfig()
    service = CanvasService(
        config=config,
        store=CanvasStore(tmp_path / "canvas"),
        event_hub=ExecutionEventHub(),
    )
    snapshot = service.create_surface(
        session_id="local",
        conversation_id="local",
        surface_id="surface",
        owner={"agent_id": "main"},
        correlation_id="turn-local",
        catalog_id=local_catalog,
    )
    assert snapshot.catalog_id == local_catalog
    validate_components(
        [
            {
                "id": "root",
                "component": "Image",
                "url": "http://127.0.0.1:8000/image.png",
                "description": "Local image",
            }
        ],
        surface_id="local",
        config=config,
    )
    with pytest.raises(CatalogValidationError, match="allowlist"):
        validate_components(
            [
                {
                    "id": "root",
                    "component": "Image",
                    "url": "http://example.com/image.png",
                    "description": "Remote image",
                }
            ],
            surface_id="remote",
            config=config,
        )
    with pytest.raises(ValueError, match="loopback"):
        CanvasConfig(
            allowed_catalogs=(local_catalog,), allow_localhost=False
        ).validate()


def test_local_canvas_catalog_endpoint_and_capability(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    save_effective_user_config({"MANA_CANVAS_ALLOW_LOCALHOST": True}, merge=False)
    client = TestClient(create_app(), base_url="http://localhost:8000")
    catalog = client.get(LOCAL_CATALOG_PATH)
    assert catalog.status_code == 200
    assert catalog.json()["catalogId"] == MANA_CATALOG_ID
    capabilities = client.get("/api/v1/canvas/capabilities").json()
    assert (
        f"http://localhost:8000{LOCAL_CATALOG_PATH}"
        in capabilities["renderer"]["catalog_ids"]
    )


def test_action_model_rejects_extra_client_permission_scope() -> None:
    with pytest.raises(PydanticValidationError):
        RendererAction.model_validate(
            {
                "session_id": "s",
                "conversation_id": "s",
                "surface_id": "x",
                "source_component_id": "root",
                "name": "go",
                "correlation_id": "c",
                "context": {},
                "permission_scope": "admin",
            }
        )
    with pytest.raises(PydanticValidationError, match="timezone"):
        RendererAction.model_validate(
            {
                "session_id": "s",
                "conversation_id": "s",
                "surface_id": "x",
                "source_component_id": "root",
                "name": "go",
                "correlation_id": "c",
                "timestamp": "2026-07-29T12:00:00",
            }
        )


def test_stale_renderer_action_is_rejected(canvas: CanvasService) -> None:
    canvas.create_surface(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="stale",
        owner={"agent_id": "main"},
        correlation_id="turn-1",
    )
    canvas.update_components(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="stale",
        components=_components(),
        correlation_id="turn-1",
    )
    with pytest.raises(CanvasStateError, match="timestamp"):
        canvas.submit_action(
            RendererAction(
                action_id="stale-action",
                session_id="session_one",
                conversation_id="session_one",
                surface_id="stale",
                source_component_id="approve",
                name="plan.press",
                correlation_id="turn-2",
                context={"priority": "high"},
                timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            )
        )


def test_validation_retry_budget_and_periodic_checkpoints(tmp_path: Path) -> None:
    service = CanvasService(
        config=CanvasConfig(validation_retry_limit=1, snapshot_interval=2),
        store=CanvasStore(tmp_path / "canvas"),
        event_hub=ExecutionEventHub(),
    )
    service.create_surface(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="bounded",
        owner={"agent_id": "main"},
        correlation_id="turn-1",
    )
    invalid = [{"id": "root", "component": "Text", "text": "x", "html": "not allowed"}]
    with pytest.raises(CatalogValidationError):
        service.update_components(
            session_id="session_one",
            conversation_id="session_one",
            surface_id="bounded",
            components=invalid,
            correlation_id="turn-invalid",
        )
    with pytest.raises(CatalogValidationError):
        service.update_components(
            session_id="session_one",
            conversation_id="session_one",
            surface_id="bounded",
            components=invalid,
            correlation_id="turn-invalid",
        )
    with pytest.raises(CanvasStateError, match="retry limit"):
        service.update_components(
            session_id="session_one",
            conversation_id="session_one",
            surface_id="bounded",
            components=_components(),
            correlation_id="turn-invalid",
        )
    service.update_components(
        session_id="session_one",
        conversation_id="session_one",
        surface_id="bounded",
        components=_components(),
        correlation_id="turn-valid",
    )
    checkpoint = (
        service.store._surface_dir("session_one", "bounded")
        / "snapshots"
        / f"{2:020d}.json"
    )
    assert checkpoint.exists()


def test_canvas_rest_snapshot_and_cross_session_action_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    monkeypatch.delenv("MANA_API_TOKEN", raising=False)
    reset_execution_event_hub_for_tests()
    root = tmp_path / "repo"
    root.mkdir()
    conversation = ConversationService(root=root).create(title="Canvas")
    service = canvas_service_for_root(root, config=CanvasConfig())
    service.create_surface(
        session_id=conversation.conversation_id,
        conversation_id=conversation.conversation_id,
        surface_id="plan",
        owner={"agent_id": "main"},
        correlation_id="turn-1",
    )
    service.update_components(
        session_id=conversation.conversation_id,
        conversation_id=conversation.conversation_id,
        surface_id="plan",
        components=_components(),
        correlation_id="turn-1",
    )
    client = TestClient(create_app())
    response = client.get(
        f"/api/v1/conversations/{conversation.conversation_id}/canvas/surfaces/plan",
        params={"root": str(root)},
    )
    assert response.status_code == 200
    assert response.json()["snapshot"]["surface_id"] == "plan"
    document = client.get(
        "/api/v1/dashboard/live-canvas",
        params={"conversation_id": conversation.conversation_id, "root": str(root)},
    )
    assert document.status_code == 200
    assert "http://127.0.0.1:*" in document.headers["Content-Security-Policy"]
    assert "http://[::1]:*" not in document.headers["Content-Security-Policy"]
    rejected = client.post(
        f"/api/v1/conversations/{conversation.conversation_id}/canvas/surfaces/other/actions",
        json={
            "action_id": "a",
            "source_component_id": "approve",
            "name": "plan.press",
            "correlation_id": "c",
            "context": {"priority": "high"},
            "root": str(root),
        },
    )
    assert rejected.status_code == 409
    injected_scope = client.post(
        f"/api/v1/conversations/{conversation.conversation_id}/canvas/surfaces/plan/actions",
        json={
            "action_id": "scope-injection",
            "source_component_id": "approve",
            "name": "plan.press",
            "correlation_id": "c",
            "context": {"priority": "high"},
            "permission_scope": "admin",
            "root": str(root),
        },
    )
    assert injected_scope.status_code == 422


def test_canvas_events_use_existing_websocket_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    monkeypatch.delenv("MANA_API_TOKEN", raising=False)
    reset_execution_event_hub_for_tests()
    root = tmp_path / "repo"
    root.mkdir()
    conversation = ConversationService(root=root).create(title="Socket Canvas")
    service = canvas_service_for_root(root, config=CanvasConfig())
    service.create_surface(
        session_id=conversation.conversation_id,
        conversation_id=conversation.conversation_id,
        surface_id="socket-plan",
        owner={"agent_id": "main"},
        correlation_id="turn-socket",
    )
    client = TestClient(create_app())
    with client.websocket_connect(
        f"/api/v1/ws/conversations/{conversation.conversation_id}?root={root}&replay_limit=50"
    ) as websocket:
        assert websocket.receive_json()["type"] == "socket.ready"
        canvas_replay = None
        for _ in range(10):
            packet = websocket.receive_json()
            if (
                packet.get("type") == "event.replay"
                and packet["event"]["type"] == "canvas.createSurface"
            ):
                canvas_replay = packet["event"]
            if packet.get("type") == "socket.replay_complete":
                break
        assert canvas_replay is not None
        assert canvas_replay["metadata"]["canvas_event"]["surface_id"] == "socket-plan"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_browser_canvas_reducer_suite() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["node", "--test", "tests/dashboard/live_canvas_reducer.test.mjs"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_renderer_has_safe_fallback_and_accessibility_contracts() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/mana_agent/dashboard/components/live_canvas.js"
    ).read_text(encoding="utf-8")
    assert "Unsupported component" in source
    assert "Waiting for surface content" in source
    assert "Surface generation did not complete" in source
    assert 'setAttribute("role","alert")' in source
    assert 'aria-label="Canvas surface"' in source
    assert "eval(" not in source
    assert "new Function" not in source
