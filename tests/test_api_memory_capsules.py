from __future__ import annotations

import getpass
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from mana_agent.api.app import create_app
from mana_agent.memory import CapsuleScope, CapsuleTaskContext, CapsuleService, MemoryPrincipal
from mana_agent.workspaces.paths import repository_id_for_path


def test_standalone_api_binds_local_identity_for_user_capsules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    monkeypatch.setenv("MANA_DASHBOARD_ROOT", str(tmp_path / "repo"))
    monkeypatch.delenv("MANA_API_TOKEN", raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    user_id = getpass.getuser()
    project_id = repository_id_for_path(root)
    task_id = f"api-{project_id}"
    principal = MemoryPrincipal(
        user_id=user_id,
        project_id=project_id,
        task_id=task_id,
        agent_id="api:local",
        capabilities=frozenset({"memory.capsule.write.user"}),
    )
    context = CapsuleTaskContext(
        user_id=user_id,
        organisation_id=None,
        project_id=project_id,
        team_ids=frozenset(),
        task_id=task_id,
        agent_id="api:local",
    )
    CapsuleService(root).create_capsule(
        principal=principal,
        context=context,
        scope=CapsuleScope.USER,
        title="Local capsule",
        summary="Visible to the standalone API",
        content={"source": "test"},
        origin_type="test",
        origin_id="local-api",
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/memory/capsules/query",
        json={"allowed_scopes": ["user"]},
    )

    assert response.status_code == 200
    assert [item["title"] for item in response.json()] == ["Local capsule"]


def test_api_reuses_a_chat_gateway_capsule_service(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    root = tmp_path / "repo"
    root.mkdir()
    capsule_service = CapsuleService(root)
    gateway = SimpleNamespace(
        root=root,
        _stack=SimpleNamespace(memory_service=SimpleNamespace(capsules=capsule_service)),
    )

    app = create_app(chat_gateway=gateway)

    assert app.state.capsule_service is capsule_service
    assert callable(app.state.capsule_identity_resolver)
