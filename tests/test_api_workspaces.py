from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from mana_agent.api.app import create_app


def _repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("API_WORKSPACE_NEEDLE\n", encoding="utf-8")
    return path


def test_workspace_api_enforces_allowed_roots_and_searches(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))
    allowed = tmp_path / "allowed"
    repo = _repo(allowed / "api")
    monkeypatch.setenv("MANA_WORKSPACE_ALLOWED_ROOTS", str(allowed))
    client = TestClient(create_app())

    created = client.post("/api/v1/workspaces", json={"name": "product", "roots": [str(allowed)], "discover": True})
    assert created.status_code == 201
    workspace = created.json()
    assert len(workspace["repository_ids"]) == 1
    searched = client.post(
        f"/api/v1/workspaces/{workspace['workspace_id']}/search",
        json={"query": "API_WORKSPACE_NEEDLE", "mode": "text"},
    )
    assert searched.status_code == 200
    assert searched.json()["results"][0]["qualified_path"] == "api::README.md"

    blocked = client.post(
        f"/api/v1/workspaces/{workspace['workspace_id']}/repositories",
        json={"path": str(tmp_path / "outside")},
    )
    assert blocked.status_code == 403


def test_packaged_dashboard_module_is_discoverable() -> None:
    import importlib.util

    spec = importlib.util.find_spec("mana_agent.dashboard.app")
    assert spec is not None
    assert spec.origin and spec.origin.replace("\\", "/").endswith("mana_agent/dashboard/app.py")
