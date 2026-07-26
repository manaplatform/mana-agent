from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import asyncio
import os
import subprocess
import sys

import pytest

from mana_agent.fleet.capabilities import canonical_inventory_payload, validate_inventory_update
from mana_agent.fleet.config import FleetConfig
from mana_agent.fleet.errors import FleetCapabilityError, FleetSelectionError, FleetStateError
from mana_agent.fleet.models import (
    FleetSelectionRequest, FleetWorker, RuntimeRequirements, WorkerCapabilities,
    WorkerHealth, WorkerIdentity, WorkerLabels, WorkerStatus, utc_now,
)
from mana_agent.fleet.permissions import FleetPermissionGrant, FleetPermissionRequest
from mana_agent.fleet.registry import FleetRegistry
from mana_agent.fleet.selector import select_workers
from mana_agent.fleet.store import FleetStore
from mana_agent.fleet.events import FleetEvent
from mana_agent.fleet.errors import FleetPersistenceError
from mana_agent.fleet.service import FleetService
from mana_agent.execution.config import ExecutionConfig, build_provider_registry
from mana_agent.execution.manager import ExecutionManager
from mana_agent.execution.store import SandboxStore
from mana_agent.remote_execution.installers.linux import systemd_user_unit
from mana_agent.remote_execution.installers.windows import task_scheduler_xml


def worker(
    worker_id: str, platform: str, *,
    labels: frozenset[str] = frozenset({"trusted"}),
    status: WorkerStatus = WorkerStatus.CONNECTED,
    python: tuple[str, ...] = ("3.12",),
) -> FleetWorker:
    capabilities = WorkerCapabilities(
        worker_id=worker_id,
        platform=platform,
        platform_release="test",
        architecture="x86_64",
        python_versions=python,
        available_tools=frozenset({"git", "pytest"}),
        labels=WorkerLabels(values=labels),
        workspace_backends=frozenset({"execution-fabric"}),
        execution_providers=frozenset({"local-process"}),
        last_probe_at=utc_now(),
    )
    return FleetWorker(
        identity=WorkerIdentity(
            worker_id=worker_id,
            identity_fingerprint=f"identity-{worker_id}",
            credential_status="valid",
        ),
        capabilities=capabilities,
        capability_fingerprint=capabilities.fingerprint,
        health=WorkerHealth(
            status=status,
            last_heartbeat=utc_now(),
            identity_status="valid",
        ),
    )


def test_selection_is_deterministic_and_covers_required_platforms(tmp_path: Path) -> None:
    config = FleetConfig(enabled=True, root=tmp_path)
    request = FleetSelectionRequest(
        decision_id="decision",
        required_platforms=frozenset({"linux", "windows"}),
        allowed_platforms=frozenset({"linux", "windows"}),
        runtime=RuntimeRequirements(python=frozenset({"3.12"})),
        required_tools=frozenset({"pytest"}),
        maximum_workers=2,
    )
    candidates = [worker("worker_z", "linux"), worker("worker_a", "windows")]
    first = select_workers(request, candidates, config)
    second = select_workers(request, list(reversed(candidates)), config)
    assert first.model_dump(exclude={"decided_at"}) == second.model_dump(exclude={"decided_at"})
    assert first.platform_coverage == {"linux", "windows"}


@pytest.mark.parametrize("status", [WorkerStatus.OFFLINE, WorkerStatus.DRAINING, WorkerStatus.REVOKED])
def test_unhealthy_worker_is_rejected_without_local_fallback(tmp_path: Path, status: WorkerStatus) -> None:
    config = FleetConfig(enabled=True, root=tmp_path)
    request = FleetSelectionRequest(
        decision_id="decision",
        required_platforms=frozenset({"windows"}),
        allowed_platforms=frozenset({"windows"}),
    )
    with pytest.raises(FleetSelectionError, match="No local fallback"):
        select_workers(request, [worker("worker_windows", "windows", status=status)], config)


def test_hard_capability_mismatch_rejects_worker(tmp_path: Path) -> None:
    with pytest.raises(FleetSelectionError, match="required fleet platform coverage"):
        select_workers(
            FleetSelectionRequest(
                decision_id="decision",
                required_platforms=frozenset({"linux"}),
                required_tools=frozenset({"ruff"}),
            ),
            [worker("worker_linux", "linux")],
            FleetConfig(enabled=True, root=tmp_path),
        )


def test_authenticated_capability_update_rejects_stale_and_bad_signature() -> None:
    inventory = worker("worker_linux", "linux").capabilities
    payload = canonical_inventory_payload(inventory)

    def accepted(worker_id: str, signed: bytes, signature: bytes) -> None:
        assert worker_id == "worker_linux"
        assert signed == payload
        assert signature == b"valid"

    result = validate_inventory_update(
        payload,
        authenticated_worker_id="worker_linux",
        verify_signature=accepted,
        signature=b"valid",
        capability_ttl_seconds=300,
    )
    assert result.fingerprint == inventory.fingerprint

    stale = inventory.model_copy(update={"last_probe_at": utc_now() - timedelta(hours=1)})
    with pytest.raises(FleetCapabilityError, match="stale"):
        validate_inventory_update(
            canonical_inventory_payload(stale),
            authenticated_worker_id="worker_linux",
            verify_signature=lambda *_: None,
            signature=b"valid",
            capability_ttl_seconds=300,
        )
    with pytest.raises(FleetCapabilityError, match="authentication"):
        validate_inventory_update(
            payload,
            authenticated_worker_id="worker_linux",
            verify_signature=lambda *_: (_ for _ in ()).throw(PermissionError()),
            signature=b"invalid",
            capability_ttl_seconds=300,
        )


def test_capability_fingerprint_is_stable_across_hash_seeds() -> None:
    script = """
from datetime import datetime, timezone
from mana_agent.fleet.models import WorkerCapabilities, WorkerLabels

capabilities = WorkerCapabilities(
    worker_id="worker_linux",
    platform="linux",
    platform_release="test",
    architecture="x86_64",
    python_versions=("3.12",),
    available_tools=frozenset({"git", "pytest"}),
    labels=WorkerLabels(values=frozenset({"trusted", "isolated"})),
    workspace_backends=frozenset({"execution-fabric", "local"}),
    execution_providers=frozenset({"local-process", "reverse-worker"}),
    last_probe_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
)
print(capabilities.fingerprint)
"""
    fingerprints = {
        subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("1", "4")
    }

    assert len(fingerprints) == 1


def test_registry_persists_revoke_and_never_reenables(tmp_path: Path) -> None:
    config = FleetConfig(enabled=True, root=tmp_path)
    registry = FleetRegistry(FleetStore(tmp_path), config)
    selected = worker("worker_linux", "linux")
    registry.accept_capabilities(selected.capabilities, selected.identity)
    registry.set_status("worker_linux", WorkerStatus.REVOKED)
    restored = FleetRegistry(FleetStore(tmp_path), config)
    assert restored.require("worker_linux").health.status is WorkerStatus.REVOKED
    with pytest.raises(FleetStateError, match="cannot be re-enabled"):
        restored.set_status("worker_linux", WorkerStatus.OFFLINE)


def test_event_sequence_and_cross_process_cancellation_are_persisted(tmp_path: Path) -> None:
    store = FleetStore(tmp_path)
    store.append_event(FleetEvent(sequence=1, kind="fleet.run.created"))
    store.append_event(FleetEvent(sequence=2, kind="fleet.job.queued"))
    with pytest.raises(FleetPersistenceError, match="sequence"):
        store.append_event(FleetEvent(sequence=2, kind="fleet.job.queued"))
    assert [item.sequence for item in store.events(after_sequence=1)] == [2]
    store.request_cancellation("fleet_job_1")
    assert FleetStore(tmp_path).cancellation_requested("fleet_job_1")
    store.clear_cancellation("fleet_job_1")
    assert not store.cancellation_requested("fleet_job_1")


def test_permission_grant_is_bound_to_exact_repository_worker_and_command() -> None:
    request = FleetPermissionRequest(
        permission_request_id="permission_1",
        scope="fleet.verify.execute",
        repository_id="repo",
        repository_commit="abc",
        worker_ids=("worker_1",),
        commands=(("pytest", "-q"),),
    )
    grant = FleetPermissionGrant(
        permission_request_id=request.permission_request_id,
        scope=request.scope,
        exact_action_key=request.exact_action_key,
    )
    assert grant.authorizes(request)
    changed = request.model_copy(update={"repository_commit": "def"})
    assert not grant.authorizes(changed)


def test_linux_and_windows_installers_do_not_embed_credentials(tmp_path: Path) -> None:
    unit = systemd_user_unit(
        executable="/opt/Mana Agent/mana-agent", state_dir=tmp_path / "state",
    )
    xml = task_scheduler_xml(
        executable=r"C:\Program Files\Mana Agent\mana-agent.exe",
        state_dir=Path(r"C:\Users\Test\AppData\Local\ManaAgent\state"),
    )
    for payload in (unit, xml):
        assert "--token" not in payload.lower()
        assert "enrollment" not in payload.lower()
        assert "credential" not in payload.lower()
        assert "worker run" in payload


def test_fleet_service_executes_through_execution_manager_and_cleans_workspace(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "fleet@example.test"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Fleet Test"], cwd=repository, check=True)
    (repository / "README.md").write_text("fleet\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repository, check=True, capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    root = tmp_path / "fleet"
    config = FleetConfig(enabled=True, root=root)
    store = FleetStore(root)
    registry = FleetRegistry(store, config)
    selected = worker("worker_linux", "linux")
    registry.accept_capabilities(selected.capabilities, selected.identity)
    registry.heartbeat("worker_linux")
    execution_config = ExecutionConfig()
    manager = ExecutionManager(
        build_provider_registry(execution_config), execution_config,
        store=SandboxStore(root / "sandboxes"),
    )
    service = FleetService(
        config=config, registry=registry, execution_manager=manager, store=store,
    )
    request = FleetSelectionRequest(
        decision_id="decision",
        required_platforms=frozenset({"linux"}),
        allowed_platforms=frozenset({"linux"}),
    )
    plan = service.create_plan(
        request=request,
        repository_path=repository,
        repository_commit=commit,
        commands=[[sys.executable, "-c", "print('fleet-ok')"]],
    )
    completed = asyncio.run(service.execute(service.create_run(plan)))
    assert completed.summary is not None
    assert completed.summary.outcome.value == "fully_verified"
    assert "fleet-ok" in completed.results[0].stdout
    assert not (root / "workspaces" / completed.jobs[0].job_id).exists()
