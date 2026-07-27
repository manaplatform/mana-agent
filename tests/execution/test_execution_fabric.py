from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mana_agent.execution.artifacts import collect_local_artifacts
from mana_agent.execution.config import ExecutionConfig, build_provider_registry
from mana_agent.execution.errors import ArtifactError, CapabilityMismatchError, ExecutionTimeoutError, LifecycleTransitionError, SnapshotError
from mana_agent.execution.lifecycle import validate_transition
from mana_agent.execution.manager import ExecutionManager
from mana_agent.execution.models import (
    ArtifactRequest,
    EnforcementStrength,
    ExecutionRequest,
    NetworkMode,
    NetworkPolicy,
    RoutingRequest,
    SandboxSpec,
    SandboxState,
    SecretInjection,
    SnapshotRequest,
)
from mana_agent.execution.providers.local_process import LocalProcessProvider, _decode_captured_output
from mana_agent.execution.registry import ProviderRegistry
from mana_agent.execution.router import ExecutionRouter
from mana_agent.execution.snapshots import restore_archive_snapshot
from mana_agent.execution.store import SandboxStore


def run(value):
    return asyncio.run(value)


def local_config() -> ExecutionConfig:
    return ExecutionConfig(providers={name: {"enabled": name == "local-process"} for name in (
        "local-process", "local-docker", "remote-ssh", "kubernetes", "modal", "custom-http-runtime"
    )})


def test_all_six_providers_share_one_registry() -> None:
    registry = build_provider_registry(local_config())
    assert set(registry.names()) == {"local-process", "local-docker", "remote-ssh", "kubernetes", "modal", "custom-http-runtime"}
    for provider in registry.providers():
        for method in ("capabilities", "provision", "start", "execute", "upload", "download_artifacts", "snapshot", "restore", "suspend", "resume", "terminate", "cleanup", "healthcheck"):
            assert callable(getattr(provider, method))


def test_models_and_lifecycle_validate_strictly(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SandboxSpec(repository_source=tmp_path, unknown_provider_blob={})
    with pytest.raises(ValidationError):
        RoutingRequest(trust_level="trusted", risk_level="low")
    validate_transition(SandboxState.REQUESTED, SandboxState.PROVISIONING)
    with pytest.raises(LifecycleTransitionError):
        validate_transition(SandboxState.REQUESTED, SandboxState.RUNNING)
    with pytest.raises(LifecycleTransitionError):
        validate_transition(SandboxState.CLEANED, SandboxState.READY)


def test_registry_rejects_duplicates() -> None:
    registry = ProviderRegistry()
    registry.register(LocalProcessProvider())
    with pytest.raises(Exception, match="already registered"):
        registry.register(LocalProcessProvider())


def test_local_provider_normalizes_windows_subprocess_newlines() -> None:
    assert _decode_captured_output(b"stdout\r\nnext\r\n") == "stdout\nnext\n"
    assert _decode_captured_output(b"progress\rupdate") == "progress\rupdate"


def test_router_never_weakens_security_boundary() -> None:
    config = local_config()
    router = ExecutionRouter(build_provider_registry(config), config)
    with pytest.raises(CapabilityMismatchError, match="no provider satisfies"):
        run(router.route(RoutingRequest(decision_id="model-1", trust_level="untrusted", risk_level="high")))
    with pytest.raises(CapabilityMismatchError, match="local-docker"):
        run(router.route(RoutingRequest(decision_id="model-2", explicit_provider="local-docker", trust_level="trusted", risk_level="low")))


def test_local_provider_shared_contract_and_persistence(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = SandboxStore(tmp_path / "state")
    registry = ProviderRegistry()
    provider = LocalProcessProvider(snapshot_root=tmp_path / "snapshots")
    registry.register(provider)
    manager = ExecutionManager(registry, local_config(), store=store)
    routing = RoutingRequest(decision_id="model-local", explicit_provider="local-process", trust_level="trusted", risk_level="low")
    spec = SandboxSpec(repository_source=workspace, artifact_paths=["result.txt"], cleanup_policy="retain")
    context = run(manager.create(spec, routing))
    assert context.handle.state == SandboxState.READY
    result = run(manager.execute(context, ExecutionRequest(argv=["python3", "-c", "from pathlib import Path; Path('result.txt').write_text('first'); print('ok')"])))
    assert result.exit_code == 0 and result.stdout == "ok\n"
    snapshot = run(manager.snapshot(context, SnapshotRequest(name="before-mutation")))
    (workspace / "result.txt").write_text("mutated", encoding="utf-8")
    (workspace / "upload.txt").write_text("upload", encoding="utf-8")
    run(provider.upload(context.handle, workspace / "upload.txt", "copied.txt"))
    assert (workspace / "copied.txt").read_text(encoding="utf-8") == "upload"
    artifacts = run(provider.download_artifacts(context.handle, ArtifactRequest(paths=["result.txt"], destination=tmp_path / "artifacts")))
    assert artifacts[0].sha256 and artifacts[0].size_bytes == 7
    run(manager.suspend(context))
    run(manager.resume(context))
    assert context.handle.state == SandboxState.READY
    assert store.get(context.handle.sandbox_id).snapshot_refs == [snapshot.snapshot_id]
    run(manager.terminate_and_cleanup(context))
    run(manager.terminate_and_cleanup(context))
    assert store.get(context.handle.sandbox_id).state == SandboxState.CLEANED


def test_timeout_network_secrets_artifacts_and_snapshot_safety(tmp_path: Path, monkeypatch) -> None:
    provider = LocalProcessProvider(snapshot_root=tmp_path / "snapshots")
    handle = run(provider.start(run(provider.provision(SandboxSpec(repository_source=tmp_path)))))
    with pytest.raises(ExecutionTimeoutError):
        run(provider.execute(handle, ExecutionRequest(argv=["python3", "-c", "import time; time.sleep(10)"], timeout_seconds=1)))
    assert not provider._processes
    with pytest.raises(Exception, match="cannot enforce"):
        run(provider.provision(SandboxSpec(repository_source=tmp_path, network=NetworkPolicy(mode=NetworkMode.DENY_ALL, required_enforcement=EnforcementStrength.ENFORCED))))

    monkeypatch.setenv("TEST_FABRIC_SECRET", "top-secret-value")
    secret_spec = SandboxSpec(repository_source=tmp_path, secrets=[SecretInjection(reference="TEST_FABRIC_SECRET", target="TOKEN")])
    secret_handle = run(provider.start(run(provider.provision(secret_spec))))
    secret_handle.metadata["secrets"] = [item.model_dump(mode="json") for item in secret_spec.secrets]
    result = run(provider.execute(secret_handle, ExecutionRequest(argv=["python3", "-c", "import os; print(os.environ['TOKEN'])"])))
    assert "top-secret-value" not in result.stdout and "REDACTED" in result.stdout
    assert "top-secret-value" not in secret_spec.model_dump_json()

    outside = tmp_path.parent / "outside.txt"
    outside.write_text("no", encoding="utf-8")
    with pytest.raises(ArtifactError):
        collect_local_artifacts(tmp_path, ArtifactRequest(paths=["../outside.txt"]))
    if hasattr(os, "symlink"):
        (tmp_path / "link").symlink_to(outside)
        with pytest.raises(ArtifactError):
            collect_local_artifacts(tmp_path, ArtifactRequest(paths=["link"]))

    (tmp_path / "data.txt").write_text("original", encoding="utf-8")
    snapshot = run(provider.snapshot(handle, SnapshotRequest()))
    Path(snapshot.location).write_bytes(b"corrupt")
    with pytest.raises(SnapshotError, match="checksum"):
        restore_archive_snapshot(snapshot, tmp_path / "restore")
