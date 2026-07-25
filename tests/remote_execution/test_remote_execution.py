from __future__ import annotations
import asyncio
from pathlib import Path
import pytest
from mana_agent.remote_execution.models import RemoteCommand, RemoteExecutionEvent, RemoteExecutionRequest, SSHAuthentication, SSHTarget, WorkerRegistration
from mana_agent.remote_execution.providers.local_ssh import build_ssh_argv
from mana_agent.remote_execution.permissions import required_permission
from mana_agent.remote_execution.service import RemoteExecutionService
from mana_agent.remote_execution.target_policy import TargetPolicy, TargetPolicyMode
from mana_agent.remote_execution.transport_errors import TransportFailure, classify_ssh_failure, permits_external_worker_failover
from mana_agent.remote_execution.worker import WorkerRegistry, store_worker_credential
from mana_agent.chat_commands import CommandDispatcher, build_default_registry
from mana_agent.chat_commands.models import CommandContext

def request(*, provider: str = "local_ssh", command: list[str] | None = None) -> RemoteExecutionRequest:
    return RemoteExecutionRequest(job_id="job-1", session_id="session-1", worker_id="worker-1", provider=provider, target=SSHTarget(host="example.test", user="root"), authentication=SSHAuthentication(mode="key_path", key_path="~/.ssh/id_test"), command=RemoteCommand(argv=command or ["tail", "-n", "20", "/tmp/a file"]))

def test_operation_not_permitted_is_sandbox_restriction() -> None:
    assert classify_ssh_failure("ssh: connect to host example.test port 22: Operation not permitted") is TransportFailure.SANDBOX_RESTRICTION
    assert permits_external_worker_failover(TransportFailure.SANDBOX_RESTRICTION)

def test_authentication_failure_never_allows_worker_failover() -> None:
    assert classify_ssh_failure("Permission denied (publickey).", 255) is TransportFailure.AUTHENTICATION_FAILURE
    assert not permits_external_worker_failover(TransportFailure.AUTHENTICATION_FAILURE)

def test_ssh_uses_argv_and_worker_only_key_path_expansion() -> None:
    args = build_ssh_argv(request())
    assert args[:7] == ["ssh", "-p", "22", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
    assert "tail -n 20 '/tmp/a file'" in args[-1]
    assert "PRIVATE KEY" not in " ".join(args)

def test_default_target_policy_requires_exact_action_approval() -> None:
    policy, first = TargetPolicy(), request()
    policy.approve_action(first)
    assert not policy.requires_approval(first)
    assert policy.requires_approval(request(command=["rm", "-f", "/tmp/a"]))
    assert TargetPolicy(TargetPolicyMode.APPROVED_TARGETS).requires_approval(first)


def test_permission_request_is_bound_to_exact_remote_job() -> None:
    service = RemoteExecutionService(target_policy=TargetPolicy())
    job = service.submit(request())
    pending = service.pending_permissions()
    assert job.state.value == "awaiting_permission"
    assert pending[0]["job_id"] == job.request.job_id
    approved = service.approve_permission(pending[0]["permission_request_id"])
    assert approved.state.value == "queued"
    with pytest.raises(ValueError):
        service.approve_permission(pending[0]["permission_request_id"])

def test_command_risk_is_not_trusted_from_read_only_flag() -> None:
    assert required_permission(request(command=["rm", "-f", "/tmp/a"])).value == "privileged_or_destructive"
    assert required_permission(request(command=["touch", "/tmp/a"])).value == "remote_write"

def test_worker_enrolment_is_one_time_and_credential_is_private(tmp_path: Path) -> None:
    registry = WorkerRegistry()
    token = registry.issue_enrolment_token("worker-1")
    registration = WorkerRegistration(worker_id="worker-1", display_name="host", capabilities={}, operating_system="test", ssh_available=True)
    credential = registry.enrol(token, registration)
    with pytest.raises(PermissionError):
        registry.enrol(token, registration)
    path = tmp_path / "credential"
    store_worker_credential(path, credential)
    assert path.stat().st_mode & 0o077 == 0

def test_sandbox_selects_external_worker_and_deduplicates_events() -> None:
    seen: list[RemoteExecutionEvent] = []
    registry = WorkerRegistry()
    credential = registry.enrol(registry.issue_enrolment_token("worker-1"), WorkerRegistration(worker_id="worker-1", display_name="host", capabilities={}, operating_system="test", ssh_available=True))
    assigned: list[RemoteExecutionRequest] = []
    registry.connect("worker-1", credential, assigned.append)
    service = RemoteExecutionService(workers=registry, target_policy=TargetPolicy(TargetPolicyMode.UNRESTRICTED), event_sink=seen.append, outbound_tcp_available=False)
    job = service.submit(request())
    asyncio.run(service.execute(job.request.job_id))
    assert job.state.value == "assigned" and assigned == [job.request]
    event = RemoteExecutionEvent(job_id="job-1", session_id="session-1", kind="stdout", data={"chunk": "same"})
    service._emit(job, event)
    service._emit(job, event)
    assert len([item for item in seen if item.kind == "stdout"]) == 1


def test_auto_worker_selection_requires_one_connected_trusted_worker() -> None:
    registry = WorkerRegistry()
    credential = registry.enrol(registry.issue_enrolment_token("worker-1"), WorkerRegistration(worker_id="worker-1", display_name="host", capabilities={}, operating_system="test", ssh_available=True))
    registry.connect("worker-1", credential, lambda item: None)
    assert registry.select_connected_worker().registration.worker_id == "worker-1"
    registry.disconnect("worker-1")
    with pytest.raises(LookupError, match="exactly one"):
        registry.select_connected_worker()

def test_worker_disconnection_safely_finishes_active_job() -> None:
    registry = WorkerRegistry()
    credential = registry.enrol(registry.issue_enrolment_token("worker-1"), WorkerRegistration(worker_id="worker-1", display_name="host", capabilities={}, operating_system="test", ssh_available=True))
    registry.connect("worker-1", credential, lambda item: None)
    service = RemoteExecutionService(workers=registry, target_policy=TargetPolicy(TargetPolicyMode.UNRESTRICTED))
    job = service.submit(request(provider="external_worker"))
    asyncio.run(service.execute(job.request.job_id))
    service.worker_disconnected("worker-1")
    assert job.state.value == "worker_disconnected"


def test_no_connected_worker_never_falls_back_to_local_ssh() -> None:
    service = RemoteExecutionService(target_policy=TargetPolicy(TargetPolicyMode.UNRESTRICTED), outbound_tcp_available=False)
    job = service.submit(request(provider="external_worker"))
    with pytest.raises(RuntimeError, match="No local SSH fallback"):
        asyncio.run(service.execute(job.request.job_id))
    assert job.state.value == "failed"


def test_remote_worker_lifecycle_is_a_typed_chat_command_not_a_cli_route() -> None:
    class Gateway:
        def remote_worker_command(self, action: str, worker_id: str):
            assert (action, worker_id) == ("register", "host-worker")
            return {"message": "issued", "status": "enrolment_issued"}

    registry = build_default_registry()
    result = CommandDispatcher(registry).dispatch(
        "/remote-worker register host-worker",
        CommandContext(frontend="cli", session_id="session", capabilities={"gateway"}, gateway=Gateway()),
    )
    assert result is not None
    assert result.status == "success"
    assert result.data == {"status": "enrolment_issued"}
