from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mana_agent.commands import server_cli
from mana_agent.remote_execution.providers.local_ssh import build_ssh_argv
from mana_agent.server.audit import ServerAuditLog
from mana_agent.server.connection import ServerConnectionFactory
from mana_agent.server.executor import ServerApprovalRequired, ServerDecisionError, ServerExecutor, action_key
from mana_agent.server.firewall import FirewallPlan, FirewallRule
from mana_agent.server.models import ServerActionDecision, ServerApproval, ServerDefinition, ServerPlan, ServerPlanStep
from mana_agent.server.packages import (
    detect_package_manager,
    package_install_auto_argv,
    package_install_argv,
)
from mana_agent.server.plans import ServerPlanExecutor
from mana_agent.server.registry import ServerRegistry
from mana_agent.server.runtime_tools import build_tool_argv, validate_tool_arguments
from mana_agent.server.tools import SERVER_TOOL_SPECS, validate_tool_decision


def server(tmp_path: Path, **updates) -> ServerDefinition:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("example ssh-ed25519 AAAA\n", encoding="utf-8")
    payload = {
        "server_id": "server-1",
        "name": "production-api",
        "host": "example.test",
        "username": "root",
        "auth_method": "ssh_agent",
        "mode": "managed_admin",
        "host_key_fingerprint": "256 SHA256:test example (ED25519)",
        "known_hosts_file": str(known_hosts),
        "allowed_capabilities": {"inspect", "package.write", "power"},
    }
    payload.update(updates)
    return ServerDefinition.model_validate(payload)


def decision(**updates) -> ServerActionDecision:
    payload = {
        "decision_id": "decision-1",
        "server_id": "server-1",
        "action": "inspect",
        "tool_name": "server_inspect",
        "arguments": {},
        "required_capability": "inspect",
        "read_only": True,
        "consequential": False,
        "safe_to_continue": True,
        "reason": "Collect bounded health evidence.",
    }
    payload.update(updates)
    return ServerActionDecision.model_validate(payload)


def test_registry_round_trip_is_atomic_and_non_secret(tmp_path: Path) -> None:
    registry = ServerRegistry(tmp_path / "registry.json")
    registered = registry.add(server(tmp_path))
    assert registry.get("production-api") == registered
    content = registry.path.read_text(encoding="utf-8")
    assert "PRIVATE KEY" not in content
    assert registry.remove("server-1").server_id == "server-1"


def test_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    registry = ServerRegistry(tmp_path / "registry.json")
    registry.add(server(tmp_path))
    with pytest.raises(ValueError, match="already enrolled"):
        registry.add(server(tmp_path, server_id="server-2"))


def test_host_key_pin_is_mandatory(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="host_key_fingerprint"):
        server(tmp_path, host_key_fingerprint="")


def test_connection_uses_pinned_hosts_keepalive_jump_and_pool(tmp_path: Path) -> None:
    enrolled = server(tmp_path, jump_host="bastion.example", agent_forwarding=True, auth_method="ssh_agent")
    request = ServerConnectionFactory().request(
        enrolled,
        command_id="command-1",
        session_id="session-1",
        argv=["uname", "-a"],
        environment={"LC_ALL": "C"},
    )
    argv = build_ssh_argv(request)
    assert "StrictHostKeyChecking=yes" in argv
    assert f"UserKnownHostsFile={enrolled.known_hosts_file}" in argv
    assert "ControlMaster=auto" in argv
    assert "bastion.example" in argv
    assert "env LC_ALL=C uname -a" in argv[-1]


def test_tool_decision_contract_rejects_false_read_only_claim() -> None:
    invalid = decision(
        action="package",
        tool_name="server_package_install",
        required_capability="package.write",
        affected_resources=["package:nginx"],
    )
    with pytest.raises(ValueError, match="tool contract"):
        validate_tool_decision(invalid)


def test_unsafe_decision_stops_without_execution(tmp_path: Path) -> None:
    registry = ServerRegistry(tmp_path / "registry.json")
    registry.add(server(tmp_path))
    executor = ServerExecutor(registry=registry, audit=ServerAuditLog(tmp_path / "audit.jsonl"))
    with pytest.raises(ServerDecisionError, match="No server action was executed"):
        executor.validate_decision(decision(safe_to_continue=False))


def test_missing_capability_returns_explicit_authorization_guidance(tmp_path: Path) -> None:
    registry = ServerRegistry(tmp_path / "registry.json")
    registry.add(server(tmp_path, allowed_capabilities={"inspect"}))
    executor = ServerExecutor(registry=registry, audit=ServerAuditLog(tmp_path / "audit.jsonl"))
    install = decision(
        action="package",
        tool_name="server_package_install",
        required_capability="package.write",
        read_only=False,
        consequential=True,
        affected_resources=["package:nginx"],
        safe_to_continue=False,
    )

    with pytest.raises(
        ServerDecisionError,
        match=r"server authorize server-1 --capability package\.write",
    ):
        executor.validate_decision(install)


def test_server_authorize_adds_an_explicit_known_capability(tmp_path: Path, monkeypatch) -> None:
    registry = ServerRegistry(tmp_path / "registry.json")
    registry.add(server(tmp_path, allowed_capabilities={"inspect"}))
    service = type(
        "ServerServiceStub",
        (),
        {"registry": registry, "server": staticmethod(registry.get)},
    )()
    monkeypatch.setattr(server_cli, "ServerManagementService", lambda: service)

    result = CliRunner().invoke(
        server_cli.server_app,
        ["authorize", "server-1", "--capability", "package.write", "--yes"],
    )

    assert result.exit_code == 0
    assert registry.get("server-1").allowed_capabilities == {"inspect", "package.write"}


def test_consequential_action_requires_exact_approval(tmp_path: Path) -> None:
    registry = ServerRegistry(tmp_path / "registry.json")
    registry.add(server(tmp_path))
    executor = ServerExecutor(registry=registry, audit=ServerAuditLog(tmp_path / "audit.jsonl"))
    install = decision(
        action="package",
        tool_name="server_package_install",
        required_capability="package.write",
        read_only=False,
        consequential=True,
        affected_resources=["package:nginx"],
    )
    with pytest.raises(ServerApprovalRequired) as pending:
        executor.validate_decision(install)
    approval = ServerApproval(
        approval_id="approval-1",
        decision_id=install.decision_id,
        server_id=install.server_id,
        exact_action_key=pending.value.exact_action_key,
    )
    assert executor.validate_decision(install, approval)[1] == action_key(install)


def test_audit_redacts_secret_values(tmp_path: Path) -> None:
    audit = ServerAuditLog(tmp_path / "audit.jsonl")
    audit.append({"server_id": "server-1", "password": "do-not-store", "authorization": "Bearer token"})
    content = audit.path.read_text(encoding="utf-8")
    assert "do-not-store" not in content
    assert "Bearer token" not in content


def test_package_manager_requires_unambiguous_evidence() -> None:
    assert detect_package_manager("ubuntu", {"apt"}) == "apt"
    with pytest.raises(ValueError, match="Multiple"):
        detect_package_manager("fedora", {"dnf", "yum"})
    assert package_install_argv("apt", ["nginx"])[-1] == "nginx"


def test_package_install_requires_typed_arguments_before_execution() -> None:
    install = decision(
        action="package",
        tool_name="server_package_install",
        arguments={"packages": ["nginx"]},
        required_capability="package.write",
        read_only=False,
        consequential=True,
        affected_resources=["package:nginx"],
    )

    with pytest.raises(ValueError, match="package manager must be one of"):
        validate_tool_arguments(install)


def test_package_install_auto_manager_uses_bounded_remote_discovery() -> None:
    install = decision(
        action="package",
        tool_name="server_package_install",
        arguments={"manager": "auto", "packages": ["nginx"]},
        required_capability="package.write",
        read_only=False,
        consequential=True,
        affected_resources=["package:nginx"],
    )

    argv = build_tool_argv(install)

    assert argv[:2] == ["sh", "-c"]
    assert "manager_count" in argv[2]
    assert "Expected exactly one supported package manager" in argv[2]
    assert "sudo apt-get install -y -- nginx" in argv[2]
    assert package_install_auto_argv(["nginx"]) == argv


def test_shell_execute_requires_an_exact_non_empty_argv_list() -> None:
    shell = decision(
        action="shell",
        tool_name="server_shell_execute",
        arguments={"argv": ["mkdir", "-p", "mana-agent-test"]},
        required_capability="shell",
        read_only=False,
        consequential=True,
        affected_resources=["directory:home/mana-agent-test"],
    )

    validate_tool_arguments(shell)
    assert build_tool_argv(shell) == ["mkdir", "-p", "mana-agent-test"]

    invalid = shell.model_copy(
        update={"arguments": {"argv": "mkdir -p mana-agent-test"}}
    )
    with pytest.raises(ValueError, match="exact argv string list"):
        validate_tool_arguments(invalid)


def test_firewall_plan_preserves_management_port() -> None:
    with pytest.raises(ValueError, match="management"):
        FirewallPlan(manager="ufw", allow=[FirewallRule(port=443)], management_port=22).validate_management_access()


def test_plan_skips_converged_steps_and_rolls_back_on_failed_verification() -> None:
    calls: list[str] = []
    plan = ServerPlan(
        plan_id="plan-1",
        name="production",
        server_ids=["server-1"],
        steps=[
            ServerPlanStep(step_id="converged", tool_name="server_package_install", rollback={"tool": "server_package_remove"}),
            ServerPlanStep(step_id="drifted", tool_name="server_service_restart", rollback={"tool": "server_service_restart"}),
        ],
    )

    async def drift(step):
        return step.step_id == "drifted"

    async def apply(step):
        calls.append(f"apply:{step.step_id}")

    async def verify(step):
        return False

    async def rollback(step):
        calls.append(f"rollback:{step.step_id}")

    runner = ServerPlanExecutor(inspect_drift=drift, apply_step=apply, verify_step=verify, rollback_step=rollback)
    with pytest.raises(RuntimeError, match="Verification failed"):
        asyncio.run(runner.apply(plan))
    assert calls == ["apply:drifted", "rollback:drifted"]


def test_all_server_tools_have_unique_typed_contracts() -> None:
    assert len(SERVER_TOOL_SPECS) >= 35
    assert len(SERVER_TOOL_SPECS) == len(set(SERVER_TOOL_SPECS))
