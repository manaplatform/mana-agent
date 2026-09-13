from __future__ import annotations

from pathlib import Path

import pytest

from typer.testing import CliRunner

from mana_agent.commands.ssh_cli import ssh_app
from mana_agent.config.user_config import get_ssh_password, save_ssh_password
from mana_agent.remote_execution.models import RemoteCommand, RemoteExecutionEvent, RemoteExecutionRequest
from mana_agent.remote_execution.profiles import SSHProfile, get_profile, list_profiles, remove_profile, save_profile
from mana_agent.remote_execution.providers.local_ssh import LocalSSHProvider, build_ssh_argv
from mana_agent.remote_execution.service import RemoteExecutionService
from mana_agent.remote_execution.target_policy import TargetPolicy, TargetPolicyMode


def test_profile_persistence_keeps_only_identity_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    key = tmp_path / "id_ed25519"
    key.write_text("not a real key", encoding="utf-8")
    profile = SSHProfile(name="production", host="example.test", user="root", identity_file=str(key))
    save_profile(profile)
    loaded = get_profile("production")
    assert loaded.identity_file == str(key)
    assert [item.name for item in list_profiles()] == ["production"]
    assert "not a real key" not in (tmp_path / "mana" / "config.toml").read_text(encoding="utf-8")
    remove_profile("production")
    with pytest.raises(LookupError):
        get_profile("production")


def test_agent_profile_constructs_safe_ssh_arguments() -> None:
    profile = SSHProfile(name="office", host="office.example.test", user="ali", port=2200, use_agent=True)
    request = RemoteExecutionRequest(
        job_id="job", session_id="session", provider="remote-ssh", target=profile.target(),
        authentication=profile.authentication(), command=RemoteCommand(argv=["uname", "-a"]),
    )
    argv = build_ssh_argv(request, connect_timeout_seconds=profile.connect_timeout_seconds)
    assert argv[:7] == ["ssh", "-p", "2200", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
    assert "-i" not in argv
    assert "ConnectTimeout=15" in argv


def test_explicit_direct_ssh_never_silently_switches_to_worker() -> None:
    request = RemoteExecutionRequest(
        job_id="job", session_id="session", provider="remote-ssh", target={"host": "example.test", "user": "root"},
        authentication={"mode": "agent"}, command={"argv": ["true"]},
    )
    service = RemoteExecutionService(target_policy=TargetPolicy(TargetPolicyMode.UNRESTRICTED), outbound_tcp_available=False)
    service.submit(request)
    with pytest.raises(RuntimeError, match="host-process sandbox restriction"):
        import asyncio
        asyncio.run(service.execute(request.job_id))


def test_password_profile_persistence_and_safe_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    save_ssh_password("staging", "super-secret-pass")

    profile = SSHProfile(
        name="staging",
        host="staging.example.test",
        user="ubuntu",
        port=2222,
        auth_mode="password",
        password_ref="secret://ssh/staging",
    )
    save_profile(profile)
    loaded = get_profile("staging")
    assert loaded.auth_mode == "password"
    assert loaded.password_ref == "secret://ssh/staging"
    assert loaded.identity_file is None
    assert not loaded.use_agent

    # Verify secret is stored securely in secrets.toml and NOT in config.toml
    config_text = (tmp_path / "mana" / "config.toml").read_text(encoding="utf-8")
    secrets_text = (tmp_path / "mana" / "secrets.toml").read_text(encoding="utf-8")
    assert "super-secret-pass" not in config_text
    assert "super-secret-pass" in secrets_text

    request = RemoteExecutionRequest(
        job_id="job-pw",
        session_id="session-pw",
        provider="remote-ssh",
        target=loaded.target(),
        authentication=loaded.authentication(),
        command=RemoteCommand(argv=["ls", "-la"]),
    )
    assert request.authentication.mode == "password"
    assert request.authentication.password_ref == "secret://ssh/staging"

    argv = build_ssh_argv(request, connect_timeout_seconds=loaded.connect_timeout_seconds)
    assert argv[:5] == ["ssh", "-p", "2222", "-o", "BatchMode=no"]
    assert "-o" in argv and "StrictHostKeyChecking=yes" in argv
    assert "PreferredAuthentications=password,keyboard-interactive" in argv
    assert "PubkeyAuthentication=no" in argv
    assert "NumberOfPasswordPrompts=1" in argv
    assert "-i" not in argv
    assert "super-secret-pass" not in " ".join(argv)


def test_no_silent_downgrade_or_cross_mode_configuration(tmp_path: Path) -> None:
    key = tmp_path / "id_rsa"
    key.write_text("fake-key", encoding="utf-8")

    # Key profile cannot have password_ref
    with pytest.raises(ValueError, match="Key authentication profile cannot set password_ref"):
        SSHProfile(
            name="test-key",
            host="example.test",
            user="root",
            auth_mode="key",
            identity_file=str(key),
            password_ref="secret://ssh/test",
        )

    # Password profile cannot have identity_file or use_agent
    with pytest.raises(ValueError, match="Password authentication profile cannot set identity_file or use_agent"):
        SSHProfile(
            name="test-pw-id",
            host="example.test",
            user="root",
            auth_mode="password",
            identity_file=str(key),
        )

    with pytest.raises(ValueError, match="Password authentication profile cannot set identity_file or use_agent"):
        SSHProfile(
            name="test-pw-agent",
            host="example.test",
            user="root",
            auth_mode="password",
            use_agent=True,
        )


def test_password_auth_missing_password_fails_safely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    profile = SSHProfile(
        name="no-pw",
        host="missing.example.test",
        user="guest",
        auth_mode="password",
        password_ref="env://NON_EXISTENT_PASSWORD_VAR",
    )
    request = RemoteExecutionRequest(
        job_id="job-missing",
        session_id="session-missing",
        provider="remote-ssh",
        target=profile.target(),
        authentication=profile.authentication(),
        command=RemoteCommand(argv=["whoami"]),
    )
    with pytest.raises(RuntimeError, match="Missing password credential"):
        import asyncio
        asyncio.run(LocalSSHProvider().execute(request, lambda e: None, asyncio.Event()))


def test_successful_password_authenticated_ssh_run_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    save_ssh_password("prod-pw", "correct-password")
    profile = SSHProfile(
        name="prod-pw",
        host="prod.example.test",
        user="deploy",
        auth_mode="password",
        password_ref="secret://ssh/prod-pw",
    )
    save_profile(profile)

    executed_requests: list[RemoteExecutionRequest] = []

    async def fake_execute(self, request, emit, cancel):
        executed_requests.append(request)
        emit(RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="stdout", data={"chunk": "hello from remote"}))
        return 0, "hello from remote", ""

    monkeypatch.setattr(LocalSSHProvider, "execute", fake_execute)

    result = CliRunner().invoke(ssh_app, ["run", "prod-pw", "--", "echo", "hello"])
    assert result.exit_code == 0
    assert "hello from remote" in result.output
    assert len(executed_requests) == 1
    assert executed_requests[0].authentication.mode == "password"


def test_ssh_add_and_set_password_cli_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    runner = CliRunner()

    result = runner.invoke(
        ssh_app,
        ["add", "web", "--host", "web.example.test", "--user", "admin", "--password"],
        input="my-interactive-password\n",
    )
    assert result.exit_code == 0
    assert "Saved SSH target 'web' (auth: password)" in result.output

    web_profile = get_profile("web")
    assert web_profile.auth_mode == "password"
    assert get_ssh_password(web_profile.password_ref) == "my-interactive-password"

    result = runner.invoke(
        ssh_app,
        ["set-password", "web"],
        input="new-updated-password\n",
    )
    assert result.exit_code == 0
    assert "Saved password for SSH target 'web'" in result.output
    assert get_ssh_password(web_profile.password_ref) == "new-updated-password"
