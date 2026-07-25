from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.remote_execution.models import RemoteCommand, RemoteExecutionRequest
from mana_agent.remote_execution.profiles import SSHProfile, get_profile, list_profiles, remove_profile, save_profile
from mana_agent.remote_execution.providers.local_ssh import build_ssh_argv
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
