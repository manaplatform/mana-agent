"""Persistent, non-secret direct-SSH connection profiles."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import Field, field_validator

from mana_agent.config.user_config import load_user_config, save_user_config
from mana_agent.remote_execution.models import SSHAuthentication, SSHTarget, StrictModel


class SSHProfile(StrictModel):
    """Only connection metadata is persisted; private-key material is never read."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    host: str = Field(min_length=1)
    port: int = Field(default=22, ge=1, le=65535)
    user: str = Field(min_length=1)
    identity_file: str | None = None
    use_agent: bool = False
    connect_timeout_seconds: int = Field(default=15, gt=0, le=600)
    strict_host_key_checking: bool = True
    known_hosts_file: str | None = None
    labels: list[str] = Field(default_factory=list)
    ssh_only: bool = False
    host_key_fingerprint: str = ""
    last_successful_connection: datetime | None = None

    @field_validator("identity_file")
    @classmethod
    def validate_identity_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError("identity_file must reference an existing private-key path")
        return str(path)

    def authentication(self) -> SSHAuthentication:
        if self.use_agent:
            return SSHAuthentication(mode="agent")
        if not self.identity_file:
            raise ValueError("SSH profile requires --identity or --use-agent")
        return SSHAuthentication(mode="key_path", key_path=self.identity_file)

    def target(self) -> SSHTarget:
        return SSHTarget(host=self.host, port=self.port, user=self.user)

    def known_hosts_path(self) -> Path:
        return Path(self.known_hosts_file or "~/.ssh/known_hosts").expanduser()


def _profiles_config() -> dict[str, object]:
    config = load_user_config()
    ssh = config.get("ssh", {})
    if not isinstance(ssh, dict):
        return {}
    profiles = ssh.get("profiles", {})
    return dict(profiles) if isinstance(profiles, dict) else {}


def list_profiles() -> list[SSHProfile]:
    profiles: list[SSHProfile] = []
    for name, payload in _profiles_config().items():
        if not isinstance(payload, dict):
            continue
        profiles.append(SSHProfile.model_validate({"name": name, **payload}))
    return sorted(profiles, key=lambda item: item.name)


def get_profile(name: str) -> SSHProfile:
    payload = _profiles_config().get(name)
    if not isinstance(payload, dict):
        raise LookupError(f"SSH profile {name!r} does not exist.")
    return SSHProfile.model_validate({"name": name, **payload})


def save_profile(profile: SSHProfile) -> None:
    config = load_user_config()
    ssh = config.get("ssh", {})
    if not isinstance(ssh, dict):
        ssh = {}
    profiles = ssh.get("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
    payload = profile.model_dump(exclude={"name"}, mode="json", exclude_none=True)
    profiles[profile.name] = payload
    ssh["profiles"] = profiles
    config["ssh"] = ssh
    save_user_config(config, merge=False)


def remove_profile(name: str) -> None:
    config = load_user_config()
    ssh = config.get("ssh", {})
    profiles = ssh.get("profiles", {}) if isinstance(ssh, dict) else {}
    if not isinstance(profiles, dict) or name not in profiles:
        raise LookupError(f"SSH profile {name!r} does not exist.")
    del profiles[name]
    ssh["profiles"] = profiles
    config["ssh"] = ssh
    save_user_config(config, merge=False)


def find_ssh_executable(binary: str = "ssh") -> str:
    resolved = shutil.which(binary)
    if not resolved:
        raise RuntimeError("No compatible system SSH client was found. Install OpenSSH and ensure `ssh` is on PATH.")
    return resolved


def inspect_host_key(profile: SSHProfile, *, keyscan_binary: str = "ssh-keyscan", timeout: int = 15) -> tuple[str, str]:
    """Return the scanned host-key line and SHA256 fingerprint without trusting it."""
    keyscan = shutil.which(keyscan_binary)
    keygen = shutil.which("ssh-keygen")
    if not keyscan or not keygen:
        raise RuntimeError("Host-key review requires system ssh-keyscan and ssh-keygen executables.")
    result = subprocess.run(
        [keyscan, "-p", str(profile.port), "-T", str(timeout), profile.host],
        capture_output=True, text=True, timeout=timeout + 2, check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
    if not lines:
        raise RuntimeError(result.stderr.strip() or "No SSH host key was returned by the target.")
    key_line = lines[0]
    fingerprint = subprocess.run([keygen, "-lf", "-"], input=key_line + "\n", capture_output=True, text=True, check=False)
    if fingerprint.returncode:
        raise RuntimeError("Could not calculate the scanned SSH host-key fingerprint.")
    return key_line, fingerprint.stdout.strip()


def trust_scanned_host_key(profile: SSHProfile, key_line: str, fingerprint: str) -> SSHProfile:
    """Append an approved key exactly once after its fingerprint was shown to the user."""
    known_hosts = profile.known_hosts_path()
    known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing = known_hosts.read_text(encoding="utf-8") if known_hosts.exists() else ""
    if key_line not in existing.splitlines():
        with known_hosts.open("a", encoding="utf-8") as handle:
            handle.write(key_line.rstrip() + "\n")
        os.chmod(known_hosts, 0o600)
    profile.host_key_fingerprint = fingerprint
    save_profile(profile)
    return profile


def fingerprint_matches(profile: SSHProfile, fingerprint: str) -> bool:
    return not profile.host_key_fingerprint or profile.host_key_fingerprint == fingerprint


def command_audit_key(profile: SSHProfile, argv: list[str]) -> str:
    """Stable audit reference with no key contents or command output."""
    raw = "\0".join([profile.name, profile.host, str(profile.port), profile.user, *argv])
    return hashlib.sha256(raw.encode()).hexdigest()
