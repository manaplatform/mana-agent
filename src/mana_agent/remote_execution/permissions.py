"""Conservative remote command-risk classification for permission selection."""

from __future__ import annotations

from mana_agent.remote_execution.models import PermissionCategory, RemoteExecutionRequest


_PRIVILEGED = frozenset({"sudo", "su", "rm", "reboot", "shutdown", "systemctl", "apt", "apt-get", "yum", "dnf", "pip", "npm", "docker", "kubectl"})
_WRITE = frozenset({"mv", "cp", "touch", "mkdir", "chmod", "chown", "tee", "sed", "dd", "truncate"})


def required_permission(request: RemoteExecutionRequest) -> PermissionCategory:
    """Classify from structured argv; `read_only` is never a security proof."""
    command = request.command.argv[0].rsplit("/", 1)[-1]
    if request.pty:
        return PermissionCategory.INTERACTIVE
    if command in _PRIVILEGED:
        return PermissionCategory.PRIVILEGED
    if command in _WRITE or any(token in {">", ">>", "|"} for token in request.command.argv):
        return PermissionCategory.REMOTE_WRITE
    return PermissionCategory.READ_ONLY
