"""Canonical server tool metadata shared by routing and execution."""

from __future__ import annotations

from dataclasses import dataclass

from mana_agent.server.models import ServerActionDecision, ServerActionKind


@dataclass(frozen=True, slots=True)
class ServerToolSpec:
    name: str
    action: ServerActionKind
    capability: str
    read_only: bool
    consequential: bool = False
    destructive: bool = False
    arguments_json_example: str = ""


def _spec(
    name: str,
    action: ServerActionKind,
    capability: str,
    read_only: bool,
    consequential: bool = False,
    destructive: bool = False,
    arguments_json_example: str = "",
) -> ServerToolSpec:
    return ServerToolSpec(
        name,
        action,
        capability,
        read_only,
        consequential,
        destructive,
        arguments_json_example,
    )


SERVER_TOOL_SPECS = {
    item.name: item
    for item in (
        _spec("server_connect", ServerActionKind.INSPECT, "inspect", True),
        _spec("server_inspect", ServerActionKind.INSPECT, "inspect", True),
        _spec("server_shell_execute", ServerActionKind.SHELL, "shell", False, True),
        _spec("server_shell_session_start", ServerActionKind.SHELL, "shell", False, True),
        _spec("server_shell_session_write", ServerActionKind.SHELL, "shell", False, True),
        _spec("server_shell_session_resize", ServerActionKind.SHELL, "shell", False, True),
        _spec("server_shell_session_stop", ServerActionKind.SHELL, "shell", False, True),
        _spec("server_file_read", ServerActionKind.FILE_READ, "filesystem.read", True),
        _spec("server_file_write", ServerActionKind.FILE_WRITE, "filesystem.write", False, True),
        _spec("server_file_patch", ServerActionKind.FILE_WRITE, "filesystem.write", False, True),
        _spec("server_file_upload", ServerActionKind.FILE_WRITE, "filesystem.write", False, True),
        _spec("server_file_download", ServerActionKind.FILE_READ, "filesystem.read", True),
        _spec("server_directory_list", ServerActionKind.FILE_READ, "filesystem.read", True),
        _spec("server_process_list", ServerActionKind.INSPECT, "process.read", True),
        _spec("server_process_stop", ServerActionKind.PROCESS, "process.write", False, True),
        _spec("server_service_status", ServerActionKind.INSPECT, "service.read", True),
        _spec("server_service_start", ServerActionKind.SERVICE, "service.write", False, True),
        _spec("server_service_stop", ServerActionKind.SERVICE, "service.write", False, True),
        _spec("server_service_restart", ServerActionKind.SERVICE, "service.write", False, True),
        _spec("server_service_enable", ServerActionKind.SERVICE, "service.write", False, True),
        _spec("server_package_search", ServerActionKind.INSPECT, "package.read", True),
        _spec(
            "server_package_install",
            ServerActionKind.PACKAGE,
            "package.write",
            False,
            True,
            arguments_json_example=(
                '{"manager":"auto|apt|dnf|yum|pacman|apk|zypper|brew",'
                '"packages":["nginx"]}'
            ),
        ),
        _spec("server_package_remove", ServerActionKind.PACKAGE, "package.write", False, True),
        _spec("server_package_update", ServerActionKind.PACKAGE, "package.write", False, True),
        _spec("server_user_create", ServerActionKind.USER, "user.write", False, True),
        _spec("server_user_delete", ServerActionKind.USER, "user.write", False, True, True),
        _spec("server_ssh_key_add", ServerActionKind.USER, "ssh.write", False, True),
        _spec("server_firewall_status", ServerActionKind.INSPECT, "network.read", True),
        _spec("server_firewall_apply", ServerActionKind.FIREWALL, "firewall.write", False, True, True),
        _spec("server_port_check", ServerActionKind.INSPECT, "network.read", True),
        _spec("server_network_inspect", ServerActionKind.INSPECT, "network.read", True),
        _spec("server_log_read", ServerActionKind.INSPECT, "logs.read", True),
        _spec("server_log_follow", ServerActionKind.INSPECT, "logs.read", True),
        _spec("server_disk_inspect", ServerActionKind.INSPECT, "storage.read", True),
        _spec("server_database_backup", ServerActionKind.BACKUP, "database.backup", False, True),
        _spec("server_database_restore", ServerActionKind.RESTORE, "database.restore", False, True, True),
        _spec("server_container_list", ServerActionKind.INSPECT, "container.read", True),
        _spec("server_container_run", ServerActionKind.CONTAINER, "container.write", False, True),
        _spec("server_deploy", ServerActionKind.DEPLOYMENT, "deployment", False, True),
        _spec("server_reboot", ServerActionKind.REBOOT, "power", False, True, True),
        _spec("server_shutdown", ServerActionKind.SHUTDOWN, "power", False, True, True),
    )
}


def validate_tool_decision(decision: ServerActionDecision) -> ServerToolSpec:
    try:
        spec = SERVER_TOOL_SPECS[decision.tool_name]
    except KeyError as exc:
        raise ValueError(f"Unknown server tool {decision.tool_name!r}; no tool was executed.") from exc
    mismatches = []
    for field, actual, expected in (
        ("action", decision.action, spec.action),
        ("required_capability", decision.required_capability, spec.capability),
        ("read_only", decision.read_only, spec.read_only),
        ("consequential", decision.consequential, spec.consequential),
        ("destructive", decision.destructive, spec.destructive),
    ):
        if actual != expected:
            mismatches.append(field)
    if mismatches:
        raise ValueError(
            f"Server decision does not match tool contract fields: {', '.join(mismatches)}. No tool was executed."
        )
    return spec
