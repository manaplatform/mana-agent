"""Mechanical argv construction after a validated model tool decision."""

from __future__ import annotations

from .backups import postgres_backup_argv
from .containers import container_list_argv
from .filesystem import atomic_write_script, list_directory_argv, read_file_argv
from .models import ServerActionDecision
from .packages import (
    package_install_auto_argv,
    package_install_argv,
    package_remove_argv,
    validate_package_arguments,
)
from .processes import process_list_argv, process_signal_argv
from .services import service_argv, service_logs_argv
from .tools import validate_tool_decision
from .users import user_create_argv, user_delete_argv


def validate_tool_arguments(decision: ServerActionDecision) -> None:
    """Validate model-selected arguments before approval or argv construction."""
    if decision.tool_name == "server_package_install":
        validate_package_arguments(decision.arguments)


def build_tool_argv(decision: ServerActionDecision) -> list[str]:
    """Build only the exact selected tool; unsupported selections fail closed."""
    validate_tool_decision(decision)
    args = decision.arguments
    name = decision.tool_name
    if name in {"server_connect"}:
        return ["true"]
    if name == "server_shell_execute":
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise ValueError("server_shell_execute requires an exact argv string list.")
        return argv
    if name == "server_file_read":
        return read_file_argv(str(args.get("path") or ""))
    if name == "server_directory_list":
        return list_directory_argv(str(args.get("path") or ""))
    if name in {"server_file_write", "server_file_patch"}:
        return atomic_write_script(
            str(args.get("path") or ""),
            str(args.get("content") or ""),
            mode=str(args.get("mode") or "0644"),
            backup=bool(args.get("backup", True)),
        )
    if name == "server_process_list":
        return process_list_argv()
    if name == "server_process_stop":
        return process_signal_argv(int(args.get("pid") or 0), str(args.get("signal") or "TERM"))
    if name.startswith("server_service_"):
        action = name.removeprefix("server_service_")
        if action == "status":
            return service_argv(str(args["manager"]), "status", str(args["service"]))  # type: ignore[arg-type]
        return service_argv(str(args["manager"]), action, str(args["service"]))  # type: ignore[arg-type]
    if name == "server_log_read":
        return service_logs_argv(str(args["manager"]), str(args["service"]), int(args.get("lines") or 100))  # type: ignore[arg-type]
    if name == "server_package_install":
        manager, packages = validate_package_arguments(args)
        if manager == "auto":
            return package_install_auto_argv(packages)
        return package_install_argv(manager, packages)
    if name == "server_package_remove":
        return package_remove_argv(str(args["manager"]), list(args.get("packages") or []), purge=bool(args.get("purge")))  # type: ignore[arg-type]
    if name == "server_user_create":
        return user_create_argv(str(args["username"]), shell=str(args.get("shell") or "/bin/bash"))
    if name == "server_user_delete":
        return user_delete_argv(str(args["username"]), remove_home=bool(args.get("remove_home")))
    if name == "server_firewall_status":
        return [str(args.get("manager") or "ufw"), "status", "verbose"]
    if name == "server_network_inspect":
        return ["ip", "-details", "address", "show"]
    if name == "server_port_check":
        return ["ss", "-lntup"]
    if name == "server_disk_inspect":
        return ["df", "-hP"]
    if name == "server_database_backup":
        return postgres_backup_argv(str(args["database"]), str(args["destination"]))
    if name == "server_container_list":
        return container_list_argv(str(args["runtime"]))
    if name == "server_reboot":
        return ["sudo", "systemctl", "reboot"]
    if name == "server_shutdown":
        return ["sudo", "systemctl", "poweroff"]
    raise NotImplementedError(
        f"Validated server tool {name!r} has no installed runtime adapter. No fallback command was executed."
    )
