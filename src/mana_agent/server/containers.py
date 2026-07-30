"""Container runtime inspection and lifecycle builders."""

from __future__ import annotations


def container_list_argv(runtime: str) -> list[str]:
    if runtime not in {"docker", "podman"}:
        raise ValueError("Container runtime must be explicitly selected as docker or podman.")
    return [runtime, "ps", "--all", "--format", "json"]


def compose_config_argv(runtime: str, compose_file: str) -> list[str]:
    if not compose_file.startswith("/"):
        raise ValueError("Compose file must be an absolute remote path.")
    if runtime == "docker":
        return ["docker", "compose", "--file", compose_file, "config", "--quiet"]
    if runtime == "podman":
        return ["podman-compose", "--file", compose_file, "config"]
    raise ValueError("Compose runtime must be explicitly selected.")
