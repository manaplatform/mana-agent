"""Bounded worker-side capability probing and authenticated inventory validation."""

from __future__ import annotations

import asyncio
import json
import platform
import shutil
from collections.abc import Callable
from datetime import timedelta

from .errors import FleetCapabilityError
from .models import WorkerCapabilities, WorkerLabels, utc_now

MAX_INVENTORY_BYTES = 64 * 1024
PROBE_TOOLS = ("git", "pytest", "ruff", "node", "npm", "docker")


def normalize_platform(value: str) -> str:
    return {"darwin": "macos", "win32": "windows", "windows": "windows", "linux": "linux"}.get(
        value.strip().lower(), "unknown"
    )


async def _version(argv: tuple[str, ...], timeout_seconds: float) -> str:
    executable = shutil.which(argv[0])
    if not executable:
        return ""
    try:
        process = await asyncio.create_subprocess_exec(
            executable, *argv[1:],
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except (OSError, asyncio.TimeoutError):
        return ""
    if process.returncode != 0:
        return ""
    text = (stdout or stderr).decode(errors="replace").strip().splitlines()
    return text[0][:160] if text else ""


def _numeric_version(value: str) -> str:
    for token in value.replace("v", " ").split():
        if token and token[0].isdigit():
            return ".".join(token.split(".")[:2])
    return ""


async def probe_worker_capabilities(
    worker_id: str, *,
    labels: set[str] | frozenset[str] = frozenset(),
    max_concurrency: int = 1,
    timeout_seconds: float = 5,
    execution_providers: set[str] | frozenset[str] = frozenset({"reverse-worker"}),
) -> WorkerCapabilities:
    """Inspect only an allowlisted, non-secret set of runtime facts."""
    if timeout_seconds <= 0 or timeout_seconds > 30:
        raise ValueError("capability probe timeout must be between 0 and 30 seconds")
    python_value, node_value, docker_value = await asyncio.gather(
        _version(("python", "--version"), timeout_seconds),
        _version(("node", "--version"), timeout_seconds),
        _version(("docker", "--version"), timeout_seconds),
    )
    available = frozenset(name for name in PROBE_TOOLS if shutil.which(name))
    return WorkerCapabilities(
        worker_id=worker_id,
        platform=normalize_platform(platform.system()),
        platform_release=platform.release(),
        architecture=platform.machine() or "unknown",
        python_versions=tuple(filter(None, [_numeric_version(python_value)])),
        node_versions=tuple(filter(None, [_numeric_version(node_value)])),
        available_tools=available,
        docker=bool(docker_value) if shutil.which("docker") else None,
        max_concurrency=max_concurrency,
        labels=WorkerLabels(values=frozenset(labels)),
        workspace_backends=frozenset({"execution-fabric"}),
        execution_providers=frozenset(execution_providers),
        last_probe_at=utc_now(),
    )


def canonical_inventory_payload(capabilities: WorkerCapabilities) -> bytes:
    return json.dumps(
        capabilities.model_dump(mode="json"),
        sort_keys=True, separators=(",", ":"),
    ).encode()


def validate_inventory_update(
    payload: bytes, *,
    authenticated_worker_id: str,
    verify_signature: Callable[[str, bytes, bytes], None],
    signature: bytes,
    capability_ttl_seconds: int,
    now=None,
) -> WorkerCapabilities:
    if len(payload) > MAX_INVENTORY_BYTES:
        raise FleetCapabilityError("worker capability inventory exceeds the 64 KiB limit")
    try:
        verify_signature(authenticated_worker_id, payload, signature)
    except Exception as exc:
        raise FleetCapabilityError("worker capability inventory authentication failed") from exc
    try:
        capabilities = WorkerCapabilities.model_validate_json(payload)
    except Exception as exc:
        raise FleetCapabilityError("worker capability inventory is malformed or incompatible") from exc
    if capabilities.worker_id != authenticated_worker_id:
        raise FleetCapabilityError("worker capability inventory identity does not match authenticated worker")
    current = now or utc_now()
    age = current - capabilities.last_probe_at
    if age < timedelta(minutes=-5):
        raise FleetCapabilityError("worker capability inventory timestamp is in the future")
    if age > timedelta(seconds=capability_ttl_seconds):
        raise FleetCapabilityError("worker capability inventory is stale")
    return capabilities
