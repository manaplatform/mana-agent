from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from mana_agent.utils.redaction import redact_secrets

from .models import (
    ActionIntent,
    ActionPreview,
    BlastRadius,
    CompensationEvidence,
    DataDisclosure,
    Reversibility,
    VerificationEvidence,
)


class ActionInvalidatedError(RuntimeError):
    """Previewed resources changed before execution; prior approval is void."""


class ActionAdapter(ABC):
    native_idempotency = False

    @abstractmethod
    def build_intent(self) -> ActionIntent: ...

    @abstractmethod
    def preview(self, action: ActionIntent) -> ActionPreview: ...

    @abstractmethod
    def execute(self, action: ActionIntent) -> dict[str, Any]: ...

    @abstractmethod
    def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence: ...

    def compensate(self, action: ActionIntent) -> CompensationEvidence:
        return CompensationEvidence(complete=False, summary="This adapter does not define a safe compensation.")

    def persistable_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return redact_secrets(result)


class CompensationActionAdapter(ActionAdapter):
    """Represent compensation as a distinct policy-gated action, never rollback by assertion."""

    def __init__(self, original: ActionIntent, delegate: ActionAdapter) -> None:
        self.original, self.delegate = original, delegate

    def build_intent(self) -> ActionIntent:
        compensation_operations = {
            "create": "delete",
            "edit": "edit",
            "delete": "create",
            "move": "move",
        }
        return ActionIntent(
            parent_task_id=self.original.parent_task_id,
            transaction_id=self.original.transaction_id,
            actor="transaction_coordinator",
            originating_agent=self.original.originating_agent,
            tool_name=self.original.tool_name,
            operation_name=(
                compensation_operations.get(self.original.operation_name, f"compensate_{self.original.operation_name}")
                if self.original.tool_name == "file"
                else f"compensate_{self.original.operation_name}"
            ),
            target_resources=self.original.target_resources,
            normalized_arguments={"compensates_action_id": self.original.action_id, "original_binding_digest": self.original.binding_digest()},
            requested_capabilities=[f"{self.original.tool_name}.compensate"],
            expected_side_effects=[f"compensate action {self.original.action_id}"],
            data_disclosure=DataDisclosure.NONE,
            blast_radius=self.original.blast_radius,
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            idempotency_key=f"compensate:{self.original.action_id}",
            verification_plan=["run adapter compensation verifier", "record compensation evidence"],
            compensation_strategy="No recursive automatic compensation.",
        )

    def preview(self, action: ActionIntent) -> ActionPreview:
        return ActionPreview(
            summary=f"compensate action {self.original.action_id}",
            resources=[{"resource": item, "change": "compensation"} for item in self.original.target_resources],
            exact_invocation=action.normalized_arguments,
            expected_side_effects=action.expected_side_effects,
            risks=["compensation is a new side effect and is not atomic rollback"],
        )

    def execute(self, action: ActionIntent) -> dict[str, Any]:
        return self.delegate.compensate(self.original).model_dump(mode="json")

    def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
        complete = bool(result.get("complete"))
        return VerificationEvidence(
            complete=complete,
            summary=str(result.get("summary") or "Compensation verification was unavailable."),
            checks=list(result.get("checks") or []),
        )


class FileActionAdapter(ActionAdapter):
    """Create, replace, move, or delete one workspace file with snapshot evidence."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        operation: str,
        path: str,
        content: bytes | str | None = None,
        destination: str = "",
        parent_task_id: str,
        actor: str,
        originating_agent: str,
        idempotency_key: str,
        transaction_id: str = "",
        snapshot_root: Path | None = None,
        desired_mode: int | None = None,
    ) -> None:
        if operation not in {"create", "edit", "move", "delete"}:
            raise ValueError("unsupported file action")
        self.workspace_root = workspace_root.expanduser().resolve()
        self.operation = operation
        self.path = self._target(path)
        self.destination = self._target(destination) if destination else None
        self.content = content.encode("utf-8") if isinstance(content, str) else content
        self.parent_task_id = parent_task_id
        self.actor = actor
        self.originating_agent = originating_agent
        self.idempotency_key = idempotency_key
        self.transaction_id = transaction_id
        self.desired_mode = desired_mode
        self.snapshot_root = (snapshot_root or self.workspace_root / ".mana" / "action_snapshots").resolve()
        self._before: bytes | None = None
        self._before_mode: int | None = None

    def _target(self, raw: str) -> Path:
        candidate = Path(raw)
        target = candidate.resolve() if candidate.is_absolute() else (self.workspace_root / candidate).resolve()
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError("file action path escapes workspace") from exc
        return target

    def build_intent(self) -> ActionIntent:
        targets = [str(self.path)] + ([str(self.destination)] if self.destination else [])
        reversible = Reversibility.FULLY_REVERSIBLE
        if self.operation == "create":
            effects = [f"create {self.path}"]
        elif self.operation == "edit":
            effects = [f"replace content of {self.path}"]
        elif self.operation == "move":
            effects = [f"move {self.path} to {self.destination}"]
        else:
            effects = [f"delete {self.path} while retaining recovery evidence"]
        return ActionIntent(
            parent_task_id=self.parent_task_id,
            transaction_id=self.transaction_id,
            actor=self.actor,
            originating_agent=self.originating_agent,
            tool_name="file",
            operation_name=self.operation,
            target_resources=targets,
            normalized_arguments={
                "path": str(self.path),
                "destination": str(self.destination or ""),
                "content_sha256": _sha(self.content) if self.content is not None else "",
                "content_bytes": len(self.content or b""),
                "desired_mode": self.desired_mode,
            },
            requested_capabilities=[f"file.{self.operation}"],
            expected_side_effects=effects,
            data_disclosure=DataDisclosure.NONE,
            blast_radius=BlastRadius.MULTIPLE_RESOURCES if self.destination else BlastRadius.SINGLE_RESOURCE,
            reversibility=reversible,
            idempotency_key=self.idempotency_key,
            verification_plan=["verify target existence", "verify expected content hash and metadata"],
            compensation_strategy="Restore the verified pre-execution snapshot; this is rollback only for local file state.",
        )

    def preview(self, action: ActionIntent) -> ActionPreview:
        exists = self.path.exists()
        before = self.path.read_bytes() if exists and self.path.is_file() else b""
        self._before = before if exists else None
        self._before_mode = stat.S_IMODE(self.path.stat().st_mode) if exists else None
        after = self.content or b""
        diff = ""
        if self.operation in {"create", "edit"} and _text(before) is not None and _text(after) is not None:
            diff = "".join(difflib.unified_diff(
                (_text(before) or "").splitlines(keepends=True),
                (_text(after) or "").splitlines(keepends=True),
                fromfile=str(self.path), tofile=str(self.path),
            ))
        risks: list[str] = []
        if self.operation == "create" and exists:
            risks.append("target already exists; execution will be denied")
        if self.operation in {"edit", "move", "delete"} and not exists:
            risks.append("source does not exist; execution will be denied")
        if self.destination and self.destination.exists():
            risks.append("destination exists; execution will be denied")
        return ActionPreview(
            summary=f"{self.operation} local file",
            resources=[
                {"path": str(self.path), "change": self.operation, "exists": exists, "before_sha256": _sha(before) if exists else ""},
                *([{"path": str(self.destination), "change": "destination", "exists": self.destination.exists()}] if self.destination else []),
            ],
            diff=diff,
            exact_invocation=action.normalized_arguments,
            expected_side_effects=action.expected_side_effects,
            risks=risks,
            supports_native_idempotency=False,
        )

    def execute(self, action: ActionIntent) -> dict[str, Any]:
        self._restore_and_validate_pre_execution_evidence(action)
        snapshot = self._persist_snapshot(action)
        if self.operation == "create":
            if self.path.exists() or self.content is None:
                raise FileExistsError("create requires a missing path and content")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("xb") as stream:
                stream.write(self.content)
        elif self.operation == "edit":
            if not self.path.is_file() or self.content is None:
                raise FileNotFoundError("edit requires an existing file and content")
            _atomic_write(self.path, self.content)
        elif self.operation == "move":
            if not self.path.exists() or self.destination is None or self.destination.exists():
                raise FileExistsError("move requires an existing source and missing destination")
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.path, self.destination)
        else:
            if not self.path.is_file():
                raise FileNotFoundError("delete requires an existing file")
            self.path.unlink()
        observed = self.destination if self.operation == "move" else self.path
        if self.desired_mode is not None and observed.exists():
            observed.chmod(self.desired_mode)
        return {
            "path": str(observed),
            "operation": self.operation,
            "sha256": _sha(observed.read_bytes()) if observed.is_file() else "",
            "snapshot_reference": snapshot,
            "snapshot_mode": self._before_mode,
        }

    def _restore_and_validate_pre_execution_evidence(self, action: ActionIntent) -> None:
        resource = (action.preview.resources[0] if action.preview and action.preview.resources else {})
        preview_existed = bool(resource.get("exists"))
        exists = self.path.is_file()
        if exists != preview_existed:
            raise ActionInvalidatedError("file state changed after preview; approval and policy must be reevaluated")
        if not exists:
            return
        current = self.path.read_bytes()
        expected_hash = str(resource.get("before_sha256") or "")
        if expected_hash and _sha(current) != expected_hash:
            raise ActionInvalidatedError("file content changed after preview; approval and policy must be reevaluated")
        self._before = current
        self._before_mode = stat.S_IMODE(self.path.stat().st_mode)

    def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
        observed = self.destination if self.operation == "move" else self.path
        should_exist = self.operation != "delete"
        exists = bool(observed and observed.is_file())
        checks = [{"check": "existence", "expected": should_exist, "observed": exists}]
        complete = exists == should_exist
        expected_hash = str(action.normalized_arguments.get("content_sha256") or "")
        if should_exist and expected_hash:
            actual = _sha(observed.read_bytes()) if observed else ""
            checks.append({"check": "sha256", "expected": expected_hash, "observed": actual})
            complete = complete and actual == expected_hash
        if self.operation == "move":
            source_absent = not self.path.exists()
            checks.append({"check": "source_absent", "expected": True, "observed": source_absent})
            complete = complete and source_absent
        if should_exist and self.desired_mode is not None and observed:
            actual_mode = stat.S_IMODE(observed.stat().st_mode) if observed.exists() else None
            checks.append({"check": "mode", "expected": self.desired_mode, "observed": actual_mode})
            complete = complete and actual_mode == self.desired_mode
        return VerificationEvidence(complete=complete, summary="File state matches the requested action." if complete else "File verification did not match the requested state.", checks=checks)

    def compensate(self, action: ActionIntent) -> CompensationEvidence:
        try:
            snapshot = self._load_snapshot(action)
            snapshot_mode = action.execution_result.get("snapshot_mode")
            result_hash = str(action.execution_result.get("sha256") or "")
            post_action_path = self.destination if self.operation == "move" else self.path
            if self.operation in {"create", "edit", "move"}:
                if post_action_path is None or not post_action_path.is_file():
                    return CompensationEvidence(
                        complete=False,
                        summary="The post-action resource is missing; automatic compensation is unsafe.",
                    )
                if result_hash and _sha(post_action_path.read_bytes()) != result_hash:
                    return CompensationEvidence(
                        complete=False,
                        summary="The post-action resource changed; automatic compensation is unsafe.",
                    )
            if self.operation == "delete" and self.path.exists():
                return CompensationEvidence(
                    complete=False,
                    summary="The deleted path was recreated; automatic compensation is unsafe.",
                )
            if self.operation == "create":
                self.path.unlink()
            elif self.operation == "edit":
                if snapshot is None:
                    return CompensationEvidence(complete=False, summary="No pre-edit snapshot is available.")
                _atomic_write(self.path, snapshot)
            elif self.operation == "move":
                if self.destination is None or not self.destination.exists() or self.path.exists():
                    return CompensationEvidence(complete=False, summary="Move compensation preconditions no longer hold.")
                os.replace(self.destination, self.path)
            elif snapshot is not None:
                _atomic_write(self.path, snapshot)
            else:
                return CompensationEvidence(complete=False, summary="No pre-delete snapshot is available.")
            if snapshot_mode is not None and self.path.exists():
                self.path.chmod(int(snapshot_mode))
            checks: list[dict[str, Any]] = []
            if self.operation == "create":
                absent = not self.path.exists()
                checks.append({"check": "created_path_absent", "expected": True, "observed": absent})
                complete = absent
            else:
                exists = self.path.is_file()
                checks.append({"check": "restored_path_exists", "expected": True, "observed": exists})
                complete = exists
                if snapshot is not None:
                    observed_hash = _sha(self.path.read_bytes()) if exists else "missing"
                    expected_hash = _sha(snapshot)
                    checks.append({"check": "restored_sha256", "expected": expected_hash, "observed": observed_hash})
                    complete = complete and observed_hash == expected_hash
                if self.operation == "move":
                    destination_absent = bool(self.destination and not self.destination.exists())
                    checks.append({"check": "destination_absent", "expected": True, "observed": destination_absent})
                    complete = complete and destination_absent
                if snapshot_mode is not None and exists:
                    observed_mode = stat.S_IMODE(self.path.stat().st_mode)
                    checks.append({"check": "restored_mode", "expected": int(snapshot_mode), "observed": observed_mode})
                    complete = complete and observed_mode == int(snapshot_mode)
            return CompensationEvidence(
                complete=complete,
                summary=(
                    "Local file state was restored from verified pre-action evidence."
                    if complete else "Local file compensation did not match the pre-action evidence."
                ),
                checks=checks,
            )
        except OSError as exc:
            return CompensationEvidence(complete=False, summary=f"File compensation failed: {type(exc).__name__}")

    def _persist_snapshot(self, action: ActionIntent) -> str:
        if self._before is None:
            return ""
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_root / f"{action.action_id}.snapshot"
        _atomic_write(path, self._before)
        return str(path)

    def _load_snapshot(self, action: ActionIntent) -> bytes | None:
        reference = str(action.execution_result.get("snapshot_reference") or "")
        if not reference:
            return None
        path = Path(reference).expanduser().resolve()
        expected = (self.snapshot_root / f"{action.action_id}.snapshot").resolve()
        if path != expected or not path.is_file():
            return None
        content = path.read_bytes()
        resource = action.preview.resources[0] if action.preview and action.preview.resources else {}
        expected_hash = str(resource.get("before_sha256") or "")
        if expected_hash and _sha(content) != expected_hash:
            return None
        return content


class ShellActionAdapter(ActionAdapter):
    def __init__(self, *, argv: list[str], cwd: Path, environment: dict[str, str] | None, expected_outputs: list[str], parent_task_id: str, actor: str, originating_agent: str, idempotency_key: str, transaction_id: str = "", timeout_seconds: int = 120, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run, policy_context: dict[str, Any] | None = None, allow_command_result_verification: bool = False) -> None:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("shell action requires a non-empty argv list")
        self.argv, self.cwd, self.environment = list(argv), cwd.resolve(), dict(environment or {})
        self.expected_outputs = list(expected_outputs)
        self.parent_task_id, self.actor, self.originating_agent = parent_task_id, actor, originating_agent
        self.idempotency_key, self.transaction_id = idempotency_key, transaction_id
        self.timeout_seconds, self.runner = timeout_seconds, runner
        self.policy_context = redact_secrets(dict(policy_context or {}))
        self.allow_command_result_verification = bool(allow_command_result_verification)

    def build_intent(self) -> ActionIntent:
        argv_fingerprint = hashlib.sha256(json.dumps(self.argv, ensure_ascii=False).encode("utf-8")).hexdigest()
        environment_fingerprint = hashlib.sha256(json.dumps(self.environment, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return ActionIntent(parent_task_id=self.parent_task_id, transaction_id=self.transaction_id, actor=self.actor, originating_agent=self.originating_agent, tool_name="shell", operation_name="execute", target_resources=[str(self.cwd), *self.expected_outputs], normalized_arguments={"argv": _redact_argv(self.argv), "argv_fingerprint": argv_fingerprint, "cwd": str(self.cwd), "environment": redact_secrets(self.environment), "environment_fingerprint": environment_fingerprint, "timeout_seconds": self.timeout_seconds, "network_access": "unknown", "policy_context": self.policy_context, "allow_command_result_verification": self.allow_command_result_verification}, requested_capabilities=["process.execute"], expected_side_effects=["start a process", *[f"produce {item}" for item in self.expected_outputs]], data_disclosure=DataDisclosure.UNKNOWN if self.environment else DataDisclosure.NONE, blast_radius=BlastRadius.WORKSPACE, reversibility=Reversibility.UNKNOWN, idempotency_key=self.idempotency_key, verification_plan=["verify exit status", "verify each declared output or a trusted command-result receipt"], compensation_strategy="")

    def preview(self, action: ActionIntent) -> ActionPreview:
        destructive = Path(self.argv[0]).name in {"rm", "sudo", "dd", "mkfs", "shutdown", "reboot"}
        return ActionPreview(summary="execute shell argv", resources=[{"cwd": str(self.cwd), "network_access": "unknown"}, *[{"path": item, "change": "declared output"} for item in self.expected_outputs]], exact_invocation=action.normalized_arguments, expected_side_effects=action.expected_side_effects, risks=["destructive executable"] if destructive else ["process effects may exceed declared outputs", "network access requirements are unknown"], supports_native_idempotency=False)

    def execute(self, action: ActionIntent) -> dict[str, Any]:
        environment = {**os.environ, **self.environment}
        completed = self.runner(self.argv, cwd=self.cwd, env=environment, capture_output=True, text=True, shell=False, timeout=self.timeout_seconds, check=False)
        return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}

    def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
        checks = [{"check": "exit_status", "expected": 0, "observed": result.get("returncode")}]
        complete = result.get("returncode") == 0
        for raw in self.expected_outputs:
            path = Path(raw) if Path(raw).is_absolute() else self.cwd / raw
            exists = path.exists()
            checks.append({"check": "declared_output_exists", "path": str(path), "expected": True, "observed": exists})
            complete = complete and exists
        if not self.expected_outputs:
            if self.allow_command_result_verification:
                stdout_hash = _sha(str(result.get("stdout") or "").encode("utf-8"))
                stderr_hash = _sha(str(result.get("stderr") or "").encode("utf-8"))
                checks.append({"check": "stdout_sha256", "observed": stdout_hash})
                checks.append({"check": "stderr_sha256", "observed": stderr_hash})
            else:
                complete = False
        return VerificationEvidence(complete=complete, summary="Exit status and declared outputs were verified." if complete else "Shell verification is incomplete or failed.", checks=checks)

    def persistable_result(self, result: dict[str, Any]) -> dict[str, Any]:
        stdout = str(result.get("stdout") or "").encode("utf-8")
        stderr = str(result.get("stderr") or "").encode("utf-8")
        return {
            "returncode": int(result.get("returncode") or 0),
            "stdout_sha256": _sha(stdout),
            "stderr_sha256": _sha(stderr),
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
        }


class HttpActionAdapter(ActionAdapter):
    def __init__(self, *, method: str, url: str, headers: dict[str, str] | None, body: bytes | str | dict[str, Any] | None, parent_task_id: str, actor: str, originating_agent: str, idempotency_key: str, expected_statuses: tuple[int, ...] = (200, 201, 202, 204), transaction_id: str = "", verification: Callable[[dict[str, Any]], VerificationEvidence] | None = None, transport: Callable[[urllib.request.Request], dict[str, Any]] | None = None) -> None:
        self.method = method.upper()
        if self.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("HTTP action adapter supports POST, PUT, PATCH, and DELETE")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("HTTP action requires an absolute credential-free URL")
        self.url, self.headers = url, dict(headers or {})
        self.body = json.dumps(body, separators=(",", ":")).encode() if isinstance(body, dict) else body.encode() if isinstance(body, str) else body
        self.parent_task_id, self.actor, self.originating_agent = parent_task_id, actor, originating_agent
        self.idempotency_key, self.transaction_id = idempotency_key, transaction_id
        self.expected_statuses, self.verification_callback, self.transport = expected_statuses, verification, transport
        self.native_idempotency = any(key.lower() == "idempotency-key" for key in self.headers)

    def build_intent(self) -> ActionIntent:
        parsed = urlsplit(self.url)
        redacted_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "<redacted>" if parsed.query else "", ""))
        return ActionIntent(parent_task_id=self.parent_task_id, transaction_id=self.transaction_id, actor=self.actor, originating_agent=self.originating_agent, tool_name="http", operation_name=self.method, target_resources=[f"{parsed.scheme}://{parsed.hostname}{parsed.path}"], normalized_arguments={"method": self.method, "url": redacted_url, "host": parsed.hostname or "", "headers": redact_secrets(self.headers), "body_sha256": _sha(self.body or b""), "body_bytes": len(self.body or b"")}, requested_capabilities=["network.write", "data.disclose" if self.body else "network.connect"], expected_side_effects=[f"mutate remote resource with HTTP {self.method}"], data_disclosure=DataDisclosure.EXTERNAL_PRIVATE if self.body else DataDisclosure.NONE, blast_radius=BlastRadius.EXTERNAL_ACCOUNT, reversibility=Reversibility.IRREVERSIBLE if self.method == "DELETE" else Reversibility.UNKNOWN, idempotency_key=self.idempotency_key, verification_plan=["verify response semantics", "query remote state when an adapter verifier is configured"], compensation_strategy="No generic rollback exists; any provider-specific compensation is a separately gated action.")

    def preview(self, action: ActionIntent) -> ActionPreview:
        return ActionPreview(summary=f"HTTP {self.method} remote mutation", resources=[{"url": action.normalized_arguments["url"], "host": action.normalized_arguments["host"], "change": "remote mutation"}], exact_invocation=action.normalized_arguments, expected_side_effects=action.expected_side_effects, disclosed_data=[f"request body sha256={action.normalized_arguments['body_sha256']} bytes={action.normalized_arguments['body_bytes']}"] if self.body else [], risks=["remote mutation may be irreversible", "request-body disclosure cannot be recalled" if self.body else ""], supports_native_idempotency=self.native_idempotency)

    def execute(self, action: ActionIntent) -> dict[str, Any]:
        request = urllib.request.Request(self.url, data=self.body, headers=self.headers, method=self.method)
        if self.transport:
            return self.transport(request)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read(1024 * 1024)
                return {"status_code": response.status, "headers": redact_secrets(dict(response.headers.items())), "body_sha256": _sha(body), "body_bytes": len(body)}
        except urllib.error.HTTPError as exc:
            body = exc.read(1024 * 1024)
            return {"status_code": exc.code, "headers": redact_secrets(dict(exc.headers.items())), "body_sha256": _sha(body), "body_bytes": len(body)}

    def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
        if self.verification_callback:
            return self.verification_callback(result)
        status = int(result.get("status_code") or 0)
        accepted = status in self.expected_statuses
        return VerificationEvidence(
            complete=accepted,
            summary=(
                "HTTP response semantics were verified; the endpoint did not expose an independent state-query verifier."
                if accepted else "HTTP response semantics indicate failure."
            ),
            checks=[
                {"check": "response_status", "expected": list(self.expected_statuses), "observed": status, "accepted": accepted},
                {"check": "remote_state_query", "observed": "not_available", "complete": False},
            ],
        )

    def persistable_result(self, result: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            key: value
            for key, value in result.items()
            if key not in {"body_preview", "json_body", "text_body"}
        }
        return redact_secrets(allowed)


def _sha(value: bytes | None) -> str:
    return hashlib.sha256(value or b"").hexdigest()


def _text(value: bytes) -> str | None:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _redact_argv(argv: list[str]) -> list[str]:
    sensitive_flags = {
        "--api-key", "--api_key", "--authorization", "--password", "--secret", "--token"
    }
    redacted: list[str] = []
    hide_next = False
    for raw in argv:
        item = str(raw)
        if hide_next:
            redacted.append("***REDACTED***")
            hide_next = False
            continue
        lowered = item.lower()
        if lowered in sensitive_flags:
            redacted.append(item)
            hide_next = True
            continue
        if any(lowered.startswith(f"{flag}=") for flag in sensitive_flags):
            redacted.append(item.split("=", 1)[0] + "=***REDACTED***")
            continue
        item = re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1***REDACTED***@", item)
        redacted.append(str(redact_secrets(item)))
    return redacted


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.transaction.tmp")
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
