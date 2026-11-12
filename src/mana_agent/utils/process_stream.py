"""Bounded subprocess capture with independent streams and owner cancellation."""
from __future__ import annotations

import codecs
import os
import queue
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_SHARED_PROCESS_GROUP: ContextVar[bool] = ContextVar("mana_ipc_process_group", default=False)


@contextmanager
def shared_process_group():
    """IPC descendants stay in the supervised worker's cancellable group."""
    token = _SHARED_PROCESS_GROUP.set(True)
    try:
        yield
    finally:
        _SHARED_PROCESS_GROUP.reset(token)


def run_streaming_process(argv, **kwargs) -> subprocess.CompletedProcess:
    """Record each actual subprocess separately, including Git/provider helpers."""
    from mana_agent.multi_agent.agents.tool_runner_agent import current_operation, RunningOperation
    parent = current_operation()
    protected_values = tuple(kwargs.pop("_protected_values", ()))
    if parent is None:
        return _run_process(argv, **kwargs)
    import uuid
    identity = f"{parent.request.operation_id}:process:{uuid.uuid4().hex}"
    request = parent.request.model_copy(update={"operation_id": identity, "operation_attempt_id": identity,
        "tool_call_id": identity, "parent_execution_id": parent.request.operation_id,
        "arguments": {"argv": list(argv), "cwd": str(kwargs.get("cwd") or parent.request.workspace_root)},
        "timeout_seconds": float(kwargs.get("timeout") or parent.request.timeout_seconds)})
    operation = RunningOperation(request, parent.authority, parent.runner)
    from mana_agent.utils.redaction import SENSITIVE_KEYS
    environment_values = tuple(str(value) for key, value in (kwargs.get("env") or {}).items()
        if str(key).lower() in SENSITIVE_KEYS or any(str(key).lower().endswith("_" + suffix) for suffix in SENSITIVE_KEYS))
    operation.protected = (*operation.protected, *protected_values, *environment_values)
    parent.protected = (*parent.protected, *protected_values, *environment_values)
    from mana_agent.multi_agent.agents.tool_runner_agent import _public_value
    operation.public = _public_value(operation.public, operation.protected)
    parent.public.setdefault("child_operation_ids", []).append(identity)
    with operation.scope():
        operation.emit("command.started")
        try:
            result = _run_process(argv, **kwargs)
        except BaseException as exc:
            operation.emit("command.cancelled" if isinstance(exc, InterruptedError) else "command.failed",
                status="cancelled" if isinstance(exc, InterruptedError) else "timed_out" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "failed",
                termination_reason=str(exc))
            raise
        operation.emit("command.completed", status="success" if result.returncode == 0 else "failed", exit_code=result.returncode)
        return result


def _run_process(argv, *, cwd=None, env=None, timeout=None, input=None,
                          capture_output=True, text=True, check=False, **kwargs) -> subprocess.CompletedProcess:
    """Adapter for Mana's captured text commands; display strings are never executed."""
    from mana_agent.multi_agent.agents.tool_runner_agent import current_operation
    operation = current_operation()
    limit = operation.runner.capture_limit if operation else 10_485_760
    popen_kwargs: dict[str, Any] = dict(kwargs)
    popen_kwargs.pop("encoding", None)
    popen_kwargs.pop("errors", None)
    popen_kwargs.pop("shell", None)
    owns_group = not _SHARED_PROCESS_GROUP.get()
    if not owns_group and os.name == "posix":
        popen_kwargs["start_new_session"] = False
    if os.name == "posix" and owns_group:
        popen_kwargs.setdefault("start_new_session", True)
    elif os.name == "nt":
        popen_kwargs.setdefault("creationflags", subprocess.CREATE_NEW_PROCESS_GROUP)
    process = subprocess.Popen(argv, cwd=cwd, env=env, shell=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL, **popen_kwargs)
    messages: queue.Queue[tuple[str, bytes | None]] = queue.Queue(maxsize=64)
    def read(channel, stream):
        try:
            while True:
                chunk = os.read(stream.fileno(), 4096)
                if not chunk:
                    break
                messages.put((channel, chunk))
        finally:
            stream.close()
            messages.put((channel, None))
    readers = [threading.Thread(target=read, args=(channel, stream), daemon=True)
               for channel, stream in (("stdout", process.stdout), ("stderr", process.stderr))]
    for reader in readers:
        reader.start()
    if input is not None:
        def write_input():
            try:
                process.stdin.write(input.encode() if isinstance(input, str) else input)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                process.stdin.close()
        threading.Thread(target=write_input, daemon=True).start()
    decoders = {name: codecs.getincrementaldecoder("utf-8")("replace") for name in ("stdout", "stderr")}
    pending = {"stdout": "", "stderr": ""}
    suppressing = {"stdout": False, "stderr": False}
    from mana_agent.utils.redaction import PrivateKeyStreamFilter
    key_filters = {channel: PrivateKeyStreamFilter() for channel in pending}
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    ended: set[str] = set()
    start = time.monotonic()
    termination: BaseException | None = None
    def stop(force_group=False):
        if process.poll() is not None and not force_group:
            return
        try:
            if os.name == "posix" and owns_group:
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass
    def publish(channel, value, final=False):
        if suppressing[channel]:
            boundary = value.find("\n")
            if boundary < 0 and not final:
                if operation:
                    operation.omitted_bytes += len(value.encode())
                return
            if operation:
                operation.omitted_bytes += len((value[:boundary + 1] if boundary >= 0 else value).encode())
            value = "[oversized stream token omitted]\n" + (value[boundary + 1:] if boundary >= 0 else "")
            suppressing[channel] = False
        pending[channel] += value
        # Retain an unfinished token so credentials split across arbitrary pipe
        # chunks are redacted as a whole. Bound a pathological unbroken token.
        boundary = len(pending[channel]) if final else pending[channel].rfind("\n") + 1
        if boundary:
            chunk, pending[channel] = pending[channel][:boundary], pending[channel][boundary:]
            if operation:
                from mana_agent.multi_agent.agents.tool_runner_agent import _public_value
                safe = str(_public_value(key_filters[channel].filter(chunk), operation.protected))
                for offset in range(0, len(safe), 3000):
                    operation.retain(channel, safe[offset:offset + 3000])
        if len(pending[channel]) > 65536:
            # Unknown unbounded tokens cannot be safely released a prefix at a time.
            if operation:
                operation.omitted_bytes += len(pending[channel].encode())
            pending[channel] = ""
            suppressing[channel] = True
            if operation:
                operation.trace_complete = False
    try:
        while len(ended) < 2:
            if termination is None:
                try:
                    if operation:
                        operation.validate_live()
                    if timeout is not None and time.monotonic() - start >= timeout:
                        raise subprocess.TimeoutExpired(argv, timeout)
                except (InterruptedError, TimeoutError, subprocess.TimeoutExpired) as exc:
                    termination = exc
                    stop(force_group=True)
                except Exception as exc:
                    termination = exc
                    stop(force_group=True)
            try:
                channel, body = messages.get(timeout=0.05)
            except queue.Empty:
                continue
            if body is None:
                publish(channel, decoders[channel].decode(b"", final=True), final=True)
                ended.add(channel)
                continue
            remaining = max(0, limit - len(captured["stdout"]) - len(captured["stderr"]))
            captured[channel].extend(body[:remaining])
            publish(channel, decoders[channel].decode(body))
        process.wait()
    finally:
        stop()
        process.wait()
    stdout = bytes(captured["stdout"]).decode("utf-8", errors="replace")
    stderr = bytes(captured["stderr"]).decode("utf-8", errors="replace")
    if isinstance(termination, subprocess.TimeoutExpired):
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr)
    if termination:
        termination.stdout, termination.stderr = stdout, stderr
        raise termination
    result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result
