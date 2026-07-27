"""Async JSON-RPC client for the official ``codex app-server`` protocol."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from mana_agent._version import get_version
from mana_agent.integrations.codex.exceptions import CodexProtocolError, CodexUnavailableError
from mana_agent.utils.redaction import redact_json_line


class AsyncCodexAppServer:
    """Own one Codex app-server subprocess and its request/notification stream."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        provider_name: str = "",
        model: str = "",
        request_timeout_seconds: int = 30,
    ) -> None:
        self.command = tuple(str(part) for part in command)
        self._environment = dict(environment) if environment is not None else None
        self._secrets = tuple(
            value
            for key, value in (self._environment or {}).items()
            if key == "MANA_CODEX_API_KEY" and value
        )
        self._provider_name = str(provider_name or "selected provider")
        self._model = str(model or "selected model")
        self.request_timeout_seconds = max(1, int(request_timeout_seconds))
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._notifications: defaultdict[str, asyncio.Queue[dict[str, Any]]] = defaultdict(asyncio.Queue)
        self._stderr: list[str] = []
        self._write_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        if self.running:
            return
        if not self.command:
            raise CodexUnavailableError("Codex app-server command is empty")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._environment,
            )
        except (FileNotFoundError, OSError) as exc:
            raise CodexUnavailableError(f"Unable to start Codex app-server: {exc}") from exc
        self._reader_task = asyncio.create_task(self._read_stdout(), name="codex-app-server-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="codex-app-server-stderr")
        await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "mana-agent",
                    "title": "Mana-Agent",
                    "version": get_version(),
                },
                "capabilities": {},
            },
        )
        await self.notify("initialized", {})

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.running or self._process is None:
            raise CodexUnavailableError("Codex app-server is not running")
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout_seconds)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise CodexProtocolError(f"Codex request timed out: {method}") from exc

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def notifications(self, thread_id: str) -> AsyncIterator[dict[str, Any]]:
        queue = self._notifications[str(thread_id)]
        while True:
            notification = await queue.get()
            yield notification
            method = str(notification.get("method") or "")
            if method in {"turn/completed", "turn/failed", "turn/cancelled"}:
                return

    async def interrupt(self, *, thread_id: str, turn_id: str) -> None:
        await self.request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id})

    async def deny_server_request(self, request: dict[str, Any]) -> None:
        """Reject an app-server approval request without granting permissions."""

        request_id = request.get("id")
        if request_id is None:
            return
        method = str(request.get("method") or "")
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "execCommandApproval",
            "applyPatchApproval",
        }:
            await self._write({"jsonrpc": "2.0", "id": request_id, "result": {"decision": "cancel"}})
            return
        await self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32001, "message": "Mana-Agent denied the permission request"},
            }
        )

    async def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
        self._reader_task = None
        self._stderr_task = None
        error = CodexUnavailableError("Codex app-server closed before responding")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def _write(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise CodexUnavailableError("Codex app-server stdin is unavailable")
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            process.stdin.write(encoded)
            await process.stdin.drain()

    async def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        while line := await process.stdout.readline():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            payload = self._sanitize_payload(payload)
            if "id" in payload and ("result" in payload or "error" in payload):
                self._resolve_response(payload)
                continue
            method = str(payload.get("method") or "")
            if method:
                thread_id = self._notification_thread_id(payload)
                await self._notifications[thread_id].put(payload)
        if process.returncode is None:
            await process.wait()
        detail = self._stderr[-1] if self._stderr else "process exited"
        error = CodexUnavailableError(f"Codex app-server stopped: {detail}")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)

    async def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while line := await process.stderr.readline():
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                self._stderr.append(self._redact(text)[:1000])
                del self._stderr[:-20]

    def _resolve_response(self, payload: dict[str, Any]) -> None:
        try:
            request_id = int(payload["id"])
        except (KeyError, TypeError, ValueError):
            return
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return
        error = payload.get("error")
        if error:
            future.set_exception(CodexProtocolError(self._format_provider_error(str(error))))
            return
        result = payload.get("result")
        future.set_result(dict(result) if isinstance(result, dict) else {"value": result})

    @staticmethod
    def _notification_thread_id(payload: dict[str, Any]) -> str:
        params = payload.get("params")
        if not isinstance(params, dict):
            return ""
        direct = params.get("threadId") or params.get("thread_id")
        if direct:
            return str(direct)
        thread = params.get("thread")
        if isinstance(thread, dict) and thread.get("id"):
            return str(thread["id"])
        return ""

    def _redact(self, value: str) -> str:
        redacted = redact_json_line(value)
        for secret in self._secrets:
            redacted = redacted.replace(secret, "***REDACTED***")
        return redacted

    def _format_provider_error(self, value: str) -> str:
        detail = self._redact(value)
        lowered = detail.lower()
        if "401" in lowered or "unauthorized" in lowered or "authentication" in lowered:
            return f"Provider authentication failed for {self._provider_name}. Codex detail: {detail}"
        if "model_not_found" in lowered or "model not found" in lowered or "does not exist" in lowered:
            return (
                f"Provider {self._provider_name} could not find configured model {self._model}. "
                f"Codex detail: {detail}"
            )
        return f"Codex JSON-RPC error: {detail}"

    def _sanitize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = self._redact(json.dumps(payload, ensure_ascii=False))
        try:
            sanitized = json.loads(encoded)
        except json.JSONDecodeError:
            return {}
        if not isinstance(sanitized, dict):
            return {}
        if str(sanitized.get("method") or "") == "turn/failed":
            params = sanitized.get("params")
            if isinstance(params, dict):
                for key in ("message", "error"):
                    if isinstance(params.get(key), str):
                        params[key] = self._format_provider_error(params[key])
        return sanitized


__all__ = ["AsyncCodexAppServer"]
