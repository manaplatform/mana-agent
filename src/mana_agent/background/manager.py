from __future__ import annotations

import os
import logging
import signal
import subprocess
import sys
import time
import hashlib
from pathlib import Path
from typing import Any

from mana_agent.background.commands import get_registered_command
from mana_agent.background.models import ProcessRecord, utc_iso
from mana_agent.background.store import ProcessStore
from mana_agent.workspaces.paths import mana_home

logger = logging.getLogger(__name__)


def _identity(pid: int) -> str:
    if pid <= 0:
        return ""
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", f"(Get-Process -Id {pid}).StartTime.ToUniversalTime().Ticks"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


class BackgroundProcessManager:
    def __init__(self, store: ProcessStore | None = None, *, event_sink: Any | None = None, max_log_bytes: int = 1_000_000) -> None:
        self.store = store or ProcessStore()
        self.event_sink = event_sink
        self.max_log_bytes = max(4096, int(max_log_bytes))

    def start(
        self,
        command_identifier: str,
        *,
        process_type: str,
        arguments: dict[str, str] | None = None,
        singleton_key: str = "",
        repository_id: str = "",
        workspace_id: str = "",
        session_id: str = "",
        ownership: str = "global",
        transient: bool = False,
        restart_policy: str = "never",
    ) -> ProcessRecord:
        spec = get_registered_command(command_identifier)
        clean_args = {str(k): str(v) for k, v in (arguments or {}).items() if "token" not in str(k).lower() and "secret" not in str(k).lower()}
        self.recover_stale()
        singleton_lock: Path | None = None
        if singleton_key:
            singleton_lock = self._claim_singleton(singleton_key)
        record = ProcessRecord(
            process_type=process_type, command_identifier=command_identifier,
            sanitized_arguments=clean_args, singleton_key=singleton_key,
            repository_id=repository_id, workspace_id=workspace_id, session_id=session_id,
            ownership=ownership, transient=transient, restart_policy=restart_policy,
            state="starting",
        )
        directory = self.store.directory(record.process_id)
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "process.log"
        record.stdout_log = str(log_path)
        self.store.save(record)
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {
                "PATH", "PYTHONPATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
                "LANG", "LC_ALL", "HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR",
                "DBUS_SESSION_BUS_ADDRESS",
            }
        }
        env["MANA_HOME"] = str(mana_home())
        env["MANA_PROCESS_MAX_LOG_BYTES"] = str(self.max_log_bytes)
        for name in spec.secret_environment_names(clean_args):
            if name in os.environ:
                env[name] = os.environ[name]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            with log_path.open("ab") as output:
                process = subprocess.Popen(
                    [sys.executable, "-m", "mana_agent.background.worker", "run", record.process_id],
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    env=env, close_fds=True, start_new_session=os.name != "nt", creationflags=flags,
                )
        except OSError as exc:
            record.state = "failed"
            record.last_error_summary = str(exc)[:240]
            record.stopped_at = utc_iso()
            self.store.save(record)
            if singleton_lock is not None:
                singleton_lock.unlink(missing_ok=True)
            raise RuntimeError(f"background process failed to start: {exc}") from exc
        record.os_pid = process.pid
        record.process_identity = _identity(process.pid)
        record.started_at = utc_iso()
        record.heartbeat_at = record.started_at
        record.state = "running"
        record.health = "healthy"
        self.store.save(record)
        self._emit("background.started", record)
        return record

    def inspect(self, process_id: str) -> ProcessRecord:
        self.recover_stale(process_id)
        return self.store.get(process_id)

    def list(self, *, session_id: str = "") -> list[ProcessRecord]:
        rows = self.store.list()
        return [row for row in rows if not session_id or row.session_id == session_id]

    def stop(self, process_id: str, *, timeout: float = 5.0) -> ProcessRecord:
        row = self.store.get(process_id)
        if row.state not in {"starting", "running", "stopping"} or not row.os_pid:
            return row
        if not self._matches(row):
            row.state = "stale"
            row.health = "unhealthy"
            row.last_error_summary = "PID is absent or belongs to another process; no signal was sent."
            saved = self.store.save(row)
            self._release_singleton(saved)
            return saved
        row.state = "stopping"
        self.store.save(row)
        try:
            if os.name == "nt":
                os.kill(row.os_pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(row.os_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline and self._matches(row):
            time.sleep(0.05)
        if self._matches(row):
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(row.os_pid), "/T", "/F"], capture_output=True, check=False)
            else:
                os.killpg(row.os_pid, signal.SIGKILL)
        row.state = "stopped"
        row.health = "unknown"
        row.stopped_at = utc_iso()
        row.heartbeat_at = row.stopped_at
        self.store.save(row)
        self._emit("background.stopped", row)
        self._release_singleton(row)
        return row

    def restart(self, process_id: str) -> ProcessRecord:
        old = self.stop(process_id)
        replacement = self.start(
            old.command_identifier, process_type=old.process_type, arguments=old.sanitized_arguments,
            singleton_key=old.singleton_key, repository_id=old.repository_id,
            workspace_id=old.workspace_id, session_id=old.session_id,
            ownership=old.ownership, transient=old.transient, restart_policy=old.restart_policy,
        )
        replacement.restart_count = old.restart_count + 1
        return self.store.save(replacement)

    def logs(self, process_id: str, *, max_bytes: int = 64_000) -> str:
        path = Path(self.store.get(process_id).stdout_log)
        if not path.exists():
            return ""
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - min(max_bytes, self.max_log_bytes)))
            return handle.read().decode("utf-8", errors="replace")

    def recover_stale(self, process_id: str = "") -> list[ProcessRecord]:
        changed: list[ProcessRecord] = []
        rows = [self.store.get(process_id)] if process_id else self.store.list()
        for row in rows:
            if row.state in {"starting", "running", "stopping"} and not self._matches(row):
                row.state = "stale"
                row.health = "unhealthy"
                row.stopped_at = utc_iso()
                row.last_error_summary = "Process disappeared or PID identity changed."
                self.store.save(row)
                self._release_singleton(row)
                changed.append(row)
        return changed

    def cleanup(self) -> int:
        self.recover_stale()
        removable = [row for row in self.store.list() if row.state in {"stopped", "stale", "failed"}]
        for row in removable:
            self.store.delete(row.process_id)
        return len(removable)

    def stop_session_processes(self, session_id: str, *, transient_only: bool = True) -> None:
        for row in self.list(session_id=session_id):
            if row.state in {"starting", "running"} and (not transient_only or row.transient):
                self.stop(row.process_id)

    def _matches(self, row: ProcessRecord) -> bool:
        if not row.os_pid:
            return False
        current = _identity(row.os_pid)
        return bool(current and row.process_identity and current == row.process_identity)

    def _claim_singleton(self, singleton_key: str) -> Path:
        root = self.store.root / "singletons"
        root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(singleton_key.encode("utf-8")).hexdigest()
        path = root / f"{digest}.lock"
        for _attempt in range(2):
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                duplicate = next(
                    (
                        row for row in self.list()
                        if row.singleton_key == singleton_key and row.state in {"starting", "running"}
                    ),
                    None,
                )
                if duplicate is not None:
                    raise RuntimeError(
                        f"background process singleton is already running: {duplicate.process_id}"
                    )
                try:
                    age = time.time() - path.stat().st_mtime
                except OSError:
                    age = 0
                if age < 30:
                    raise RuntimeError("background process singleton startup is already in progress")
                path.unlink(missing_ok=True)
                continue
            else:
                os.close(descriptor)
                return path
        raise RuntimeError("background process singleton lock could not be acquired")

    def _release_singleton(self, row: ProcessRecord) -> None:
        if not row.singleton_key:
            return
        digest = hashlib.sha256(row.singleton_key.encode("utf-8")).hexdigest()
        (self.store.root / "singletons" / f"{digest}.lock").unlink(missing_ok=True)

    def _emit(self, event: str, row: ProcessRecord) -> None:
        if self.event_sink is not None:
            try:
                self.event_sink(event, row.model_dump(mode="json"))
            except Exception:
                logger.debug("background event sink failed", exc_info=True)
        from mana_agent.services.execution_event_hub import get_execution_event_hub

        get_execution_event_hub().publish(
            {
                "type": event,
                "title": f"Background process {row.state}",
                "status": "failed" if row.state == "failed" else "success",
                "message": f"{row.process_type} {row.process_id}",
                "metadata": {"process": row.model_dump(mode="json")},
            },
            conversation_id=row.session_id,
            repository_id=row.repository_id,
            persist=bool(row.session_id and row.repository_id),
        )
