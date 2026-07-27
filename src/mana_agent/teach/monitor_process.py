"""Lifecycle for the persistent native desktop recording subprocess."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .models import AuditEntry, SessionState, TeachError, TeachSession
from .storage import LocalTeachStorage


class DesktopMonitorProcess:
    def __init__(self, storage: LocalTeachStorage):
        self.storage = storage

    def start(self, session: TeachSession) -> int:
        ready = self._signal_path(session.id, "ready")
        error = self._signal_path(session.id, "error")
        stop = self._signal_path(session.id, "stop")
        ready.unlink(missing_ok=True)
        error.unlink(missing_ok=True)
        stop.unlink(missing_ok=True)
        process = subprocess.Popen(
            [sys.executable, "-m", "mana_agent.teach.monitor_worker", session.id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        session.monitor_pid = process.pid
        session.audit_trail.append(AuditEntry(action="monitor.spawned", detail=f"pid={process.pid}"))
        self.storage.save_session(session)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if ready.exists():
                ready.unlink(missing_ok=True)
                return process.pid
            if error.exists() or process.poll() is not None:
                detail = error.read_text(encoding="utf-8").strip() if error.exists() else "recorder process exited"
                error.unlink(missing_ok=True)
                recovered = self.storage.load_session(session.id)
                if recovered.state == SessionState.RECORDING:
                    recovered.transition(SessionState.FAILED, detail)
                    self.storage.save_session(recovered)
                raise TeachError(f"Desktop recorder failed to attach: {detail}")
            time.sleep(0.05)
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        recovered = self.storage.load_session(session.id)
        if recovered.state == SessionState.RECORDING:
            recovered.transition(SessionState.FAILED, "Desktop recorder readiness timeout.")
        recovered.monitor_pid = None
        self.storage.save_session(recovered)
        raise TeachError("Desktop recorder did not become ready within three seconds.")

    def stop(self, session: TeachSession) -> TeachSession:
        pid = session.monitor_pid
        if not pid:
            return session
        self._request_stop(pid, session.id)
        recovered = self.storage.load_session(session.id)
        recovered.monitor_pid = None
        recovered.audit_trail.append(AuditEntry(action="monitor.stopped", detail=f"pid={pid}"))
        self.storage.save_session(recovered)
        return recovered

    @staticmethod
    def _terminate(pid: int, session_id: str) -> None:
        if not DesktopMonitorProcess._is_expected_process(pid, session_id):
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)

    def _request_stop(self, pid: int, session_id: str) -> None:
        stop = self._signal_path(session_id, "stop")
        stop.write_text("stop\n", encoding="utf-8")
        stop.chmod(0o600)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                stop.unlink(missing_ok=True)
                return
            time.sleep(0.05)
        self._terminate(pid, session_id)
        stop.unlink(missing_ok=True)

    @staticmethod
    def _is_expected_process(pid: int, session_id: str) -> bool:
        if os.name == "nt":
            # The Windows implementation cannot safely inspect another
            # process's argv without optional APIs, so stale PIDs are never
            # signalled there.
            return False
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
        command = result.stdout.strip()
        return (
            result.returncode == 0
            and "mana_agent.teach.monitor_worker" in command
            and session_id in command
        )

    def _signal_path(self, session_id: str, kind: str) -> Path:
        return self.storage.root / "sessions" / f".{session_id}.monitor.{kind}"
