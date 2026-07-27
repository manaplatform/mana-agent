"""Private long-running worker for a desktop Teach Mode session."""

from __future__ import annotations

import signal
import sys
import time

from .config import TeachSettings
from .desktop_recorder import NativeDesktopRecorder
from .models import AuditEntry, SessionState
from .service import TeachService
from .storage import LocalTeachStorage


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    session_id = sys.argv[1]
    settings = TeachSettings.load()
    storage = LocalTeachStorage(settings.storage_path)
    service = TeachService(settings=settings, storage=storage)
    recorder = NativeDesktopRecorder()
    ready = storage.root / "sessions" / f".{session_id}.monitor.ready"
    error = storage.root / "sessions" / f".{session_id}.monitor.error"
    stop = storage.root / "sessions" / f".{session_id}.monitor.stop"
    stopping = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        session = storage.load_session(session_id)
        recorder.start(session, service.record_event)
        ready.write_text("ready\n", encoding="utf-8")
        ready.chmod(0o600)
        last_state = session.state
        while not stopping and not stop.exists():
            recorder.poll_desktop_context()
            current = storage.load_session(session_id)
            if current.state not in {SessionState.RECORDING, SessionState.PAUSED}:
                break
            if current.state != last_state:
                recorder.pause() if current.state == SessionState.PAUSED else recorder.resume()
                last_state = current.state
            time.sleep(0.2)
        recorder.stop()
        stop.unlink(missing_ok=True)
        current = storage.load_session(session_id)
        current.monitor_pid = None
        current.audit_trail.append(AuditEntry(action="monitor.exited", detail="Desktop recorder stopped."))
        storage.save_session(current)
        return 0
    except Exception as exc:
        error.write_text(str(exc)[:1000] + "\n", encoding="utf-8")
        error.chmod(0o600)
        try:
            session = storage.load_session(session_id)
            session.monitor_pid = None
            session.audit_trail.append(AuditEntry(action="monitor.failed", detail=str(exc)[:500]))
            if session.state == SessionState.RECORDING:
                session.transition(SessionState.FAILED, str(exc)[:500])
            storage.save_session(session)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
