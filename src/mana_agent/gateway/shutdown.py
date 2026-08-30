"""Authoritative shutdown coordinator and signal-aware task cancellation lifecycle."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

ShutdownSource = Literal[
    "ctrl_c",
    "exit",
    "quit",
    "sigterm",
    "sigint",
    "tui_quit",
    "api_shutdown",
    "runtime_close",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class CancellationMetadata:
    cancellation_reason: str = "user_interrupt"
    cancellation_source: str = "ctrl_c"
    cancelled_at: str = field(default_factory=_utc_now_iso)
    execution_id: str = ""
    task_id: str = ""
    session_id: str = ""
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    cleanup_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cancellation_reason": self.cancellation_reason,
            "cancellation_source": self.cancellation_source,
            "cancelled_at": self.cancelled_at,
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "provider_metadata": dict(self.provider_metadata),
            "cleanup_metadata": dict(self.cleanup_metadata),
        }


class ShutdownCoordinator:
    """Authoritative coordinator for graceful, signal-safe shutdown and task cancellation.

    Sequence:
    1. shutdown requested
    2. mark runtime shutting_down
    3. stop accepting new tasks
    4. identify active task/execution
    5. request cancellation
    6. propagate cancellation to provider / worker / subagents / tools
    7. persist terminal task state
    8. cancel pending retries / timers
    9. shutdown executors / workers
    10. close runtime resources
    11. exit process
    """

    def __init__(
        self,
        gateway: Any | None = None,
        *,
        event_sink: Callable[[str, str, dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.gateway = gateway
        self.event_sink = event_sink
        self._lock = threading.RLock()
        self._is_shutting_down = False
        self._shutdown_requested_count = 0
        self._active_sessions: set[str] = set()
        self._installed_signals: dict[int, Any] = {}
        self._cancellation_records: dict[str, CancellationMetadata] = {}

    @property
    def is_shutting_down(self) -> bool:
        with self._lock:
            return self._is_shutting_down

    def register_session(self, session_id: str) -> None:
        with self._lock:
            if session_id:
                self._active_sessions.add(session_id)

    def unregister_session(self, session_id: str) -> None:
        with self._lock:
            self._active_sessions.discard(session_id)

    def emit(self, event_type: str, message: str = "", metadata: dict[str, Any] | None = None) -> None:
        sink = self.event_sink or getattr(self.gateway, "_event_sink", None)
        if callable(sink):
            try:
                sink(event_type, message or event_type, metadata or {})
            except Exception:
                logger.debug("Failed to emit shutdown event %s", event_type, exc_info=True)

    def request_shutdown(
        self,
        *,
        source: ShutdownSource = "ctrl_c",
        session_id: str = "",
        reason: str = "user_interrupt",
    ) -> bool:
        """Trigger authoritative shutdown and cancel active tasks before runtime teardown."""
        with self._lock:
            self._shutdown_requested_count += 1
            is_first = not self._is_shutting_down
            self._is_shutting_down = True

        if is_first:
            self.emit(
                "shutdown.requested",
                f"Shutdown requested via {source}",
                {"source": source, "reason": reason, "session_id": session_id},
            )

        if self.gateway is not None and hasattr(self.gateway, "request_shutdown"):
            try:
                return self.gateway.request_shutdown(source=source, session_id=session_id, reason=reason)
            except Exception:
                logger.warning("Error during gateway request_shutdown", exc_info=True)
                return False

        # If gateway is not present or does not have request_shutdown:
        target_sessions = [session_id] if session_id else list(self._active_sessions)
        if not target_sessions and self.gateway is not None:
            active_sid = getattr(self.gateway, "_chat_session_id", None)
            if active_sid:
                target_sessions = [active_sid]

        self.emit("runtime.shutdown.started", "Runtime shutdown started", {"source": source})

        if self.gateway is not None:
            setattr(self.gateway, "_shutting_down", True)
            if hasattr(self.gateway, "cancel_active"):
                for sid in target_sessions:
                    try:
                        self.gateway.cancel_active(
                            session_id=sid,
                            reason=reason,
                            source=source,
                        )
                    except Exception:
                        logger.warning("Error cancelling active session %s on shutdown", sid, exc_info=True)
            elif hasattr(self.gateway, "cancel"):
                for sid in target_sessions:
                    try:
                        self.gateway.cancel(sid)
                    except Exception:
                        logger.warning("Error cancelling session %s on shutdown", sid, exc_info=True)

        self.emit("runtime.shutdown.completed", "Runtime shutdown completed", {"source": source})
        return True


    def install_signal_handlers(self) -> None:
        """Install SIGINT and SIGTERM handlers for process-wide signal safety."""
        with self._lock:
            if self._installed_signals:
                return

            def _handle_signal(signum: int, frame: Any) -> None:
                sig_name = "ctrl_c" if signum == signal.SIGINT else "sigterm"
                with self._lock:
                    count = self._shutdown_requested_count + 1

                if count > 1:
                    # Repeated Ctrl+C: force exit immediately
                    logger.warning("Forced exit on repeated signal %s", signum)
                    sys.stderr.write("\nForced exit requested.\n")
                    sys.stderr.flush()
                    os._exit(130 if signum == signal.SIGINT else 143)

                logger.info("Signal %s received, requesting graceful cancellation", signum)
                self.request_shutdown(source=sig_name, reason=f"received signal {signum}")

                # If this was SIGINT on main thread in interactive mode, raise KeyboardInterrupt
                # so synchronous input() / loops unblock
                if signum == signal.SIGINT and threading.current_thread() is threading.main_thread():
                    raise KeyboardInterrupt()

            try:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    prev = signal.signal(sig, _handle_signal)
                    self._installed_signals[sig] = prev
            except (ValueError, AttributeError) as exc:
                logger.debug("Could not install signal handlers (not main thread or OS unsupported): %s", exc)

    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        with self._lock:
            for sig, prev in self._installed_signals.items():
                try:
                    signal.signal(sig, prev)
                except Exception:
                    pass
            self._installed_signals.clear()


__all__ = [
    "CancellationMetadata",
    "ShutdownCoordinator",
    "ShutdownSource",
]
