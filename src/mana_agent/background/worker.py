from __future__ import annotations

import sys
import os
import threading

from mana_agent.background.commands import get_registered_command, idle_retirement
from mana_agent.background.models import utc_iso
from mana_agent.background.store import ProcessStore


def run(process_id: str) -> int:
    store = ProcessStore()
    record = store.get(process_id)
    command = get_registered_command(record.command_identifier)
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(5):
            try:
                current = store.get(process_id)
                current.heartbeat_at = utc_iso()
                store.save(current)
                log_path = store.directory(process_id) / "process.log"
                limit = max(4096, int(os.getenv("MANA_PROCESS_MAX_LOG_BYTES", "1000000")))
                if log_path.exists() and log_path.stat().st_size > limit:
                    with log_path.open("rb") as source:
                        source.seek(-limit, 2)
                        tail = source.read()
                    with log_path.open("wb") as target:
                        target.write(tail)
            except (OSError, ValueError):
                return

    thread = threading.Thread(target=heartbeat, name="mana-background-heartbeat", daemon=True)
    thread.start()
    retirement_token = None
    if record.command_identifier == "batch.watch":
        from pathlib import Path
        from mana_agent.batch.watcher import watcher_lifecycle_lock
        from mana_agent.background.manager import BackgroundProcessManager

        def retire_if_idle(predicate):
            # Submitters hold this same lock while finding/starting a watcher.
            # Recheck durable demand before relinquishing singleton ownership.
            with watcher_lifecycle_lock(Path(record.sanitized_arguments["root"]), store.root):
                if not predicate():
                    return False
                stopped.set()
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError("Background heartbeat did not stop at the idle retirement boundary")
                current = store.get(process_id)
                current.state = "stopping"
                store.save(current)
                BackgroundProcessManager(store)._release_singleton(current)
                return True

        retirement_token = idle_retirement.set(retire_if_idle)
    try:
        command.runner(record.sanitized_arguments)
    except BaseException as exc:
        from mana_agent.utils.redaction import redact_secrets
        current = store.get(process_id)
        current.state = "failed"
        current.health = "unhealthy"
        current.last_error_summary = f"{type(exc).__name__}: {redact_secrets(str(exc))}"[:240]
        current.stopped_at = utc_iso()
        store.save(current)
        return 1
    finally:
        stopped.set()
        if retirement_token is not None:
            idle_retirement.reset(retirement_token)
    current = store.get(process_id)
    current.state = "stopped"
    current.stopped_at = utc_iso()
    current.health = "unknown"
    store.save(current)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        raise SystemExit("Usage: python -m mana_agent.background.worker run <process-id>")
    raise SystemExit(run(sys.argv[2]))
