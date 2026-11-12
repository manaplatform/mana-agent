from __future__ import annotations

from threading import Lock
from contextlib import contextmanager
import errno
import hashlib
import os
import time


class LockTable:
    def __init__(self) -> None:
        self._guard = Lock()
        self._locks: dict[str, Lock] = {}

    def lock_for(self, key: str) -> Lock:
        with self._guard:
            if key not in self._locks:
                self._locks[key] = Lock()
            return self._locks[key]

    @contextmanager
    def workspace_lock(self, root: str, *, validate):
        """Extend the existing lock table across IPC/application processes."""
        from mana_agent.workspaces.paths import mana_home
        local = self.lock_for(f"workspace:{root}")
        while not local.acquire(timeout=0.05):
            validate()
        handle = None
        acquired = False
        try:
            validate()
            directory = mana_home() / "locks" / "tool-workspaces"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{hashlib.sha256(root.encode()).hexdigest()}.lock"
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
            handle = os.fdopen(descriptor, "r+b")
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            while not acquired:
                validate()
                try:
                    if os.name == "nt":
                        import msvcrt
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(0.05)
            yield
        finally:
            try:
                if handle is not None:
                    try:
                        if acquired:
                            from mana_agent.execution_supervisor.store import _release_file_lock
                            _release_file_lock(handle)
                    finally:
                        handle.close()
            finally:
                local.release()
