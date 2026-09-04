from __future__ import annotations

import builtins
import gc
import io
import logging
import os
import shutil
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Ensure test imports resolve to local source tree, not an installed wheel/editable.
# Insert src first so `mana_agent` resolves under src/, then the repo root so
# tests can import non-packaged helpers such as `scripts.swe_bench.runner`.
# Relying only on "" / cwd fails when pytest is invoked with a different
# working directory or without the root on sys.path.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1 if sys.path and sys.path[0] == str(SRC_DIR) else 0, str(PROJECT_ROOT))


# Capture this before pytest changes HOME.  The value is deliberately never
# removed or otherwise modified by this module.
REAL_MANA_HOME = (Path.home() / ".mana").resolve()
_TEST_MANA_HOME: Path | None = None


def _resolved_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _targets_real_mana_home(path: str | os.PathLike[str]) -> bool:
    if isinstance(path, int):
        return False
    try:
        _resolved_path(path).relative_to(REAL_MANA_HOME)
    except ValueError:
        return False
    return True


def _reject_real_mana_write(path: str | os.PathLike[str]) -> None:
    if _targets_real_mana_home(path):
        raise PermissionError(
            f"pytest isolation rejected a write to the real Mana home: {path}. "
            "Use the MANA_HOME supplied by the test session instead."
        )


def _install_real_mana_write_guard() -> None:
    """Block common filesystem write APIs from touching the user's Mana home."""

    original_open = builtins.open
    original_io_open = io.open
    original_path_open = Path.open
    original_os_open = os.open
    original_mkdir = os.mkdir
    original_makedirs = os.makedirs
    original_unlink = os.unlink
    original_remove = os.remove
    original_rename = os.rename
    original_replace = os.replace
    original_chmod = os.chmod
    original_symlink = os.symlink
    original_link = os.link
    original_rmtree = shutil.rmtree

    def guarded_open(file, mode="r", *args, **kwargs):  # noqa: ANN001
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            _reject_real_mana_write(file)
        return original_open(file, mode, *args, **kwargs)

    def guarded_io_open(file, mode="r", *args, **kwargs):  # noqa: ANN001
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            _reject_real_mana_write(file)
        return original_io_open(file, mode, *args, **kwargs)

    def guarded_path_open(path, mode="r", *args, **kwargs):  # noqa: ANN001
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            _reject_real_mana_write(path)
        return original_path_open(path, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):  # noqa: ANN001
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
        if flags & write_flags:
            _reject_real_mana_write(path)
        return original_os_open(path, flags, *args, **kwargs)

    def guarded_one_path(operation, path, *args, **kwargs):  # noqa: ANN001
        _reject_real_mana_write(path)
        return operation(path, *args, **kwargs)

    def guarded_two_paths(operation, source, destination, *args, **kwargs):  # noqa: ANN001
        _reject_real_mana_write(source)
        _reject_real_mana_write(destination)
        return operation(source, destination, *args, **kwargs)

    def guarded_mkdir(path, *args, **kwargs):  # noqa: ANN001
        return guarded_one_path(original_mkdir, path, *args, **kwargs)

    def guarded_makedirs(path, *args, **kwargs):  # noqa: ANN001
        return guarded_one_path(original_makedirs, path, *args, **kwargs)

    def guarded_unlink(path, *args, **kwargs):  # noqa: ANN001
        return guarded_one_path(original_unlink, path, *args, **kwargs)

    def guarded_remove(path, *args, **kwargs):  # noqa: ANN001
        return guarded_one_path(original_remove, path, *args, **kwargs)

    def guarded_rename(source, destination, *args, **kwargs):  # noqa: ANN001
        return guarded_two_paths(original_rename, source, destination, *args, **kwargs)

    def guarded_replace(source, destination, *args, **kwargs):  # noqa: ANN001
        return guarded_two_paths(original_replace, source, destination, *args, **kwargs)

    def guarded_chmod(path, *args, **kwargs):  # noqa: ANN001
        return guarded_one_path(original_chmod, path, *args, **kwargs)

    def guarded_symlink(source, destination, *args, **kwargs):  # noqa: ANN001
        return guarded_two_paths(original_symlink, source, destination, *args, **kwargs)

    def guarded_link(source, destination, *args, **kwargs):  # noqa: ANN001
        return guarded_two_paths(original_link, source, destination, *args, **kwargs)

    def guarded_rmtree(path, *args, **kwargs):  # noqa: ANN001
        return guarded_one_path(original_rmtree, path, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    Path.open = guarded_path_open
    os.open = guarded_os_open
    os.mkdir = guarded_mkdir
    os.makedirs = guarded_makedirs
    os.unlink = guarded_unlink
    os.remove = guarded_remove
    os.rename = guarded_rename
    os.replace = guarded_replace
    os.chmod = guarded_chmod
    os.symlink = guarded_symlink
    os.link = guarded_link
    shutil.rmtree = guarded_rmtree


def pytest_sessionstart(session: pytest.Session) -> None:
    """Isolate all process and subprocess Mana state before test collection."""

    global _TEST_MANA_HOME
    factory = session.config._tmp_path_factory  # pytest creates this before collection.
    base = factory.mktemp("mana-agent-home", numbered=True)
    _TEST_MANA_HOME = base / ".mana"
    _TEST_MANA_HOME.mkdir(mode=0o700)

    # MANA_HOME covers Mana's path resolver. HOME/USERPROFILE also protect
    # legacy tests and third-party code that construct Path.home() / ".mana".
    os.environ["MANA_HOME"] = str(_TEST_MANA_HOME)
    os.environ["HOME"] = str(base)
    os.environ["USERPROFILE"] = str(base)
    _install_real_mana_write_guard()


@pytest.fixture(autouse=True)
def _cleanup_workspace_dbs_per_test() -> Iterator[None]:
    yield
    try:
        from mana_agent.persistence.workspace_db import close_all_workspace_dbs

        close_all_workspace_dbs()
    except Exception:
        pass


def _remove_test_home(path: Path) -> None:
    def retry_onerror(function, failed_path, exc_info):  # noqa: ANN001
        try:
            os.chmod(failed_path, stat.S_IWRITE | stat.S_IREAD)
            function(failed_path)
        except OSError:
            pass

    for attempt in range(5):
        try:
            shutil.rmtree(path, onerror=retry_onerror)
            if not path.exists():
                return
        except OSError:
            pass
        gc.collect()
        time.sleep(0.1 * (attempt + 1))

    def strict_onerror(function, failed_path, exc_info):  # noqa: ANN001
        try:
            os.chmod(failed_path, stat.S_IWRITE | stat.S_IREAD)
            function(failed_path)
        except OSError as cleanup_error:
            raise RuntimeError(f"Could not remove isolated Mana test home {path}: {failed_path}") from cleanup_error

    if path.exists():
        try:
            shutil.rmtree(path, onerror=strict_onerror)
        except OSError as exc:
            raise RuntimeError(f"Could not remove isolated Mana test home {path}") from exc
    if path.exists():
        raise RuntimeError(f"Could not remove isolated Mana test home {path}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Release resources and remove the complete isolated Mana test home."""

    if _TEST_MANA_HOME is None:
        return
    try:
        from mana_agent.persistence.workspace_db import close_all_workspace_dbs

        close_all_workspace_dbs()
    except Exception:
        pass
    gc.collect()
    logging.shutdown()
    _remove_test_home(_TEST_MANA_HOME)


@pytest.fixture(scope="session")
def mana_test_home() -> Path:
    assert _TEST_MANA_HOME is not None
    return _TEST_MANA_HOME


@pytest.fixture(scope="session")
def real_mana_home() -> Path:
    """The protected user-level path, exposed only for guard regression tests."""

    return REAL_MANA_HOME
