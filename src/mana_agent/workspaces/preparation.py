"""Shared preparation boundary for coding workspaces and Git repositories."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from mana_agent.workspaces.models import RepositoryRecord, WorkspaceRecord
from mana_agent.workspaces.paths import mana_home

if TYPE_CHECKING:
    from mana_agent.workspaces.service import WorkspaceService

if os.name == "nt":  # pragma: no cover - exercised on Windows CI
    import msvcrt
else:  # pragma: no cover - platform selection is deterministic
    import fcntl

logger = logging.getLogger(__name__)
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class PreparedRepository:
    """Resolved coding paths and their persisted Mana identities."""

    requested_workspace_path: Path
    working_directory: Path
    repository_root: Path
    git_common_dir: Path
    repository_existed: bool
    initialized: bool
    reused_parent_repository: bool
    repository: RepositoryRecord
    workspace: WorkspaceRecord

    @property
    def repository_id(self) -> str:
        return self.repository.repository_id

    @property
    def workspace_id(self) -> str:
        return self.workspace.workspace_id


class RepositoryPreparationError(RuntimeError):
    """Base error with an explicit preparation phase and safe user message."""

    def __init__(self, path: Path, phase: str, detail: str) -> None:
        self.workspace_path = Path(path)
        self.phase = phase
        self.detail = _one_line(detail)
        super().__init__(
            f"Coding workspace preparation failed for '{self.workspace_path}' during "
            f"{self.phase}: {self.detail}. The coding agent was not started."
        )


class WorkspaceResolutionError(RepositoryPreparationError):
    pass


class GitUnavailableError(RepositoryPreparationError):
    pass


class GitInitializationError(RepositoryPreparationError):
    pass


class RepositoryValidationError(RepositoryPreparationError):
    pass


class RepositoryPersistenceError(RepositoryPreparationError):
    pass


def _one_line(value: str) -> str:
    return " ".join(str(value or "unknown error").strip().splitlines())[:1000]


def _run_git(
    git_executable: str,
    args: list[str],
    *,
    cwd: Path,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [git_executable, *args],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _git_error(result: subprocess.CompletedProcess[str]) -> str:
    return _one_line(result.stderr or result.stdout or f"Git exited with status {result.returncode}")


def _has_write_bit(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _thread_lock(key: str) -> threading.RLock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _preparation_lock(path: Path) -> Iterator[None]:
    identity = hashlib.sha256(os.path.normcase(str(path)).encode("utf-8")).hexdigest()[:24]
    lock_path = mana_home() / "locks" / f"repository-preparation-{identity}.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _thread_lock(identity), lock_path.open("a+b") as handle:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                handle.seek(0)
                if handle.read(1) == b"":
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover - platform selection is deterministic
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - platform selection is deterministic
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except RepositoryPreparationError:
        raise
    except OSError as exc:
        raise RepositoryPreparationError(path, "preparation lock acquisition", str(exc)) from exc


def _resolve_existing_repository(
    git_executable: str,
    working_directory: Path,
    *,
    requested_path: Path,
) -> tuple[Path | None, bool]:
    inside = _run_git(
        git_executable,
        ["rev-parse", "--is-inside-work-tree"],
        cwd=working_directory,
        timeout=10,
    )
    if inside.returncode == 0 and inside.stdout.strip().lower() == "true":
        root_result = _run_git(
            git_executable,
            ["rev-parse", "--show-toplevel"],
            cwd=working_directory,
            timeout=10,
        )
        if root_result.returncode != 0 or not root_result.stdout.strip():
            raise RepositoryValidationError(
                requested_path,
                "Git root resolution",
                _git_error(root_result),
            )
        root = Path(root_result.stdout.strip()).resolve()
        broad_roots = {Path(working_directory.anchor).resolve(), Path.home().resolve()}
        if root != working_directory and root in broad_roots:
            logger.warning(
                "Ignored unsafe broad ancestor Git repository workspace=%s ancestor=%s",
                working_directory,
                root,
            )
            return None, False
        return root, root != working_directory

    bare = _run_git(
        git_executable,
        ["rev-parse", "--is-bare-repository"],
        cwd=working_directory,
        timeout=10,
    )
    if bare.returncode == 0 and bare.stdout.strip().lower() == "true":
        raise RepositoryValidationError(
            requested_path,
            "repository validation",
            "a bare Git repository is not a writable coding workspace",
        )

    git_marker = working_directory / ".git"
    if git_marker.exists() or git_marker.is_symlink():
        raise RepositoryValidationError(
            requested_path,
            "repository validation",
            f"Git metadata is invalid or corrupt: {_git_error(inside)}",
        )
    return None, False


def _initialize_repository(git_executable: str, path: Path, requested_path: Path) -> None:
    preferred = _run_git(git_executable, ["init", "-b", "main", str(path)], cwd=path.parent)
    if preferred.returncode == 0:
        return
    diagnostic = (preferred.stderr or preferred.stdout).lower()
    unsupported = any(
        marker in diagnostic
        for marker in ("unknown option", "unknown switch", "unrecognized option", "usage: git init")
    )
    if not unsupported:
        raise GitInitializationError(requested_path, "Git initialization", _git_error(preferred))

    compatibility = _run_git(git_executable, ["init", str(path)], cwd=path.parent)
    if compatibility.returncode != 0:
        raise GitInitializationError(
            requested_path,
            "Git initialization compatibility fallback",
            _git_error(compatibility),
        )
    branch = _run_git(git_executable, ["-C", str(path), "branch", "-M", "main"], cwd=path.parent)
    if branch.returncode != 0:
        raise GitInitializationError(
            requested_path,
            "initial branch selection",
            _git_error(branch),
        )


def _resolve_git_common_dir(
    git_executable: str,
    working_directory: Path,
    requested_path: Path,
) -> Path:
    result = _run_git(
        git_executable,
        ["rev-parse", "--git-common-dir"],
        cwd=working_directory,
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RepositoryValidationError(
            requested_path,
            "Git common directory resolution",
            _git_error(result),
        )
    value = Path(result.stdout.strip())
    return (value if value.is_absolute() else working_directory / value).resolve()


def _workspace_allows_path(workspace: WorkspaceRecord, path: Path, repository_root: Path) -> bool:
    boundaries = [
        Path(value).expanduser().resolve()
        for value in [*workspace.allowed_roots, *workspace.discovery.roots]
        if str(value).strip()
    ]
    if not boundaries:
        return True
    return any(
        candidate == boundary or boundary in candidate.parents
        for candidate in (path, repository_root)
        for boundary in boundaries
    )


def _persist_prepared_repository(
    service: "WorkspaceService",
    repository_root: Path,
    working_directory: Path,
    *,
    expected_workspace_id: str | None,
    requested_path: Path,
) -> tuple[RepositoryRecord, WorkspaceRecord]:
    try:
        repository = service.register_repository(repository_root, refresh=True)
        if expected_workspace_id:
            workspace = service.store.get_workspace(expected_workspace_id)
            if not _workspace_allows_path(workspace, working_directory, repository_root):
                raise WorkspaceResolutionError(
                    requested_path,
                    "workspace boundary validation",
                    f"path is outside persisted workspace {expected_workspace_id}",
                )
            if repository.repository_id not in workspace.repository_ids:
                workspace.repository_ids.append(repository.repository_id)
            workspace.primary_repository_id = workspace.primary_repository_id or repository.repository_id
            workspace.updated_at = repository.updated_at
            service.store.save_workspace(workspace)
        else:
            workspace = service.workspace_for_repository(repository.repository_id)
        return repository, workspace
    except RepositoryPreparationError:
        raise
    except Exception as exc:
        raise RepositoryPersistenceError(
            requested_path,
            "repository persistence",
            str(exc),
        ) from exc


def prepare_repository(
    service: "WorkspaceService",
    workspace_path: str | Path,
    *,
    allow_create: bool,
    initialize_if_missing: bool,
    expected_workspace_id: str | None = None,
    entry_point: str = "coding",
) -> PreparedRepository:
    """Resolve, initialize when authorized, and persist one coding workspace."""

    requested = Path(workspace_path).expanduser()
    requested_display = requested.absolute()
    git_executable = shutil.which("git")
    if not git_executable:
        raise GitUnavailableError(
            requested_display,
            "Git availability check",
            "Git is not installed or is not available on PATH",
        )
    if requested_display == Path(requested_display.anchor):
        raise WorkspaceResolutionError(
            requested_display,
            "path safety validation",
            "the filesystem root cannot be used as an automatically prepared coding workspace",
        )
    tentative_workspace = requested_display.resolve(strict=False)
    if expected_workspace_id:
        try:
            expected_workspace = service.store.get_workspace(expected_workspace_id)
        except Exception as exc:
            raise WorkspaceResolutionError(
                requested_display,
                "persisted workspace resolution",
                f"workspace {expected_workspace_id} is unavailable: {exc}",
            ) from exc
        if not _workspace_allows_path(
            expected_workspace,
            tentative_workspace,
            tentative_workspace,
        ):
            raise WorkspaceResolutionError(
                requested_display,
                "workspace boundary validation",
                f"path is outside persisted workspace {expected_workspace_id}",
            )
    if requested_display.exists() and not requested_display.is_dir():
        raise WorkspaceResolutionError(
            requested_display,
            "workspace resolution",
            "the selected path is a file, not a directory",
        )
    if not requested_display.exists():
        if not allow_create:
            raise WorkspaceResolutionError(
                requested_display,
                "workspace resolution",
                "the persisted workspace path no longer exists and was not recreated",
            )
        parent = requested_display.parent
        if not parent.is_dir() or not _has_write_bit(parent):
            raise WorkspaceResolutionError(
                requested_display,
                "workspace creation",
                f"parent directory is missing or not writable: {parent}",
            )
        try:
            requested_display.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            raise WorkspaceResolutionError(
                requested_display,
                "workspace creation",
                str(exc),
            ) from exc

    canonical_workspace = requested_display.resolve()
    existed_as_repository = False
    initialized = False
    reused_parent = False
    with _preparation_lock(canonical_workspace):
        if not canonical_workspace.is_dir():
            raise WorkspaceResolutionError(
                requested_display,
                "workspace revalidation",
                "the selected directory changed or disappeared during preparation",
            )
        try:
            repository_root, reused_parent = _resolve_existing_repository(
                git_executable,
                canonical_workspace,
                requested_path=requested_display,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RepositoryValidationError(
                requested_display,
                "Git repository inspection",
                str(exc),
            ) from exc
        existed_as_repository = repository_root is not None
        if repository_root is None:
            if not initialize_if_missing:
                raise RepositoryValidationError(
                    requested_display,
                    "repository validation",
                    "the selected coding workspace is not a Git repository",
                )
            if not _has_write_bit(canonical_workspace):
                raise GitInitializationError(
                    requested_display,
                    "Git initialization",
                    "the selected directory is not writable",
                )
            try:
                _initialize_repository(git_executable, canonical_workspace, requested_display)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise GitInitializationError(
                    requested_display,
                    "Git initialization",
                    str(exc),
                ) from exc
            try:
                repository_root, reused_parent = _resolve_existing_repository(
                    git_executable,
                    canonical_workspace,
                    requested_path=requested_display,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise GitInitializationError(
                    requested_display,
                    "Git initialization verification",
                    str(exc),
                ) from exc
            if repository_root != canonical_workspace:
                raise GitInitializationError(
                    requested_display,
                    "Git initialization verification",
                    "Git did not create a repository at the selected workspace",
                )
            initialized = True

        repository, workspace = _persist_prepared_repository(
            service,
            repository_root,
            canonical_workspace,
            expected_workspace_id=expected_workspace_id,
            requested_path=requested_display,
        )
        try:
            git_common_dir = _resolve_git_common_dir(
                git_executable,
                canonical_workspace,
                requested_display,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RepositoryValidationError(
                requested_display,
                "Git common directory resolution",
                str(exc),
            ) from exc

    prepared = PreparedRepository(
        requested_workspace_path=requested_display,
        working_directory=canonical_workspace,
        repository_root=repository_root,
        git_common_dir=git_common_dir,
        repository_existed=existed_as_repository,
        initialized=initialized,
        reused_parent_repository=reused_parent,
        repository=repository,
        workspace=workspace,
    )
    logger.info(
        "Prepared coding workspace requested=%s canonical=%s repository_root=%s "
        "repository_existed=%s reused_parent=%s initialized=%s workspace_id=%s "
        "repository_id=%s entry_point=%s",
        requested_display,
        canonical_workspace,
        repository_root,
        existed_as_repository,
        reused_parent,
        initialized,
        workspace.workspace_id,
        repository.repository_id,
        entry_point,
    )
    return prepared


def validate_prepared_repository(
    repository_root: str | Path,
    working_directory: str | Path,
) -> None:
    """Defensively validate direct Codex calls without initializing anything."""

    root = Path(repository_root).expanduser().resolve()
    working = Path(working_directory).expanduser().resolve()
    if not root.is_dir() or not working.is_dir():
        raise RepositoryValidationError(
            working,
            "Codex boundary validation",
            "the prepared repository root or working directory does not exist",
        )
    try:
        working.relative_to(root)
    except ValueError as exc:
        raise RepositoryValidationError(
            working,
            "Codex boundary validation",
            f"working directory is outside repository root {root}",
        ) from exc
    git_executable = shutil.which("git")
    if not git_executable:
        raise GitUnavailableError(
            working,
            "Codex boundary Git availability check",
            "Git is not installed or is not available on PATH",
        )
    try:
        resolved, _reused_parent = _resolve_existing_repository(
            git_executable,
            working,
            requested_path=working,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepositoryValidationError(
            working,
            "Codex boundary Git inspection",
            str(exc),
        ) from exc
    if resolved != root:
        raise RepositoryValidationError(
            working,
            "Codex boundary validation",
            f"expected Git repository root {root}, resolved {resolved or 'none'}",
        )


__all__ = [
    "GitInitializationError",
    "GitUnavailableError",
    "PreparedRepository",
    "RepositoryPersistenceError",
    "RepositoryPreparationError",
    "RepositoryValidationError",
    "WorkspaceResolutionError",
    "prepare_repository",
    "validate_prepared_repository",
]
