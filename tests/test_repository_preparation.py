from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mana_agent.workspaces.preparation import (
    GitInitializationError,
    GitUnavailableError,
    RepositoryPersistenceError,
    RepositoryValidationError,
    WorkspaceResolutionError,
)
from mana_agent.workspaces.service import WorkspaceService


def _git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        text=True,
        capture_output=True,
        check=check,
    )


def _init(path: Path, *, commit: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    if commit:
        _git(path, "config", "user.name", "Mana Test")
        _git(path, "config", "user.email", "mana@example.invalid")
        (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
        _git(path, "add", "tracked.txt")
        _git(path, "commit", "-q", "-m", "initial")


def _prepare(service: WorkspaceService, path: Path, **kwargs):
    return service.prepare_repository(
        path,
        allow_create=kwargs.pop("allow_create", False),
        initialize_if_missing=kwargs.pop("initialize_if_missing", True),
        entry_point="test",
        **kwargs,
    )


def test_existing_repository_is_reused_without_mutating_git_state(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _init(repository, commit=True)
    _git(repository, "checkout", "-q", "-b", "feature/existing")
    _git(repository, "remote", "add", "origin", "https://example.invalid/project.git")
    _git(repository, "config", "custom.preserved", "yes")
    before_branch = _git(repository, "branch", "--show-current").stdout
    before_remotes = _git(repository, "remote", "-v").stdout
    before_config = _git(repository, "config", "--local", "--list").stdout

    prepared = _prepare(WorkspaceService(), repository)

    assert prepared.repository_root == repository.resolve()
    assert prepared.repository_existed is True
    assert prepared.initialized is False
    assert _git(repository, "branch", "--show-current").stdout == before_branch
    assert _git(repository, "remote", "-v").stdout == before_remotes
    assert _git(repository, "config", "--local", "--list").stdout == before_config


@pytest.mark.parametrize("with_file", [False, True])
def test_non_git_directory_is_initialized_without_staging_or_committing(
    tmp_path: Path,
    with_file: bool,
) -> None:
    workspace = tmp_path / ("non-empty workspace" if with_file else "empty workspace")
    workspace.mkdir()
    existing = workspace / "existing-π.txt"
    payload = b"existing\x00content\n"
    if with_file:
        existing.write_bytes(payload)

    prepared = _prepare(WorkspaceService(), workspace)

    assert prepared.initialized is True
    assert prepared.repository_root == workspace.resolve()
    assert (workspace / ".git").exists()
    assert _git(workspace, "symbolic-ref", "--short", "HEAD").stdout.strip() == "main"
    assert _git(workspace, "diff", "--cached", "--name-only").stdout == ""
    assert _git(workspace, "rev-parse", "--verify", "HEAD", check=False).returncode != 0
    if with_file:
        assert existing.read_bytes() == payload
        assert _git(workspace, "status", "--short", "-z").stdout == "?? existing-π.txt\x00"


def test_initialization_compatibility_fallback_selects_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mana_agent.workspaces import preparation

    workspace = tmp_path / "legacy git"
    workspace.mkdir()
    original = preparation._run_git
    calls: list[list[str]] = []

    def without_init_branch(git_executable, args, **kwargs):  # noqa: ANN001
        calls.append(list(args))
        if args[:3] == ["init", "-b", "main"]:
            return subprocess.CompletedProcess(
                [git_executable, *args],
                129,
                "",
                "error: unknown option `b'",
            )
        return original(git_executable, args, **kwargs)

    monkeypatch.setattr(preparation, "_run_git", without_init_branch)
    prepared = _prepare(WorkspaceService(), workspace)

    assert prepared.initialized is True
    assert any(args[:1] == ["init"] and "-b" not in args for args in calls)
    assert ["-C", str(workspace.resolve()), "branch", "-M", "main"] in calls
    assert _git(workspace, "symbolic-ref", "--short", "HEAD").stdout.strip() == "main"


def test_repeated_and_concurrent_preparation_is_idempotent(tmp_path: Path) -> None:
    workspace = tmp_path / "concurrent"
    workspace.mkdir()

    def prepare_once(_index: int):
        return _prepare(WorkspaceService(), workspace)

    with ThreadPoolExecutor(max_workers=6) as executor:
        prepared = list(executor.map(prepare_once, range(12)))

    assert sum(item.initialized for item in prepared) == 1
    assert len({item.repository_id for item in prepared}) == 1
    matching = [
        record
        for record in WorkspaceService().store.list_repositories()
        if Path(record.canonical_path) == workspace.resolve()
    ]
    assert len(matching) == 1


def test_selected_subdirectory_reuses_parent_repository_without_nested_git(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    selected = repository / "packages" / "example"
    selected.mkdir(parents=True)
    _init(repository)

    prepared = _prepare(WorkspaceService(), selected)

    assert prepared.working_directory == selected.resolve()
    assert prepared.repository_root == repository.resolve()
    assert prepared.reused_parent_repository is True
    assert prepared.initialized is False
    assert not (selected / ".git").exists()


def test_git_worktree_is_recognized_without_reinitialization(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    worktree = tmp_path / "linked worktree"
    _init(repository, commit=True)
    _git(repository, "worktree", "add", "-q", "-b", "linked", str(worktree))
    assert (worktree / ".git").is_file()

    prepared = _prepare(WorkspaceService(), worktree)

    assert prepared.repository_root == worktree.resolve()
    assert prepared.repository_existed is True
    assert prepared.initialized is False
    assert prepared.git_common_dir == (repository / ".git").resolve()
    assert (worktree / ".git").is_file()


def test_bare_and_corrupt_repositories_fail_without_repair(tmp_path: Path) -> None:
    bare = tmp_path / "bare.git"
    bare.mkdir()
    _git(bare, "init", "--bare", "-q")
    with pytest.raises(RepositoryValidationError, match="bare Git repository"):
        _prepare(WorkspaceService(), bare)

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    marker = corrupt / ".git"
    marker.mkdir()
    sentinel = marker / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(RepositoryValidationError, match="invalid or corrupt"):
        _prepare(WorkspaceService(), corrupt)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_file_missing_stale_and_explicit_missing_paths_are_distinguished(tmp_path: Path) -> None:
    regular_file = tmp_path / "file.txt"
    regular_file.write_text("content", encoding="utf-8")
    with pytest.raises(WorkspaceResolutionError, match="file, not a directory"):
        _prepare(WorkspaceService(), regular_file)

    stale = tmp_path / "stale"
    with pytest.raises(WorkspaceResolutionError, match="was not recreated"):
        _prepare(WorkspaceService(), stale, allow_create=False)
    assert not stale.exists()

    explicit = tmp_path / "explicit"
    prepared = _prepare(WorkspaceService(), explicit, allow_create=True)
    assert explicit.is_dir()
    assert prepared.initialized is True


def test_persisted_workspace_boundary_is_checked_before_initialization(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    outside = tmp_path / "outside"
    approved.mkdir()
    outside.mkdir()
    service = WorkspaceService()
    workspace = service.create_workspace(
        "bounded",
        roots=[approved],
        allowed_roots=[approved],
    )

    with pytest.raises(WorkspaceResolutionError, match="outside persisted workspace"):
        _prepare(
            service,
            outside,
            expected_workspace_id=workspace.workspace_id,
        )

    assert not (outside / ".git").exists()


def test_git_unavailable_and_read_only_workspace_fail_before_coding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr("mana_agent.workspaces.preparation.shutil.which", lambda _name: None)
    with pytest.raises(GitUnavailableError, match="not installed or is not available on PATH"):
        _prepare(WorkspaceService(), workspace)
    assert not (workspace / ".git").exists()

    monkeypatch.undo()
    workspace.chmod(0o500)
    try:
        with pytest.raises(GitInitializationError, match="not writable"):
            _prepare(WorkspaceService(), workspace)
    finally:
        workspace.chmod(0o700)


def test_persistence_failure_keeps_git_and_next_call_reconciles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "persistence"
    workspace.mkdir()
    service = WorkspaceService()
    original = service.store.save_repository
    failed = False

    def fail_once(record):  # noqa: ANN001
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("persistence unavailable")
        return original(record)

    monkeypatch.setattr(service.store, "save_repository", fail_once)
    with pytest.raises(RepositoryPersistenceError, match="persistence unavailable"):
        _prepare(service, workspace)
    assert (workspace / ".git").exists()

    prepared = _prepare(service, workspace)
    assert prepared.initialized is False
    assert prepared.repository_existed is True


def test_symlink_and_canonical_paths_share_one_persistence_record(tmp_path: Path) -> None:
    workspace = tmp_path / "real"
    alias = tmp_path / "alias"
    workspace.mkdir()
    try:
        alias.symlink_to(workspace, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - restricted Windows hosts
        pytest.skip(f"symlinks unavailable: {exc}")
    service = WorkspaceService()

    first = _prepare(service, alias)
    second = _prepare(service, workspace)

    assert first.repository_id == second.repository_id
    assert first.repository_root == second.repository_root == workspace.resolve()


def test_git_initialization_failure_preserves_underlying_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mana_agent.workspaces import preparation

    workspace = tmp_path / "failed init"
    workspace.mkdir()
    original = preparation._run_git

    def fail_init(git_executable, args, **kwargs):  # noqa: ANN001
        if args[:1] == ["init"]:
            return subprocess.CompletedProcess(
                [git_executable, *args],
                1,
                "",
                "fatal: simulated filesystem failure",
            )
        return original(git_executable, args, **kwargs)

    monkeypatch.setattr(preparation, "_run_git", fail_init)
    with pytest.raises(GitInitializationError, match="simulated filesystem failure"):
        _prepare(WorkspaceService(), workspace)
    assert not (workspace / ".git").exists()


def test_unsafe_filesystem_root_is_rejected() -> None:
    with pytest.raises(WorkspaceResolutionError, match="filesystem root"):
        _prepare(WorkspaceService(), Path(os.path.abspath(os.sep)))
