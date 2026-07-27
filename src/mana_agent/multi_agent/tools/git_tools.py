"""Model-selectable Git tools with shared safety policy."""

from __future__ import annotations

import hashlib
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from mana_agent.utils.redaction import redact_json_line, redact_secrets


class GitRiskLevel(str, Enum):
    READ_ONLY = "READ_ONLY"
    LOCAL_SAFE_WRITE = "LOCAL_SAFE_WRITE"
    LOCAL_HISTORY_WRITE = "LOCAL_HISTORY_WRITE"
    REMOTE_WRITE = "REMOTE_WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"
    HISTORY_REWRITE = "HISTORY_REWRITE"


@dataclass(frozen=True)
class GitExecutionResult:
    ok: bool
    command: list[str]
    repo_root: str
    risk_level: str
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    blocked: bool = False
    error: str = ""
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GitObservation:
    current_branch: str = ""
    head: str = ""
    status_porcelain: str = ""
    status_hash: str = ""
    operation_state: str = ""

    def fingerprint(self) -> str:
        payload = "\n".join([self.current_branch, self.head, self.status_porcelain, self.operation_state])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"fingerprint": self.fingerprint()}


class GitStateMemory:
    """Session-local Git observation cache invalidated by repository state."""

    def __init__(self) -> None:
        self._observations: dict[str, GitObservation] = {}
        self.created_branch_names: list[str] = []
        self.commit_messages: list[str] = []
        self.files_staged_or_committed: list[str] = []
        self.last_pushed_branch: str = ""

    def remember(self, repo_root: Path, observation: GitObservation) -> dict[str, Any]:
        key = str(repo_root.resolve())
        previous = self._observations.get(key)
        self._observations[key] = observation
        return {
            "cache_hit": previous is not None and previous.fingerprint() == observation.fingerprint(),
            "invalidated": previous is not None and previous.fingerprint() != observation.fingerprint(),
            "observation": observation.to_dict(),
        }

    def record_branch(self, name: str) -> None:
        if name and name not in self.created_branch_names:
            self.created_branch_names.append(name)

    def record_commit(self, message: str, files: list[str]) -> None:
        if message:
            self.commit_messages.append(message)
        for item in files:
            if item not in self.files_staged_or_committed:
                self.files_staged_or_committed.append(item)

    def record_push(self, branch: str) -> None:
        self.last_pushed_branch = branch


_SESSION_COMMAND_CACHE: dict[tuple[str, str], list[str]] = {}
_DEFAULT_TIMEOUT_SECONDS = 120
_READ_ONLY = {"status", "diff", "log", "show", "branch", "remote", "config", "help", "rev-parse"}
_LOCAL_SAFE_WRITE = {"add", "restore", "stash", "switch", "checkout", "fetch"}
_LOCAL_HISTORY_WRITE = {"commit", "merge", "revert", "tag"}
_REMOTE_WRITE = {"push", "pull"}
_HISTORY_REWRITE = {"rebase", "filter-branch"}
_DESTRUCTIVE = {"clean", "reset", "update-ref"}
_PROTECTED_PATTERNS = (
    ("reset", "--hard"),
    ("clean", "-fd"),
    ("clean", "-fdx"),
    ("branch", "-D"),
    ("push", "--force"),
    ("push", "--force-with-lease"),
    ("push", "--mirror"),
    ("push", "--delete"),
    ("rebase", "--onto"),
    ("filter-branch",),
    ("update-ref",),
    ("reflog", "expire"),
    ("gc", "--prune=now"),
)
_SAFE_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]{1,120}$")


def _redact_text(text: str) -> str:
    return redact_json_line(str(text or ""))


def _run_raw_git(args: list[str], *, cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def resolve_repo_root(repo_path: str | Path | None = None) -> Path:
    start = Path(repo_path or ".").expanduser().resolve()
    if start.is_file():
        start = start.parent
    completed = _run_raw_git(["rev-parse", "--show-toplevel"], cwd=start, timeout=10)
    if completed.returncode != 0:
        raise ValueError(f"not a git repository: {start}")
    return Path(completed.stdout.strip()).resolve()


def git_version(repo_root: Path | None = None) -> str:
    cwd = Path(repo_root or ".").resolve()
    completed = _run_raw_git(["--version"], cwd=cwd, timeout=10)
    return completed.stdout.strip() if completed.returncode == 0 else "git version unknown"


def discover_git_commands(repo_path: str | Path | None = None, *, refresh: bool = False) -> list[str]:
    repo_root = resolve_repo_root(repo_path)
    key = (git_version(repo_root), str(repo_root))
    if not refresh and key in _SESSION_COMMAND_CACHE:
        return list(_SESSION_COMMAND_CACHE[key])
    completed = _run_raw_git(["help", "-a"], cwd=repo_root, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git help -a failed")
    commands: set[str] = set()
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":") or stripped.startswith(("usage:", "See ")):
            continue
        for token in re.split(r"\s+", stripped):
            if re.fullmatch(r"[a-z][a-z0-9-]*", token):
                commands.add(token)
    discovered = sorted(commands)
    _SESSION_COMMAND_CACHE[key] = discovered
    return list(discovered)


def observe_git_state(repo_path: str | Path | None = None) -> GitObservation:
    repo_root = resolve_repo_root(repo_path)

    def _out(args: list[str]) -> str:
        completed = _run_raw_git(args, cwd=repo_root, timeout=15)
        return completed.stdout.strip() if completed.returncode == 0 else ""

    operation = ""
    git_dir = repo_root / ".git"
    if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
        operation = "rebase"
    elif (git_dir / "MERGE_HEAD").exists():
        operation = "merge"
    elif (git_dir / "CHERRY_PICK_HEAD").exists():
        operation = "cherry-pick"
    status = _out(["status", "--porcelain=v1"])
    return GitObservation(
        current_branch=_out(["branch", "--show-current"]),
        head=_out(["rev-parse", "HEAD"]),
        status_porcelain=status,
        status_hash=hashlib.sha256(status.encode("utf-8")).hexdigest(),
        operation_state=operation,
    )


def classify_git_risk(args: list[str]) -> GitRiskLevel:
    command = str(args[0] if args else "").strip()
    lowered = [str(item).strip().lower() for item in args]
    if _matches_protected(lowered):
        if command == "push" and any(item in lowered for item in {"--force", "--force-with-lease", "--mirror", "--delete"}):
            return GitRiskLevel.HISTORY_REWRITE
        return GitRiskLevel.DESTRUCTIVE if command not in _HISTORY_REWRITE else GitRiskLevel.HISTORY_REWRITE
    if command in _READ_ONLY:
        return GitRiskLevel.READ_ONLY
    if command in _LOCAL_SAFE_WRITE:
        return GitRiskLevel.LOCAL_SAFE_WRITE
    if command in _LOCAL_HISTORY_WRITE:
        return GitRiskLevel.LOCAL_HISTORY_WRITE
    if command in _REMOTE_WRITE:
        return GitRiskLevel.REMOTE_WRITE
    if command in _HISTORY_REWRITE:
        return GitRiskLevel.HISTORY_REWRITE
    if command in _DESTRUCTIVE:
        return GitRiskLevel.DESTRUCTIVE
    return GitRiskLevel.LOCAL_SAFE_WRITE


def _matches_protected(lowered_args: list[str]) -> bool:
    joined = " ".join(lowered_args)
    for pattern in _PROTECTED_PATTERNS:
        if all(part in lowered_args or part in joined for part in pattern):
            return True
    return False


def _validate_args(args: list[str]) -> list[str]:
    if not isinstance(args, list) or not args:
        raise ValueError("git args must be a non-empty list")
    normalized = [str(item) for item in args]
    if normalized[0] == "git":
        raise ValueError("args must omit the leading git executable")
    if any("\x00" in item for item in normalized):
        raise ValueError("git args must not contain NUL bytes")
    return normalized


def run_git(
    args: list[str],
    *,
    repo_path: str | Path | None = None,
    timeout: int | None = None,
    allow_protected: bool = False,
    memory: GitStateMemory | None = None,
) -> dict[str, Any]:
    normalized = _validate_args(args)
    repo_root = resolve_repo_root(repo_path)
    risk = classify_git_risk(normalized)
    lowered = [item.lower() for item in normalized]
    if _matches_protected(lowered) and not allow_protected:
        state = observe_git_state(repo_root).to_dict()
        return GitExecutionResult(
            ok=False,
            command=["git", *normalized],
            repo_root=str(repo_root),
            risk_level=risk.value,
            returncode=None,
            blocked=True,
            error="protected git command blocked; explicit user intent is required",
            state=state,
        ).to_dict()
    started = time.perf_counter()
    try:
        completed = _run_raw_git(
            normalized,
            cwd=repo_root,
            timeout=max(1, min(int(timeout or _DEFAULT_TIMEOUT_SECONDS), 900)),
        )
        duration = round((time.perf_counter() - started) * 1000.0, 3)
        observation = observe_git_state(repo_root)
        state = observation.to_dict()
        if memory is not None:
            state["memory"] = memory.remember(repo_root, observation)
        return GitExecutionResult(
            ok=completed.returncode == 0,
            command=["git", *normalized],
            repo_root=str(repo_root),
            risk_level=risk.value,
            returncode=completed.returncode,
            stdout=_redact_text(completed.stdout),
            stderr=_redact_text(completed.stderr),
            duration_ms=duration,
            state=redact_secrets(state),
        ).to_dict()
    except subprocess.TimeoutExpired as exc:
        duration = round((time.perf_counter() - started) * 1000.0, 3)
        return GitExecutionResult(
            ok=False,
            command=["git", *normalized],
            repo_root=str(repo_root),
            risk_level=risk.value,
            returncode=None,
            stdout=_redact_text(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"git command timed out after {timeout or _DEFAULT_TIMEOUT_SECONDS}s",
            duration_ms=duration,
            error="timeout",
        ).to_dict()


def git_help(
    *,
    command: str | None = None,
    all: bool = False,  # noqa: A002 - public tool schema uses all
    repo_path: str | Path | None = None,
    refresh: bool = False,
    timeout: int | None = None,
) -> dict[str, Any]:
    if all:
        commands = discover_git_commands(repo_path, refresh=refresh)
        return {"ok": True, "commands": commands, "count": len(commands), "cache_refresh": bool(refresh)}
    if command:
        return run_git(["help", str(command)], repo_path=repo_path, timeout=timeout)
    return run_git(["help"], repo_path=repo_path, timeout=timeout)


def status(*, repo_path: str | Path | None = None, short: bool = True, porcelain: bool = False) -> dict[str, Any]:
    args = ["status"]
    if porcelain:
        args.append("--porcelain=v1")
    elif short:
        args.append("--short")
    return run_git(args, repo_path=repo_path)


def diff(*, repo_path: str | Path | None = None, path: str = "", staged: bool = False) -> dict[str, Any]:
    args = ["diff"]
    if staged:
        args.append("--staged")
    if path:
        args.extend(["--", path])
    return run_git(args, repo_path=repo_path)


def log(*, repo_path: str | Path | None = None, limit: int = 10, oneline: bool = True) -> dict[str, Any]:
    args = ["log", f"-{max(1, min(int(limit or 10), 100))}"]
    if oneline:
        args.append("--oneline")
    return run_git(args, repo_path=repo_path)


def show(*, repo_path: str | Path | None = None, revision: str = "HEAD", stat: bool = False) -> dict[str, Any]:
    args = ["show", str(revision or "HEAD")]
    if stat:
        args.append("--stat")
    return run_git(args, repo_path=repo_path)


def branch(*, repo_path: str | Path | None = None, all: bool = False) -> dict[str, Any]:  # noqa: A002
    args = ["branch"]
    if all:
        args.append("--all")
    return run_git(args, repo_path=repo_path)


def switch(*, branch_name: str, repo_path: str | Path | None = None) -> dict[str, Any]:
    _validate_branch_name(branch_name)
    return run_git(["switch", branch_name], repo_path=repo_path)


def checkout(*, target: str, repo_path: str | Path | None = None, new_branch: bool = False) -> dict[str, Any]:
    _validate_branch_name(target)
    args = ["checkout", "-b", target] if new_branch else ["checkout", target]
    return run_git(args, repo_path=repo_path)


def create_branch(*, branch_name: str, repo_path: str | Path | None = None, switch_to: bool = True) -> dict[str, Any]:
    _validate_branch_name(branch_name)
    preflight = {
        "status": run_git(["status", "--short"], repo_path=repo_path),
        "current_branch": run_git(["branch", "--show-current"], repo_path=repo_path),
        "branches": run_git(["branch", "--list"], repo_path=repo_path),
    }
    args = ["switch", "-c", branch_name] if switch_to else ["branch", branch_name]
    result = run_git(args, repo_path=repo_path)
    result["preflight"] = preflight
    return result


def add(*, paths: list[str], repo_path: str | Path | None = None) -> dict[str, Any]:
    clean_paths = _validate_paths(paths)
    return run_git(["add", "--", *clean_paths], repo_path=repo_path)


def restore(*, paths: list[str], repo_path: str | Path | None = None, staged: bool = False) -> dict[str, Any]:
    clean_paths = _validate_paths(paths)
    args = ["restore"]
    if staged:
        args.append("--staged")
    args.extend(["--", *clean_paths])
    return run_git(args, repo_path=repo_path)


def stash(*, repo_path: str | Path | None = None, message: str = "", include_untracked: bool = False) -> dict[str, Any]:
    args = ["stash", "push"]
    if include_untracked:
        args.append("--include-untracked")
    if message:
        args.extend(["-m", message])
    return run_git(args, repo_path=repo_path)


def commit(*, message: str, repo_path: str | Path | None = None, amend: bool = False) -> dict[str, Any]:
    text = str(message or "").strip()
    if not text:
        raise ValueError("commit message is required")
    preflight = {
        "status_short": run_git(["status", "--short"], repo_path=repo_path),
        "diff": run_git(["diff"], repo_path=repo_path),
        "diff_staged": run_git(["diff", "--staged"], repo_path=repo_path),
        "diff_staged_stat": run_git(["diff", "--staged", "--stat"], repo_path=repo_path),
    }
    args = ["commit"]
    if amend:
        args.append("--amend")
    args.extend(["-m", text])
    result = run_git(args, repo_path=repo_path)
    result["preflight"] = preflight
    return result


def push(
    *,
    repo_path: str | Path | None = None,
    remote: str = "",
    branch_name: str = "",
    set_upstream: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    args = ["push"]
    if force:
        args.append("--force-with-lease")
    preflight = {
        "status_short": run_git(["status", "--short"], repo_path=repo_path),
        "current_branch": run_git(["branch", "--show-current"], repo_path=repo_path),
        "remotes": run_git(["remote", "-v"], repo_path=repo_path),
        "upstream": run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], repo_path=repo_path),
    }
    if not remote and not branch_name and not set_upstream:
        repo_root = resolve_repo_root(repo_path)
        upstream = _run_raw_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=repo_root, timeout=15)
        if upstream.returncode != 0:
            current = _run_raw_git(["branch", "--show-current"], cwd=repo_root, timeout=15)
            current_branch = current.stdout.strip()
            if current.returncode == 0 and current_branch:
                args.extend(["-u", "origin", current_branch])
                result = run_git(args, repo_path=repo_root)
                result["preflight"] = preflight
                return result
    if set_upstream:
        args.append("-u")
    if remote:
        args.append(remote)
    if branch_name:
        _validate_branch_name(branch_name)
        args.append(branch_name)
    result = run_git(args, repo_path=repo_path)
    result["preflight"] = preflight
    return result


def pull(*, repo_path: str | Path | None = None, rebase: bool = False) -> dict[str, Any]:
    args = ["pull"]
    if rebase:
        args.append("--rebase")
    return run_git(args, repo_path=repo_path)


def fetch(*, repo_path: str | Path | None = None, remote: str = "", prune: bool = False) -> dict[str, Any]:
    args = ["fetch"]
    if prune:
        args.append("--prune")
    if remote:
        args.append(remote)
    return run_git(args, repo_path=repo_path)


def remote(*, repo_path: str | Path | None = None, verbose: bool = True) -> dict[str, Any]:
    args = ["remote"]
    if verbose:
        args.append("-v")
    return run_git(args, repo_path=repo_path)


def tag(*, repo_path: str | Path | None = None, name: str = "", message: str = "") -> dict[str, Any]:
    args = ["tag"]
    if name:
        _validate_branch_name(name)
        args.append(name)
    if message:
        args.extend(["-m", message])
    return run_git(args, repo_path=repo_path)


def merge(*, target: str, repo_path: str | Path | None = None, no_ff: bool = False) -> dict[str, Any]:
    _validate_branch_name(target)
    args = ["merge"]
    if no_ff:
        args.append("--no-ff")
    args.append(target)
    return run_git(args, repo_path=repo_path)


def rebase(*, target: str = "", repo_path: str | Path | None = None, continue_: bool = False, abort: bool = False) -> dict[str, Any]:
    args = ["rebase"]
    if continue_:
        args.append("--continue")
    elif abort:
        args.append("--abort")
    elif target:
        _validate_branch_name(target)
        args.append(target)
    else:
        raise ValueError("rebase requires target, continue_, or abort")
    return run_git(args, repo_path=repo_path)


def revert(*, revision: str, repo_path: str | Path | None = None, no_commit: bool = False) -> dict[str, Any]:
    args = ["revert"]
    if no_commit:
        args.append("--no-commit")
    args.append(str(revision or "HEAD"))
    return run_git(args, repo_path=repo_path)


def reset(*, repo_path: str | Path | None = None, mode: str = "--mixed", target: str = "HEAD", allow_protected: bool = False) -> dict[str, Any]:
    if mode not in {"--soft", "--mixed", "--hard"}:
        raise ValueError("reset mode must be --soft, --mixed, or --hard")
    return run_git(["reset", mode, target], repo_path=repo_path, allow_protected=allow_protected)


def clean(*, repo_path: str | Path | None = None, force: bool = False, directories: bool = False, allow_protected: bool = False) -> dict[str, Any]:
    flags = "-"
    if force:
        flags += "f"
    if directories:
        flags += "d"
    if flags == "-":
        flags = "-n"
    return run_git(["clean", flags], repo_path=repo_path, allow_protected=allow_protected)


def config(*, repo_path: str | Path | None = None, key: str = "", value: str = "", get: bool = True) -> dict[str, Any]:
    args = ["config"]
    if key:
        args.append(key)
    if value and not get:
        args.append(value)
    return run_git(args, repo_path=repo_path)


def generic(
    *,
    args: list[str],
    repo_path: str | Path | None = None,
    timeout: int | None = None,
    allow_protected: bool = False,
    memory: GitStateMemory | None = None,
) -> dict[str, Any]:
    return run_git(args, repo_path=repo_path, timeout=timeout, allow_protected=allow_protected, memory=memory)


def execute_tool(tool_name: str, args: dict[str, Any] | None = None, *, repo_path: str | Path | None = None) -> dict[str, Any]:
    payload = dict(args or {})
    payload.setdefault("repo_path", repo_path)
    name = str(tool_name or "").strip().replace("_", ".")
    mapping = {
        "git.status": status,
        "git.diff": diff,
        "git.log": log,
        "git.show": show,
        "git.branch": branch,
        "git.switch": switch,
        "git.checkout": checkout,
        "git.create.branch": create_branch,
        "git.add": add,
        "git.restore": restore,
        "git.stash": stash,
        "git.commit": commit,
        "git.push": push,
        "git.pull": pull,
        "git.fetch": fetch,
        "git.remote": remote,
        "git.tag": tag,
        "git.merge": merge,
        "git.rebase": rebase,
        "git.revert": revert,
        "git.reset": reset,
        "git.clean": clean,
        "git.config": config,
        "git.generic": generic,
        "git.help": git_help,
    }
    if name == "git.create_branch":
        name = "git.create.branch"
    func = mapping.get(name)
    if func is None:
        raise ValueError(f"unsupported git tool: {tool_name}")
    return func(**payload)


def _validate_branch_name(value: str) -> None:
    text = str(value or "").strip()
    if not text or ".." in text or text.startswith(("-", "/")) or text.endswith(("/", ".lock")):
        raise ValueError("unsafe git ref name")
    if not _SAFE_BRANCH_RE.fullmatch(text):
        raise ValueError("unsafe git ref name")


def _validate_paths(paths: list[str]) -> list[str]:
    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list")
    clean: list[str] = []
    for item in paths:
        text = str(item or "").strip().replace("\\", "/")
        if not text or text.startswith("/") or "\x00" in text or ".." in Path(text).parts:
            raise ValueError("unsafe repository path")
        clean.append(text)
    return clean
