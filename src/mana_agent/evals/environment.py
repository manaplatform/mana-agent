from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

from mana_agent import __version__

from .ids import stable_hash
from .models import EnvironmentSnapshot

LOCKFILES = ("pyproject.toml", "uv.lock", "poetry.lock", "requirements.txt", "package-lock.json", "pnpm-lock.yaml", "yarn.lock")
SAFE_ENV_PREFIXES = ("MANA_", "OPENAI_", "CODEX_", "CI", "GITHUB_")


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def capture_environment(
    root: str | Path,
    *,
    workspace_backend: str,
    prompt_hashes: dict[str, str] | None = None,
    tool_policy_hash: str = "",
    lane_policy_hash: str = "",
    environment_image: str = "",
    deterministic_task: bool = True,
    has_test_command: bool = True,
) -> EnvironmentSnapshot:
    root = Path(root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    diff = _git(root, "diff", "--binary", check=False)
    branch = _git(root, "branch", "--show-current", check=False)
    remote = _git(root, "config", "--get", "remote.origin.url", check=False)
    lockfile_hashes = {
        name: stable_hash((root / name).read_bytes().hex())
        for name in LOCKFILES
        if (root / name).is_file()
    }
    reasons: list[str] = []
    if not commit:
        reasons.append("repository commit is unavailable")
    if status:
        reasons.append("starting workspace is dirty")
    if workspace_backend not in {"local-worktree", "execution-fabric", "fleet"}:
        reasons.append(f"workspace backend {workspace_backend!r} is unsupported")
    if not deterministic_task:
        reasons.append("task definition is not deterministic")
    if not has_test_command:
        reasons.append("task has no test command")
    packages = sorted(f"{item.metadata['Name']}=={item.version}" for item in importlib.metadata.distributions() if item.metadata.get("Name"))
    return EnvironmentSnapshot(
        repository_commit=commit,
        base_branch=branch,
        remote_url=remote,
        dirty=bool(status),
        starting_diff_hash=stable_hash(diff),
        python_version=sys.version.split()[0],
        mana_agent_version=__version__,
        operating_system=platform.system(),
        architecture=platform.machine(),
        packages=packages,
        lockfile_hashes=lockfile_hashes,
        environment_image=environment_image,
        workspace_backend=workspace_backend,
        git_version=subprocess.run(["git", "--version"], capture_output=True, text=True, check=False).stdout.strip(),
        prompt_hashes=dict(prompt_hashes or {}),
        tool_policy_hash=tool_policy_hash,
        lane_policy_hash=lane_policy_hash,
        environment_variable_names=sorted(name for name in os.environ if name.startswith(SAFE_ENV_PREFIXES)),
        reproducible=not reasons,
        non_reproducible_reasons=reasons,
    )
