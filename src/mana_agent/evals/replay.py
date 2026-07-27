from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from .config import EvalSuite
from .models import EvalRun, EvalVariant
from .runner import EvalRunner
from .storage import EvalStorage
from .workspace import workspace_backend


@dataclass(frozen=True, slots=True)
class TrajectoryDivergence:
    sequence: int
    expected_status: str
    actual_status: str
    expected_output_hash: str
    actual_output_hash: str
    repository_diff_changed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def task_replay(*, storage: EvalStorage, runner: EvalRunner, suite: EvalSuite, run_id: str, variant: EvalVariant, experiment_id: str) -> EvalRun:
    original = storage.load_run(run_id)
    task = next((item for item in suite.tasks if item.task_id == original.task_id), None)
    if task is None:
        raise ValueError(f"suite does not contain original task {original.task_id}")
    return runner.run(
        suite=suite, task=task, variant=variant, trial_number=original.trial_number,
        experiment_id=experiment_id, replayed_from_run_id=original.run_id,
        force_execute=True,
    )


def trajectory_replay(*, storage: EvalStorage, run_id: str, retain_workspace: bool = False) -> dict[str, Any]:
    original = storage.load_run(run_id)
    config_path = storage.run_dir(run_id) / "config.yaml"
    try:
        import yaml
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot load replay configuration: {exc}") from exc
    task = config["task"]
    backend = workspace_backend("local-worktree", storage.root / "workspaces", retain=retain_workspace)
    workspace = backend.create(task["repository"], original.repository_commit, run_id=f"replay-{run_id}")
    divergences: list[dict[str, Any]] = []
    try:
        for sequence, command in enumerate(original.commands, start=1):
            result = subprocess.run(command.command, cwd=workspace.path, shell=True, capture_output=True, text=True)
            actual_hash = hashlib.sha256((result.stdout + result.stderr).encode()).hexdigest()
            expected_hash = hashlib.sha256((command.stdout + command.stderr).encode()).hexdigest()
            expected_status = "success" if command.exit_code == 0 and not command.timed_out else "failed"
            actual_status = "success" if result.returncode == 0 else "failed"
            if expected_status != actual_status or expected_hash != actual_hash:
                diff = subprocess.run(["git", "diff", "--binary"], cwd=workspace.path, capture_output=True, text=True).stdout
                divergence = TrajectoryDivergence(sequence, expected_status, actual_status, expected_hash, actual_hash, bool(diff))
                divergences.append(divergence.to_dict())
                break
        return {"run_id": run_id, "replayed_commands": sequence if original.commands else 0, "diverged": bool(divergences), "first_divergence": divergences[0] if divergences else None, "model_called": False}
    finally:
        backend.cleanup(workspace)
