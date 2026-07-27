"""Persistent, explicitly deployed automation schedules.

This module owns the on-disk schedule contract and its local-cron/GitHub
deployment adapters.  It deliberately does not select an action on the
user's behalf: callers must supply a validated definition or an explicit
action requested by the user.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mana_agent.workspaces.paths import repository_dir, repository_id_for_path


ScheduleTarget = Literal["local", "github"]
BUILTIN_ACTIONS = frozenset({"analyze", "daily_report", "self_improvement", "fleet-verify"})
CONFIG_VERSION = 2
_CRON_FIELDS = re.compile(r"^[0-9*/?,\-]+$")
_WORKFLOW_PREFIX = "mana-agent-schedule-"


class AutomationValidationError(ValueError):
    """Raised when an explicit schedule cannot be safely deployed."""


@dataclass
class ScheduleDefinition:
    id: str
    name: str
    action: str
    cron: str
    targets: list[ScheduleTarget]
    command: str | None = None
    action_config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: str = field(default_factory=lambda: _timestamp())
    updated_at: str = field(default_factory=lambda: _timestamp())
    deployment: dict[str, Any] = field(default_factory=dict)
    last_run: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        name: str,
        action: str,
        cron: str,
        targets: list[str],
        command: str | None = None,
        action_config: dict[str, Any] | None = None,
    ) -> "ScheduleDefinition":
        definition = cls(
            id=f"sch_{uuid.uuid4().hex[:12]}",
            name=name.strip(),
            action=action.strip().lower(),
            cron=" ".join(cron.split()),
            targets=list(dict.fromkeys(targets)),  # type: ignore[arg-type]
            command=command.strip() if command else None,
            action_config=dict(action_config or {}),
        )
        definition.validate()
        return definition

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScheduleDefinition":
        schedule = cls(
            id=str(value.get("id", "")),
            name=str(value.get("name", "")),
            action=str(value.get("action", "")),
            cron=str(value.get("cron", "")),
            targets=list(value.get("targets", [])),
            command=value.get("command"),
            action_config=dict(value.get("action_config") or {}),
            enabled=bool(value.get("enabled", True)),
            created_at=str(value.get("created_at", _timestamp())),
            updated_at=str(value.get("updated_at", _timestamp())),
            deployment=dict(value.get("deployment", {})),
            last_run=value.get("last_run"),
        )
        schedule.validate()
        return schedule

    def validate(self) -> None:
        if not re.fullmatch(r"sch_[a-f0-9]{12}", self.id):
            raise AutomationValidationError("Schedule id is invalid.")
        if not self.name or len(self.name) > 120 or "\n" in self.name:
            raise AutomationValidationError("Schedule name must be 1-120 characters on one line.")
        validate_cron(self.cron)
        if not self.targets or any(target not in {"local", "github"} for target in self.targets):
            raise AutomationValidationError("Choose at least one valid deployment target.")
        if self.action not in BUILTIN_ACTIONS and self.action != "custom":
            raise AutomationValidationError("Action must be a supported built-in action or custom.")
        if self.action == "custom":
            validate_custom_command(self.command)
        elif self.command:
            raise AutomationValidationError("Built-in actions cannot include a custom command.")
        if self.action == "fleet-verify":
            platforms = self.action_config.get("platforms")
            commands = self.action_config.get("commands")
            if (
                not isinstance(platforms, list)
                or not platforms
                or set(platforms) - {"linux", "windows", "macos"}
            ):
                raise AutomationValidationError(
                    "fleet-verify schedules require explicit linux/windows/macos platforms."
                )
            if not isinstance(commands, list) or not commands or any(
                not isinstance(item, str) or not item.strip() for item in commands
            ):
                raise AutomationValidationError(
                    "fleet-verify schedules require at least one explicit command."
                )
        elif self.action_config:
            raise AutomationValidationError(
                "Action configuration is supported only for fleet-verify schedules."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_cron(expression: str) -> None:
    fields = expression.split()
    if len(fields) != 5 or any(not _CRON_FIELDS.fullmatch(field) for field in fields):
        raise AutomationValidationError("Cron must be a five-field POSIX expression.")


def validate_custom_command(command: str | None) -> None:
    if not command or not command.strip():
        raise AutomationValidationError("Custom schedules require a command.")
    if "\n" in command or "\x00" in command:
        raise AutomationValidationError("Custom commands must be a single non-null line.")


def config_path(root: Path) -> Path:
    return repository_dir(repository_id_for_path(root)) / "automations" / "config.json"


def load_config(root: Path) -> dict[str, Any]:
    path = config_path(root)
    if not path.exists():
        return {"version": CONFIG_VERSION, "schedules": [], "automations": [], "runs": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutomationValidationError(f"Automation configuration is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise AutomationValidationError("Automation configuration must be an object.")
    payload.setdefault("version", CONFIG_VERSION)
    payload.setdefault("schedules", [])
    payload.setdefault("automations", [])
    payload.setdefault("runs", [])
    return payload


def save_config(root: Path, payload: dict[str, Any]) -> None:
    path = config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["version"] = CONFIG_VERSION
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def list_schedules(root: Path) -> list[ScheduleDefinition]:
    return [ScheduleDefinition.from_dict(item) for item in load_config(root)["schedules"]]


def get_schedule(root: Path, schedule_id: str) -> ScheduleDefinition:
    for schedule in list_schedules(root):
        if schedule.id == schedule_id:
            return schedule
    raise AutomationValidationError(f"Schedule not found: {schedule_id}")


def upsert_schedule(root: Path, schedule: ScheduleDefinition) -> None:
    schedule.validate()
    payload = load_config(root)
    records = [item for item in payload["schedules"] if item.get("id") != schedule.id]
    records.append(schedule.to_dict())
    payload["schedules"] = records
    save_config(root, payload)


def delete_schedule(root: Path, schedule_id: str) -> ScheduleDefinition:
    schedule = get_schedule(root, schedule_id)
    payload = load_config(root)
    payload["schedules"] = [item for item in payload["schedules"] if item.get("id") != schedule_id]
    save_config(root, payload)
    return schedule


def managed_cron_marker(schedule: ScheduleDefinition) -> str:
    return f"# mana-agent:{schedule.id}"


def schedule_command(schedule: ScheduleDefinition, root: Path) -> str:
    if schedule.action == "custom":
        assert schedule.command is not None
        return schedule.command
    root_text = str(root.resolve()).replace("'", "'\\''")
    if schedule.action == "fleet-verify":
        return shlex.join(_fleet_verify_argv(schedule, str(root.resolve())))
    return f"mana-agent automation execute --action {schedule.action} --root-dir '{root_text}'"


def _fleet_verify_argv(schedule: ScheduleDefinition, root: str) -> list[str]:
    argv = ["mana-agent", "fleet", "verify", "--root-dir", root]
    for value in schedule.action_config["platforms"]:
        argv += ["--platform", str(value)]
    for value in schedule.action_config["commands"]:
        argv += ["--command", str(value)]
    argv += [
        "--maximum-workers", str(int(schedule.action_config.get("maximum_workers", 4))),
        "--timeout", str(int(schedule.action_config.get("timeout", 1800))),
        "--json",
    ]
    return argv


def read_crontab(runner=subprocess.run) -> str:
    result = runner(["crontab", "-l"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return result.stdout
    if "no crontab" in (result.stderr or "").lower():
        return ""
    raise AutomationValidationError(f"Unable to inspect crontab: {(result.stderr or '').strip()}")


def _replace_cron_entry(contents: str, schedule: ScheduleDefinition, root: Path, *, present: bool) -> str:
    marker = managed_cron_marker(schedule)
    lines = [line for line in contents.splitlines() if marker not in line]
    if present:
        lines.append(f"{schedule.cron} {schedule_command(schedule, root)} {marker}")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def deploy_local(schedule: ScheduleDefinition, root: Path, runner=subprocess.run) -> dict[str, Any]:
    if shutil.which("crontab") is None:
        raise AutomationValidationError("Local deployment unavailable: crontab is not installed.")
    contents = read_crontab(runner)
    desired = _replace_cron_entry(contents, schedule, root, present=schedule.enabled)
    result = runner(["crontab", "-"], input=desired, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AutomationValidationError(f"Unable to update crontab: {(result.stderr or '').strip()}")
    return {"status": "deployed" if schedule.enabled else "disabled", "updated_at": _timestamp()}


def remove_local(schedule: ScheduleDefinition, root: Path, runner=subprocess.run) -> None:
    if shutil.which("crontab") is None:
        return
    contents = read_crontab(runner)
    desired = _replace_cron_entry(contents, schedule, root, present=False)
    runner(["crontab", "-"], input=desired, capture_output=True, text=True, check=False)


def workflow_path(root: Path, schedule: ScheduleDefinition) -> Path:
    return root / ".github" / "workflows" / f"{_WORKFLOW_PREFIX}{schedule.id}.yml"


def render_workflow(schedule: ScheduleDefinition) -> str:
    if schedule.action == "custom":
        command = schedule.command
    elif schedule.action == "fleet-verify":
        command = shlex.join(_fleet_verify_argv(schedule, "."))
    else:
        command = f"mana-agent automation execute --action {schedule.action} --root-dir ."
    assert command
    return "\n".join(
        [
            "# Managed by mana-agent. Do not edit manually.",
            f"name: {json.dumps(f'Mana Agent schedule: {schedule.name}')}",
            "",
            "on:",
            "  schedule:",
            f'    - cron: "{schedule.cron}"',
            "  workflow_dispatch:",
            "",
            "permissions:",
            "  contents: read",
            "",
            "jobs:",
            "  run:",
            "    runs-on: ubuntu-latest",
            "    env:",
            "      MANA_HOME: ${{ runner.temp }}/mana",
            "    steps:",
            "      - uses: actions/checkout@v4",
            "      - uses: actions/setup-python@v5",
            "        with:",
            '          python-version: "3.12"',
            "      - name: Install Mana Agent",
            '        run: python -m pip install ".[full]"',
            "      - name: Run schedule",
            "        run: |",
            f"          {command}",
            "      - name: Upload Mana Agent results",
            "        if: always()",
            "        uses: actions/upload-artifact@v4",
            "        with:",
            f"          name: mana-agent-{schedule.id}-${{{{ github.run_id }}}}",
            "          path: ${{ runner.temp }}/mana/",
            "          if-no-files-found: ignore",
            "",
        ]
    )


def deploy_github(schedule: ScheduleDefinition, root: Path, runner=subprocess.run) -> dict[str, Any]:
    path = workflow_path(root, schedule)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_workflow(schedule), encoding="utf-8")
    status = {"status": "written", "path": str(path.relative_to(root)), "updated_at": _timestamp()}
    if not schedule.enabled:
        path.unlink(missing_ok=True)
        return {"status": "disabled", "path": str(path.relative_to(root)), "updated_at": _timestamp()}
    if shutil.which("gh") is None:
        status["publish"] = "blocked: GitHub CLI (gh) is not installed"
        return status
    default_branch = runner(
        ["gh", "repo", "view", "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if default_branch.returncode != 0 or not default_branch.stdout.strip():
        raise AutomationValidationError(f"Unable to determine GitHub default branch: {default_branch.stderr.strip()}")
    branch = runner(["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True, check=False)
    if branch.returncode != 0 or not branch.stdout.strip():
        raise AutomationValidationError("GitHub deployment requires a checked-out feature branch.")
    branch_name = branch.stdout.strip()
    add = runner(["git", "add", "--", str(path.relative_to(root))], cwd=root, capture_output=True, text=True, check=False)
    if add.returncode != 0:
        raise AutomationValidationError(f"Unable to stage workflow: {add.stderr.strip()}")
    commit = runner(["git", "commit", "-m", f"Add Mana Agent schedule {schedule.name}", "--", str(path.relative_to(root))], cwd=root, capture_output=True, text=True, check=False)
    if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr).lower():
        raise AutomationValidationError(f"Unable to commit workflow: {commit.stderr.strip()}")
    push = runner(["git", "push", "-u", "origin", branch_name], cwd=root, capture_output=True, text=True, check=False)
    if push.returncode != 0:
        raise AutomationValidationError(f"Unable to push workflow branch: {push.stderr.strip()}")
    pr = runner(["gh", "pr", "create", "--base", default_branch.stdout.strip(), "--head", branch_name, "--title", f"Add Mana Agent schedule: {schedule.name}", "--body", "Managed schedule workflow. It activates after merge to the default branch."], cwd=root, capture_output=True, text=True, check=False)
    status["publish"] = "pushed"
    status["pull_request"] = pr.stdout.strip() if pr.returncode == 0 else f"blocked: {pr.stderr.strip()}"
    return status


def deploy_schedule(schedule: ScheduleDefinition, root: Path, *, runner=subprocess.run) -> ScheduleDefinition:
    schedule.validate()
    deployment: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for target in schedule.targets:
        try:
            deployment[target] = deploy_local(schedule, root, runner) if target == "local" else deploy_github(schedule, root, runner)
        except AutomationValidationError as exc:
            errors[target] = str(exc)
    schedule.deployment = {**deployment, **({"errors": errors} if errors else {})}
    schedule.updated_at = _timestamp()
    upsert_schedule(root, schedule)
    return schedule


def remove_deployment(schedule: ScheduleDefinition, root: Path, *, runner=subprocess.run) -> None:
    if "local" in schedule.targets:
        remove_local(schedule, root, runner)
    if "github" in schedule.targets:
        workflow_path(root, schedule).unlink(missing_ok=True)


def deployment_status(schedule: ScheduleDefinition, root: Path, *, runner=subprocess.run) -> dict[str, Any]:
    result: dict[str, Any] = {"id": schedule.id, "name": schedule.name, "targets": {}}
    if "local" in schedule.targets:
        try:
            result["targets"]["local"] = {"status": "deployed" if managed_cron_marker(schedule) in read_crontab(runner) else "drifted"}
        except AutomationValidationError as exc:
            result["targets"]["local"] = {"status": "unavailable", "reason": str(exc)}
    if "github" in schedule.targets:
        path = workflow_path(root, schedule)
        deployed = schedule.deployment.get("github", {})
        status = "written" if path.exists() else "drifted"
        if "github" in schedule.deployment.get("errors", {}):
            status = "failed"
        result["targets"]["github"] = {
            "status": status,
            "path": str(path.relative_to(root)),
            "schedule_timezone": "UTC",
            "deployment": deployed,
        }
    return result


def execute_builtin_action(
    action: str, root: Path, *, action_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if action not in BUILTIN_ACTIONS:
        raise AutomationValidationError(f"Unsupported built-in action: {action}")
    if action == "self_improvement":
        from mana_agent.automations.self_improvement import run_self_improvement_loop
        return {"ok": True, "created": len(run_self_improvement_loop(root) or [])}
    if action == "fleet-verify":
        config = dict(action_config or {})
        platforms = list(config.get("platforms") or [])
        commands = list(config.get("commands") or [])
        if not platforms or not commands:
            raise AutomationValidationError(
                "fleet-verify requires persisted explicit platforms and commands"
            )
        argv = ["mana-agent", "fleet", "verify", "--root-dir", str(root.resolve())]
        for value in platforms:
            argv += ["--platform", str(value)]
        for value in commands:
            argv += ["--command", str(value)]
        argv += [
            "--maximum-workers", str(int(config.get("maximum_workers", 4))),
            "--timeout", str(int(config.get("timeout", 1800))),
            "--json",
        ]
        result = subprocess.run(
            argv, cwd=root, capture_output=True, text=True,
            timeout=int(config.get("timeout", 1800)) + 60, check=False,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": (result.stdout or "")[-4000:],
            "stderr": (result.stderr or "")[-2000:],
        }
    from mana_agent.ui.streamlit_helpers import trigger_automation
    return trigger_automation("analyze" if action in {"analyze", "daily_report"} else action, root=root)


def run_schedule_now(schedule: ScheduleDefinition, root: Path, *, runner=subprocess.run) -> dict[str, Any]:
    """Run an explicitly selected saved schedule once and persist its result."""
    schedule.validate()
    if schedule.action == "custom":
        assert schedule.command is not None
        completed = runner(
            schedule.command,
            cwd=root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        result: dict[str, Any] = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "")[-2000:],
            "stderr": (completed.stderr or "")[-1000:],
        }
    else:
        result = execute_builtin_action(
            schedule.action, root, action_config=schedule.action_config,
        )
    schedule.last_run = {"at": _timestamp(), **result}
    schedule.updated_at = _timestamp()
    upsert_schedule(root, schedule)
    return result
