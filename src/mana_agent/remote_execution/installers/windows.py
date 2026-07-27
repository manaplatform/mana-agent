"""Explicit user-mode Windows Task Scheduler worker installer."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

TASK_NAME = "ManaAgentWorker"


def task_scheduler_xml(*, executable: str, state_dir: Path) -> str:
    command = escape(executable)
    arguments = escape(f'worker run --state-dir "{state_dir}"')
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>
  <Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><RestartOnFailure><Interval>PT1M</Interval><Count>5</Count></RestartOnFailure><ExecutionTimeLimit>PT0S</ExecutionTimeLimit></Settings>
  <Actions Context="Author"><Exec><Command>{command}</Command><Arguments>{arguments}</Arguments></Exec></Actions>
</Task>
"""


class WindowsTaskSchedulerInstaller:
    def __init__(self, *, local_app_data: Path | None = None, runner=subprocess.run) -> None:
        base = local_app_data or Path(os.environ.get("LOCALAPPDATA", ""))
        if not str(base):
            raise RuntimeError("LOCALAPPDATA is unavailable; cannot create owner-scoped worker state")
        self.root = base.resolve() / "ManaAgent"
        self.state_dir = self.root / "state"
        self.log_dir = self.root / "logs"
        self.task_xml = self.root / "worker-task.xml"
        self.runner = runner

    def install(self, *, executable: str | None = None) -> Path:
        executable = executable or shutil.which("mana-agent.exe") or shutil.which("mana-agent")
        if not executable:
            raise RuntimeError("Mana-Agent executable could not be located")
        for path in (self.root, self.state_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._restrict_acl(self.root)
        try:
            self.task_xml.write_text(
                task_scheduler_xml(executable=executable, state_dir=self.state_dir),
                encoding="utf-16",
            )
            self._restrict_acl(self.task_xml)
            self.runner(
                ["schtasks.exe", "/Create", "/TN", TASK_NAME, "/XML", str(self.task_xml), "/F"],
                capture_output=True, text=True, check=True,
            )
            self.runner(
                ["schtasks.exe", "/Run", "/TN", TASK_NAME],
                capture_output=True, text=True, check=True,
            )
            return self.task_xml
        except Exception:
            self.runner(
                ["schtasks.exe", "/Delete", "/TN", TASK_NAME, "/F"],
                capture_output=True, text=True, check=False,
            )
            self.task_xml.unlink(missing_ok=True)
            raise

    def uninstall(self) -> None:
        self.runner(
            ["schtasks.exe", "/Delete", "/TN", TASK_NAME, "/F"],
            capture_output=True, text=True, check=False,
        )
        self.task_xml.unlink(missing_ok=True)

    def status(self) -> bool:
        result = self.runner(
            ["schtasks.exe", "/Query", "/TN", TASK_NAME],
            capture_output=True, text=True, check=False,
        )
        return result.returncode == 0

    def logs(self) -> str:
        rows = []
        for path in sorted(self.log_dir.glob("*.log")):
            rows.append(path.read_text(encoding="utf-8", errors="replace")[-50_000:])
        return "\n".join(rows)[-100_000:]

    def reconnect(self) -> None:
        self.runner(
            ["schtasks.exe", "/End", "/TN", TASK_NAME],
            capture_output=True, text=True, check=False,
        )
        self.runner(
            ["schtasks.exe", "/Run", "/TN", TASK_NAME],
            capture_output=True, text=True, check=True,
        )

    def _restrict_acl(self, path: Path) -> None:
        username = os.environ.get("USERNAME")
        if not username:
            raise RuntimeError("Windows user identity is unavailable for ACL validation")
        result = self.runner(
            ["icacls.exe", str(path), "/inheritance:r", "/grant:r", f"{username}:(OI)(CI)F"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Unable to restrict worker state ACLs to the current user. "
                "Run from an account permitted to manage its LOCALAPPDATA ACLs."
            )
