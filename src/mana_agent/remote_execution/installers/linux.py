"""Owner-scoped Linux systemd user service installer."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from mana_agent.remote_execution.service_errors import WorkerServiceError

UNIT_NAME = "mana-agent-worker.service"


def systemd_user_unit(*, executable: str, state_dir: Path) -> str:
    if "\n" in executable or "\n" in str(state_dir):
        raise ValueError("systemd unit values cannot contain newlines")
    return "\n".join([
        "[Unit]",
        "Description=Mana-Agent reverse worker",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f'ExecStart="{executable}" worker run --state-dir "{state_dir}"',
        "Restart=on-failure",
        "RestartSec=5s",
        "StartLimitIntervalSec=300",
        "StartLimitBurst=10",
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ])


class LinuxSystemdInstaller:
    def __init__(self, *, home: Path | None = None, runner=subprocess.run) -> None:
        self.home = (home or Path.home()).resolve()
        self.state_dir = self.home / ".local/share/mana-agent"
        self.unit_dir = self.home / ".config/systemd/user"
        self.unit_path = self.unit_dir / UNIT_NAME
        self.runner = runner

    def install(self, *, executable: str | None = None) -> Path:
        executable = executable or shutil.which("mana-agent")
        if not executable or (not Path(executable).exists() and shutil.which(executable) is None):
            raise RuntimeError("Mana-Agent executable could not be located")
        self.unit_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.unit_dir, 0o700)
        os.chmod(self.state_dir, 0o700)
        try:
            self.unit_path.write_text(
                systemd_user_unit(executable=executable, state_dir=self.state_dir),
                encoding="utf-8",
            )
            os.chmod(self.unit_path, 0o600)
            self._systemctl("daemon-reload")
            self._systemctl("enable", "--now", UNIT_NAME)
            return self.unit_path
        except Exception:
            self._systemctl("disable", "--now", UNIT_NAME, check=False)
            self.unit_path.unlink(missing_ok=True)
            raise

    def uninstall(self) -> None:
        self._systemctl("disable", "--now", UNIT_NAME, check=False)
        self.unit_path.unlink(missing_ok=True)
        self._systemctl("daemon-reload", check=False)

    def start(self) -> None:
        self._require_installed()
        self._systemctl("start", UNIT_NAME)

    def stop(self) -> None:
        self._require_installed()
        self._systemctl("stop", UNIT_NAME)

    def restart(self) -> None:
        self._require_installed()
        self._systemctl("restart", UNIT_NAME)

    def status(self) -> bool:
        return self._systemctl("is-active", "--quiet", UNIT_NAME, check=False).returncode == 0

    def logs(self) -> str:
        result = self.runner(
            ["journalctl", "--user", "-u", UNIT_NAME, "-n", "200", "--no-pager"],
            capture_output=True, text=True, check=False,
        )
        return (result.stdout or result.stderr)[-100_000:]

    def reconnect(self) -> None:
        self.restart()

    def _require_installed(self) -> None:
        if not self.unit_path.is_file():
            raise WorkerServiceError(
                "Worker service is not installed. Run `mana-agent worker install` "
                "with a coordinator URL, enrollment token, and worker ID first."
            )

    def _systemctl(self, *args: str, check: bool = True):
        try:
            return self.runner(
                ["systemctl", "--user", *args],
                capture_output=True, text=True, check=check,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()[-2_000:]
            suffix = f": {detail}" if detail else ""
            raise WorkerServiceError(
                f"Unable to {args[0]} the Linux worker service{suffix}"
            ) from exc
