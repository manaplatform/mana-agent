"""Platform installer boundary; macOS LaunchAgent implementation is production-ready."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mana_agent.remote_execution.service_errors import WorkerServiceError

LABEL = "net.manaplatform.mana-agent.worker"


@dataclass(frozen=True)
class MacOSPaths:
    home: Path

    @property
    def support(self) -> Path: return self.home / "Library/Application Support/ManaAgent"
    @property
    def logs(self) -> Path: return self.home / "Library/Logs/ManaAgent"
    @property
    def plist(self) -> Path: return self.home / "Library/LaunchAgents" / f"{LABEL}.plist"


def launchagent_payload(*, executable: str, state_dir: Path, log_dir: Path) -> dict:
    """No enrollment token, credential, or private key is ever placed here."""
    return {"Label": LABEL, "ProgramArguments": [executable, "worker", "run", "--state-dir", str(state_dir)],
            "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False}, "ProcessType": "Background",
            "StandardOutPath": str(log_dir / "worker.out.log"), "StandardErrorPath": str(log_dir / "worker.err.log")}


class MacOSInstaller:
    def __init__(
        self,
        *,
        home: Path | None = None,
        runner=subprocess.run,
        user_id: int | None = None,
    ) -> None:
        if user_id is not None and user_id < 0:
            raise ValueError("macOS user ID must be non-negative")
        self.paths = MacOSPaths((home or Path.home()).resolve())
        self.runner = runner
        self.user_id = user_id

    def install(self, *, executable: str | None = None) -> Path:
        executable = executable or shutil.which("mana-agent")
        if not executable:
            raise RuntimeError("Mana-Agent executable could not be located")
        if not Path(executable).exists() and shutil.which(executable) is None:
            raise RuntimeError("Mana-Agent executable could not be located")
        created: list[Path] = []
        try:
            for path in (self.paths.support, self.paths.logs, self.paths.plist.parent):
                path.mkdir(parents=True, exist_ok=True)
                os.chmod(path, 0o700)
                created.append(path)
            with self.paths.plist.open("wb") as stream:
                plistlib.dump(launchagent_payload(executable=executable, state_dir=self.paths.support, log_dir=self.paths.logs), stream)
            os.chmod(self.paths.plist, 0o600)
            self._launch("bootstrap", self._user_domain(), str(self.paths.plist))
            self._launch("kickstart", "-k", self._service_domain())
            return self.paths.plist
        except Exception:
            self._bootout(ignore_errors=True)
            self.paths.plist.unlink(missing_ok=True)
            raise

    def uninstall(self) -> None:
        self._bootout(ignore_errors=True)
        self.paths.plist.unlink(missing_ok=True)

    def start(self) -> None:
        self._require_installed()
        if not self.status():
            self._launch("bootstrap", self._user_domain(), str(self.paths.plist))
        self._launch("kickstart", "-k", self._service_domain())

    def stop(self) -> None:
        self._require_installed()
        self._bootout(ignore_errors=False)

    def restart(self) -> None:
        self._require_installed()
        self._bootout(ignore_errors=True)
        self._launch("bootstrap", self._user_domain(), str(self.paths.plist))
        self._launch("kickstart", "-k", self._service_domain())

    def status(self) -> bool:
        result = self.runner(
            ["launchctl", "print", self._service_domain()],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    def _require_installed(self) -> None:
        if not self.paths.plist.is_file():
            raise WorkerServiceError(
                "Worker service is not installed. Run `mana-agent worker install` "
                "with a coordinator URL and enrollment token first."
            )

    def _bootout(self, *, ignore_errors: bool) -> None:
        self._launch(
            "bootout",
            self._user_domain(),
            str(self.paths.plist),
            check=not ignore_errors,
        )

    def _user_domain(self) -> str:
        user_id = self.user_id
        if user_id is None:
            getuid = getattr(os, "getuid", None)
            if getuid is None:
                raise WorkerServiceError(
                    "macOS worker service control requires a POSIX user ID"
                )
            user_id = int(getuid())
        return f"gui/{user_id}"

    def _service_domain(self) -> str:
        return f"{self._user_domain()}/{LABEL}"

    def _launch(self, *args: str, check: bool = True) -> None:
        try:
            self.runner(["launchctl", *args], capture_output=True, text=True, check=check)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()[-2_000:]
            suffix = f": {detail}" if detail else ""
            raise WorkerServiceError(
                f"Unable to {args[0]} the macOS worker service{suffix}"
            ) from exc
