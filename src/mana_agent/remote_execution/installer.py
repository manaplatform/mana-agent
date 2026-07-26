"""Platform installer boundary; macOS LaunchAgent implementation is production-ready."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

LABEL = "net.manaplatform.mana-agent.worker"


class WorkerServiceError(RuntimeError):
    """Raised when an installed worker service cannot be controlled."""


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
    def __init__(self, *, home: Path | None = None, runner=subprocess.run) -> None:
        self.paths = MacOSPaths((home or Path.home()).resolve())
        self.runner = runner

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
            self._launch("bootstrap", f"gui/{os.getuid()}", str(self.paths.plist))
            self._launch("kickstart", "-k", f"gui/{os.getuid()}/{LABEL}")
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
            self._launch("bootstrap", f"gui/{os.getuid()}", str(self.paths.plist))
        self._launch("kickstart", "-k", f"gui/{os.getuid()}/{LABEL}")

    def stop(self) -> None:
        self._require_installed()
        self._bootout(ignore_errors=False)

    def restart(self) -> None:
        self._require_installed()
        self._bootout(ignore_errors=True)
        self._launch("bootstrap", f"gui/{os.getuid()}", str(self.paths.plist))
        self._launch("kickstart", "-k", f"gui/{os.getuid()}/{LABEL}")

    def status(self) -> bool:
        result = self.runner(["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"], capture_output=True, text=True, check=False)
        return result.returncode == 0

    def _require_installed(self) -> None:
        if not self.paths.plist.is_file():
            raise WorkerServiceError(
                "Worker service is not installed. Run `mana-agent worker install` "
                "with a coordinator URL and enrollment token first."
            )

    def _bootout(self, *, ignore_errors: bool) -> None:
        self._launch("bootout", f"gui/{os.getuid()}", str(self.paths.plist), check=not ignore_errors)

    def _launch(self, *args: str, check: bool = True) -> None:
        try:
            self.runner(["launchctl", *args], capture_output=True, text=True, check=check)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()[-2_000:]
            suffix = f": {detail}" if detail else ""
            raise WorkerServiceError(
                f"Unable to {args[0]} the macOS worker service{suffix}"
            ) from exc
