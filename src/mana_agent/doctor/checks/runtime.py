from __future__ import annotations

import importlib.metadata
import shutil
import sys

from mana_agent._version import get_version
from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity


def python_version(context: DoctorContext) -> list[DoctorFinding]:
    del context
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if not (3, 10) <= sys.version_info[:2] <= (3, 14):
        return [DoctorFinding("runtime/python-version", Severity.ERROR, "Unsupported Python version", f"Python {version} is outside Mana-Agent's supported range (3.10–3.14).")]
    return [DoctorFinding("runtime/python-version", Severity.INFO, "Python runtime", f"Python {version} ({sys.executable})")]


def installation(context: DoctorContext) -> list[DoctorFinding]:
    del context
    executable = shutil.which("mana-agent")
    if not executable:
        return [DoctorFinding("installation/executable-path", Severity.WARNING, "mana-agent executable not on PATH", "The installed console script could not be resolved.", "Run Mana-Agent through its intended environment or reinstall it.")]
    return [DoctorFinding("installation/executable-path", Severity.INFO, "mana-agent executable", executable)]


def version_consistency(context: DoctorContext) -> list[DoctorFinding]:
    del context
    runtime = get_version()
    try:
        installed = importlib.metadata.version("mana-agent")
    except importlib.metadata.PackageNotFoundError:
        installed = runtime
    severity = Severity.INFO if runtime == installed else Severity.WARNING
    return [DoctorFinding("runtime/package-version", severity, "Mana-Agent version", f"Runtime {runtime}; installed metadata {installed}.", "Reinstall Mana-Agent if these versions were not intentionally different." if severity is Severity.WARNING else None)]


def git_available(context: DoctorContext) -> list[DoctorFinding]:
    del context
    if shutil.which("git"):
        return [DoctorFinding("tools/requirements", Severity.INFO, "Git executable", "git is available.")]
    return [DoctorFinding("tools/requirements", Severity.ERROR, "Git executable missing", "Git is required for repository workflows.", "Install Git and ensure it is on PATH.")]
