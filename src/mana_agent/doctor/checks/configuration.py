from __future__ import annotations

import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

from mana_agent.config import user_config
from mana_agent.doctor.models import DoctorContext, DoctorFinding, RepairResult, Severity


def parse(context: DoctorContext) -> list[DoctorFinding]:
    path = context.home / "config.toml"
    if not path.exists():
        return [DoctorFinding("config/parse", Severity.WARNING, "Configuration file missing", f"No configuration file exists at {path}.", "Run mana-agent --configure to create configuration.")]
    try:
        user_config.load_user_config()
    except Exception as exc:
        return [DoctorFinding("config/parse", Severity.ERROR, "Configuration cannot be parsed", str(exc), path=str(path))]
    return [DoctorFinding("config/parse", Severity.INFO, "Configuration parsed", f"{path} parsed successfully.", path=str(path))]


def schema(context: DoctorContext) -> list[DoctorFinding]:
    try:
        values = user_config.load_effective_settings()
        user_config.validate_config_values(values)
    except Exception as exc:
        return [DoctorFinding("config/schema", Severity.ERROR, "Configuration schema is invalid", str(exc), "Run mana-agent --configure to correct the configuration.")]
    return [DoctorFinding("config/schema", Severity.INFO, "Configuration schema", "Configuration values validate successfully.")]


def permissions(context: DoctorContext) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    for name in ("config.toml", "secrets.toml"):
        path = context.home / name
        if not path.exists():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            findings.append(DoctorFinding("config/file-permissions", Severity.WARNING, "Configuration permissions are too broad", f"{path} has mode {mode:04o}; owner-only access is required.", "Run mana-agent doctor --fix to tighten permissions.", path=str(path), repairable=True))
    return findings or [DoctorFinding("config/file-permissions", Severity.INFO, "Configuration permissions", "Sensitive configuration files are owner-only.")]


def repair_permissions(context: DoctorContext, finding: DoctorFinding) -> RepairResult:
    path = Path(finding.path or "")
    if not path.is_file():
        return RepairResult(finding.check_id, False, False, "Configuration file no longer exists.")
    backup = path.with_name(f"{path.name}.doctor-{datetime.now(timezone.utc):%Y%m%d%H%M%S}.bak")
    shutil.copy2(path, backup)
    os.chmod(path, 0o600)
    return RepairResult(finding.check_id, True, stat.S_IMODE(path.stat().st_mode) == 0o600, "Set configuration permissions to 0600.", str(backup))
