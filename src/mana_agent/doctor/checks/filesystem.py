from __future__ import annotations

import os

from mana_agent.doctor.models import DoctorContext, DoctorFinding, RepairResult, Severity


def state_path(context: DoctorContext) -> list[DoctorFinding]:
    if context.home.is_dir() and os.access(context.home, os.W_OK):
        return [DoctorFinding("persistence/state-path", Severity.INFO, "Mana state directory", f"{context.home} is writable.", path=str(context.home))]
    return [DoctorFinding("persistence/state-path", Severity.WARNING, "Mana state directory missing or unwritable", f"{context.home} does not exist or is not writable.", "Run mana-agent doctor --fix to create the state directory.", path=str(context.home), repairable=True)]


def repair_state_path(context: DoctorContext, finding: DoctorFinding) -> RepairResult:
    try:
        context.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        return RepairResult(finding.check_id, True, context.home.is_dir() and os.access(context.home, os.W_OK), "Created Mana state directory.")
    except OSError as exc:
        return RepairResult(finding.check_id, False, False, str(exc))
