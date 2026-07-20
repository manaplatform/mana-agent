from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mana_agent._version import get_version
from mana_agent.config.settings import mana_home
from mana_agent.doctor.models import DoctorContext, DoctorFinding, RepairResult, Severity
from mana_agent.doctor.redaction import redact
from mana_agent.doctor.registry import select_checks


@dataclass(frozen=True, slots=True)
class DoctorReport:
    findings: list[DoctorFinding]
    repairs: list[RepairResult]
    checks_run: int
    checks_skipped: int
    deep: bool

    @property
    def ok(self) -> bool:
        return not any(item.severity in {Severity.WARNING, Severity.ERROR} for item in self.findings)

    def as_dict(self) -> dict:
        def finding(item: DoctorFinding) -> dict:
            return {"checkId": item.check_id, "severity": item.severity.value, "title": item.title, "message": item.message, "fixHint": item.fix_hint, "path": item.path, "repairable": item.repairable, "details": redact(item.details)}
        return redact({"ok": self.ok, "version": "1", "manaAgentVersion": get_version(), "mode": "deep" if self.deep else "normal", "checksRun": self.checks_run, "checksSkipped": self.checks_skipped, "summary": {"passed": sum(item.severity is Severity.INFO for item in self.findings), "warnings": sum(item.severity is Severity.WARNING for item in self.findings), "errors": sum(item.severity is Severity.ERROR for item in self.findings), "repairsApplied": sum(item.success and item.changed for item in self.repairs), "repairsFailed": sum(not item.success for item in self.repairs)}, "findings": [finding(item) for item in self.findings], "repairs": [{"checkId": item.check_id, "changed": item.changed, "success": item.success, "message": item.message, "backupPath": item.backup_path, "details": redact(item.details)} for item in self.repairs]})


def run_doctor(*, deep: bool = False, only: list[str] | None = None, skip: list[str] | None = None, fix: bool = False, repository: Path | None = None) -> DoctorReport:
    selected, unknown = select_checks(only or [], skip or [], deep=deep)
    if unknown:
        raise ValueError("Unknown doctor check ID(s): " + ", ".join(unknown))
    context = DoctorContext(home=mana_home(), repository=(repository or Path.cwd()).resolve(), deep=deep)
    findings: list[DoctorFinding] = []
    repairs: list[RepairResult] = []
    for check in selected:
        try:
            current = check.detect(context)
        except Exception as exc:
            current = [DoctorFinding(check.check_id, Severity.ERROR, "Diagnostic check failed", redact(str(exc)), "Inspect this subsystem's configuration and logs.")]
        findings.extend(current)
        if fix and check.repair:
            for item in current:
                if item.repairable:
                    repairs.append(check.repair(context, item))
    if fix and repairs:
        rerun = run_doctor(deep=deep, only=only, skip=skip, repository=repository)
        return DoctorReport(rerun.findings, repairs, rerun.checks_run, rerun.checks_skipped, rerun.deep)
    return DoctorReport(findings, repairs, len(selected), len(select_checks([], [], deep=True)[0]) - len(selected), deep)
