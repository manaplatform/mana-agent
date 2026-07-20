from __future__ import annotations

from mana_agent.doctor.checks import codex, configuration, filesystem, runtime
from mana_agent.doctor.models import DoctorCheck


CHECKS: tuple[DoctorCheck, ...] = (
    DoctorCheck("runtime/python-version", "Runtime", "Validate Python version.", runtime.python_version),
    DoctorCheck("runtime/package-version", "Runtime", "Compare package versions.", runtime.version_consistency),
    DoctorCheck("installation/executable-path", "Installation", "Locate console script.", runtime.installation),
    DoctorCheck("tools/requirements", "Tools", "Locate Git.", runtime.git_available),
    DoctorCheck("config/parse", "Configuration", "Parse managed configuration.", configuration.parse),
    DoctorCheck("config/schema", "Configuration", "Validate managed configuration.", configuration.schema),
    DoctorCheck("config/file-permissions", "Configuration", "Validate sensitive file modes.", configuration.permissions, configuration.repair_permissions),
    DoctorCheck("persistence/state-path", "Persistence", "Validate Mana state path.", filesystem.state_path, filesystem.repair_state_path),
    DoctorCheck("integrations/codex-binary", "Codex", "Locate configured Codex.", codex.binary),
    DoctorCheck("integrations/codex-protocol", "Codex", "Run Codex app-server preflight.", codex.protocol, deep=True),
)

_BY_ID = {check.check_id: check for check in CHECKS}
if len(_BY_ID) != len(CHECKS):
    raise RuntimeError("Doctor registry contains duplicate check IDs.")


def select_checks(only: list[str], skip: list[str], *, deep: bool) -> tuple[list[DoctorCheck], list[str]]:
    unknown = sorted((set(only) | set(skip)) - set(_BY_ID))
    if unknown:
        return [], unknown
    selected = [check for check in CHECKS if (not only or check.check_id in only) and check.check_id not in skip and (deep or not check.deep)]
    return selected, []
