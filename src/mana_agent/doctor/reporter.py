from __future__ import annotations

import json
from collections import defaultdict

from mana_agent.doctor.models import Severity
from mana_agent.doctor.runner import DoctorReport


def render(report: DoctorReport, *, json_mode: bool = False) -> str:
    if json_mode:
        return json.dumps(report.as_dict(), indent=2, sort_keys=True)
    sections: dict[str, list] = defaultdict(list)
    for finding in report.findings:
        sections[finding.check_id.split("/", 1)[0].replace("-", " ").title()].append(finding)
    lines = ["Mana-Agent Doctor", ""]
    markers = {Severity.INFO: "✓", Severity.WARNING: "!", Severity.ERROR: "✗"}
    for section, findings in sections.items():
        lines.append(section)
        for finding in findings:
            lines.append(f"  {markers[finding.severity]} {finding.title}")
            if finding.severity is not Severity.INFO:
                lines.append(f"    {finding.message}")
                if finding.fix_hint:
                    lines.append(f"    Fix: {finding.fix_hint}")
        lines.append("")
    data = report.as_dict()["summary"]
    lines.append(f"Summary: {data['passed']} passed, {data['warnings']} warnings, {data['errors']} errors")
    if any(item.repairable for item in report.findings):
        lines.append("Run: mana-agent doctor --fix")
    return "\n".join(lines)
