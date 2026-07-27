from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .leaderboard import leaderboard_csv, leaderboard_markdown
from .redaction import redact, redact_text
from .regression import GateResult
from .storage import atomic_write


def write_leaderboard(root: str | Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    root = Path(root)
    rows = redact(rows)
    paths = {
        "json": root / "leaderboard.json",
        "csv": root / "leaderboard.csv",
        "markdown": root / "leaderboard.md",
    }
    atomic_write(paths["json"], json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n")
    atomic_write(paths["csv"], leaderboard_csv(rows))
    atomic_write(paths["markdown"], redact_text(leaderboard_markdown(rows)))
    return paths


def regression_markdown(comparison: dict[str, Any], gate: GateResult) -> str:
    lines = [
        "# Mana Eval Lab Regression Report", "",
        f"**Gate: {'PASS' if gate.passed else 'FAIL'}**", "",
        f"Baseline tasks: {comparison.get('baseline_task_count', 0)}  ",
        f"Candidate tasks: {comparison.get('candidate_task_count', 0)}  ",
        f"Pass rate: {comparison.get('baseline_pass_rate', 0):.1%} → {comparison.get('candidate_pass_rate', 0):.1%}  ",
        f"Mean score: {comparison.get('baseline_mean_score', 0):.2f} → {comparison.get('candidate_mean_score', 0):.2f}", "",
    ]
    if gate.reasons:
        lines.extend(["## Gate failures", "", *(f"- {reason}" for reason in gate.reasons), ""])
    for classification in ("regressed", "improved", "missing", "inconclusive"):
        tasks = [item for item in comparison.get("tasks", []) if item.get("classification") == classification]
        if tasks:
            lines.extend([f"## {classification.title()} tasks", ""])
            lines.extend(f"- `{item['task_id']}`: {item.get('baseline_score')} → {item.get('candidate_score')}" for item in tasks)
            lines.append("")
    routing = [item for item in comparison.get("tasks", []) if item.get("routing_changed")]
    if routing:
        lines.extend(["## Routing changes", "", *(f"- `{item['task_id']}`" for item in routing), ""])
    return "\n".join(lines) + "\n"


def regression_junit(comparison: dict[str, Any], gate: GateResult) -> str:
    tasks = comparison.get("tasks", [])
    suite = ET.Element("testsuite", name="mana-eval-regression", tests=str(len(tasks)), failures=str(sum(item.get("classification") in {"regressed", "missing", "inconclusive"} for item in tasks)))
    for item in tasks:
        case = ET.SubElement(suite, "testcase", classname="mana.eval.regression", name=str(item.get("task_id")))
        if item.get("classification") in {"regressed", "missing", "inconclusive"}:
            failure = ET.SubElement(case, "failure", message=f"task {item.get('classification')}")
            failure.text = json.dumps(item, sort_keys=True)
    if not gate.passed and not tasks:
        case = ET.SubElement(suite, "testcase", classname="mana.eval.regression", name="gate")
        failure = ET.SubElement(case, "failure", message="regression gate failed")
        failure.text = "\n".join(gate.reasons)
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def write_regression(root: str | Path, comparison: dict[str, Any], gate: GateResult) -> dict[str, Path]:
    root = Path(root)
    comparison = redact(comparison)
    payload = {**comparison, "gate": redact(gate.to_dict())}
    paths = {"json": root / "regression.json", "markdown": root / "regression.md", "junit": root / "regression.junit.xml"}
    atomic_write(paths["json"], json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    atomic_write(paths["markdown"], regression_markdown(comparison, gate))
    atomic_write(paths["junit"], regression_junit(comparison, gate))
    return paths
