"""Typed, side-effect-free contracts for the doctor command."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DoctorFinding:
    check_id: str
    severity: Severity
    title: str
    message: str
    fix_hint: str | None = None
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    repairable: bool = False


@dataclass(frozen=True, slots=True)
class RepairResult:
    check_id: str
    changed: bool
    success: bool
    message: str
    backup_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DoctorContext:
    home: Path
    repository: Path
    deep: bool = False


Detect = Callable[[DoctorContext], list[DoctorFinding]]
Repair = Callable[[DoctorContext, DoctorFinding], RepairResult]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    check_id: str
    section: str
    description: str
    detect: Detect
    repair: Repair | None = None
    deep: bool = False
    network: bool = False
