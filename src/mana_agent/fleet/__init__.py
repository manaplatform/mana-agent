"""Mana Fleet distributed cross-platform verification."""

from .config import FleetConfig
from .models import (
    CapabilityMismatch, FleetJob, FleetJobResult, FleetRun, FleetRunSummary,
    FleetSelectionDecision, FleetSelectionRequest, FleetVerificationPlan,
    FleetWorker, PlatformResult, WorkerCapabilities, WorkerHealth,
    WorkerIdentity, WorkerLabels,
)
from .registry import FleetRegistry
from .selector import select_workers
from .service import FleetService
from .store import FleetStore

__all__ = [
    "CapabilityMismatch", "FleetConfig", "FleetJob", "FleetJobResult",
    "FleetRegistry", "FleetRun", "FleetRunSummary", "FleetSelectionDecision",
    "FleetSelectionRequest", "FleetService", "FleetStore",
    "FleetVerificationPlan", "FleetWorker", "PlatformResult",
    "WorkerCapabilities", "WorkerHealth", "WorkerIdentity", "WorkerLabels",
    "select_workers",
]
