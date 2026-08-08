"""Durable connector health and incident storage under ~/.mana/connectors."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from mana_agent.utils.redaction import redact_secrets

from .models import (
    ConnectorHealthSnapshot,
    ConnectorIncident,
    DeliveryReceipt,
    IncidentEvent,
    ProbeResult,
    utc_now,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
_atomic_replace = os.replace


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        _atomic_replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _safe_id(value: str) -> str:
    if not _SAFE_ID.fullmatch(value or ""):
        raise ValueError(f"invalid connector health identity: {value!r}")
    return value


class ConnectorHealthStore:
    """Persist snapshots, incidents, probe results, and delivery receipts."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.health_dir = self.root / "health"
        self.incidents_dir = self.root / "incidents"
        self.probes_dir = self.root / "probes"
        self.receipts_dir = self.root / "receipts"
        self.events_path = self.root / "events.jsonl"
        self._lock = threading.RLock()
        for directory in (
            self.root,
            self.health_dir,
            self.incidents_dir,
            self.probes_dir,
            self.receipts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass

    def save_snapshot(self, snapshot: ConnectorHealthSnapshot) -> None:
        with self._lock:
            path = self.health_dir / f"{_safe_id(snapshot.connector_id)}.json"
            payload = redact_secrets(json.loads(snapshot.model_dump_json()))
            _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, default=str))

    def load_snapshot(self, connector_id: str) -> ConnectorHealthSnapshot | None:
        path = self.health_dir / f"{_safe_id(connector_id)}.json"
        if not path.exists():
            return None
        return ConnectorHealthSnapshot.model_validate_json(path.read_text(encoding="utf-8"))

    def list_snapshots(self) -> list[ConnectorHealthSnapshot]:
        rows: list[ConnectorHealthSnapshot] = []
        for path in sorted(self.health_dir.glob("*.json")):
            try:
                rows.append(ConnectorHealthSnapshot.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return rows

    def save_incident(self, incident: ConnectorIncident) -> None:
        with self._lock:
            path = self.incidents_dir / f"{_safe_id(incident.incident_id)}.json"
            payload = redact_secrets(json.loads(incident.model_dump_json()))
            _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, default=str))
            # Secondary index by connector for listing
            index = self.incidents_dir / f"by_connector_{_safe_id(incident.connector_id)}.jsonl"
            with index.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"incident_id": incident.incident_id, "started_at": incident.started_at.isoformat()}) + "\n")

    def load_incident(self, incident_id: str) -> ConnectorIncident | None:
        path = self.incidents_dir / f"{_safe_id(incident_id)}.json"
        if not path.exists():
            return None
        return ConnectorIncident.model_validate_json(path.read_text(encoding="utf-8"))

    def list_incidents(
        self,
        *,
        connector_id: str | None = None,
        include_open: bool = True,
        include_closed: bool = True,
        limit: int = 100,
    ) -> list[ConnectorIncident]:
        incidents: list[ConnectorIncident] = []
        for path in sorted(self.incidents_dir.glob("incident_*.json"), reverse=True):
            try:
                incident = ConnectorIncident.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if connector_id and incident.connector_id != connector_id:
                continue
            if incident.open and not include_open:
                continue
            if not incident.open and not include_closed:
                continue
            incidents.append(incident)
            if len(incidents) >= limit:
                break
        return incidents

    def append_probe_result(self, connector_id: str, result: ProbeResult) -> None:
        with self._lock:
            path = self.probes_dir / f"{_safe_id(connector_id)}.jsonl"
            payload = redact_secrets(json.loads(result.model_dump_json()))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")

    def load_probe_results(self, connector_id: str, *, limit: int = 200) -> list[ProbeResult]:
        path = self.probes_dir / f"{_safe_id(connector_id)}.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        results: list[ProbeResult] = []
        for line in lines[-limit:]:
            try:
                results.append(ProbeResult.model_validate_json(line))
            except Exception:
                continue
        return results

    def save_receipt(self, receipt: DeliveryReceipt) -> None:
        with self._lock:
            path = self.receipts_dir / f"{_safe_id(receipt.connector_id)}_{_safe_id(receipt.message_id)}.json"
            payload = redact_secrets(json.loads(receipt.model_dump_json()))
            _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, default=str))

    def list_receipts(self, connector_id: str, *, limit: int = 50) -> list[DeliveryReceipt]:
        prefix = f"{_safe_id(connector_id)}_"
        receipts: list[DeliveryReceipt] = []
        for path in sorted(self.receipts_dir.glob(f"{prefix}*.json"), reverse=True):
            try:
                receipts.append(DeliveryReceipt.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if len(receipts) >= limit:
                break
        return receipts

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            safe = redact_secrets({"event_type": event_type, "occurred_at": utc_now().isoformat(), **payload})
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe, default=str) + "\n")
            try:
                os.chmod(self.events_path, 0o600)
            except OSError:
                pass

    def rotate(
        self,
        *,
        incident_retention_days: int = 30,
        probe_retention_days: int = 14,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """Bounded retention so probe logs cannot grow forever."""
        clock = now or utc_now()
        removed = {"incidents": 0, "probes": 0, "receipts": 0}
        incident_cutoff = clock - timedelta(days=incident_retention_days)
        probe_cutoff = clock - timedelta(days=probe_retention_days)
        with self._lock:
            for path in self.incidents_dir.glob("incident_*.json"):
                try:
                    incident = ConnectorIncident.model_validate_json(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if incident.ended_at and incident.ended_at < incident_cutoff:
                    path.unlink(missing_ok=True)
                    removed["incidents"] += 1
            for path in self.probes_dir.glob("*.jsonl"):
                kept = self._filter_jsonl_by_time(path, probe_cutoff, time_key="checked_at")
                if kept is not None:
                    removed["probes"] += kept
            for path in self.receipts_dir.glob("*.json"):
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=clock.tzinfo)
                except OSError:
                    continue
                if mtime < probe_cutoff:
                    path.unlink(missing_ok=True)
                    removed["receipts"] += 1
        return removed

    def _filter_jsonl_by_time(self, path: Path, cutoff: datetime, *, time_key: str) -> int | None:
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        kept: list[str] = []
        dropped = 0
        for line in lines:
            try:
                payload = json.loads(line)
                stamp = datetime.fromisoformat(str(payload.get(time_key)))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=cutoff.tzinfo)
                if stamp >= cutoff:
                    kept.append(line)
                else:
                    dropped += 1
            except Exception:
                kept.append(line)
        if dropped:
            _atomic_write(path, "\n".join(kept) + ("\n" if kept else ""))
        return dropped
