"""Durable event and snapshot store using Mana's user-level state root."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Iterable

from mana_agent.canvas.models import (
    CanvasEventEnvelope,
    RendererAction,
    SurfaceSnapshot,
)
from mana_agent.workspaces.paths import mana_home
from mana_agent.workspaces.store import atomic_write_json


class CanvasStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or mana_home() / "canvas").expanduser().resolve()
        self._lock = threading.RLock()

    def _session_dir(self, session_id: str) -> Path:
        key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / "sessions" / key

    def _surface_dir(self, session_id: str, surface_id: str) -> Path:
        key = hashlib.sha256(surface_id.encode("utf-8")).hexdigest()
        return self._session_dir(session_id) / "surfaces" / key

    def append_event(self, event: CanvasEventEnvelope) -> None:
        path = self._surface_dir(event.session_id, event.surface_id) / "events.jsonl"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")

    def save_snapshot(
        self, snapshot: SurfaceSnapshot, *, checkpoint: bool = False
    ) -> None:
        path = (
            self._surface_dir(snapshot.session_id, snapshot.surface_id)
            / "snapshot.json"
        )
        with self._lock:
            atomic_write_json(path, snapshot.model_dump(mode="json"))
            if checkpoint:
                checkpoint_path = (
                    path.parent / "snapshots" / f"{snapshot.last_sequence:020d}.json"
                )
                atomic_write_json(checkpoint_path, snapshot.model_dump(mode="json"))
            self._update_index(snapshot)

    def load_snapshot(self, session_id: str, surface_id: str) -> SurfaceSnapshot | None:
        path = self._surface_dir(session_id, surface_id) / "snapshot.json"
        if not path.exists():
            return None
        try:
            return SurfaceSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("Canvas snapshot recovery failed.") from exc

    def events(
        self, session_id: str, surface_id: str, *, after_sequence: int = 0
    ) -> list[CanvasEventEnvelope]:
        path = self._surface_dir(session_id, surface_id) / "events.jsonl"
        if not path.exists():
            return []
        rows: list[CanvasEventEnvelope] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError("Canvas event recovery failed.") from exc
        for line in lines:
            try:
                event = CanvasEventEnvelope.model_validate_json(line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "Canvas event log contains an invalid record."
                ) from exc
            if event.sequence > after_sequence:
                rows.append(event)
        return sorted(rows, key=lambda item: item.sequence)

    def list_snapshots(self, session_id: str) -> list[SurfaceSnapshot]:
        path = self._session_dir(session_id) / "index.json"
        if not path.exists():
            return []
        try:
            identifiers = json.loads(path.read_text(encoding="utf-8")).get(
                "surface_ids", []
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Canvas session index recovery failed.") from exc
        return [
            snapshot
            for surface_id in identifiers
            if (snapshot := self.load_snapshot(session_id, surface_id))
        ]

    def record_action(
        self,
        action: RendererAction,
        *,
        status: str,
        permission_request_id: str | None = None,
    ) -> None:
        path = self._surface_dir(action.session_id, action.surface_id) / "actions.jsonl"
        record = {
            **action.model_dump(mode="json"),
            "status": status,
            "permission_request_id": permission_request_id,
        }
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def action_seen(self, session_id: str, surface_id: str, action_id: str) -> bool:
        return self.action_record(session_id, surface_id, action_id) is not None

    def action_record(
        self, session_id: str, surface_id: str, action_id: str
    ) -> dict[str, object] | None:
        path = self._surface_dir(session_id, surface_id) / "actions.jsonl"
        if not path.exists():
            return None
        try:
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            return next(
                (
                    record
                    for record in reversed(records)
                    if record.get("action_id") == action_id
                ),
                None,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Canvas action log contains an invalid record.") from exc

    def remove_expired(self, snapshots: Iterable[SurfaceSnapshot]) -> int:
        from datetime import datetime, timezone
        import shutil

        now = datetime.now(timezone.utc)
        removed = 0
        for snapshot in snapshots:
            if snapshot.expires_at <= now:
                directory = self._surface_dir(snapshot.session_id, snapshot.surface_id)
                if directory.exists():
                    shutil.rmtree(directory)
                    removed += 1
        return removed

    def delete_session(self, session_id: str) -> None:
        """Delete Canvas state only as part of the canonical session deletion flow."""
        import shutil

        directory = self._session_dir(session_id)
        if directory.exists():
            shutil.rmtree(directory)

    def _update_index(self, snapshot: SurfaceSnapshot) -> None:
        path = self._session_dir(snapshot.session_id) / "index.json"
        identifiers: list[str] = []
        if path.exists():
            try:
                identifiers = list(
                    json.loads(path.read_text(encoding="utf-8")).get("surface_ids", [])
                )
            except (OSError, json.JSONDecodeError):
                identifiers = []
        if snapshot.surface_id not in identifiers:
            identifiers.append(snapshot.surface_id)
        atomic_write_json(
            path, {"session_id": snapshot.session_id, "surface_ids": identifiers}
        )
