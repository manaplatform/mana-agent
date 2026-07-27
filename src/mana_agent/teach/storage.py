"""Atomic, owner-only Teach Mode persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import ManaFlow, RecordedEvent, TeachError, TeachSession


class LocalTeachStorage:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        for name in ("sessions", "recordings", "flows", "packages", "indexes"):
            path = self.root / name
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                path.chmod(0o700)
            except OSError:
                pass

    def _atomic_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_session(self, session: TeachSession) -> None:
        self._atomic_text(
            self.root / "sessions" / f"{session.id}.json",
            session.model_dump_json(indent=2) + "\n",
        )

    def load_session(self, session_id: str) -> TeachSession:
        return self._load_model(self.root / "sessions" / f"{_safe_id(session_id)}.json", TeachSession)

    def list_sessions(self) -> list[TeachSession]:
        return [self._load_model(path, TeachSession) for path in sorted((self.root / "sessions").glob("teach_*.json"))]

    def append_raw_event(self, event: RecordedEvent) -> None:
        path = self.root / "recordings" / f"{_safe_id(event.session_id)}.raw.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def save_normalized_events(self, session_id: str, events: list[RecordedEvent]) -> None:
        text = "".join(event.model_dump_json() + "\n" for event in events)
        self._atomic_text(self.root / "recordings" / f"{_safe_id(session_id)}.normalized.jsonl", text)

    def load_events(self, session_id: str, *, normalized: bool = False) -> list[RecordedEvent]:
        suffix = "normalized" if normalized else "raw"
        path = self.root / "recordings" / f"{_safe_id(session_id)}.{suffix}.jsonl"
        if not path.exists():
            return []
        try:
            return [RecordedEvent.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, ValidationError, ValueError) as exc:
            raise TeachError(f"Teach recording is unreadable: {exc}") from exc

    def save_flow(self, flow: ManaFlow) -> None:
        directory = self.root / "flows" / _safe_id(flow.id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = flow.model_dump(mode="json", by_alias=True, exclude_none=True)
        self._atomic_text(directory / f"v{flow.version}.yaml", yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        self._atomic_text(directory / "latest", f"{flow.version}\n")
        index = [{"id": item.id, "version": item.version, "name": item.name, "status": item.status} for item in self.list_flows()]
        self._atomic_text(self.root / "indexes" / "flows.json", json.dumps(index, indent=2, sort_keys=True) + "\n")

    def load_flow(self, flow_id: str, version: int | None = None) -> ManaFlow:
        directory = self.root / "flows" / _safe_id(flow_id)
        if version is None:
            try:
                version = int((directory / "latest").read_text(encoding="utf-8").strip())
            except (OSError, ValueError) as exc:
                raise TeachError(f"Flow not found: {flow_id}") from exc
        return self._load_model(directory / f"v{version}.yaml", ManaFlow, yaml_mode=True)

    def list_flows(self) -> list[ManaFlow]:
        result: list[ManaFlow] = []
        for latest in sorted((self.root / "flows").glob("*/latest")):
            result.append(self.load_flow(latest.parent.name))
        return result

    def _load_model(self, path: Path, model: Any, *, yaml_mode: bool = False):
        try:
            text = path.read_text(encoding="utf-8")
            value = yaml.safe_load(text) if yaml_mode else json.loads(text)
            return model.model_validate(value)
        except FileNotFoundError as exc:
            raise TeachError(f"Teach Mode record not found: {path.stem}") from exc
        except (OSError, ValidationError, ValueError, yaml.YAMLError) as exc:
            raise TeachError(f"Teach Mode record is unreadable: {exc}") from exc


def _safe_id(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise TeachError("Invalid Teach Mode identifier.")
    return value
