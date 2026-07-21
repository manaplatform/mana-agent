from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Protocol

from mana_agent.model_routing.models import RoutingOutcome


class RoutingHistory(Protocol):
    def record(self, outcome: RoutingOutcome) -> None: ...
    def query(self, *, provider: str, model_id: str, task_category: str | None = None) -> tuple[RoutingOutcome, ...]: ...
    def healthy(self) -> bool: ...


class InMemoryRoutingHistory:
    def __init__(self, outcomes: tuple[RoutingOutcome, ...] = ()) -> None:
        self._outcomes = list(outcomes)

    def record(self, outcome: RoutingOutcome) -> None:
        self._outcomes.append(outcome)

    def query(self, *, provider: str, model_id: str, task_category: str | None = None) -> tuple[RoutingOutcome, ...]:
        return tuple(item for item in self._outcomes if item.provider == provider and item.model_id == model_id and (task_category is None or item.task_category == task_category))

    def healthy(self) -> bool:
        return True


class JsonlRoutingHistory(InMemoryRoutingHistory):
    """Local, append-only evidence store containing metadata but no source text."""

    def __init__(self, path: Path, *, retention_days: int = 90) -> None:
        self.path = path
        self.retention_days = max(1, retention_days)
        self._load_error = ""
        super().__init__(self._load())

    def _load(self) -> tuple[RoutingOutcome, ...]:
        if not self.path.exists():
            return ()
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        outcomes: list[RoutingOutcome] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                data = json.loads(line)
                occurred = datetime.fromisoformat(str(data.pop("occurred_at")).replace("Z", "+00:00"))
                item = RoutingOutcome(**data, occurred_at=occurred)
                if item.occurred_at >= cutoff:
                    outcomes.append(item)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._load_error = str(exc)
            return ()
        return tuple(outcomes)

    def record(self, outcome: RoutingOutcome) -> None:
        safe = replace(outcome, model_configuration=dict(outcome.safe_dict()["model_configuration"]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe.safe_dict(), sort_keys=True, separators=(",", ":")) + "\n")
        super().record(safe)

    def healthy(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return self.path.parent.is_dir() and not self._load_error
        except OSError:
            return False
