"""Independent, scope-separated Mana and operating-system permission checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from mana_agent.config.settings import mana_home
from mana_agent.integrations.computer_control.config import ComputerControlSettings
from mana_agent.integrations.computer_control.errors import ManaPermissionDenied
from mana_agent.integrations.computer_control.models import PermissionDecision, PermissionStatus


class PermissionService:
    def __init__(
        self,
        settings: ComputerControlSettings,
        *,
        store_path: Path | None = None,
        os_checker: Callable[[str], PermissionStatus] | None = None,
    ) -> None:
        self.settings = settings
        self.store_path = store_path or mana_home() / "computer_control_permissions.json"
        self.os_checker = os_checker
        self._once: set[str] = set()
        self._session: set[str] = set()
        self._persistent = self._load()

    def _load(self) -> dict[str, str]:
        if not self.store_path.exists():
            return {}
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(key): str(value) for key, value in raw.items()} if isinstance(raw, dict) else {}

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.store_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._persistent, indent=2, sort_keys=True), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.store_path)

    def decision_for(self, scope: str) -> PermissionDecision:
        if scope in self._once:
            return PermissionDecision.ALLOW_ONCE
        if scope in self._session:
            return PermissionDecision.ALLOW_SESSION
        raw = self._persistent.get(scope, self.settings.permissions.get(scope, "ask"))
        if raw in {"allow_once", "allow_session"}:
            # Ephemeral grants are never restored from persistent configuration.
            raw = "ask"
        try:
            return PermissionDecision(raw)
        except ValueError:
            return PermissionDecision.DENIED

    def status(self, scope: str) -> PermissionStatus:
        decision = self.decision_for(scope)
        granted = decision in {
            PermissionDecision.ALLOW_ONCE,
            PermissionDecision.ALLOW_SESSION,
            PermissionDecision.ALWAYS_ALLOW,
        }
        if granted and self.os_checker is not None:
            os_status = self.os_checker(scope)
            if not os_status.granted:
                return os_status
        return PermissionStatus(scope=scope, decision=decision, granted=granted)

    def grant(self, scope: str, decision: PermissionDecision) -> PermissionStatus:
        if decision is PermissionDecision.ALLOW_ONCE:
            self._once.add(scope)
        elif decision is PermissionDecision.ALLOW_SESSION:
            self._session.add(scope)
        elif decision is PermissionDecision.ALWAYS_ALLOW:
            self._persistent[scope] = decision.value
            self._save()
        elif decision in {PermissionDecision.DENIED, PermissionDecision.ASK_EVERY_TIME}:
            self._persistent[scope] = decision.value
            self._save()
        else:
            raise ValueError(f"{decision.value!r} cannot be granted by Mana-Agent")
        return self.status(scope)

    def require(self, scope: str) -> PermissionStatus:
        status = self.status(scope)
        if not status.granted:
            raise ManaPermissionDenied(
                f"Mana-Agent permission {scope!r} is {status.decision.value}.",
                corrective_action=f"Grant only the {scope} scope in computer-control settings.",
            )
        return status

    def consume_once(self, scope: str, status: PermissionStatus) -> None:
        if status.decision is PermissionDecision.ALLOW_ONCE:
            self._once.discard(scope)

    def clear_session(self) -> None:
        self._once.clear()
        self._session.clear()
