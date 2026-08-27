"""Durable API integration registry under Mana's managed state directory."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mana_agent.api_manager.errors import (
    IntegrationAlreadyExistsError,
    IntegrationNotFoundError,
    OperationNotFoundError,
)
from mana_agent.api_manager.models import ApiIntegration, ApiOperation
from mana_agent.config.settings import mana_home


RegistryEventSink = Callable[[str, dict[str, Any]], None]


class ApiIntegrationRegistry:
    def __init__(
        self,
        path: Path | None = None,
        *,
        event_sink: RegistryEventSink | None = None,
    ) -> None:
        self.path = path or (mana_home() / "api_manager" / "integrations")
        self.event_sink = event_sink
        self._lock = threading.RLock()
        self._ephemeral: dict[str, ApiIntegration] = {}

    def _path(self, integration_id: str) -> Path:
        if not integration_id.startswith("api_") or not integration_id[4:].isalnum():
            raise IntegrationNotFoundError("Invalid integration identifier.")
        return self.path / f"{integration_id}.json"

    def _emit(self, kind: str, integration: ApiIntegration, **details: Any) -> None:
        if self.event_sink:
            self.event_sink(
                kind,
                {
                    "integration_id": integration.integration_id,
                    "version": integration.active_version,
                    **details,
                },
            )

    def save(self, integration: ApiIntegration, *, replace: bool = False) -> ApiIntegration:
        _reject_secret_material(integration.model_dump(mode="json", by_alias=True))
        target = self._path(integration.integration_id)
        with self._lock:
            if target.exists() and not replace:
                raise IntegrationAlreadyExistsError(
                    f"Integration {integration.integration_id!r} already exists; retry the "
                    "documentation import with that exact refresh_integration_id.",
                    details={
                        "refresh_integration_id": integration.integration_id,
                        "integration_id": integration.integration_id,
                    },
                )
            self.path.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path, 0o700)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                integration.model_dump_json(indent=2, by_alias=True),
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            temporary.replace(target)
        self._emit("api.integration.saved", integration)
        return integration

    def get(self, integration_id: str) -> ApiIntegration:
        ephemeral = self._ephemeral.get(integration_id)
        if ephemeral is not None:
            return ephemeral.model_copy(deep=True)
        target = self._path(integration_id)
        try:
            return ApiIntegration.model_validate_json(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise IntegrationNotFoundError(
                f"API integration {integration_id!r} was not found.",
                details={"integration_id": integration_id},
            ) from exc
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise IntegrationNotFoundError(
                f"API integration {integration_id!r} could not be loaded safely.",
                details={"integration_id": integration_id},
            ) from exc

    def list(self, *, include_disabled: bool = True) -> list[ApiIntegration]:
        records: list[ApiIntegration] = [
            item.model_copy(deep=True) for item in self._ephemeral.values()
        ]
        targets = sorted(self.path.glob("api_*.json")) if self.path.exists() else ()
        for target in targets:
            try:
                integration = ApiIntegration.model_validate_json(target.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise IntegrationNotFoundError(
                    f"API integration record {target.name!r} is malformed; registry listing stopped."
                ) from exc
            if include_disabled or integration.enabled:
                records.append(integration)
        return records

    def save_ephemeral(self, integration: ApiIntegration) -> ApiIntegration:
        transient = integration.model_copy(update={"ephemeral": True}, deep=True)
        self._ephemeral[transient.integration_id] = transient
        self._emit("api.integration.ephemeral_created", transient)
        return transient

    def discard_ephemeral(self, integration_id: str) -> None:
        self._ephemeral.pop(integration_id, None)

    def update(self, integration_id: str, changes: dict[str, Any]) -> ApiIntegration:
        forbidden = {"integration_id", "created_at", "versions", "active_version"}
        unknown_forbidden = forbidden.intersection(changes)
        if unknown_forbidden:
            raise ValueError(f"Immutable integration fields: {', '.join(sorted(unknown_forbidden))}")
        existing = self.get(integration_id)
        payload = existing.model_dump(mode="python", by_alias=True)
        payload.update({**changes, "updated_at": datetime.now(timezone.utc)})
        updated = ApiIntegration.model_validate(payload)
        self.save(updated, replace=True)
        self._emit("api.integration.updated", updated, changed_fields=sorted(changes))
        return updated

    def enable(self, integration_id: str) -> ApiIntegration:
        return self.update(integration_id, {"enabled": True})

    def disable(self, integration_id: str) -> ApiIntegration:
        return self.update(integration_id, {"enabled": False})

    def refresh(
        self,
        integration_id: str,
        imported: ApiIntegration,
    ) -> ApiIntegration:
        existing = self.get(integration_id)
        next_number = max(version.number for version in existing.versions) + 1
        source_digest = imported.documentation_sources[-1].content_sha256
        version = imported.versions[-1].model_copy(
            update={"number": next_number, "source_sha256": source_digest}
        )
        payload = imported.model_dump(mode="python", by_alias=True)
        payload.update(
            {
                "integration_id": existing.integration_id,
                "name": existing.name,
                "enabled": existing.enabled,
                "created_at": existing.created_at,
                "updated_at": datetime.now(timezone.utc),
                "documentation_sources": (
                    *existing.documentation_sources,
                    *imported.documentation_sources,
                ),
                "versions": (*existing.versions, version),
                "active_version": next_number,
            }
        )
        refreshed = ApiIntegration.model_validate(payload)
        self.save(refreshed, replace=True)
        self._emit("api.integration.refreshed", refreshed)
        return refreshed

    def delete(self, integration_id: str, *, explicit: bool) -> dict[str, Any]:
        if not explicit:
            raise PermissionError("Explicit delete intent is required for an API integration.")
        integration = self.get(integration_id)
        if integration.ephemeral:
            self._ephemeral.pop(integration_id, None)
            self._emit("api.integration.deleted", integration)
            return {"integration_id": integration_id, "deleted": True}
        target = self._path(integration_id)
        with self._lock:
            target.unlink()
        self._emit("api.integration.deleted", integration)
        return {"integration_id": integration_id, "deleted": True}

    def operation(self, integration_id: str, operation_id: str) -> tuple[ApiIntegration, ApiOperation]:
        integration = self.get(integration_id)
        operation = next(
            (item for item in integration.operations if item.operation_id == operation_id),
            None,
        )
        if operation is None:
            raise OperationNotFoundError(
                f"Operation {operation_id!r} was not found in integration {integration_id!r}.",
                details={"integration_id": integration_id, "operation_id": operation_id},
            )
        return integration, operation


def _reject_secret_material(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {
                "api_key",
                "apikey",
                "access_token",
                "refresh_token",
                "bearer_token",
                "client_secret",
                "password",
                "secret_value",
            }:
                raise ValueError(
                    f"Raw secret material is forbidden in API integration records ({path}.{key}). "
                    "Store only a credential_reference."
                )
            _reject_secret_material(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secret_material(item, path=f"{path}[{index}]")
