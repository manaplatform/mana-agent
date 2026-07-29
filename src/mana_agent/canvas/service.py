"""Canvas domain service bridging validated state, persistence, events, and actions."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.canvas.catalog import (
    CatalogValidationError,
    validate_components,
    validate_data_model,
)
from mana_agent.canvas.config import CanvasConfig, MANA_CATALOG_ID, WIRE_VERSION
from mana_agent.canvas.models import (
    CanvasActionResult,
    CanvasEventEnvelope,
    CanvasEventType,
    CanvasSource,
    Component,
    OwnerRef,
    RendererAction,
    SurfaceSnapshot,
)
from mana_agent.canvas.reducer import CanvasStateError, reduce_canvas_event
from mana_agent.canvas.store import CanvasStore
from mana_agent.services.execution_event_hub import (
    ExecutionEventHub,
    get_execution_event_hub,
)


ActionHandler = Callable[[RendererAction, SurfaceSnapshot], None]
PermissionAuthorizer = Callable[[RendererAction, SurfaceSnapshot, str], str]
PermissionVerifier = Callable[[RendererAction, SurfaceSnapshot, str], bool]


class CanvasService:
    """Only supported path for publishing or acting on A2UI surfaces."""

    def __init__(
        self,
        *,
        config: CanvasConfig | None = None,
        store: CanvasStore | None = None,
        event_hub: ExecutionEventHub | None = None,
        repository_id: str = "",
        permission_authorizer: PermissionAuthorizer | None = None,
        permission_verifier: PermissionVerifier | None = None,
    ) -> None:
        self.config = config or CanvasConfig()
        self.config.validate()
        self.store = store or CanvasStore()
        self.event_hub = event_hub or get_execution_event_hub()
        self.repository_id = repository_id
        self.permission_authorizer = permission_authorizer
        self.permission_verifier = permission_verifier
        self._lock = threading.RLock()
        self._handlers: dict[tuple[str, str], ActionHandler] = {}
        self._actions: dict[tuple[str, str, str], deque[RendererAction]] = defaultdict(
            deque
        )
        self._conditions: dict[tuple[str, str, str], threading.Condition] = {}
        self._update_times: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._validation_failures: dict[tuple[str, str, str], int] = defaultdict(int)

    def create_surface(
        self,
        *,
        session_id: str,
        conversation_id: str,
        surface_id: str,
        owner: OwnerRef | dict[str, Any],
        correlation_id: str,
        source: CanvasSource = CanvasSource.AGENT,
        catalog_id: str = MANA_CATALOG_ID,
        retain_on_complete: bool = True,
        workflow_id: str | None = None,
        node_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> SurfaceSnapshot:
        self._require_enabled()
        active = [
            item
            for item in self.list_surfaces(session_id)
            if not item.deleted and not self._expired(item)
        ]
        if len(active) >= self.config.max_active_surfaces_per_session:
            raise CanvasStateError("Session has reached its active surface limit.")
        if self.store.load_snapshot(session_id, surface_id) is not None:
            raise CanvasStateError(
                "Surface already exists; delete it before reusing the identifier."
            )
        owner_ref = (
            owner if isinstance(owner, OwnerRef) else OwnerRef.model_validate(owner)
        )
        payload = {
            "version": WIRE_VERSION,
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": catalog_id,
                "sendDataModel": True,
            },
        }
        return self._publish(
            session_id=session_id,
            conversation_id=conversation_id,
            surface_id=surface_id,
            correlation_id=correlation_id,
            source=source,
            event_type=CanvasEventType.CREATE,
            payload=payload,
            workflow_id=workflow_id or owner_ref.workflow_id,
            node_id=node_id or owner_ref.node_id,
            task_id=task_id or owner_ref.task_id,
            agent_id=agent_id or owner_ref.agent_id,
            automation_id=owner_ref.automation_id,
            retain_on_complete=retain_on_complete,
        )

    def update_components(
        self,
        *,
        session_id: str,
        conversation_id: str,
        surface_id: str,
        components: Iterable[Component | dict[str, Any]],
        correlation_id: str,
        source: CanvasSource = CanvasSource.AGENT,
        **routing: Any,
    ) -> SurfaceSnapshot:
        failure_key = (session_id, surface_id, correlation_id)
        self._assert_validation_budget(failure_key)
        try:
            rows = validate_components(
                components,
                surface_id=surface_id,
                config=self.config,
                require_root=False,
            )
        except (ValueError, CatalogValidationError):
            self._validation_failures[failure_key] += 1
            raise
        payload = {
            "version": WIRE_VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    item.model_dump(mode="json", exclude_none=True) for item in rows
                ],
            },
        }
        return self._publish(
            session_id=session_id,
            conversation_id=conversation_id,
            surface_id=surface_id,
            correlation_id=correlation_id,
            source=source,
            event_type=CanvasEventType.COMPONENTS,
            payload=payload,
            **routing,
        )

    def update_data(
        self,
        *,
        session_id: str,
        conversation_id: str,
        surface_id: str,
        value: dict[str, Any],
        correlation_id: str,
        path: str = "/",
        source: CanvasSource = CanvasSource.AGENT,
        **routing: Any,
    ) -> SurfaceSnapshot:
        failure_key = (session_id, surface_id, correlation_id)
        self._assert_validation_budget(failure_key)
        if path in {"", "/"}:
            try:
                validate_data_model(value, surface_id=surface_id)
            except (ValueError, CatalogValidationError):
                self._validation_failures[failure_key] += 1
                raise
        payload = {
            "version": WIRE_VERSION,
            "updateDataModel": {"surfaceId": surface_id, "path": path, "value": value},
        }
        return self._publish(
            session_id=session_id,
            conversation_id=conversation_id,
            surface_id=surface_id,
            correlation_id=correlation_id,
            source=source,
            event_type=CanvasEventType.DATA,
            payload=payload,
            **routing,
        )

    def delete_surface(
        self,
        *,
        session_id: str,
        conversation_id: str,
        surface_id: str,
        correlation_id: str,
        source: CanvasSource = CanvasSource.AGENT,
        **routing: Any,
    ) -> SurfaceSnapshot:
        return self._publish(
            session_id=session_id,
            conversation_id=conversation_id,
            surface_id=surface_id,
            correlation_id=correlation_id,
            source=source,
            event_type=CanvasEventType.DELETE,
            payload={
                "version": WIRE_VERSION,
                "deleteSurface": {"surfaceId": surface_id},
            },
            **routing,
        )

    def complete_surface(
        self,
        *,
        session_id: str,
        conversation_id: str,
        surface_id: str,
        correlation_id: str,
        source: CanvasSource = CanvasSource.AGENT,
        **routing: Any,
    ) -> SurfaceSnapshot:
        return self._publish(
            session_id=session_id,
            conversation_id=conversation_id,
            surface_id=surface_id,
            correlation_id=correlation_id,
            source=source,
            event_type=CanvasEventType.COMPLETE,
            payload={
                "version": WIRE_VERSION,
                "streamComplete": {"surfaceId": surface_id},
            },
            **routing,
        )

    def get_surface(self, session_id: str, surface_id: str) -> SurfaceSnapshot:
        snapshot = self.store.load_snapshot(session_id, surface_id)
        if snapshot is None:
            raise CanvasStateError("Unknown canvas surface.")
        if self._expired(snapshot):
            raise CanvasStateError("Canvas surface has expired.")
        return snapshot

    def list_surfaces(
        self, session_id: str, *, include_deleted: bool = False
    ) -> list[SurfaceSnapshot]:
        rows = self.store.list_snapshots(session_id)
        if self.store.remove_expired(rows):
            rows = self.store.list_snapshots(session_id)
        return [item for item in rows if include_deleted or not item.deleted]

    def replay(
        self, session_id: str, surface_id: str, *, after_sequence: int = 0
    ) -> tuple[SurfaceSnapshot, list[CanvasEventEnvelope]]:
        snapshot = self.get_surface(session_id, surface_id)
        events = self.store.events(
            session_id,
            surface_id,
            after_sequence=max(after_sequence, snapshot.last_sequence),
        )
        return snapshot, events

    def register_action_handler(
        self, owner: OwnerRef, handler: ActionHandler
    ) -> Callable[[], None]:
        keys = self._owner_keys(owner)
        with self._lock:
            for key in keys:
                self._handlers[key] = handler

        def unregister() -> None:
            with self._lock:
                for key in keys:
                    if self._handlers.get(key) is handler:
                        self._handlers.pop(key, None)

        return unregister

    def submit_action(
        self, action: RendererAction | dict[str, Any]
    ) -> CanvasActionResult:
        self._require_enabled()
        item = (
            action
            if isinstance(action, RendererAction)
            else RendererAction.model_validate(action)
        )
        snapshot = self.get_surface(item.session_id, item.surface_id)
        if item.conversation_id != snapshot.conversation_id:
            raise CanvasStateError("Cross-conversation canvas action rejected.")
        if self.store.action_seen(item.session_id, item.surface_id, item.action_id):
            raise CanvasStateError("Replayed canvas action rejected.")
        age = (datetime.now(timezone.utc) - item.timestamp).total_seconds()
        if age < -30 or age > self.config.action_timeout_seconds:
            raise CanvasStateError(
                "Canvas action timestamp is outside the accepted window."
            )
        component = next(
            (row for row in snapshot.components if row.id == item.source_component_id),
            None,
        )
        if component is None:
            raise CanvasStateError("Canvas action references an unknown component.")
        declaration = next(
            (row for row in component.actions if row.name == item.name), None
        )
        if declaration is None:
            raise CanvasStateError("Canvas action is not declared for this component.")
        unexpected = set(item.context) - set(declaration.context)
        if unexpected:
            raise CanvasStateError("Canvas action contains undeclared context fields.")
        encoded = json.dumps(item.context, ensure_ascii=False, default=str)
        if len(encoded.encode("utf-8")) > min(
            self.config.max_event_payload_bytes, 65_536
        ):
            raise CanvasStateError(
                "Canvas action context exceeds the configured limit."
            )
        validate_data_model(dict(item.context), surface_id=item.surface_id)

        if declaration.side_effect:
            if self.permission_authorizer is None:
                raise CanvasStateError(
                    "Side-effecting canvas action requires the permission broker."
                )
            request_id = self.permission_authorizer(
                item, snapshot, str(declaration.permission_scope)
            )
            self.store.record_action(
                item,
                status="permission_required",
                permission_request_id=request_id,
            )
            self._emit_action_activity(
                item, snapshot, "permission_required", request_id
            )
            return CanvasActionResult(
                action_id=item.action_id,
                status="permission_required",
                routed_to=snapshot.owner,
                permission_request_id=request_id,
            )

        self.store.record_action(item, status="accepted")
        self._deliver_action(item, snapshot)
        return CanvasActionResult(
            action_id=item.action_id, status="delivered", routed_to=snapshot.owner
        )

    def deliver_authorized_action(
        self, action: RendererAction, *, permission_request_id: str
    ) -> CanvasActionResult:
        snapshot = self.get_surface(action.session_id, action.surface_id)
        record = self.store.action_record(
            action.session_id, action.surface_id, action.action_id
        )
        expected_action = action.model_dump(mode="json")
        if (
            record is None
            or record.get("status") != "permission_required"
            or record.get("permission_request_id") != permission_request_id
            or any(record.get(key) != value for key, value in expected_action.items())
        ):
            raise CanvasStateError(
                "Canvas permission request does not match the pending action."
            )
        if self.permission_verifier is None:
            raise CanvasStateError(
                "Authorized canvas delivery requires the permission verifier."
            )
        if not self.permission_verifier(action, snapshot, permission_request_id):
            raise CanvasStateError("Canvas permission verification failed.")
        self.store.record_action(action, status="delivered")
        self._deliver_action(action, snapshot)
        return CanvasActionResult(
            action_id=action.action_id, status="delivered", routed_to=snapshot.owner
        )

    def wait_for_action(
        self,
        *,
        session_id: str,
        surface_id: str,
        action_name: str,
        timeout: float | None = None,
    ) -> RendererAction:
        key = (session_id, surface_id, action_name)
        wait_seconds = min(
            float(
                timeout if timeout is not None else self.config.action_timeout_seconds
            ),
            float(self.config.action_timeout_seconds),
        )
        with self._lock:
            condition = self._conditions.setdefault(
                key, threading.Condition(self._lock)
            )
            deadline = time.monotonic() + max(0.0, wait_seconds)
            while not self._actions[key]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a canvas action.")
                condition.wait(remaining)
            return self._actions[key].popleft()

    def _publish(self, **values: Any) -> SurfaceSnapshot:
        self._require_enabled()
        session_id = str(values["session_id"])
        surface_id = str(values["surface_id"])
        payload = dict(values["payload"])
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        if len(encoded) > self.config.max_event_payload_bytes:
            raise CanvasStateError("Canvas event payload exceeds the configured limit.")
        with self._lock:
            self._enforce_rate(session_id, surface_id)
            failure_key = (session_id, surface_id, str(values["correlation_id"]))
            if values["event_type"] != CanvasEventType.DELETE:
                self._assert_validation_budget(failure_key)
            current = self.store.load_snapshot(session_id, surface_id)
            event = CanvasEventEnvelope(
                **values,
                sequence=1 if current is None else current.last_sequence + 1,
            )
            try:
                updated = reduce_canvas_event(current, event, config=self.config)
            except (CanvasStateError, CatalogValidationError) as exc:
                self._validation_failures[failure_key] += 1
                self._emit_validation_failure(event, exc)
                raise
            self.store.append_event(event)
            checkpoint = (
                event.sequence == 1
                or event.sequence % self.config.snapshot_interval == 0
                or event.event_type
                in {CanvasEventType.DELETE, CanvasEventType.COMPLETE}
            )
            self.store.save_snapshot(updated, checkpoint=checkpoint)
            self._validation_failures.pop(failure_key, None)
        self._emit_canvas_event(event, updated)
        return updated

    def _deliver_action(
        self, action: RendererAction, snapshot: SurfaceSnapshot
    ) -> None:
        key = (action.session_id, action.surface_id, action.name)
        with self._lock:
            self._actions[key].append(action)
            condition = self._conditions.get(key)
            if condition:
                condition.notify_all()
            handlers = [
                self._handlers[item]
                for item in self._owner_keys(snapshot.owner)
                if item in self._handlers
            ]
        for handler in dict.fromkeys(handlers):
            handler(action, snapshot)
        self._emit_action_activity(action, snapshot, "delivered", None)

    def _emit_canvas_event(
        self, event: CanvasEventEnvelope, snapshot: SurfaceSnapshot
    ) -> None:
        self.event_hub.emit(
            f"canvas.{event.event_type.value}",
            title=f"Canvas {event.event_type.value}",
            conversation_id=event.conversation_id,
            execution_id=event.correlation_id,
            repository_id=self.repository_id,
            status="success"
            if event.event_type in {CanvasEventType.DELETE, CanvasEventType.COMPLETE}
            else "running",
            agent_id=event.agent_id,
            metadata={
                "kind": "canvas",
                "canvas_event": event.model_dump(mode="json"),
                "surface_id": event.surface_id,
                "surface_version": snapshot.version,
                "owner": snapshot.owner.model_dump(mode="json", exclude_none=True),
            },
            event_id=event.event_id,
        )

    def _emit_validation_failure(
        self, event: CanvasEventEnvelope, error: Exception
    ) -> None:
        details = (
            [item.model_dump(mode="json") for item in error.errors]
            if isinstance(error, CatalogValidationError)
            else [
                {
                    "code": "INVALID_TRANSITION",
                    "message": str(error),
                    "surface_id": event.surface_id,
                }
            ]
        )
        self.event_hub.emit(
            "canvas.validation_failed",
            title="Canvas validation failed",
            conversation_id=event.conversation_id,
            execution_id=event.correlation_id,
            repository_id=self.repository_id,
            status="failed",
            metadata={
                "kind": "canvas",
                "surface_id": event.surface_id,
                "errors": details,
            },
        )

    def _emit_action_activity(
        self,
        action: RendererAction,
        snapshot: SurfaceSnapshot,
        status: str,
        permission_request_id: str | None,
    ) -> None:
        self.event_hub.emit(
            "canvas.action",
            title="Canvas action",
            conversation_id=action.conversation_id,
            execution_id=action.correlation_id,
            repository_id=self.repository_id,
            status="running" if status == "permission_required" else "success",
            metadata={
                "kind": "canvas",
                "surface_id": action.surface_id,
                "action_id": action.action_id,
                "action_name": action.name,
                "component_id": action.source_component_id,
                "routing_status": status,
                "permission_request_id": permission_request_id,
                "owner": snapshot.owner.model_dump(mode="json", exclude_none=True),
            },
        )

    def _enforce_rate(self, session_id: str, surface_id: str) -> None:
        now = time.monotonic()
        rows = self._update_times[(session_id, surface_id)]
        while rows and rows[0] <= now - 1.0:
            rows.popleft()
        if len(rows) >= self.config.max_updates_per_second:
            raise CanvasStateError("Canvas update rate exceeds the configured limit.")
        rows.append(now)

    def _assert_validation_budget(self, key: tuple[str, str, str]) -> None:
        if self._validation_failures[key] > self.config.validation_retry_limit:
            raise CanvasStateError("Canvas validation retry limit was exhausted.")

    @staticmethod
    def _owner_keys(owner: OwnerRef) -> list[tuple[str, str]]:
        return [
            (name, value)
            for name, value in (
                ("agent", owner.agent_id),
                ("task", owner.task_id),
                ("workflow", owner.workflow_id),
                ("node", owner.node_id),
                ("automation", owner.automation_id),
            )
            if value
        ]

    @staticmethod
    def _expired(snapshot: SurfaceSnapshot) -> bool:
        return snapshot.expires_at <= datetime.now(timezone.utc)

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise CanvasStateError("Live Canvas is disabled.")


def canvas_service_for_root(
    root: str | Path,
    *,
    config: CanvasConfig | None = None,
    permission_authorizer: PermissionAuthorizer | None = None,
    permission_verifier: PermissionVerifier | None = None,
) -> CanvasService:
    from mana_agent.workspaces.paths import repository_id_for_path
    from mana_agent.workspaces.paths import mana_home

    resolved = Path(root).resolve()
    repository_id = repository_id_for_path(resolved)
    key = (
        repository_id,
        str(mana_home()),
        str((config or CanvasConfig()).allowed_catalogs),
    )
    with _SERVICE_LOCK:
        service = _SERVICES.get(key)
        if service is None:
            service = CanvasService(
                config=config,
                repository_id=repository_id,
                permission_authorizer=permission_authorizer,
                permission_verifier=permission_verifier,
            )
            _SERVICES[key] = service
        elif permission_authorizer is not None:
            service.permission_authorizer = permission_authorizer
        if permission_verifier is not None:
            service.permission_verifier = permission_verifier
        return service


_SERVICES: dict[tuple[str, str, str], CanvasService] = {}
_SERVICE_LOCK = threading.Lock()
