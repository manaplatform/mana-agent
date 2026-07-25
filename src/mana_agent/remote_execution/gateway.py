"""Coordinator gateway: enrollment, authenticated WebSockets, and job events."""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from mana_agent.remote_execution.models import RemoteExecutionRequest, WorkerRegistration
from mana_agent.remote_execution.protocol import MessageType, WorkerMessage
from mana_agent.remote_execution.service import RemoteExecutionService
from mana_agent.remote_execution.worker import WorkerRegistry

logger = logging.getLogger(__name__)


class WorkerGatewayConfig(BaseModel):
    enabled: bool = False
    public_url: str = ""
    bind_host: str = "127.0.0.1"
    bind_port: int = Field(default=8765, ge=1, le=65535)
    heartbeat_interval_seconds: int = Field(default=15, ge=5, le=300)
    offline_after_seconds: int = Field(default=45, ge=10, le=900)
    bootstrap_token_ttl_seconds: int = Field(default=900, ge=60, le=86400)
    require_manual_approval: bool = False
    allow_insecure_local_development: bool = False

    def validate_public_url(self) -> None:
        parsed = urlparse(self.public_url)
        if parsed.scheme == "https":
            return
        if self.allow_insecure_local_development and parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
            return
        raise ValueError("worker gateway requires HTTPS; HTTP is only allowed for explicit localhost development")


class EnrollmentRequest(BaseModel):
    token: str = Field(repr=False, min_length=16)
    registration: WorkerRegistration


class EnrollmentResponse(BaseModel):
    worker_id: str
    credential: str = Field(repr=False)
    protocol_version: int = 1
    websocket_url: str


class EnrollmentCreateRequest(BaseModel):
    worker_id: str = Field(min_length=3, max_length=128)
    name: str = Field(default="", max_length=128)
    expires_in_seconds: int | None = Field(default=None, ge=60, le=86400)
    labels: list[str] = Field(default_factory=list, max_length=32)
    capability_restrictions: list[str] = Field(default_factory=list, max_length=32)


@dataclass
class _Session:
    websocket: WebSocket
    outbound: asyncio.Queue[WorkerMessage] = field(default_factory=lambda: asyncio.Queue(maxsize=1000))


class WorkerGateway:
    """One gateway integrated into the API application, never an inbound worker server."""

    def __init__(self, config: WorkerGatewayConfig | None = None, *, registry: WorkerRegistry | None = None,
                 execution: RemoteExecutionService | None = None) -> None:
        self.config = config or WorkerGatewayConfig()
        self.registry = registry or WorkerRegistry()
        self.execution = execution or RemoteExecutionService(workers=self.registry, event_sink=self._execution_event)
        self.sessions: dict[str, _Session] = {}
        self.audit_events: list[dict[str, Any]] = []

    def audit(self, event: str, **data: Any) -> None:
        self.audit_events.append({"event": event, "at": datetime.now(timezone.utc).isoformat(), **data})
        self.audit_events[:] = self.audit_events[-10_000:]

    def create_enrollment(self, *, worker_id: str, name: str = "", ttl_seconds: int | None = None,
                          labels: list[str] | None = None, capability_restrictions: list[str] | None = None) -> str:
        if not self.config.enabled:
            raise RuntimeError("worker gateway is disabled")
        self.config.validate_public_url()
        token = self.registry.issue_enrolment_token(worker_id, ttl_seconds=ttl_seconds or self.config.bootstrap_token_ttl_seconds,
                                                    name=name, labels=labels, capability_restrictions=capability_restrictions)
        self.audit("worker.enrollment_created", worker_id=worker_id, name=name)
        return token

    def enroll(self, request: EnrollmentRequest) -> EnrollmentResponse:
        if not request.registration.public_key_pem:
            raise PermissionError("worker enrollment requires a generated public identity key")
        credential = self.registry.enrol(request.token, request.registration)
        self.audit("worker.enrollment_succeeded", worker_id=request.registration.worker_id)
        return EnrollmentResponse(worker_id=request.registration.worker_id, credential=credential,
                                  websocket_url=self.config.public_url.rstrip("/") + "/api/v1/workers/connect")

    def offer(self, request: RemoteExecutionRequest) -> None:
        session = self.sessions.get(request.worker_id)
        if session is None:
            raise LookupError("selected reverse worker has no active session")
        message = WorkerMessage(type=MessageType.OFFER, worker_id=request.worker_id, job_id=request.job_id,
                                correlation_id=request.job_id, payload={"request": request.model_dump(mode="json")})
        try:
            session.outbound.put_nowait(message)
        except asyncio.QueueFull as exc:
            raise RuntimeError("worker outbound event queue is full") from exc
        self.audit("job.offered", worker_id=request.worker_id, job_id=request.job_id)

    def _execution_event(self, event: Any) -> None:
        self.audit("job.event", job_id=event.job_id, kind=event.kind)

    def mark_offline_workers(self) -> list[str]:
        now = datetime.now(timezone.utc)
        offline: list[str] = []
        for worker_id in list(self.sessions):
            connected = self.registry._workers.get(worker_id)  # registry-owned status read
            if connected and (now - connected.last_heartbeat).total_seconds() > self.config.offline_after_seconds:
                self.registry.disconnect(worker_id)
                self.sessions.pop(worker_id, None)
                self.execution.worker_disconnected(worker_id)
                self.audit("worker.heartbeat_missed", worker_id=worker_id)
                offline.append(worker_id)
        return offline

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        worker_id = ""
        try:
            first = WorkerMessage.parse_frame(await websocket.receive_text())
            if first.type is not MessageType.HELLO or not first.worker_id:
                raise PermissionError("first worker message must be worker.hello")
            credential = str(first.payload.get("credential", ""))
            self.registry.connect(first.worker_id, credential, lambda request: self.offer(request))
            worker_id = first.worker_id
            session = _Session(websocket)
            self.sessions[worker_id] = session
            await websocket.send_text(WorkerMessage(type=MessageType.AUTHENTICATED, worker_id=worker_id,
                                                     correlation_id=first.message_id,
                                                     payload={"connection_generation": self.registry._workers[worker_id].connection_generation}).model_dump_json())
            self.audit("worker.authenticated", worker_id=worker_id)

            async def sender() -> None:
                while True:
                    await websocket.send_text((await session.outbound.get()).model_dump_json())

            sender_task = asyncio.create_task(sender())
            try:
                while True:
                    message = WorkerMessage.parse_frame(await websocket.receive_text())
                    if message.worker_id != worker_id:
                        raise PermissionError("worker ID changed during session")
                    if not self.registry.accept_message(worker_id, message.message_id):
                        continue
                    if message.signature:
                        self.registry.verify_signature(worker_id, message.signing_bytes(), base64.b64decode(message.signature))
                    if message.type is MessageType.HEARTBEAT:
                        self.registry.heartbeat(worker_id)
                    elif message.type in {MessageType.COMPLETED, MessageType.FAILED, MessageType.CANCELLED}:
                        job = self.execution.jobs.get(message.job_id)
                        if job and job.state not in {job.state.SUCCEEDED, job.state.FAILED, job.state.CANCELLED}:
                            job.state = {MessageType.COMPLETED: job.state.SUCCEEDED, MessageType.FAILED: job.state.FAILED,
                                         MessageType.CANCELLED: job.state.CANCELLED}[message.type]
                    self.audit(message.type.value, worker_id=worker_id, job_id=message.job_id)
            finally:
                sender_task.cancel()
        except (WebSocketDisconnect, asyncio.CancelledError):
            raise
        except Exception as exc:
            logger.info("reverse worker connection rejected: %s", exc)
            try:
                await websocket.send_text(WorkerMessage(type=MessageType.ERROR, worker_id=worker_id,
                                                         payload={"reason": str(exc)}).model_dump_json())
            except Exception:
                pass
        finally:
            if worker_id:
                self.registry.disconnect(worker_id)
                self.sessions.pop(worker_id, None)
                self.execution.worker_disconnected(worker_id)
                self.audit("worker.disconnected", worker_id=worker_id)


def build_worker_router(gateway: WorkerGateway) -> APIRouter:
    router = APIRouter(prefix="/api/v1/workers", tags=["workers"])

    def require_mutation_token(authorization: str | None) -> None:
        # Reuse the API's standard mutation token; enrollment tokens themselves
        # are not authorization to mint further enrollment tokens.
        from mana_agent.api.routes.workspaces import _require_mutation_token
        _require_mutation_token(authorization)

    @router.post("/enroll", response_model=EnrollmentResponse)
    def enroll(payload: EnrollmentRequest) -> EnrollmentResponse:
        try:
            return gateway.enroll(payload)
        except PermissionError as exc:
            gateway.audit("worker.enrollment_failed")
            raise HTTPException(status_code=401, detail="worker enrollment was rejected") from exc

    @router.post("/enrollments")
    def create_enrollment(payload: EnrollmentCreateRequest, authorization: str | None = Header(None)) -> dict[str, Any]:
        require_mutation_token(authorization)
        try:
            token = gateway.create_enrollment(worker_id=payload.worker_id, name=payload.name,
                                              ttl_seconds=payload.expires_in_seconds, labels=payload.labels,
                                              capability_restrictions=payload.capability_restrictions)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"worker_id": payload.worker_id, "token": token, "expires_in_seconds": payload.expires_in_seconds or gateway.config.bootstrap_token_ttl_seconds,
                "install_command": f"mana-agent worker install --coordinator {gateway.config.public_url} --token <sensitive-token> --name {payload.name or payload.worker_id}"}

    @router.websocket("/connect")
    async def connect(websocket: WebSocket) -> None:
        await gateway.connect(websocket)

    @router.get("")
    def list_workers() -> list[dict[str, Any]]:
        return [{"worker_id": worker_id, "status": item.status, "last_heartbeat": item.last_heartbeat,
                 "registration": item.registration.model_dump(mode="json"), "revoked": item.revoked}
                for worker_id, item in gateway.registry._workers.items()]

    @router.get("/{worker_id}")
    def show_worker(worker_id: str) -> dict[str, Any]:
        worker = gateway.registry._workers.get(worker_id)
        if worker is None:
            raise HTTPException(status_code=404, detail="worker not found")
        return {"worker_id": worker_id, "status": worker.status, "last_heartbeat": worker.last_heartbeat,
                "registration": worker.registration.model_dump(mode="json"), "revoked": worker.revoked,
                "connection_generation": worker.connection_generation}

    @router.post("/{worker_id}/revoke")
    def revoke_worker(worker_id: str, authorization: str | None = Header(None)) -> dict[str, bool]:
        require_mutation_token(authorization)
        if gateway.registry.registration(worker_id) is None:
            raise HTTPException(status_code=404, detail="worker not found")
        gateway.registry.revoke(worker_id)
        gateway.audit("worker.revoked", worker_id=worker_id)
        return {"revoked": True}

    @router.post("/{worker_id}/rotate-identity")
    def rotate_identity(worker_id: str, authorization: str | None = Header(None)) -> dict[str, str]:
        require_mutation_token(authorization)
        try:
            credential = gateway.registry.rotate_credential(worker_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="worker not found or revoked") from exc
        gateway.audit("worker.identity_rotated", worker_id=worker_id)
        return {"worker_id": worker_id, "credential": credential}

    return router
