"""Versioned, JSON-only protocol for reverse-connected Mana workers.

The protocol deliberately contains data, never executable Python objects.  Both
ends validate every frame before it reaches scheduling or execution code.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from mana_agent.remote_execution.models import StrictModel

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576


class MessageType(str, Enum):
    HELLO = "worker.hello"
    CHALLENGE = "worker.challenge"
    AUTHENTICATED = "worker.authenticated"
    HEARTBEAT = "worker.heartbeat"
    CAPABILITIES = "worker.capabilities"
    STATUS = "worker.status"
    OFFER = "job.offer"
    ACCEPTED = "job.accepted"
    REJECTED = "job.rejected"
    STARTED = "job.started"
    PROGRESS = "job.progress"
    STDOUT = "job.stdout"
    STDERR = "job.stderr"
    PERMISSION_REQUIRED = "job.permission_required"
    PERMISSION_RESULT = "job.permission_result"
    ARTIFACT = "job.artifact"
    COMPLETED = "job.completed"
    FAILED = "job.failed"
    CANCEL = "job.cancel"
    CANCELLED = "job.cancelled"
    UPDATE_AVAILABLE = "worker.update_available"
    SHUTDOWN = "worker.shutdown"
    ERROR = "protocol.error"


class WorkerMessage(StrictModel):
    protocol_version: int = Field(default=PROTOCOL_VERSION, ge=1)
    message_id: str = Field(default_factory=lambda: secrets.token_urlsafe(18), min_length=12, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    type: MessageType
    worker_id: str = Field(default="", max_length=128)
    job_id: str = Field(default="", max_length=128)
    correlation_id: str = Field(default="", max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)
    signature: str = Field(default="", max_length=1024)

    @field_validator("payload")
    @classmethod
    def payload_is_bounded_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise ValueError("protocol payload exceeds message size limit")
        return value

    def signing_bytes(self) -> bytes:
        data = self.model_dump(mode="json", exclude={"signature"})
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    @classmethod
    def parse_frame(cls, raw: str | bytes) -> "WorkerMessage":
        data = raw.encode("utf-8") if isinstance(raw, str) else raw
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError("protocol frame exceeds message size limit")
        message = cls.model_validate_json(data)
        if message.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported worker protocol version {message.protocol_version}")
        return message
