"""Deserialization and conversion helpers for multi-agent core types."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

from mana_agent.multi_agent.core.types import (
    AgentMessage,
    DecisionRecord,
    DecisionStatus,
    DiscussionStatus,
    DiscussionThread,
    HandoffRecord,
    MessageType,
    QueueJob,
    QueueJobStatus,
    QueueJobType,
    RiskLevel,
    TaskBoardItem,
    TaskStatus,
    VerificationResult,
    parse_dt,
    to_jsonable,
)

T = TypeVar("T")


def _enum(enum_cls, value):
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name in payload:
            kwargs[item.name] = payload[item.name]
    return cls(**kwargs)  # type: ignore[arg-type]


def handoff_from_dict(payload: dict[str, Any]) -> HandoffRecord:
    payload = dict(payload)
    payload["created_at"] = parse_dt(payload.get("created_at"))
    return _dataclass_from_dict(HandoffRecord, payload)


def verification_from_dict(payload: dict[str, Any]) -> VerificationResult:
    payload = dict(payload)
    payload["created_at"] = parse_dt(payload.get("created_at"))
    return _dataclass_from_dict(VerificationResult, payload)


def task_from_dict(payload: dict[str, Any]) -> TaskBoardItem:
    payload = dict(payload)
    payload["status"] = _enum(TaskStatus, payload.get("status", TaskStatus.NEW.value))
    payload["risk_level"] = _enum(RiskLevel, payload.get("risk_level", RiskLevel.LOW.value))
    payload["handoff_records"] = [
        handoff_from_dict(item) for item in payload.get("handoff_records", []) if isinstance(item, dict)
    ]
    payload["verification_results"] = [
        verification_from_dict(item)
        for item in payload.get("verification_results", [])
        if isinstance(item, dict)
    ]
    payload["created_at"] = parse_dt(payload.get("created_at"))
    payload["updated_at"] = parse_dt(payload.get("updated_at"))
    return _dataclass_from_dict(TaskBoardItem, payload)


def message_from_dict(payload: dict[str, Any]) -> AgentMessage:
    payload = dict(payload)
    payload["message_type"] = _enum(MessageType, payload.get("message_type"))
    payload["created_at"] = parse_dt(payload.get("created_at"))
    return _dataclass_from_dict(AgentMessage, payload)


def discussion_from_dict(payload: dict[str, Any]) -> DiscussionThread:
    payload = dict(payload)
    payload["status"] = _enum(DiscussionStatus, payload.get("status", DiscussionStatus.OPEN.value))
    payload["created_at"] = parse_dt(payload.get("created_at"))
    payload["updated_at"] = parse_dt(payload.get("updated_at"))
    return _dataclass_from_dict(DiscussionThread, payload)


def decision_from_dict(payload: dict[str, Any]) -> DecisionRecord:
    payload = dict(payload)
    payload["decision_status"] = _enum(DecisionStatus, payload.get("decision_status", DecisionStatus.PROPOSED.value))
    payload["created_at"] = parse_dt(payload.get("created_at"))
    return _dataclass_from_dict(DecisionRecord, payload)


def queue_job_from_dict(payload: dict[str, Any]) -> QueueJob:
    payload = dict(payload)
    payload["job_type"] = _enum(QueueJobType, payload.get("job_type"))
    payload["status"] = _enum(QueueJobStatus, payload.get("status", QueueJobStatus.PENDING.value))
    payload["created_at"] = parse_dt(payload.get("created_at"))
    payload["updated_at"] = parse_dt(payload.get("updated_at"))
    return _dataclass_from_dict(QueueJob, payload)


def serialize(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(value)
    return to_jsonable(value)
