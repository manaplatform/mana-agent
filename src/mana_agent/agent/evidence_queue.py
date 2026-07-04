from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


EvidenceAction = Literal["search", "read", "patch", "verify", "summarize"]
EvidenceStatus = Literal["pending", "running", "done", "skipped", "failed"]


@dataclass(slots=True)
class EvidenceQueueItem:
    action_type: EvidenceAction
    target: str
    reason: str
    priority: int = 50
    required: bool = False
    status: EvidenceStatus = "pending"
    evidence: dict[str, Any] = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    retry_count: int = 0
    skip_reason: str = ""


class EvidenceQueue:
    def __init__(self, items: list[EvidenceQueueItem] | None = None) -> None:
        self._items = list(items or [])

    def add(self, item: EvidenceQueueItem) -> None:
        self._items.append(item)

    def items(self) -> list[EvidenceQueueItem]:
        return list(self._items)

    def pending(self) -> list[EvidenceQueueItem]:
        return [item for item in self._items if item.status == "pending"]

    def mark_done(self, action_type: str, target: str, *, evidence: dict[str, Any] | None = None) -> None:
        for item in self._items:
            if item.action_type == action_type and item.target == target and item.status in {"pending", "running"}:
                item.status = "done"
                item.evidence.update(evidence or {})
                return

    def skip_where(self, predicate: Callable[[EvidenceQueueItem], bool], *, reason: str) -> int:
        skipped = 0
        for item in self._items:
            if item.status == "pending" and predicate(item):
                item.status = "skipped"
                item.skip_reason = reason
                skipped += 1
        return skipped

    def trace_row(self) -> dict[str, Any]:
        return {
            "layer": "evidence_queue",
            "decision": "queue_state",
            "reason": "evidence queue snapshot",
            "items": [
                {
                    "action_type": item.action_type,
                    "target": item.target,
                    "required": item.required,
                    "status": item.status,
                    "skip_reason": item.skip_reason,
                }
                for item in self._items
            ],
        }


__all__ = ["EvidenceAction", "EvidenceQueue", "EvidenceQueueItem", "EvidenceStatus"]
