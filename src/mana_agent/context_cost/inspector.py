"""Bounded, session-scoped, metadata-only admission diagnostics.

This is not a provider request trace: admission candidates can differ from the
payload ultimately submitted by a caller. No source content or caller-supplied
identifiers are retained in the public snapshots.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, replace
import hashlib
import hmac
import secrets
from threading import RLock
from typing import Any, Sequence
from uuid import uuid4

from mana_agent.context_cost.models import ContextSegment

MAX_SNAPSHOTS = 50
MAX_SEGMENTS = 100
_CATEGORIES = frozenset({
    "system", "developer", "user", "history", "conversation", "summary",
    "schema", "tool", "tool_result", "memory", "retrieval", "repository",
    "file", "document", "evidence", "attachment", "skill", "artifact",
    "dependency", "prerequisite", "current_turn", "routing", "planning",
})
_BOUNDARIES = frozenset({"admission", "compatible_chat", "ask_agent", "codex_outer"})


@dataclass(frozen=True, slots=True)
class InspectorSegment:
    segment_id: str
    position: int
    category: str
    status: str
    token_estimate: int | None
    counting_method: str
    reason: str


@dataclass(frozen=True, slots=True)
class InspectorSnapshot:
    schema_version: int
    invocation_id: str
    boundary: str
    lifecycle: str
    model_ref: str | None
    turn_ref: str | None
    task_ref: str | None
    attempt_ref: str | None
    segments: tuple[InspectorSegment, ...]
    category_totals: tuple[tuple[str, int], ...]
    omitted_segments: int
    admitted_segment_tokens: int | None
    accounting_input_estimate: int | None
    accounting_allowance: int | None
    context_window: int | None
    provider_input_tokens: int | None = None


class ContextInspector:
    """Only an exact session/workspace/repository scope can retrieve snapshots."""

    def __init__(self, scope: tuple[str, str, str]) -> None:
        self._scope = scope
        self._lock = RLock()
        self._records: OrderedDict[str, InspectorSnapshot] = OrderedDict()
        self._key = secrets.token_bytes(32)
        self._epoch = 0

    def reset(self, scope: tuple[str, str, str]) -> None:
        with self._lock:
            self._scope = scope
            self._records.clear()
            self._key = secrets.token_bytes(32)
            self._epoch += 1

    def stamp(self) -> tuple[tuple[str, str, str], int]:
        with self._lock:
            return self._scope, self._epoch

    def capture(
        self, call_id: str, candidates: Sequence[ContextSegment],
        admitted: Sequence[ContextSegment], *, scope: tuple[str, str, str],
        boundary: str, blocked: bool, estimate: Any = None,
        context_window: int | None = None,
        epoch: int | None = None,
        model_identity: str = "", turn_id: str = "", task_id: str = "", attempt_id: str = "",
    ) -> None:
        # Identity-based matching preserves repeated equal segments and original order.
        selected_indexes: dict[int, deque[int]] = defaultdict(deque)
        for index, segment in enumerate(admitted):
            selected_indexes[id(segment)].append(index)
        components = estimate.components if estimate is not None else {}
        records = []
        segment_total = 0
        category_totals: Counter[str] = Counter()
        for position, segment in enumerate(candidates):
            category = segment.kind if segment.kind in _CATEGORIES else "other"
            included = bool(selected_indexes[id(segment)])
            if included:
                index = selected_indexes[id(segment)].popleft()
                # Keys come from the existing accounting request; never serialize them.
                count = max(0, int(components.get(f"{segment.kind}:{index}", 0))) if estimate is not None else max(0, int(segment.token_estimate))
                segment_total += count
                category_totals[category] += count
                method = "accounting tokenizer estimate" if estimate is not None else "caller estimate"
                status = "admitted; submission not observed" if not blocked else "blocked; not submitted"
                reason = "Provided to admission; upstream selection reason not observable"
            else:
                count, method = None, "unknown"
                status = "removed before admission"
                reason = "Removed by admission compaction or fitting"
            if position < MAX_SEGMENTS:
                records.append(InspectorSegment(
                    segment_id=uuid4().hex, position=position, category=category,
                    status=status, token_estimate=count, counting_method=method, reason=reason,
                ))
        input_estimate = int(estimate.input_tokens) if estimate is not None else None
        with self._lock:
            if scope != self._scope or (epoch is not None and epoch != self._epoch):
                return
            def opaque(kind: str, value: str) -> str | None:
                if not value:
                    return None
                material = (kind + "\0" + value).encode("utf-8")
                return hmac.new(self._key, material, hashlib.sha256).hexdigest()[:24]

            snapshot = InspectorSnapshot(
                schema_version=1, invocation_id=uuid4().hex,
                boundary=boundary if boundary in _BOUNDARIES else "admission",
                lifecycle="blocked" if blocked else "admitted",
                model_ref=opaque("model", model_identity), turn_ref=opaque("turn", turn_id),
                task_ref=opaque("task", task_id), attempt_ref=opaque("attempt", attempt_id),
                segments=tuple(records), category_totals=tuple(sorted(category_totals.items())),
                omitted_segments=max(0, len(candidates) - MAX_SEGMENTS),
                admitted_segment_tokens=segment_total,
                accounting_input_estimate=input_estimate,
                accounting_allowance=max(0, input_estimate - segment_total) if input_estimate is not None else None,
                context_window=context_window,
            )
            self._records[call_id] = snapshot
            self._records.move_to_end(call_id)
            while len(self._records) > MAX_SNAPSHOTS:
                self._records.popitem(last=False)

    def finish(self, call_id: str, *, lifecycle: str, provider_input_tokens: int | None = None) -> None:
        if lifecycle not in {"completed", "released"}:
            raise ValueError("invalid inspector lifecycle")
        with self._lock:
            snapshot = self._records.get(call_id)
            if snapshot is not None:
                self._records[call_id] = replace(snapshot, lifecycle=lifecycle, provider_input_tokens=provider_input_tokens)

    def history(self, *, scope: tuple[str, str, str]) -> tuple[InspectorSnapshot, ...]:
        with self._lock:
            if scope != self._scope or not all(scope):
                return ()
            return tuple(reversed(self._records.values()))
