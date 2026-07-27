from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from .events import EvalEvent
from .redaction import redact


class EvalRecorder(Protocol):
    def record(self, event_type: str, payload: dict[str, Any] | None = None) -> None: ...
    def record_run_started(self, payload: dict[str, Any]) -> None: ...
    def record_route_decision(self, payload: dict[str, Any]) -> None: ...
    def record_model_call(self, payload: dict[str, Any]) -> None: ...
    def record_tool_started(self, payload: dict[str, Any]) -> None: ...
    def record_tool_finished(self, payload: dict[str, Any]) -> None: ...
    def record_command(self, payload: dict[str, Any]) -> None: ...
    def record_patch(self, payload: dict[str, Any]) -> None: ...
    def record_test_result(self, payload: dict[str, Any]) -> None: ...
    def record_review(self, payload: dict[str, Any]) -> None: ...
    def record_verification(self, payload: dict[str, Any]) -> None: ...
    def record_run_finished(self, payload: dict[str, Any]) -> None: ...
    def record_run_failed(self, payload: dict[str, Any]) -> None: ...


class NullEvalRecorder:
    def record(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        _ = (event_type, payload)

    def __getattr__(self, name: str):
        if name.startswith("record_"):
            return lambda payload=None: None
        raise AttributeError(name)


@dataclass(frozen=True, slots=True)
class EvalExecutionContext:
    run_id: str
    task_id: str
    variant_id: str
    recorder: EvalRecorder


_CURRENT: contextvars.ContextVar[EvalExecutionContext | None] = contextvars.ContextVar(
    "mana_eval_execution_context", default=None
)


def current_eval_context() -> EvalExecutionContext | None:
    return _CURRENT.get()


@contextmanager
def use_eval_context(context: EvalExecutionContext) -> Iterator[EvalExecutionContext]:
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def record_current(event_type: str, payload: dict[str, Any] | None = None) -> None:
    context = current_eval_context()
    if context is not None:
        context.recorder.record(event_type, payload or {})


class ArtifactEvalRecorder:
    def __init__(self, run_dir: str | Path, *, run_id: str, task_id: str, variant_id: str, extra_redaction_patterns: tuple[str, ...] = ()) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.run_id = run_id
        self.task_id = task_id
        self.variant_id = variant_id
        self.extra_redaction_patterns = extra_redaction_patterns
        self._lock = threading.RLock()
        self._sequence = self._recover_sequence()

    def _recover_sequence(self) -> int:
        if not self.events_path.exists():
            return 0
        with self.events_path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def record(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._sequence += 1
            event = EvalEvent(
                sequence=self._sequence,
                event_type=event_type,
                run_id=self.run_id,
                task_id=self.task_id,
                variant_id=self.variant_id,
                payload=redact(payload or {}, self.extra_redaction_patterns),
            )
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
                handle.flush()

    def events(self) -> list[EvalEvent]:
        if not self.events_path.exists():
            return []
        return [EvalEvent.model_validate_json(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __getattr__(self, name: str):
        mapping = {
            "record_run_started": "run.started",
            "record_route_decision": "route.decision",
            "record_model_call": "model.call",
            "record_tool_started": "tool.started",
            "record_tool_finished": "tool.finished",
            "record_command": "command.finished",
            "record_patch": "patch.captured",
            "record_test_result": "test.finished",
            "record_review": "review.finished",
            "record_verification": "verification.finished",
            "record_run_finished": "run.finished",
            "record_run_failed": "run.failed",
        }
        if name in mapping:
            return lambda payload=None: self.record(mapping[name], payload or {})
        raise AttributeError(name)
