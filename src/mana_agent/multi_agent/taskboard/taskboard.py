from __future__ import annotations

from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.ids import new_task_id
from mana_agent.multi_agent.core.types import (
    DecisionRecord,
    HandoffRecord,
    RiskLevel,
    TaskBoardItem,
    TaskStatus,
    VerificationResult,
    utc_now,
)
from mana_agent.multi_agent.taskboard.store import JsonStateStore, serialize, task_from_dict
from mana_agent.multi_agent.taskboard.validators import validate_transition


def _append_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            target.append(text)
            seen.add(text)


class TaskBoard:
    def __init__(self, root: str | Path = ".") -> None:
        self.store = JsonStateStore(root)
        self.tasks: dict[str, TaskBoardItem] = {}
        self.load()

    def create_task(
        self,
        *,
        title: str,
        user_request: str,
        normalized_goal: str | None = None,
        priority: int = 100,
        risk_level: RiskLevel = RiskLevel.LOW,
        owner_agent_id: str | None = None,
    ) -> TaskBoardItem:
        task_id = new_task_id()
        task = TaskBoardItem(
            task_id=task_id,
            parent_task_id=None,
            root_task_id=task_id,
            title=title,
            user_request=user_request,
            normalized_goal=normalized_goal or user_request.strip(),
            status=TaskStatus.NEW,
            priority=priority,
            risk_level=risk_level,
            owner_agent_id=owner_agent_id,
        )
        self.tasks[task_id] = task
        self._record("task.created", task)
        self.save()
        return task

    def create_child_task(self, parent_task_id: str, *, title: str, user_request: str, owner_agent_id: str | None = None) -> TaskBoardItem:
        parent = self.get_task(parent_task_id)
        task_id = new_task_id()
        task = TaskBoardItem(
            task_id=task_id,
            parent_task_id=parent_task_id,
            root_task_id=parent.root_task_id,
            title=title,
            user_request=user_request,
            normalized_goal=user_request.strip(),
            status=TaskStatus.NEW,
            priority=parent.priority,
            risk_level=parent.risk_level,
            owner_agent_id=owner_agent_id,
        )
        self.tasks[task_id] = task
        self._record("task.created", task)
        self.save()
        return task

    def get_task(self, task_id: str) -> TaskBoardItem:
        return self.tasks[task_id]

    def update_status(self, task_id: str, status: TaskStatus, *, reason: str | None = None) -> None:
        task = self.get_task(task_id)
        validate_transition(task, status, reason=reason)
        task.status = status
        task.updated_at = utc_now()
        if status == TaskStatus.BLOCKED and reason:
            self.add_blocker(task_id, reason, save=False)
        self._record("task.updated", {"task_id": task_id, "status": status.value, "reason": reason})
        self.save()

    def assign(self, task_id: str, agent_id: str) -> None:
        task = self.get_task(task_id)
        _append_unique(task.assigned_agent_ids, [agent_id])
        task.updated_at = utc_now()
        self._record("task.assigned", {"task_id": task_id, "agent_id": agent_id})
        self.save()

    def assign_subagent(self, task_id: str, subagent_id: str) -> None:
        task = self.get_task(task_id)
        _append_unique(task.assigned_subagent_ids, [subagent_id])
        task.updated_at = utc_now()
        self._record("task.subagent_assigned", {"task_id": task_id, "subagent_id": subagent_id})
        self.save()

    def add_assumption(self, task_id: str, assumption: str) -> None:
        self._add_text(task_id, "assumptions", assumption)

    def add_decision(self, task_id: str, decision: DecisionRecord | str) -> None:
        task = self.get_task(task_id)
        decision_id = decision.decision_id if isinstance(decision, DecisionRecord) else str(decision)
        _append_unique(task.decision_ids, [decision_id])
        task.updated_at = utc_now()
        self._record("decision.recorded", {"task_id": task_id, "decision": serialize(decision)})
        self.save()

    def add_evidence(self, task_id: str, evidence: str) -> None:
        self._add_text(task_id, "evidence", evidence)

    def add_blocker(self, task_id: str, blocker: str, *, save: bool = True) -> None:
        self._add_text(task_id, "blockers", blocker, save=save)

    def add_discussion(self, task_id: str, discussion_id: str) -> None:
        task = self.get_task(task_id)
        _append_unique(task.discussion_ids, [discussion_id])
        task.updated_at = utc_now()
        self._record("discussion.opened", {"task_id": task_id, "discussion_id": discussion_id})
        self.save()

    def add_files_to_inspect(self, task_id: str, files: list[str]) -> None:
        self._add_many(task_id, "files_to_inspect", files)

    def add_files_touched(self, task_id: str, files: list[str]) -> None:
        self._add_many(task_id, "files_touched", files)

    def add_queue_job(self, task_id: str, job_id: str) -> None:
        self._add_many(task_id, "queue_job_ids", [job_id])

    def add_verification_result(self, task_id: str, result: VerificationResult) -> None:
        task = self.get_task(task_id)
        task.verification_results.append(result)
        task.updated_at = utc_now()
        self._record("verification.finished", {"task_id": task_id, "result": serialize(result)})
        self.save()

    def add_handoff(self, task_id: str, handoff: HandoffRecord) -> None:
        task = self.get_task(task_id)
        task.handoff_records.append(handoff)
        task.updated_at = utc_now()
        self._record("handoff.created", handoff)
        self.save()

    def compact_context(self, task_id: str, token_budget: int = 1200) -> str:
        task = self.get_task(task_id)
        lines = [
            f"Task {task.task_id}: {task.title}",
            f"Status: {task.status.value}",
            f"Goal: {task.normalized_goal}",
        ]
        for label, values in (
            ("Plan", task.plan),
            ("Acceptance", task.acceptance_criteria),
            ("Evidence", task.evidence),
            ("Blockers", task.blockers),
            ("Assumptions", task.assumptions),
        ):
            if values:
                lines.append(f"{label}:")
                lines.extend(f"- {item}" for item in values)
        text = "\n".join(lines)
        return text[: max(200, int(token_budget) * 4)]

    def save(self) -> None:
        self.store.save_state({"tasks": {key: serialize(value) for key, value in self.tasks.items()}})

    def load(self) -> None:
        payload = self.store.load_state()
        tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
        self.tasks = {
            task_id: task_from_dict(item)
            for task_id, item in tasks.items()
            if isinstance(item, dict)
        }

    def _add_text(self, task_id: str, field_name: str, value: str, *, save: bool = True) -> None:
        if not str(value or "").strip():
            return
        task = self.get_task(task_id)
        _append_unique(getattr(task, field_name), [value])
        task.updated_at = utc_now()
        self._record("task.updated", {"task_id": task_id, field_name: value})
        if save:
            self.save()

    def _add_many(self, task_id: str, field_name: str, values: list[str]) -> None:
        task = self.get_task(task_id)
        _append_unique(getattr(task, field_name), values)
        task.updated_at = utc_now()
        self._record("task.updated", {"task_id": task_id, field_name: values})
        self.save()

    def _record(self, event_type: str, payload: Any) -> None:
        self.store.append_history({"event_type": event_type, "payload": serialize(payload), "created_at": utc_now()})
