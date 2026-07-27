"""Typed model tools for the dedicated automation chat route."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .service import (
    AgentPromptJob,
    AutomationService,
    AutomationValidationError,
    CommandJob,
    ConnectorActionJob,
    CronTrigger,
    IntervalTrigger,
    MisfirePolicy,
    OnceTrigger,
    RetryPolicy,
    TeachFlowJob,
    ToolActionJob,
)


_CreateTrigger = Annotated[
    CronTrigger | IntervalTrigger | OnceTrigger,
    Field(discriminator="type"),
]
_CreateJob = Annotated[
    AgentPromptJob | ConnectorActionJob | ToolActionJob | TeachFlowJob | CommandJob,
    Field(discriminator="type"),
]


class _Decision(BaseModel):
    source_decision_id: str = Field(
        min_length=1, description="ID of the structured model decision selecting this operation."
    )


class _Create(_Decision):
    name: str = Field(min_length=1, max_length=160)
    description: str = ""
    trigger: _CreateTrigger
    job: _CreateJob
    timezone: str
    target_runtime: Literal["local", "github"] = "local"
    permission_references: list[str] = Field(default_factory=list)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    misfire_policy: MisfirePolicy = Field(default_factory=MisfirePolicy)
    idempotency_key: str = ""


class _Id(_Decision):
    automation_id: str


class _Update(_Id):
    changes: dict[str, Any]


def _result(operation) -> str:
    try:
        value = operation()
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        return json.dumps({"ok": True, "persisted": value}, ensure_ascii=False, default=str)
    except (AutomationValidationError, ValueError) as exc:
        return json.dumps({
            "ok": False,
            "error_code": "automation_definition_invalid",
            "message": str(exc),
            "no_automation_created": True,
        })


def build_automation_langchain_tools(root: str | Path) -> list[Any]:
    service = AutomationService(root)
    return [
        StructuredTool.from_function(
            name="automation_create",
            description=(
                "Create and deploy a durable automation from a complete typed semantic trigger and job. "
                "Call this directly when the validated operation is create; do not list existing "
                "automations first. "
                "Use interval/every_seconds for exact elapsed intervals, cron only for calendar schedules, "
                "and once/run_at for one-time jobs. Returns the persisted record and truthful deployment."
            ),
            args_schema=_Create,
            func=lambda name, description, trigger, job, timezone, target_runtime,
            permission_references, retry_policy, misfire_policy, idempotency_key,
            source_decision_id: _result(lambda: service.create(
                name=name, description=description,
                source="teach" if job.type == "teach_flow" else "chat",
                trigger=trigger, job=job, timezone_name=timezone,
                target_runtime=target_runtime, permission_references=permission_references,
                retry_policy=retry_policy, misfire_policy=misfire_policy,
                idempotency_key=idempotency_key,
            )),
        ),
        StructuredTool.from_function(
            name="automation_get", description="Get one canonical persisted automation definition.",
            args_schema=_Id,
            func=lambda automation_id, source_decision_id: _result(lambda: service.get(automation_id)),
        ),
        StructuredTool.from_function(
            name="automation_list",
            description=(
                "List canonical persisted automations only when the validated operation is list. "
                "Never use this as discovery or a prerequisite for automation creation."
            ),
            args_schema=_Decision,
            func=lambda source_decision_id: _result(
                lambda: [item.to_dict() for item in service.list()]
            ),
        ),
        StructuredTool.from_function(
            name="automation_status", description="Inspect deployment health and recent persisted runs.",
            args_schema=_Id,
            func=lambda automation_id, source_decision_id: _result(lambda: service.status(automation_id)),
        ),
        StructuredTool.from_function(
            name="automation_update",
            description="Explicitly update a canonical automation and reconcile its persistent deployment.",
            args_schema=_Update,
            func=lambda automation_id, changes, source_decision_id: _result(
                lambda: service.update(automation_id, changes)
            ),
        ),
        StructuredTool.from_function(
            name="automation_delete", description="Delete an automation and every managed scheduler artifact.",
            args_schema=_Id,
            func=lambda automation_id, source_decision_id: _result(lambda: service.delete(automation_id)),
        ),
        StructuredTool.from_function(
            name="automation_enable", description="Enable and redeploy a persisted automation.",
            args_schema=_Id,
            func=lambda automation_id, source_decision_id: _result(
                lambda: service.update(automation_id, {"enabled": True})
            ),
        ),
        StructuredTool.from_function(
            name="automation_disable", description="Disable a persisted automation and remove its active wakeup.",
            args_schema=_Id,
            func=lambda automation_id, source_decision_id: _result(
                lambda: service.update(automation_id, {"enabled": False})
            ),
        ),
        StructuredTool.from_function(
            name="automation_run_now", description="Claim and execute one persisted automation immediately.",
            args_schema=_Id,
            func=lambda automation_id, source_decision_id: _result(
                lambda: service.execute(automation_id, force=True)
            ),
        ),
    ]


AUTOMATION_TOOL_NAMES = (
    "automation_create", "automation_get", "automation_list", "automation_status",
    "automation_update", "automation_delete", "automation_enable",
    "automation_disable", "automation_run_now",
)

AUTOMATION_OPERATION_TOOLS: dict[str, tuple[str, ...]] = {
    "create": ("automation_create",),
    "get": ("automation_get",),
    "list": ("automation_list",),
    "status": ("automation_status",),
    "update": ("automation_update",),
    "delete": ("automation_delete",),
    "enable": ("automation_enable",),
    "disable": ("automation_disable",),
    "run_now": ("automation_run_now",),
}
