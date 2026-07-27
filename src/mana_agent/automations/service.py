"""Canonical, durable automation definitions, deployment, and execution.

``AutomationDefinition`` is the only persisted authoring contract.  Platform
schedulers are deliberately hidden: they wake the ID-based executor, which
reloads the definition, checks its due time, and acquires a durable lease.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from mana_agent.utils.redaction import redact_secrets
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path


CONFIG_VERSION = 3
MAX_OUTPUT_CHARS = 8_000
LEASE_SECONDS = 3_600
_ID_PATTERN = re.compile(r"^(?:aut|sch)_[A-Za-z0-9_-]{3,128}$")
_CRON_FIELD = re.compile(r"^[0-9*/,\-]+$")
_MANAGED_PREFIX = "mana-agent-automation-"
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class AutomationValidationError(ValueError):
    """A canonical definition or requested operation is invalid."""


class CronTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["cron"] = "cron"
    expression: str
    timezone: str

    @model_validator(mode="after")
    def validate_trigger(self) -> "CronTrigger":
        self.expression = " ".join(self.expression.split())
        validate_cron(self.expression)
        validate_timezone(self.timezone)
        return self


class IntervalTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["interval"] = "interval"
    every_seconds: int = Field(ge=60, le=31_536_000)
    anchor_at: datetime

    @model_validator(mode="after")
    def aware_anchor(self) -> "IntervalTrigger":
        self.anchor_at = ensure_aware(self.anchor_at)
        return self


class OnceTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["once"] = "once"
    run_at: datetime
    timezone: str

    @model_validator(mode="after")
    def validate_trigger(self) -> "OnceTrigger":
        self.run_at = ensure_aware(self.run_at)
        validate_timezone(self.timezone)
        return self


AutomationTrigger = CronTrigger | IntervalTrigger | OnceTrigger
TRIGGER_ADAPTER = TypeAdapter(AutomationTrigger)


class AgentPromptJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["agent_prompt"] = "agent_prompt"
    prompt: str = Field(min_length=1, max_length=20_000)
    route: str = "automation"


class ConnectorActionJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["connector_action"] = "connector_action"
    connector: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=160)
    arguments: dict[str, Any] = Field(default_factory=dict)
    prompt: str = Field(default="", max_length=10_000)


class ToolActionJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_action"] = "tool_action"
    tool_name: str = Field(min_length=1, max_length=160)
    arguments: dict[str, Any] = Field(default_factory=dict)


class TeachFlowJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["teach_flow"] = "teach_flow"
    flow_id: str = Field(min_length=1, max_length=240)
    flow_version: int = Field(ge=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    required_permission_scopes: list[str] = Field(default_factory=list)
    verification: dict[str, Any] = Field(default_factory=dict)


class CommandJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["command"] = "command"
    argv: list[str] = Field(min_length=1, max_length=128)
    cwd: str | None = None
    explicitly_requested: Literal[True]

    @model_validator(mode="after")
    def validate_argv(self) -> "CommandJob":
        if any(not isinstance(value, str) or not value or "\x00" in value for value in self.argv):
            raise ValueError("Command argv must contain non-empty, null-free strings.")
        return self


AutomationJob = AgentPromptJob | ConnectorActionJob | ToolActionJob | TeachFlowJob | CommandJob
JOB_ADAPTER = TypeAdapter(AutomationJob)


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maximum_attempts: int = Field(default=1, ge=1, le=10)
    backoff_seconds: int = Field(default=300, ge=1, le=86_400)


class MisfirePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["skip", "run_once", "catch_up"] = "run_once"
    grace_seconds: int = Field(default=3_600, ge=0, le=604_800)


class DeploymentState(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: Literal["pending", "deployed", "disabled", "blocked", "failed"] = "pending"
    backend: str = ""
    artifact: str = ""
    updated_at: datetime = Field(default_factory=lambda: now_utc())
    blocked_reason: str = ""


class ExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    status: Literal["running", "succeeded", "failed", "blocked", "skipped"]
    started_at: datetime
    finished_at: datetime | None = None
    output_summary: str = ""
    error: str = ""
    attempt: int = 1


class AutomationDefinition(BaseModel):
    """Versioned, secret-free source of truth for one durable automation."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    id: str = Field(default_factory=lambda: f"aut_{uuid.uuid4().hex}")
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=4_000)
    enabled: bool = True
    source: Literal["chat", "teach", "migration"]
    repository_id: str
    workspace_path: str
    session_id: str = ""
    trigger: AutomationTrigger = Field(discriminator="type")
    job: AutomationJob = Field(discriminator="type")
    timezone: str
    target_runtime: Literal["local", "github"] = "local"
    created_at: datetime = Field(default_factory=lambda: now_utc())
    updated_at: datetime = Field(default_factory=lambda: now_utc())
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    deployment: DeploymentState = Field(default_factory=DeploymentState)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    misfire_policy: MisfirePolicy = Field(default_factory=MisfirePolicy)
    permission_references: list[str] = Field(default_factory=list)
    recent_execution: ExecutionSummary | None = None
    migration_status: Literal["not_applicable", "migrated", "invalid"] = "not_applicable"
    migration_error: str = ""

    @model_validator(mode="after")
    def validate_definition(self) -> "AutomationDefinition":
        if not _ID_PATTERN.fullmatch(self.id):
            raise ValueError("Automation id is invalid.")
        validate_timezone(self.timezone)
        if self.trigger.type in {"cron", "once"} and self.trigger.timezone != self.timezone:
            raise ValueError("Trigger timezone must match the automation timezone.")
        self.created_at = ensure_aware(self.created_at)
        self.updated_at = ensure_aware(self.updated_at)
        if self.next_run_at is not None:
            self.next_run_at = ensure_aware(self.next_run_at)
        if self.last_run_at is not None:
            self.last_run_at = ensure_aware(self.last_run_at)
        reject_secret_values(self.job.model_dump(mode="json"))
        if self.job.type == "teach_flow":
            combined = set(self.permission_references) | set(self.job.required_permission_scopes)
            self.permission_references = sorted(combined)
        if self.target_runtime == "github" and self.job.type in {
            "connector_action", "teach_flow",
        }:
            raise ValueError("Authenticated connector and Teach jobs require the local runtime.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AutomationDefinition":
        return cls.model_validate(value)


class AutomationRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    automation_id: str
    status: Literal["running", "succeeded", "failed", "blocked", "skipped"] = "running"
    claimed_at: datetime = Field(default_factory=lambda: now_utc())
    started_at: datetime = Field(default_factory=lambda: now_utc())
    finished_at: datetime | None = None
    attempt: int = 1
    output_summary: str = ""
    error: str = ""
    next_run_at: datetime | None = None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Automation timestamps must include a timezone.")
    return value.astimezone(timezone.utc)


def machine_timezone() -> str:
    name = os.getenv("TZ", "").strip()
    if name:
        try:
            ZoneInfo(name)
            return name
        except ZoneInfoNotFoundError:
            pass
    local = datetime.now().astimezone().tzinfo
    key = getattr(local, "key", "")
    if key:
        return str(key)
    return "UTC"


def validate_timezone(value: str) -> None:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Invalid IANA timezone: {value}") from exc


def validate_cron(expression: str) -> None:
    fields = expression.split()
    if len(fields) != 5 or any(not _CRON_FIELD.fullmatch(field) for field in fields):
        raise AutomationValidationError("Cron must be a five-field POSIX expression.")
    ranges = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
    for field, bounds in zip(fields, ranges):
        _expand_cron_field(field, *bounds)


def _expand_cron_field(field: str, low: int, high: int) -> set[int]:
    values: set[int] = set()
    for part in field.split(","):
        base, slash, step_text = part.partition("/")
        step = int(step_text) if slash else 1
        if step < 1:
            raise AutomationValidationError("Cron steps must be positive.")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            left, right = base.split("-", 1)
            start, end = int(left), int(right)
        else:
            start = end = int(base)
        if start < low or end > high or start > end:
            raise AutomationValidationError(f"Cron field {field!r} is outside {low}-{high}.")
        values.update(range(start, end + 1, step))
    return values


def reject_secret_values(value: Any, path: str = "job") -> None:
    sensitive = re.compile(r"(?:password|secret|access[_-]?token|api[_-]?key|oauth|private[_-]?key)", re.I)
    if isinstance(value, dict):
        for key, item in value.items():
            if sensitive.search(str(key)) and item not in (None, "", [], {}):
                if not (isinstance(item, str) and item.startswith(("secret://", "permission://", "account://"))):
                    raise ValueError(f"{path}.{key} must contain a credential reference, not a secret.")
            reject_secret_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_secret_values(item, f"{path}[{index}]")


def calculate_next_run(
    trigger: AutomationTrigger,
    *,
    after: datetime | None = None,
    previous_claimed_at: datetime | None = None,
) -> datetime | None:
    """Calculate the first occurrence strictly after ``after``.

    Intervals are always an integer number of seconds from the persisted anchor,
    so midnight, restarts, and daylight-saving boundaries cannot alter cadence.
    """
    cursor = ensure_aware(after or now_utc())
    if trigger.type == "once":
        instant = trigger.run_at.astimezone(timezone.utc)
        return instant if instant > cursor else None
    if trigger.type == "interval":
        anchor = trigger.anchor_at.astimezone(timezone.utc)
        base = max(cursor, ensure_aware(previous_claimed_at) if previous_claimed_at else cursor)
        if base < anchor:
            return anchor
        elapsed = (base - anchor).total_seconds()
        steps = math.floor(elapsed / trigger.every_seconds) + 1
        return anchor + timedelta(seconds=steps * trigger.every_seconds)
    zone = ZoneInfo(trigger.timezone)
    local = cursor.astimezone(zone).replace(second=0, microsecond=0) + timedelta(minutes=1)
    minute, hour, day, month, weekday = (
        _expand_cron_field(part, *bounds)
        for part, bounds in zip(
            trigger.expression.split(),
            ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7)),
        )
    )
    for _ in range(60 * 24 * 366 * 6):
        cron_weekday = (local.weekday() + 1) % 7
        if (
            local.minute in minute and local.hour in hour and local.day in day
            and local.month in month and (cron_weekday in weekday or (cron_weekday == 0 and 7 in weekday))
        ):
            return local.astimezone(timezone.utc)
        local += timedelta(minutes=1)
    raise AutomationValidationError("Unable to calculate the next cron occurrence.")


def human_trigger(trigger: AutomationTrigger) -> str:
    if trigger.type == "interval":
        seconds = trigger.every_seconds
        if seconds % 3600 == 0:
            return f"Every {seconds // 3600} hour{'s' if seconds != 3600 else ''}"
        if seconds % 60 == 0:
            return f"Every {seconds // 60} minutes"
        return f"Every {seconds} seconds"
    if trigger.type == "once":
        return f"Once at {trigger.run_at.astimezone(ZoneInfo(trigger.timezone)).isoformat()}"
    return f"Calendar schedule ({trigger.expression})"


def config_path(root: Path) -> Path:
    return repository_dir(repository_id_for_path(root.resolve())) / "automations" / "config.json"


def _lock_path(root: Path) -> Path:
    return config_path(root).with_suffix(".lock")


@contextmanager
def _store_lock(root: Path):
    path = _lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.setdefault(str(path), threading.RLock())
    with lock:
        handle = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _empty_store() -> dict[str, Any]:
    return {"version": CONFIG_VERSION, "automations": [], "runs": [], "leases": {}, "migration_errors": []}


def _read_unlocked(root: Path) -> dict[str, Any]:
    path = config_path(root)
    if not path.exists():
        return _empty_store()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutomationValidationError(f"Automation configuration is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise AutomationValidationError("Automation configuration must be an object.")
    return value


def _write_unlocked(root: Path, payload: dict[str, Any]) -> None:
    path = config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["version"] = CONFIG_VERSION
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(root: Path) -> dict[str, Any]:
    with _store_lock(root):
        payload = _read_unlocked(root)
        migrated, changed = _migrate_payload(payload, root)
        if changed:
            _write_unlocked(root, migrated)
        return migrated


def save_config(root: Path, payload: dict[str, Any]) -> None:
    with _store_lock(root):
        migrated, _ = _migrate_payload(payload, root)
        _write_unlocked(root, migrated)


def _legacy_trigger(value: dict[str, Any], tz: str) -> AutomationTrigger:
    cron = str(value.get("cron") or "").strip()
    if cron:
        return CronTrigger(expression=cron, timezone=tz)
    trigger = str(value.get("trigger") or "")
    if trigger == "interval":
        seconds = int(value.get("every_seconds") or value.get("interval_seconds") or 3600)
        return IntervalTrigger(every_seconds=seconds, anchor_at=now_utc())
    raise AutomationValidationError("Legacy record has no valid timed trigger.")


def _legacy_job(value: dict[str, Any]) -> AutomationJob:
    action = str(value.get("action") or "").strip()
    config = dict(value.get("action_config") or {})
    if action == "teach-flow":
        return TeachFlowJob(
            flow_id=str(config.get("flow_id") or ""),
            flow_version=int(config.get("flow_version") or 0),
            inputs=dict(config.get("inputs") or {}),
            required_permission_scopes=list(config.get("required_permission_scopes") or []),
            verification=dict(config.get("verification") or {}),
        )
    if action == "custom":
        command = str(value.get("command") or "")
        return CommandJob(argv=shlex.split(command), explicitly_requested=True)
    prompts = {
        "analyze": "Analyze this repository and persist the analysis artifacts.",
        "daily_report": "Create the daily repository report.",
        "self_improvement": "Run the approved self-improvement workflow.",
        "fleet-verify": "Run the persisted fleet verification workflow.",
    }
    if action in prompts:
        return AgentPromptJob(prompt=prompts[action])
    if action:
        return ToolActionJob(tool_name=action, arguments=config)
    raise AutomationValidationError("Legacy record has no executable action.")


def _migrate_payload(payload: dict[str, Any], root: Path) -> tuple[dict[str, Any], bool]:
    if int(payload.get("version") or 0) >= CONFIG_VERSION and "schedules" not in payload:
        result = _empty_store()
        result.update(payload)
        return result, False
    result = _empty_store()
    existing_ids: set[str] = set()
    legacy_rows = list(payload.get("schedules") or []) + list(payload.get("automations") or [])
    tz = machine_timezone()
    for index, row in enumerate(legacy_rows):
        if not isinstance(row, dict):
            result["migration_errors"].append({"index": index, "record": row, "error": "record is not an object"})
            continue
        record_id = str(row.get("id") or f"aut_migrated_{uuid.uuid4().hex}")
        if record_id in existing_ids:
            continue
        existing_ids.add(record_id)
        try:
            trigger = _legacy_trigger(row, tz)
            job = _legacy_job(row)
            definition = AutomationDefinition(
                id=record_id if _ID_PATTERN.fullmatch(record_id) else f"aut_migrated_{uuid.uuid4().hex}",
                name=str(row.get("name") or record_id),
                description=str(row.get("description") or ""),
                enabled=bool(row.get("enabled", True)),
                source="migration",
                repository_id=repository_id_for_path(root.resolve()),
                workspace_path=str(root.resolve()),
                trigger=trigger,
                job=job,
                timezone=tz,
                target_runtime="github" if "github" in list(row.get("targets") or []) else "local",
                created_at=_parse_datetime(row.get("created_at")) or now_utc(),
                updated_at=_parse_datetime(row.get("updated_at")) or now_utc(),
                last_run_at=_parse_datetime((row.get("last_run") or {}).get("at")),
                next_run_at=calculate_next_run(trigger),
                deployment=_migrated_deployment(row),
                migration_status="migrated",
            )
            result["automations"].append(definition.to_dict())
        except (ValueError, TypeError, AutomationValidationError) as exc:
            result["migration_errors"].append(
                {"id": record_id, "record": redact_secrets(row), "error": str(exc), "status": "invalid"}
            )
    for row in list(payload.get("runs") or []):
        if isinstance(row, dict):
            result["runs"].append(redact_secrets(row))
        else:
            result["migration_errors"].append({"record": row, "error": "run record is not an object"})
    return result, True


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return ensure_aware(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (ValueError, TypeError):
        return None


def _migrated_deployment(row: dict[str, Any]) -> DeploymentState:
    old = dict(row.get("deployment") or {})
    status = "deployed" if old and not old.get("errors") else "pending"
    return DeploymentState(status=status, backend="legacy", artifact=json.dumps(redact_secrets(old))[:1000])


def list_automations(root: Path) -> list[AutomationDefinition]:
    rows: list[AutomationDefinition] = []
    payload = load_config(root)
    for raw in payload["automations"]:
        try:
            rows.append(AutomationDefinition.from_dict(raw))
        except ValueError as exc:
            payload["migration_errors"].append(
                {"id": raw.get("id") if isinstance(raw, dict) else "", "record": redact_secrets(raw), "error": str(exc)}
            )
    return rows


def get_automation(root: Path, automation_id: str) -> AutomationDefinition:
    for automation in list_automations(root):
        if automation.id == automation_id:
            return automation
    raise AutomationValidationError(f"Automation not found: {automation_id}")


def upsert_automation(root: Path, automation: AutomationDefinition) -> AutomationDefinition:
    automation.updated_at = now_utc()
    if automation.next_run_at is None and automation.enabled:
        automation.next_run_at = calculate_next_run(automation.trigger)
    automation = AutomationDefinition.model_validate(automation.model_dump())
    with _store_lock(root):
        payload, _ = _migrate_payload(_read_unlocked(root), root)
        payload["automations"] = [
            item for item in payload["automations"] if item.get("id") != automation.id
        ] + [automation.to_dict()]
        _write_unlocked(root, payload)
    return automation


def create_automation(
    root: Path,
    *,
    name: str,
    description: str = "",
    source: Literal["chat", "teach", "migration"] = "chat",
    trigger: AutomationTrigger | dict[str, Any],
    job: AutomationJob | dict[str, Any],
    timezone_name: str | None = None,
    session_id: str = "",
    target_runtime: Literal["local", "github"] = "local",
    permission_references: list[str] | None = None,
    retry_policy: RetryPolicy | dict[str, Any] | None = None,
    misfire_policy: MisfirePolicy | dict[str, Any] | None = None,
    idempotency_key: str = "",
    deploy: bool = True,
) -> AutomationDefinition:
    trigger_model = trigger if isinstance(trigger, (CronTrigger, IntervalTrigger, OnceTrigger)) else TRIGGER_ADAPTER.validate_python(trigger)
    job_model = job if isinstance(job, (AgentPromptJob, ConnectorActionJob, ToolActionJob, TeachFlowJob, CommandJob)) else JOB_ADAPTER.validate_python(job)
    timezone_value = timezone_name or (
        trigger_model.timezone if isinstance(trigger_model, (CronTrigger, OnceTrigger)) else machine_timezone()
    )
    next_run_at = calculate_next_run(trigger_model, after=now_utc() - timedelta(microseconds=1))
    if isinstance(trigger_model, OnceTrigger) and next_run_at is None:
        raise AutomationValidationError(
            "One-time automation run_at must be strictly in the future."
        )
    if idempotency_key:
        for existing in list_automations(root):
            if existing.description.endswith(f"[idempotency:{idempotency_key}]"):
                return existing
        description = f"{description.rstrip()} [idempotency:{idempotency_key}]".strip()
    try:
        automation = AutomationDefinition(
            name=name,
            description=description,
            source=source,
            repository_id=repository_id_for_path(root.resolve()),
            workspace_path=str(root.resolve()),
            session_id=session_id,
            trigger=trigger_model,
            job=job_model,
            timezone=timezone_value,
            target_runtime=target_runtime,
            next_run_at=next_run_at,
            permission_references=list(permission_references or []),
            retry_policy=RetryPolicy.model_validate(retry_policy or {}),
            misfire_policy=MisfirePolicy.model_validate(misfire_policy or {}),
        )
    except ValidationError as exc:
        message = "; ".join(
            str(item.get("msg") or "invalid automation definition")
            for item in exc.errors(include_url=False)
        )
        raise AutomationValidationError(message) from exc
    if automation.job.type == "teach_flow":
        validate_teach_job(automation.job)
    emit_automation_event(root, "automation.planning", automation, status="running")
    upsert_automation(root, automation)
    emit_automation_event(root, "automation.created", automation, status="completed")
    if deploy:
        automation = reconcile_deployment(root, automation.id)
    return automation


def update_automation(root: Path, automation_id: str, changes: dict[str, Any]) -> AutomationDefinition:
    forbidden = {"id", "schema_version", "repository_id", "workspace_path", "source", "created_at"}
    if forbidden & changes.keys():
        raise AutomationValidationError(f"Immutable automation fields: {', '.join(sorted(forbidden & changes.keys()))}")
    current = get_automation(root, automation_id)
    payload = current.model_dump()
    payload.update(changes)
    if "trigger" in changes:
        payload["next_run_at"] = calculate_next_run(TRIGGER_ADAPTER.validate_python(changes["trigger"]))
    updated = AutomationDefinition.model_validate(payload)
    upsert_automation(root, updated)
    updated = reconcile_deployment(root, updated.id)
    emit_automation_event(root, "automation.updated", updated, status="completed")
    return updated


def delete_automation(root: Path, automation_id: str) -> AutomationDefinition:
    automation = get_automation(root, automation_id)
    remove_deployment(automation, root)
    with _store_lock(root):
        payload, _ = _migrate_payload(_read_unlocked(root), root)
        payload["automations"] = [item for item in payload["automations"] if item.get("id") != automation_id]
        payload["leases"].pop(automation_id, None)
        _write_unlocked(root, payload)
    emit_automation_event(root, "automation.deleted", automation, status="completed")
    return automation


def list_runs(root: Path, automation_id: str = "", *, limit: int = 50) -> list[dict[str, Any]]:
    rows = list(load_config(root)["runs"])
    if automation_id:
        rows = [row for row in rows if row.get("automation_id") == automation_id]
    return rows[-max(1, limit):]


def _executor_argv(automation: AutomationDefinition, root: Path) -> list[str]:
    executable = shutil.which("mana-agent") or "mana-agent"
    prefix = [executable]
    return prefix + [
        "automation", "execute", "--automation-id", automation.id,
        "--root-dir", str(root.resolve()),
    ]


def managed_marker(automation: AutomationDefinition) -> str:
    return f"# mana-agent:{automation.id}"


def schedule_command(automation: AutomationDefinition, root: Path) -> str:
    return shlex.join(_executor_argv(automation, root))


def read_crontab(runner=subprocess.run) -> str:
    result = runner(["crontab", "-l"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return result.stdout
    if "no crontab" in (result.stderr or "").lower():
        return ""
    raise AutomationValidationError(f"Unable to inspect crontab: {(result.stderr or '').strip()}")


def _cron_wakeup(automation: AutomationDefinition) -> str:
    # Exact due-time semantics live in the store. A minute wakeup supports
    # interval/once triggers without pretending they are calendar cron rules.
    return "* * * * *"


def _replace_cron_entry(contents: str, automation: AutomationDefinition, root: Path, *, present: bool) -> str:
    marker = managed_marker(automation)
    rows = [line for line in contents.splitlines() if marker not in line]
    if present:
        rows.append(f"{_cron_wakeup(automation)} {schedule_command(automation, root)} {marker}")
    return "\n".join(rows).strip() + ("\n" if rows else "")


def deploy_local_cron(
    automation: AutomationDefinition, root: Path, *, runner=subprocess.run,
) -> DeploymentState:
    if shutil.which("crontab") is None:
        raise AutomationValidationError("Persistent local scheduling is unavailable: crontab is not installed.")
    current = read_crontab(runner)
    desired = _replace_cron_entry(current, automation, root, present=automation.enabled)
    result = runner(["crontab", "-"], input=desired, capture_output=True, text=True, check=False)
    if result.returncode:
        raise AutomationValidationError(f"Unable to update crontab: {(result.stderr or '').strip()}")
    return DeploymentState(
        status="deployed" if automation.enabled else "disabled",
        backend="cron",
        artifact=managed_marker(automation),
    )


def _launchd_path(automation: AutomationDefinition) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"com.mana-agent.automation.{automation.id}.plist"


def _systemd_paths(automation: AutomationDefinition) -> tuple[Path, Path]:
    directory = Path.home() / ".config" / "systemd" / "user"
    base = f"mana-agent-automation-{automation.id}"
    return directory / f"{base}.service", directory / f"{base}.timer"


def deploy_systemd(
    automation: AutomationDefinition, root: Path, *, runner=subprocess.run,
) -> DeploymentState:
    service_path, timer_path = _systemd_paths(automation)
    unit_name = timer_path.name
    if not automation.enabled:
        runner(["systemctl", "--user", "disable", "--now", unit_name], capture_output=True, text=True, check=False)
        service_path.unlink(missing_ok=True)
        timer_path.unlink(missing_ok=True)
        runner(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True, check=False)
        return DeploymentState(status="disabled", backend="systemd", artifact=str(timer_path))
    service_path.parent.mkdir(parents=True, exist_ok=True)
    service_path.write_text("\n".join([
        "[Unit]", f"Description=Mana-Agent automation {automation.id}",
        "[Service]", "Type=oneshot", f"ExecStart={schedule_command(automation, root)}", "",
    ]), encoding="utf-8")
    timer_path.write_text("\n".join([
        "[Unit]", f"Description=Wake Mana-Agent automation {automation.id}",
        "[Timer]", "OnBootSec=1min", "OnUnitActiveSec=1min", "AccuracySec=1s",
        "Persistent=true", f"Unit={service_path.name}",
        "[Install]", "WantedBy=timers.target", "",
    ]), encoding="utf-8")
    runner(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True, check=False)
    result = runner(
        ["systemctl", "--user", "enable", "--now", unit_name],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise AutomationValidationError(f"Unable to deploy systemd user timer: {(result.stderr or '').strip()}")
    return DeploymentState(status="deployed", backend="systemd", artifact=str(timer_path))


def deploy_launchd(
    automation: AutomationDefinition, root: Path, *, runner=subprocess.run,
) -> DeploymentState:
    path = _launchd_path(automation)
    if not automation.enabled:
        path.unlink(missing_ok=True)
        runner(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], capture_output=True, text=True, check=False)
        return DeploymentState(status="disabled", backend="launchd", artifact=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    argv = _executor_argv(automation, root)
    escaped = "".join(f"<string>{_xml_escape(value)}</string>" for value in argv)
    calendar = ""
    interval = "<key>StartInterval</key><integer>60</integer>"
    if automation.trigger.type == "cron":
        # launchd calendar dictionaries cannot express every POSIX cron union
        # safely; the minute wakeup preserves canonical due checking.
        interval = "<key>StartInterval</key><integer>60</integer>"
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict>'
        f"<key>Label</key><string>com.mana-agent.automation.{automation.id}</string>"
        f"<key>ProgramArguments</key><array>{escaped}</array>{calendar}{interval}"
        "<key>RunAtLoad</key><true/></dict></plist>"
    )
    path.write_text(content, encoding="utf-8")
    runner(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], capture_output=True, text=True, check=False)
    result = runner(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], capture_output=True, text=True, check=False)
    if result.returncode and "already loaded" not in (result.stderr or "").lower():
        raise AutomationValidationError(f"Unable to deploy launchd job: {(result.stderr or '').strip()}")
    return DeploymentState(status="deployed", backend="launchd", artifact=str(path))


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def deploy_windows(
    automation: AutomationDefinition, root: Path, *, runner=subprocess.run,
) -> DeploymentState:
    name = f"ManaAgent-{automation.id}"
    if not automation.enabled:
        runner(["schtasks", "/Delete", "/TN", name, "/F"], capture_output=True, text=True, check=False)
        return DeploymentState(status="disabled", backend="windows-task-scheduler", artifact=name)
    command = subprocess.list2cmdline(_executor_argv(automation, root))
    result = runner(
        ["schtasks", "/Create", "/TN", name, "/TR", command, "/SC", "MINUTE", "/MO", "1", "/F"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise AutomationValidationError(f"Unable to deploy Windows task: {(result.stderr or result.stdout or '').strip()}")
    return DeploymentState(status="deployed", backend="windows-task-scheduler", artifact=name)


def workflow_path(root: Path, automation: AutomationDefinition) -> Path:
    return root / ".github" / "workflows" / f"{_MANAGED_PREFIX}{automation.id}.yml"


def github_definition_path(root: Path, automation: AutomationDefinition) -> Path:
    return root / ".mana" / "automations" / "managed" / f"{automation.id}.json"


def render_workflow(automation: AutomationDefinition, root: Path | None = None) -> str:
    if automation.target_runtime != "github":
        raise AutomationValidationError("GitHub workflow deployment was not explicitly selected.")
    if automation.job.type not in {"agent_prompt", "tool_action", "command"}:
        raise AutomationValidationError("This automation requires authenticated local capabilities.")
    expression = "* * * * *"
    command = shlex.join(_executor_argv(automation, root or Path(".")))
    definition = f".mana/automations/managed/{automation.id}.json"
    return "\n".join([
        "# Managed by mana-agent. Do not edit manually.",
        f"name: {json.dumps(f'Mana Agent automation: {automation.name}')}",
        "on:",
        "  schedule:",
        f'    - cron: "{expression}"',
        "  workflow_dispatch:",
        "permissions:",
        "  contents: read",
        "jobs:",
        "  execute:",
        "    runs-on: ubuntu-latest",
        "    env:",
        "      MANA_HOME: ${{ runner.temp }}/mana",
        "    steps:",
        "      - uses: actions/checkout@v4",
        "      - uses: actions/setup-python@v5",
        "        with:",
        '          python-version: "3.12"',
        "      - run: python -m pip install .",
        "      - name: Restore canonical automation definition",
        "        run: |",
        "          python - <<'PY'",
        "          import json",
        "          from pathlib import Path",
        "          from mana_agent.automations.service import AutomationDefinition, upsert_automation",
        f"          upsert_automation(Path('.'), AutomationDefinition.model_validate(json.loads(Path('{definition}').read_text())))",
        "          PY",
        f"      - run: {command}",
        "      - if: always()",
        "        uses: actions/upload-artifact@v4",
        "        with:",
        f"          name: mana-agent-automation-{automation.id}",
        "          path: ${{ runner.temp }}/mana/",
        "",
    ])


def deploy_github(automation: AutomationDefinition, root: Path) -> DeploymentState:
    path = workflow_path(root, automation)
    definition_path = github_definition_path(root, automation)
    if not automation.enabled:
        path.unlink(missing_ok=True)
        definition_path.unlink(missing_ok=True)
        return DeploymentState(status="disabled", backend="github-actions", artifact=str(path.relative_to(root)))
    path.parent.mkdir(parents=True, exist_ok=True)
    definition_path.parent.mkdir(parents=True, exist_ok=True)
    definition_path.write_text(
        json.dumps(automation.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    path.write_text(render_workflow(automation, root), encoding="utf-8")
    return DeploymentState(
        status="blocked",
        backend="github-actions",
        artifact=str(path.relative_to(root)),
        blocked_reason="Workflow must be committed to the repository default branch before GitHub can run it.",
    )


def reconcile_deployment(
    root: Path, automation_id: str, *, runner=subprocess.run,
) -> AutomationDefinition:
    automation = get_automation(root, automation_id)
    try:
        if automation.target_runtime == "github":
            state = deploy_github(automation, root)
        elif automation.enabled and automation.job.type == "connector_action" and not automation.permission_references:
            raise AutomationValidationError(
                "Connector automation deployment is blocked until an account or permission reference is selected."
            )
        elif platform.system() == "Windows":
            state = deploy_windows(automation, root, runner=runner)
        elif platform.system() == "Darwin" and shutil.which("launchctl"):
            state = deploy_launchd(automation, root, runner=runner)
        elif platform.system() == "Linux" and shutil.which("systemctl"):
            state = deploy_systemd(automation, root, runner=runner)
        else:
            state = deploy_local_cron(automation, root, runner=runner)
    except (AutomationValidationError, OSError) as exc:
        state = DeploymentState(status="blocked", blocked_reason=str(exc))
    automation.deployment = state
    upsert_automation(root, automation)
    event = "automation.deployed" if state.status == "deployed" else "automation.deployment.blocked"
    emit_automation_event(root, event, automation, status="completed" if state.status == "deployed" else "blocked")
    return automation


def remove_deployment(automation: AutomationDefinition, root: Path, *, runner=subprocess.run) -> None:
    if automation.target_runtime == "github":
        workflow_path(root, automation).unlink(missing_ok=True)
        github_definition_path(root, automation).unlink(missing_ok=True)
        return
    if automation.deployment.backend == "launchd":
        path = _launchd_path(automation)
        runner(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], capture_output=True, text=True, check=False)
        path.unlink(missing_ok=True)
    elif automation.deployment.backend == "windows-task-scheduler":
        runner(["schtasks", "/Delete", "/TN", f"ManaAgent-{automation.id}", "/F"], capture_output=True, text=True, check=False)
    elif automation.deployment.backend == "systemd":
        service_path, timer_path = _systemd_paths(automation)
        runner(["systemctl", "--user", "disable", "--now", timer_path.name], capture_output=True, text=True, check=False)
        service_path.unlink(missing_ok=True)
        timer_path.unlink(missing_ok=True)
        runner(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True, check=False)
    elif shutil.which("crontab"):
        current = read_crontab(runner)
        desired = _replace_cron_entry(current, automation, root, present=False)
        runner(["crontab", "-"], input=desired, capture_output=True, text=True, check=False)


def deployment_status(automation: AutomationDefinition, root: Path, *, runner=subprocess.run) -> dict[str, Any]:
    state = automation.deployment.model_dump(mode="json")
    healthy = False
    if state["backend"] == "github-actions":
        healthy = state["status"] == "deployed" and workflow_path(root, automation).exists()
    elif state["backend"] == "launchd":
        healthy = _launchd_path(automation).exists()
    elif state["backend"] == "windows-task-scheduler":
        result = runner(["schtasks", "/Query", "/TN", f"ManaAgent-{automation.id}"], capture_output=True, text=True, check=False)
        healthy = result.returncode == 0
    elif state["backend"] == "systemd":
        _service_path, timer_path = _systemd_paths(automation)
        healthy = timer_path.exists()
    elif state["backend"] == "cron":
        try:
            healthy = managed_marker(automation) in read_crontab(runner)
        except AutomationValidationError:
            healthy = False
    return {
        "automation": automation.to_dict(),
        "interpreted_trigger": human_trigger(automation.trigger),
        "deployment_healthy": healthy if automation.enabled else state["status"] == "disabled",
        "recent_runs": list_runs(root, automation.id, limit=10),
    }


def _acquire_lease(root: Path, automation: AutomationDefinition, *, force: bool) -> AutomationRun | None:
    now = now_utc()
    with _store_lock(root):
        payload, _ = _migrate_payload(_read_unlocked(root), root)
        lease = dict(payload["leases"].get(automation.id) or {})
        expires = _parse_datetime(lease.get("expires_at"))
        if expires and expires > now:
            return None
        current_raw = next((item for item in payload["automations"] if item.get("id") == automation.id), None)
        if current_raw is None:
            raise AutomationValidationError(f"Automation not found: {automation.id}")
        current = AutomationDefinition.from_dict(current_raw)
        if not current.enabled and not force:
            return None
        if not force and (current.next_run_at is None or current.next_run_at > now):
            return None
        attempt = 1
        if (
            current.recent_execution is not None
            and current.recent_execution.status in {"failed", "blocked"}
            and current.recent_execution.attempt < current.retry_policy.maximum_attempts
        ):
            attempt = current.recent_execution.attempt + 1
        run = AutomationRun(automation_id=automation.id, attempt=attempt)
        payload["leases"][automation.id] = {
            "run_id": run.id,
            "claimed_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=LEASE_SECONDS)).isoformat(),
            "pid": os.getpid(),
        }
        payload["runs"].append(run.model_dump(mode="json"))
        current.recent_execution = ExecutionSummary(
            run_id=run.id, status="running", started_at=run.started_at, attempt=attempt,
        )
        payload["automations"] = [
            item if item.get("id") != automation.id else current.to_dict()
            for item in payload["automations"]
        ]
        _write_unlocked(root, payload)
        return run


def execute_automation(
    root: Path,
    automation_id: str,
    *,
    force: bool = False,
    job_executor: Callable[[AutomationDefinition, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    automation = get_automation(root, automation_id)
    if (
        not force
        and automation.next_run_at is not None
        and automation.next_run_at <= now_utc()
        and automation.misfire_policy.mode == "skip"
        and (now_utc() - automation.next_run_at).total_seconds()
        > automation.misfire_policy.grace_seconds
    ):
        skipped = _acquire_lease(root, automation, force=False)
        if skipped is None:
            return {"ok": True, "executed": False, "reason": "not_due_or_already_claimed", "automation_id": automation_id}
        completed = _finish_run(
            root, automation_id, skipped.id, status="skipped",
            output="", error="Missed occurrence exceeded the configured grace period.",
        )
        return {"ok": True, "executed": False, "reason": "misfire_skipped", **completed}
    run = _acquire_lease(root, automation, force=force)
    if run is None:
        return {"ok": True, "executed": False, "reason": "not_due_or_already_claimed", "automation_id": automation_id}
    emit_automation_event(root, "automation.run.started", automation, run_id=run.id, status="running")
    status: Literal["succeeded", "failed", "blocked"] = "succeeded"
    output = ""
    error = ""
    try:
        refreshed = get_automation(root, automation_id)
        result = (job_executor or _execute_job)(refreshed, root)
        output = _bounded(json.dumps(redact_secrets(result), ensure_ascii=False, default=str))
        if result.get("blocked"):
            status = "blocked"
            error = str(result.get("reason") or "Required capability is unavailable.")
        elif result.get("ok") is False:
            status = "failed"
            error = str(result.get("error") or "Automation job failed.")
    except Exception as exc:
        status = "failed"
        error = _bounded(str(exc), 2_000)
    try:
        completed = _finish_run(root, automation_id, run.id, status=status, output=output, error=error)
    except Exception:
        _release_lease(root, automation_id, run.id)
        raise
    event = "automation.run.completed" if status == "succeeded" else "automation.run.failed"
    emit_automation_event(root, event, get_automation(root, automation_id), run_id=run.id, status=status)
    return {"ok": status == "succeeded", "executed": True, **completed}


def _finish_run(
    root: Path, automation_id: str, run_id: str, *,
    status: Literal["succeeded", "failed", "blocked", "skipped"], output: str, error: str,
) -> dict[str, Any]:
    finished = now_utc()
    with _store_lock(root):
        payload, _ = _migrate_payload(_read_unlocked(root), root)
        raw = next(item for item in payload["automations"] if item.get("id") == automation_id)
        automation = AutomationDefinition.from_dict(raw)
        automation.last_run_at = finished
        current_attempt = (
            automation.recent_execution.attempt if automation.recent_execution else 1
        )
        retry_due = (
            status in {"failed", "blocked"}
            and current_attempt < automation.retry_policy.maximum_attempts
        )
        if retry_due:
            automation.next_run_at = finished + timedelta(
                seconds=automation.retry_policy.backoff_seconds
            )
        elif automation.trigger.type == "once":
            automation.enabled = False
            automation.next_run_at = None
        else:
            advance_after = finished
            if automation.misfire_policy.mode == "catch_up" and automation.next_run_at:
                advance_after = automation.next_run_at
            automation.next_run_at = calculate_next_run(
                automation.trigger, after=advance_after, previous_claimed_at=advance_after,
            )
        summary = ExecutionSummary(
            run_id=run_id, status=status, started_at=automation.recent_execution.started_at if automation.recent_execution else finished,
            finished_at=finished, output_summary=output, error=error,
            attempt=current_attempt,
        )
        automation.recent_execution = summary
        for row in payload["runs"]:
            if row.get("id") == run_id:
                row.update({
                    "status": status, "finished_at": finished.isoformat(),
                    "output_summary": output, "error": error,
                    "next_run_at": automation.next_run_at.isoformat() if automation.next_run_at else None,
                })
                break
        payload["automations"] = [
            item if item.get("id") != automation_id else automation.to_dict()
            for item in payload["automations"]
        ]
        payload["leases"].pop(automation_id, None)
        _write_unlocked(root, payload)
    return {
        "automation_id": automation_id, "run_id": run_id, "status": status,
        "output_summary": output, "error": error,
        "next_run_at": automation.next_run_at.isoformat() if automation.next_run_at else None,
    }


def _release_lease(root: Path, automation_id: str, run_id: str) -> None:
    with _store_lock(root):
        payload, _ = _migrate_payload(_read_unlocked(root), root)
        lease = dict(payload["leases"].get(automation_id) or {})
        if lease.get("run_id") == run_id:
            payload["leases"].pop(automation_id, None)
            _write_unlocked(root, payload)


def _execute_job(automation: AutomationDefinition, root: Path) -> dict[str, Any]:
    missing = _missing_permissions(automation.permission_references)
    if missing:
        return {"ok": False, "blocked": True, "reason": f"Missing permission references: {', '.join(missing)}"}
    job = automation.job
    if job.type == "teach_flow":
        validate_teach_job(job)
        from mana_agent.teach.service import TeachService
        result = TeachService().replay(
            job.flow_id, version=job.flow_version, mode="normal", inputs=job.inputs,
        )
        return {"ok": result.verification_status == "verified", **result.model_dump(mode="json")}
    if job.type == "command":
        cwd = Path(job.cwd).expanduser().resolve() if job.cwd else root.resolve()
        if cwd != root.resolve() and not cwd.is_relative_to(root.resolve()):
            raise AutomationValidationError("Command working directory must remain inside the workspace.")
        completed = subprocess.run(job.argv, cwd=cwd, capture_output=True, text=True, timeout=900, check=False)
        return {
            "ok": completed.returncode == 0, "returncode": completed.returncode,
            "stdout": _bounded(completed.stdout), "stderr": _bounded(completed.stderr),
        }
    # Recreate a fresh gateway; no chat object or conversation state is reused.
    from mana_agent.gateway import AgentChatGateway
    prompt = job.prompt if job.type in {"agent_prompt", "connector_action"} else (
        f"Execute the model-selected tool {job.tool_name} with these validated arguments: "
        f"{json.dumps(job.arguments, ensure_ascii=False)}"
    )
    if job.type == "connector_action":
        prompt = (
            f"Execute connector {job.connector} action {job.action} with these validated arguments: "
            f"{json.dumps(job.arguments, ensure_ascii=False)}. {job.prompt}"
        )
    gateway = AgentChatGateway(root)
    session_id = f"automation_{automation.id}_{uuid.uuid4().hex[:12]}"
    try:
        answer = gateway.send(session_id, prompt)
    finally:
        gateway.close_session(session_id)
    return {"ok": True, "answer": _bounded(answer)}


def _missing_permissions(references: list[str]) -> list[str]:
    # Permission references are opaque stable identifiers. Explicitly marked
    # missing/revoked references block; providers perform their own live check.
    return [item for item in references if item.startswith(("missing://", "revoked://"))]


def validate_teach_job(job: TeachFlowJob) -> None:
    from mana_agent.teach.models import TeachError
    from mana_agent.teach.service import TeachService
    try:
        flow = TeachService().storage.load_flow(job.flow_id, job.flow_version)
    except TeachError as exc:
        raise AutomationValidationError(str(exc)) from exc
    if not flow.steps:
        raise AutomationValidationError("Teach flows with no steps cannot be automated.")
    if flow.status not in {"verified", "active"} or flow.statistics.verified_replays < 1:
        raise AutomationValidationError(
            "Teach flow must be reviewed and have a successful verified replay before automation."
        )
    unknown = set(job.inputs) - set(flow.inputs)
    missing = {key for key, value in flow.inputs.items() if value.required and value.default is None} - set(job.inputs)
    if unknown or missing:
        raise AutomationValidationError(
            f"Teach flow inputs are invalid (unknown={sorted(unknown)}, missing={sorted(missing)})."
        )
    if set(flow.permissions) - set(job.required_permission_scopes):
        raise AutomationValidationError("Teach automation must persist every required permission scope.")
    if not job.verification:
        raise AutomationValidationError("Teach automation requires verification metadata.")


def _bounded(value: Any, limit: int = MAX_OUTPUT_CHARS) -> str:
    text = str(value or "")
    return text[-limit:]


def emit_automation_event(
    root: Path, event_type: str, automation: AutomationDefinition, *,
    run_id: str = "", status: str = "running",
) -> None:
    try:
        from mana_agent.services.execution_event_hub import get_execution_event_hub
        get_execution_event_hub().emit(
            event_type,
            title=automation.name,
            message=human_trigger(automation.trigger),
            conversation_id=automation.session_id or automation.id,
            execution_id=run_id or automation.id,
            repository_id=automation.repository_id,
            status=status,
            metadata={
                "automation_id": automation.id,
                "source": automation.source,
                "timezone": automation.timezone,
                "next_run_at": automation.next_run_at.isoformat() if automation.next_run_at else None,
                "deployment_status": automation.deployment.status,
                "blocked_reason": automation.deployment.blocked_reason,
            },
        )
    except Exception:
        # Event fan-out is observational; the canonical store remains the
        # authoritative result and must not be rolled back by a UI subscriber.
        pass


class AutomationService:
    """Small service boundary shared by chat, CLI, dashboard, and executors."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def create(self, **kwargs: Any) -> AutomationDefinition:
        return create_automation(self.root, **kwargs)

    def get(self, automation_id: str) -> AutomationDefinition:
        return get_automation(self.root, automation_id)

    def list(self) -> list[AutomationDefinition]:
        return list_automations(self.root)

    def status(self, automation_id: str) -> dict[str, Any]:
        return deployment_status(self.get(automation_id), self.root)

    def update(self, automation_id: str, changes: dict[str, Any]) -> AutomationDefinition:
        return update_automation(self.root, automation_id, changes)

    def delete(self, automation_id: str) -> AutomationDefinition:
        return delete_automation(self.root, automation_id)

    def execute(self, automation_id: str, *, force: bool = False) -> dict[str, Any]:
        return execute_automation(self.root, automation_id, force=force)
