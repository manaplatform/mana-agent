"""Typed specialist-lane contracts used by the production gateway."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from mana_agent.server.tools import SERVER_TOOL_SPECS


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class LaneId(_ValueEnum):
    ARTIFACT = "artifact"
    MEDIA = "media"
    CODING = "coding"
    RESEARCH = "research"
    REVIEW = "review"
    VERIFY = "verify"
    RELEASE = "release"
    OPERATIONS = "operations"


class LanePriority(_ValueEnum):
    CRITICAL = "critical"
    INTERACTIVE = "interactive"
    HIGH = "high"
    NORMAL = "normal"
    BACKGROUND = "background"
    MAINTENANCE = "maintenance"


PRIORITY_ORDER: dict[LanePriority, int] = {
    LanePriority.CRITICAL: 0,
    LanePriority.INTERACTIVE: 10,
    LanePriority.HIGH: 20,
    LanePriority.NORMAL: 30,
    LanePriority.BACKGROUND: 40,
    LanePriority.MAINTENANCE: 50,
}


class LockMode(_ValueEnum):
    NONE = "none"
    REPOSITORY_READ = "repository_read"
    REPOSITORY_WRITE = "repository_write"
    WORKSPACE_WRITE = "workspace_write"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"


class LaneTaskState(_ValueEnum):
    CREATED = "created"
    ROUTING = "routing"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    HANDOFF = "handoff"
    VERIFYING = "verifying"
    SELECTING_WINNER = "selecting_winner"
    APPLYING = "applying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    BUDGET_EXHAUSTED = "budget_exhausted"


ACTIVE_LANE_STATES = frozenset(
    {
        LaneTaskState.ROUTING, LaneTaskState.QUEUED, LaneTaskState.RUNNING,
        LaneTaskState.WAITING, LaneTaskState.HANDOFF, LaneTaskState.VERIFYING,
        LaneTaskState.SELECTING_WINNER, LaneTaskState.APPLYING,
    }
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 1
    retry_backoff_seconds: float = 0.0
    retryable_errors: tuple[str, ...] = ("transport", "provider_unavailable")

    @classmethod
    def from_value(cls, value: Any) -> "RetryPolicy":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("retry_policy must be an object")
        result = cls(
            max_attempts=int(value.get("max_attempts", 1)),
            retry_backoff_seconds=float(value.get("retry_backoff_seconds", 0.0)),
            retryable_errors=tuple(str(item) for item in value.get("retryable_errors", cls().retryable_errors)),
        )
        if result.max_attempts < 1 or result.retry_backoff_seconds < 0:
            raise ValueError("retry_policy values must be non-negative and max_attempts >= 1")
        return result


@dataclass(frozen=True, slots=True)
class LaneContract:
    lane_id: LaneId
    display_name: str
    description: str
    owns: tuple[str, ...]
    handoff_targets: tuple[LaneId, ...]
    allowed_tools: tuple[str, ...]
    denied_tools: tuple[str, ...]
    allowed_models: tuple[str, ...]
    max_concurrent_jobs: int
    max_subagents: int
    token_budget: int
    cost_budget: float
    default_priority: LanePriority
    can_create_subagents: bool
    requires_repository: bool
    requires_write_access: bool
    lock_policy: LockMode
    timeout_seconds: int
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> "LaneContract":
        if not self.display_name.strip() or not self.description.strip() or not self.owns:
            raise ValueError(f"lane {self.lane_id.value} requires a name, description, and ownership")
        if self.max_concurrent_jobs < 1:
            raise ValueError(f"lane {self.lane_id.value} max_concurrent_jobs must be >= 1")
        if self.max_subagents < 0 or self.token_budget < 1 or self.cost_budget < 0:
            raise ValueError(f"lane {self.lane_id.value} contains an invalid budget or subagent limit")
        if self.timeout_seconds < 1:
            raise ValueError(f"lane {self.lane_id.value} timeout_seconds must be >= 1")
        if self.requires_write_access and self.lock_policy in {LockMode.NONE, LockMode.REPOSITORY_READ, LockMode.FILE_READ}:
            raise ValueError(f"lane {self.lane_id.value} requires write access but has a read-only lock policy")
        if not self.can_create_subagents and self.max_subagents:
            raise ValueError(f"lane {self.lane_id.value} disables subagents but max_subagents is non-zero")
        return self


READ_CAPABILITIES = (
    "repository_read", "shell_read", "web_search", "browser", "git_read",
    "test_execution", "email", "calendar", "computer", "api",
)
WRITE_CAPABILITIES = (
    "repository_write", "shell_write", "git_write", "release", "deployment", "automation",
)


def default_lane_contracts() -> dict[LaneId, LaneContract]:
    contracts = {
        LaneId.ARTIFACT: LaneContract(
            lane_id=LaneId.ARTIFACT, display_name="Artifact", description="Creates and updates user artifacts outside repository workflows.",
            owns=("document artifacts", "spreadsheet artifacts", "PDF artifacts"), handoff_targets=(),
            allowed_tools=("artifact_read", "artifact_write"), denied_tools=WRITE_CAPABILITIES + ("secrets",), allowed_models=(),
            max_concurrent_jobs=2, max_subagents=0, token_budget=30_000, cost_budget=10.0,
            default_priority=LanePriority.INTERACTIVE, can_create_subagents=False, requires_repository=False,
            requires_write_access=False, lock_policy=LockMode.NONE, timeout_seconds=900,
        ),
        LaneId.MEDIA: LaneContract(
            lane_id=LaneId.MEDIA,
            display_name="Media",
            description="Executes configured image, voice, and video generation jobs.",
            owns=("image generation", "voice generation", "video generation", "media job lifecycle"),
            handoff_targets=(),
            allowed_tools=(
                "media.image.generate",
                "media.voice.generate",
                "media.video.generate",
                "media.artifact.write",
                "media.status.read",
                "media.generation.cancel",
            ),
            denied_tools=WRITE_CAPABILITIES + ("secrets",),
            allowed_models=(),
            max_concurrent_jobs=2,
            max_subagents=0,
            token_budget=10_000,
            cost_budget=100.0,
            default_priority=LanePriority.INTERACTIVE,
            can_create_subagents=False,
            requires_repository=False,
            requires_write_access=False,
            lock_policy=LockMode.NONE,
            timeout_seconds=3600,
        ),
        LaneId.CODING: LaneContract(
            lane_id=LaneId.CODING, display_name="Coding", description="Implements repository changes.",
            owns=("implementation", "repository mutations"),
            handoff_targets=(LaneId.REVIEW, LaneId.VERIFY, LaneId.RELEASE),
            allowed_tools=READ_CAPABILITIES + ("repository_write", "shell_write", "git_read"),
            denied_tools=("deployment", "release", "secrets", "email", "calendar"),
            allowed_models=(), max_concurrent_jobs=2, max_subagents=2,
            token_budget=80_000, cost_budget=25.0, default_priority=LanePriority.INTERACTIVE,
            can_create_subagents=True, requires_repository=True, requires_write_access=True,
            lock_policy=LockMode.FILE_WRITE, timeout_seconds=1800,
        ),
        LaneId.RESEARCH: LaneContract(
            lane_id=LaneId.RESEARCH, display_name="Research", description="Investigates external and repository information.",
            owns=("web research", "documentation", "dependencies", "external investigation"),
            handoff_targets=(LaneId.CODING,),
            allowed_tools=("repository_read", "shell_read", "web_search", "browser", "git_read", "email", "calendar", "api"),
            denied_tools=WRITE_CAPABILITIES + ("secrets",), allowed_models=(),
            max_concurrent_jobs=4, max_subagents=4, token_budget=50_000, cost_budget=20.0,
            default_priority=LanePriority.INTERACTIVE, can_create_subagents=True,
            requires_repository=False, requires_write_access=False, lock_policy=LockMode.NONE,
            timeout_seconds=900,
        ),
        LaneId.REVIEW: LaneContract(
            lane_id=LaneId.REVIEW, display_name="Review", description="Reviews correctness, security, and architecture.",
            owns=("correctness review", "security review", "architectural review"),
            handoff_targets=(LaneId.CODING, LaneId.VERIFY),
            allowed_tools=("repository_read", "shell_read", "git_read"), denied_tools=WRITE_CAPABILITIES + ("secrets",),
            allowed_models=(), max_concurrent_jobs=2, max_subagents=0, token_budget=35_000, cost_budget=15.0,
            default_priority=LanePriority.HIGH, can_create_subagents=False, requires_repository=True,
            requires_write_access=False, lock_policy=LockMode.REPOSITORY_READ, timeout_seconds=900,
        ),
        LaneId.VERIFY: LaneContract(
            lane_id=LaneId.VERIFY, display_name="Verify", description="Runs tests, lint, types, and validation.",
            owns=("tests", "linting", "type checking", "validation"),
            handoff_targets=(LaneId.CODING,),
            allowed_tools=("repository_read", "shell_read", "test_execution", "git_read"),
            denied_tools=("repository_write", "git_write", "deployment", "release", "secrets"),
            allowed_models=(), max_concurrent_jobs=2, max_subagents=0, token_budget=25_000, cost_budget=10.0,
            default_priority=LanePriority.HIGH, can_create_subagents=False, requires_repository=True,
            requires_write_access=False, lock_policy=LockMode.REPOSITORY_READ, timeout_seconds=1200,
        ),
        LaneId.RELEASE: LaneContract(
            lane_id=LaneId.RELEASE, display_name="Release", description="Prepares versions, changelogs, and packages.",
            owns=("versioning", "changelog", "packaging", "publishing preparation"),
            handoff_targets=(LaneId.OPERATIONS,),
            allowed_tools=("repository_read", "repository_write", "shell_read", "shell_write", "git_read", "release", "test_execution"),
            denied_tools=("deployment", "secrets", "email", "calendar"), allowed_models=(),
            max_concurrent_jobs=1, max_subagents=0, token_budget=30_000, cost_budget=12.0,
            default_priority=LanePriority.NORMAL, can_create_subagents=False, requires_repository=True,
            requires_write_access=True, lock_policy=LockMode.REPOSITORY_WRITE, timeout_seconds=1800,
        ),
        LaneId.OPERATIONS: LaneContract(
            lane_id=LaneId.OPERATIONS, display_name="Operations", description="Handles deployment, infrastructure, and monitoring.",
            owns=("deployment", "infrastructure checks", "monitoring"),
            handoff_targets=(LaneId.CODING,),
            allowed_tools=("shell_read", "shell_write", "deployment", "automation", "canvas", "browser", "git_read", "computer", "remote_ssh_execute", "server", "api", "mcp"),
            denied_tools=("repository_write", "release", "secrets", "email", "calendar"), allowed_models=(),
            max_concurrent_jobs=1, max_subagents=0, token_budget=25_000, cost_budget=12.0,
            default_priority=LanePriority.NORMAL, can_create_subagents=False, requires_repository=False,
            requires_write_access=True, lock_policy=LockMode.WORKSPACE_WRITE, timeout_seconds=1800,
        ),
    }
    return {key: value.validate() for key, value in contracts.items()}


_OVERRIDABLE_FIELDS = frozenset({
    "enabled", "max_concurrent_jobs", "max_subagents", "token_budget", "cost_budget",
    "priority", "default_priority", "timeout_seconds", "allowed_models",
})


def configured_lane_contracts(overrides: Mapping[str, Any] | None = None) -> dict[LaneId, LaneContract]:
    contracts = default_lane_contracts()
    for raw_lane, raw_values in (overrides or {}).items():
        try:
            lane_id = LaneId(str(raw_lane))
        except ValueError as exc:
            raise ValueError(f"unknown specialist lane: {raw_lane}") from exc
        if not isinstance(raw_values, Mapping):
            raise ValueError(f"lane override for {lane_id.value} must be an object")
        unknown = set(raw_values) - _OVERRIDABLE_FIELDS
        if unknown:
            raise ValueError(f"unsupported {lane_id.value} lane settings: {', '.join(sorted(unknown))}")
        values = dict(raw_values)
        if "priority" in values:
            values["default_priority"] = values.pop("priority")
        if "default_priority" in values:
            values["default_priority"] = LanePriority(str(values["default_priority"]))
        if "allowed_models" in values:
            if not isinstance(values["allowed_models"], (list, tuple)):
                raise ValueError(f"{lane_id.value}.allowed_models must be a list")
            values["allowed_models"] = tuple(str(item) for item in values["allowed_models"])
        contracts[lane_id] = replace(contracts[lane_id], **values).validate()
    return contracts


INTENT_LANES: dict[str, LaneId] = {
    "edit": LaneId.CODING,
    "plan": LaneId.CODING,
    "web_research": LaneId.RESEARCH,
    "repo_search": LaneId.RESEARCH,
    "analyze": LaneId.RESEARCH,
    "review": LaneId.REVIEW,
    "verify": LaneId.VERIFY,
}

ENTRY_ROUTE_LANES: dict[str, LaneId] = {
    "multi_task": LaneId.RESEARCH,
    "artifact": LaneId.ARTIFACT,
    "media": LaneId.MEDIA,
    "coding": LaneId.CODING,
    "mcp": LaneId.OPERATIONS,
    "browser": LaneId.RESEARCH,
    "search": LaneId.RESEARCH,
    "github": LaneId.RESEARCH,
    "repository": LaneId.RESEARCH,
    "memory": LaneId.RESEARCH,
    "gmail": LaneId.RESEARCH,
    "calendar": LaneId.RESEARCH,
    "computer": LaneId.OPERATIONS,
    "automation": LaneId.OPERATIONS,
    "api": LaneId.OPERATIONS,
    "canvas": LaneId.OPERATIONS,
    "remote_execution": LaneId.OPERATIONS,
    "server": LaneId.OPERATIONS,
    "conversation": LaneId.RESEARCH,
    "unsupported": LaneId.RESEARCH,
    "capability_error": LaneId.RESEARCH,
}


def select_lane(*, entry_route: str = "", intent: str = "", model_lane: str | LaneId | None = None) -> LaneId:
    """Select from validated decisions; no user-text keyword classifier is used."""
    if model_lane is not None:
        try:
            return model_lane if isinstance(model_lane, LaneId) else LaneId(str(model_lane))
        except ValueError:
            pass
    if intent in INTENT_LANES:
        return INTENT_LANES[intent]
    if entry_route in ENTRY_ROUTE_LANES:
        return ENTRY_ROUTE_LANES[entry_route]
    raise ValueError("No valid specialist lane decision was available. No fallback action was executed.")


TOOL_CAPABILITIES: dict[str, frozenset[str]] = {
    "repo_search": frozenset({"repository_read"}), "repo_batch_search": frozenset({"repository_read"}),
    "read_file": frozenset({"repository_read"}), "repo_batch_read": frozenset({"repository_read"}),
    "generate_image": frozenset({"media.image.generate", "media.artifact.write"}),
    "generate_voice": frozenset({"media.voice.generate", "media.artifact.write"}),
    "generate_video": frozenset({"media.video.generate", "media.artifact.write"}),
    "get_media_generation_status": frozenset({"media.status.read"}),
    "cancel_media_generation": frozenset({"media.generation.cancel"}),
    "list_files": frozenset({"repository_read"}), "find_symbols": frozenset({"repository_read"}),
    "semantic_search": frozenset({"repository_read"}), "call_graph": frozenset({"repository_read"}),
    "read_skill": frozenset({"repository_read"}),
    "edit_file": frozenset({"repository_write"}), "multi_edit_file": frozenset({"repository_write"}),
    "apply_patch": frozenset({"repository_write"}), "apply_patch_batch": frozenset({"repository_write"}),
    "write_file": frozenset({"repository_write"}), "create_file": frozenset({"repository_write"}),
    "delete_file": frozenset({"repository_write"}), "run_command": frozenset({"shell_write"}),
    "run_script_once": frozenset({"shell_write"}),
    "verify_project": frozenset({"test_execution"}), "run_tests": frozenset({"test_execution"}),
    "run_lint": frozenset({"test_execution"}), "web_search": frozenset({"web_search"}),
    "github_search": frozenset({"web_search"}), "git_status": frozenset({"git_read"}),
    "git_diff": frozenset({"git_read"}),
    "remote_ssh_execute": frozenset({"remote_ssh_execute"}),
}

for _server_tool in SERVER_TOOL_SPECS:
    TOOL_CAPABILITIES[_server_tool] = frozenset({"server"})

for _automation_tool in (
    "automation_create", "automation_get", "automation_list", "automation_status",
    "automation_update", "automation_delete", "automation_enable",
    "automation_disable", "automation_run_now",
):
    TOOL_CAPABILITIES[_automation_tool] = frozenset({"automation"})

for _api_tool in (
    "api_workflow_decide", "api_docs_inspect", "api_docs_import", "api_docs_import_semantic",
    "api_integrations_list", "api_integration_get",
    "api_integration_update", "api_integration_delete", "api_operations_search",
    "api_request_preview", "api_request_execute",
):
    TOOL_CAPABILITIES[_api_tool] = frozenset({"api"})

for _canvas_tool in (
    "canvas_create_surface", "canvas_update_components", "canvas_update_data",
    "canvas_delete_surface", "canvas_get_surface", "canvas_list_surfaces",
    "canvas_wait_for_action",
):
    TOOL_CAPABILITIES[_canvas_tool] = frozenset({"canvas"})

for _git_read_tool in (
    "git_log", "git_show", "git_branch", "git_remote", "git_help", "git_config_get",
):
    TOOL_CAPABILITIES[_git_read_tool] = frozenset({"git_read"})
for _git_write_tool in (
    "git_switch", "git_checkout", "git_create_branch", "git_add", "git_restore", "git_stash",
    "git_commit", "git_push", "git_pull", "git_fetch", "git_tag", "git_merge", "git_rebase",
    "git_revert", "git_reset", "git_clean", "git_config", "git_generic",
):
    TOOL_CAPABILITIES[_git_write_tool] = frozenset({"git_write"})
for _document_read_tool in ("document_detect", "document_read", "document_analyze", "document_query"):
    TOOL_CAPABILITIES[_document_read_tool] = frozenset({"repository_read"})
for _document_write_tool in ("document_create", "document_update", "document_delete"):
    TOOL_CAPABILITIES[_document_write_tool] = frozenset({"repository_write"})
try:
    from mana_agent.integrations.computer_control.tool_contracts import computer_tool_contracts
except ImportError:  # optional integration packaging failure remains fail-closed
    pass
else:
    for _computer_tool in computer_tool_contracts():
        TOOL_CAPABILITIES[_computer_tool.name] = frozenset({"computer"})


class LanePermissionError(PermissionError):
    pass


def validate_tool_permission(contract: LaneContract, tool_name: str, *, task_capabilities: tuple[str, ...] = ()) -> frozenset[str]:
    capabilities = TOOL_CAPABILITIES.get(tool_name)
    if capabilities is None:
        prefixes = {"browser_": "browser", "email_": "email", "calendar_": "calendar", "deploy_": "deployment", "release_": "release"}
        capability = next((value for prefix, value in prefixes.items() if tool_name.startswith(prefix)), None)
        if capability is None:
            raise LanePermissionError(f"Tool {tool_name!r} has no registered capability category")
        capabilities = frozenset({capability})
    denied = capabilities.intersection(contract.denied_tools)
    missing = capabilities.difference(contract.allowed_tools)
    task_missing = capabilities.difference(task_capabilities) if task_capabilities else frozenset()
    if denied or missing or task_missing:
        reason = sorted(denied or missing or task_missing)
        raise LanePermissionError(f"Lane {contract.lane_id.value} cannot use {tool_name}: {', '.join(reason)}")
    return capabilities
