from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class AgentRole(_ValueEnum):
    MAIN = "main"
    HEAD_DECISION = "head_decision"
    PLANNER = "planner"
    CODING = "coding"
    VERIFIER = "verifier"
    REVIEWER = "reviewer"
    TOOL = "tool"
    TOOL_WORKER = "tool_worker"
    RESEARCH = "research"
    SUMMARIZER = "summarizer"


class AgentState(_ValueEnum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"


class TaskStatus(_ValueEnum):
    NEW = "new"
    PLANNING = "planning"
    DISCUSSING = "discussing"
    ROUTED = "routed"
    WAITING_FOR_TOOLS = "waiting_for_tools"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    NEEDS_REVIEW = "needs_review"
    VERIFYING = "verifying"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class RiskLevel(_ValueEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QueueJobType(_ValueEnum):
    REPO_SEARCH = "repo_search"
    REPO_READ = "repo_read"
    REPO_BATCH_READ = "repo_batch_read"
    APPLY_PATCH = "apply_patch"
    SHELL = "shell"
    RUN_TESTS = "run_tests"
    RUN_LINT = "run_lint"
    GIT_STATUS = "git_status"
    GIT_DIFF = "git_diff"


class QueueJobStatus(_ValueEnum):
    QUEUED = "queued"
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(_ValueEnum):
    PROPOSAL = "proposal"
    QUESTION = "question"
    ANSWER = "answer"
    OBJECTION = "objection"
    EVIDENCE = "evidence"
    HANDOFF = "handoff"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL = "approval"
    REJECTION = "rejection"
    VERIFICATION_RESULT = "verification_result"
    SUMMARY = "summary"


class DiscussionStatus(_ValueEnum):
    OPEN = "open"
    WAITING = "waiting"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class DecisionStatus(_ValueEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class ToolPermissionLevel(_ValueEnum):
    READ_ONLY = "read_only"
    WRITE = "write"
    SHELL = "shell"
    GIT_HISTORY = "git_history"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return utc_now()


@dataclass
class HandoffRecord:
    from_agent_id: str
    to_agent_id: str
    task_id: str
    reason: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class VerificationResult:
    verification_id: str
    task_id: str
    verified_by_agent_id: str
    commands_run: list[str]
    passed: bool
    summary: str
    failures: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class AgentNode:
    agent_id: str
    role: AgentRole
    parent_agent_id: str | None = None
    capabilities: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    model_level: str = ""
    state: AgentState = AgentState.IDLE


@dataclass
class TaskBoardItem:
    task_id: str
    parent_task_id: str | None
    root_task_id: str
    title: str
    user_request: str
    normalized_goal: str
    status: TaskStatus
    priority: int
    risk_level: RiskLevel
    owner_agent_id: str | None = None
    supervisor_agent_id: str | None = None
    assigned_agent_ids: list[str] = field(default_factory=list)
    assigned_subagent_ids: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    files_to_inspect: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    queue_job_ids: list[str] = field(default_factory=list)
    verification_queue_job_ids: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    discussion_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    handoff_records: list[HandoffRecord] = field(default_factory=list)
    budget_records: list[dict[str, Any]] = field(default_factory=list)
    hierarchy_violations: list[dict[str, Any]] = field(default_factory=list)
    actual_tool_events: list[dict[str, Any]] = field(default_factory=list)
    delegated_by_agent_id: str | None = None
    accepted_by_agent_id: str | None = None
    executed_by_worker_agent_id: str | None = None
    reviewed_by_agent_id: str | None = None
    approved_by_agent_id: str | None = None
    budget_reserved_tokens: int = 0
    budget_used_tokens: int = 0
    budget_remaining_tokens: int = 0
    budget_reserved_ms: int = 0
    budget_used_ms: int = 0
    max_agents: int = 8
    max_subagents: int = 4
    max_queue_jobs: int = 32
    max_tool_calls: int = 32
    cost_by_agent_id: dict[str, int] = field(default_factory=dict)
    cost_by_queue_job_id: dict[str, int] = field(default_factory=dict)
    verification_commands: list[str] = field(default_factory=list)
    verification_results: list[VerificationResult] = field(default_factory=list)
    memory_status: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class QueueJob:
    job_id: str
    task_id: str
    requested_by_agent_id: str
    job_type: QueueJobType
    payload: dict[str, Any]
    root_task_id: str | None = None
    parent_task_id: str | None = None
    assigned_worker_agent_id: str | None = None
    approved_by_agent_id: str | None = None
    purpose: str = ""
    args_summary: str = ""
    budget_reserved: int = 0
    budget_reserved_ms: int = 0
    depends_on: list[str] = field(default_factory=list)
    result_summary: str | None = None
    status: QueueJobStatus = QueueJobStatus.QUEUED
    priority: int = 100
    lock_key: str | None = None
    requires_write_lock: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    token_usage: int = 0
    changed_files: list[str] = field(default_factory=list)
    cache_status: str = "unknown"
    fingerprint: str = ""
    memory_bundle_id: str | None = None
    related_files: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    @property
    def queue_job_id(self) -> str:
        return self.job_id

    @property
    def tool_name(self) -> str:
        return self.job_type.value

    @property
    def tool_args(self) -> dict[str, Any]:
        return self.payload


@dataclass
class ToolRequest:
    request_id: str
    task_id: str
    agent_id: str
    tool_type: QueueJobType
    payload: dict[str, Any]
    approved: bool = False


@dataclass
class ToolResult:
    request_id: str
    task_id: str
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AgentMessage:
    message_id: str
    discussion_id: str | None
    from_agent_id: str
    to_agent_id: str | None
    task_id: str
    message_type: MessageType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class DiscussionThread:
    discussion_id: str
    task_id: str
    title: str
    status: DiscussionStatus
    participant_agent_ids: list[str]
    message_ids: list[str] = field(default_factory=list)
    created_by_agent_id: str = ""
    final_decision_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class DecisionRecord:
    decision_id: str
    task_id: str
    discussion_id: str | None
    made_by_agent_id: str
    decision_status: DecisionStatus
    summary: str
    rationale_summary: str
    selected_route: str
    assigned_agent_ids: list[str] = field(default_factory=list)
    required_verification: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    rejected_options: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class TraceEvent:
    trace_id: str
    event_type: str
    task_id: str | None = None
    agent_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class RouteDecision:
    task_id: str
    route_name: str
    task_size: str
    required_agents: list[str]
    required_subagents: list[str]
    required_capabilities: list[str]
    requires_discussion: bool
    requires_verification: bool
    risk_level: RiskLevel
    reason_summary: str


@dataclass
class PlanResult:
    task_id: str
    plan_steps: list[str]
    acceptance_criteria: list[str]
    files_to_inspect: list[str]
    verification_commands: list[str]
    risks: list[str]
    assumptions: list[str]
