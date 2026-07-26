from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ids import stable_hash

SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


class ExpectedOutcome(StrictModel):
    intent: str | None = None
    route: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_lane: str | None = None
    forbidden_execution_paths: list[str] = Field(default_factory=list)
    required_changed_files: list[str] = Field(default_factory=list)
    forbidden_changed_files: list[str] = Field(default_factory=list)
    maximum_cost: float | None = None
    maximum_latency_seconds: float | None = None
    maximum_tokens: int | None = None
    no_repository_mutation: bool = False


class EvalTask(StrictModel):
    schema_version: int = SCHEMA_VERSION
    task_id: str
    suite_name: str
    suite_version: str
    description: str
    repository: str
    repository_commit: str
    base_branch: str = ""
    setup_commands: list[str] = Field(default_factory=list)
    test_commands: list[str] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    forbidden_files: list[str] = Field(default_factory=list)
    expected: ExpectedOutcome = Field(default_factory=ExpectedOutcome)
    labels: list[str] = Field(default_factory=list)
    difficulty: str = "unspecified"
    timeout_seconds: int = 900
    resource_limits: dict[str, Any] = Field(default_factory=dict)
    reference_patch: dict[str, Any] | None = None
    scoring_rubric: dict[str, Any] | None = None

    @field_validator("task_id", "suite_name", "suite_version", "description", "repository", "repository_commit")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("value is required")
        return value

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


class RetryPolicy(StrictModel):
    max_attempts: int = 1
    retryable_errors: list[str] = Field(default_factory=list)

    @field_validator("max_attempts")
    @classmethod
    def positive_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_attempts must be at least one")
        return value


class EvalVariant(StrictModel):
    schema_version: int = SCHEMA_VERSION
    variant_id: str
    display_name: str
    main_model: str
    router_model: str
    coding_model: str
    reviewer_model: str
    verifier_model: str
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    prompt_set: str = "current"
    routing_policy: str = "default"
    tool_policy: str = "standard"
    tool_allowlist: list[str] = Field(default_factory=list)
    tool_denylist: list[str] = Field(default_factory=list)
    lane_configuration: dict[str, Any] = Field(default_factory=dict)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    memory_mode: str = "isolated"
    context_limits: dict[str, int] = Field(default_factory=dict)
    token_budgets: dict[str, int] = Field(default_factory=dict)
    cost_budgets: dict[str, float] = Field(default_factory=dict)
    workspace_backend: Literal["local-worktree", "execution-fabric", "fleet"] = "local-worktree"
    environment_specification: str = "local"
    random_seed: int = 0
    trial_number: int = 1

    @field_validator("variant_id", "display_name", "main_model", "router_model", "coding_model", "reviewer_model", "verifier_model")
    @classmethod
    def required_variant_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("value is required")
        return value

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude={"trial_number"}))


class RouteRecord(StrictModel):
    route_input_hash: str
    router_model: str
    router_prompt_hash: str = ""
    intent: str
    execution_path: str = ""
    selected_tools: list[str] = Field(default_factory=list)
    tool_inputs: dict[str, Any] = Field(default_factory=dict)
    lane_selection: str = ""
    confidence: float = 0.0
    reasoning_summary: str = ""
    flow_action: str = "none"
    decision_verification_result: dict[str, Any] = Field(default_factory=dict)
    decision_latency_seconds: float = 0.0
    decision_token_usage: dict[str, Any] = Field(default_factory=dict)
    retry_attempts: int = 0


class ToolCallRecord(StrictModel):
    sequence: int
    tool_name: str
    owner: str = ""
    redacted_input: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    status: str = "running"
    redacted_output_preview: str = ""
    full_output_artifact: str | None = None
    error_type: str | None = None
    retry_number: int = 0
    mutated_repository: bool = False


class CommandRecord(StrictModel):
    command: str
    working_directory: str
    exit_code: int | None = None
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    lane_owner: str = ""
    environment_variable_names: list[str] = Field(default_factory=list)


class PatchRecord(StrictModel):
    changed_files: list[str] = Field(default_factory=list)
    unified_diff_artifact: str = "final.patch"
    diff_hash: str = ""
    added_lines: int = 0
    removed_lines: int = 0
    application_status: str = "captured"
    repository_state_before: str = ""
    repository_state_after: str = ""
    commit_sha: str | None = None


class UsageRecord(StrictModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    time_to_first_token_seconds: float | None = None
    provider: str
    model: str
    calculated_cost: float | None = None
    pricing_table_version: str
    lane: str = ""


class TestResult(StrictModel):
    command: str
    passed: bool
    exit_code: int | None = None
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class EvaluationResult(StrictModel):
    setup_success: bool = False
    tests: list[TestResult] = Field(default_factory=list)
    reviewer_scores: dict[str, float] = Field(default_factory=dict)
    verifier_result: dict[str, Any] = Field(default_factory=dict)
    policy_violations: list[str] = Field(default_factory=list)
    task_success: bool = False
    normalized_score: float = 0.0
    score_dimensions: dict[str, float] = Field(default_factory=dict)
    score_explanation: str = ""
    failure_category: str | None = None
    first_attempt_success: bool = False
    final_success_after_retries: bool = False


class EnvironmentSnapshot(StrictModel):
    repository_commit: str
    base_branch: str = ""
    remote_url: str = ""
    dirty: bool
    starting_diff_hash: str
    python_version: str
    mana_agent_version: str
    operating_system: str
    architecture: str
    packages: list[str] = Field(default_factory=list)
    lockfile_hashes: dict[str, str] = Field(default_factory=dict)
    environment_image: str = ""
    workspace_backend: str
    git_version: str
    config_hashes: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    tool_policy_hash: str = ""
    lane_policy_hash: str = ""
    environment_variable_names: list[str] = Field(default_factory=list)
    reproducible: bool = False
    non_reproducible_reasons: list[str] = Field(default_factory=list)


class EvalRun(StrictModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    run_fingerprint: str
    experiment_id: str
    task_id: str
    variant_id: str
    trial_number: int = 1
    replayed_from_run_id: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    status: RunStatus = RunStatus.PENDING
    repository_commit: str
    initial_dirty_state: bool = False
    environment: EnvironmentSnapshot | None = None
    routes: list[RouteRecord] = Field(default_factory=list)
    model_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    commands: list[CommandRecord] = Field(default_factory=list)
    patches: list[PatchRecord] = Field(default_factory=list)
    usage: list[UsageRecord] = Field(default_factory=list)
    latency_seconds: float = 0.0
    calculated_cost: float | None = None
    tests: list[TestResult] = Field(default_factory=list)
    reviewer_results: list[dict[str, Any]] = Field(default_factory=list)
    verifier_results: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str = ""
    outcome: EvaluationResult | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def completed_run_is_final(self) -> "EvalRun":
        if self.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED} and self.completed_at is None:
            raise ValueError("terminal runs require completed_at")
        return self
