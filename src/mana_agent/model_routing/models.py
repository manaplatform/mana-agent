from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any


class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LatencyClass(str, Enum):
    INTERACTIVE = "interactive"
    STANDARD = "standard"
    BATCH = "batch"


class RoutingMode(str, Enum):
    SINGLE = "single"
    SINGLE_WITH_VERIFICATION = "single_with_verification"
    MULTI_AGENT = "multi_agent"
    PARALLEL_CANDIDATES = "parallel_candidates"
    MULTI_AGENT_WITH_PARALLEL_CANDIDATES = "multi_agent_with_parallel_candidates"


_LEVEL_VALUE = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}


def level_value(value: Complexity | RiskLevel) -> float:
    return _LEVEL_VALUE[value.value]


_SECRET_KEY_MARKERS = ("key", "token", "secret", "password", "credential", "authorization", "cookie")
_SECRET_VALUE_RE = re.compile(r"(?i)(bearer\s+\S+|\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{8,})")


def sanitize_configuration(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitize_configuration(item)
            for key, item in value.items()
            if not any(marker in str(key).lower() for marker in _SECRET_KEY_MARKERS)
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_configuration(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub("[REDACTED]", value)
    return value


@dataclass(frozen=True, slots=True)
class RepositoryMetadata:
    languages: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    build_systems: tuple[str, ...] = ()
    file_count: int = 0
    test_file_count: int = 0
    changed_files: tuple[str, ...] = ()
    sensitive_areas: tuple[str, ...] = ()
    fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class RoutingBudgets:
    task_token_limit: int | None = 32_000
    task_cost_limit: float | None = None
    session_cost_remaining: float | None = None
    competition_cost_limit: float | None = None
    verification_cost_limit: float | None = None
    retry_cost_limit: float | None = None
    verification_reserve_ratio: float = 0.15
    allow_controlled_override: bool = False

    def __post_init__(self) -> None:
        for name in ("task_token_limit", "task_cost_limit", "session_cost_remaining", "competition_cost_limit", "verification_cost_limit", "retry_cost_limit"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0.0 <= self.verification_reserve_ratio <= 1.0:
            raise ValueError("verification_reserve_ratio must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    provider: str
    model_id: str
    supported_roles: frozenset[str]
    supported_tools: frozenset[str] = frozenset()
    reasoning_settings: frozenset[str] = frozenset({"none"})
    context_window: int = 128_000
    latency_class: LatencyClass = LatencyClass.STANDARD
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    logical_cost_per_1k_tokens: float = 1.0
    reliability_score: float = 0.8
    supported_languages: frozenset[str] = frozenset()
    benchmark_scores: dict[str, float] = field(default_factory=dict)
    can_patch: bool = True
    can_structured_output: bool = True
    can_tool_call: bool = True
    can_verify: bool = True
    available: bool = True
    configuration: dict[str, Any] = field(default_factory=dict, compare=False)
    source_level: str = ""

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model_id.strip() or not self.supported_roles:
            raise ValueError("model profile requires provider, model_id, and supported_roles")
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if min(self.input_cost_per_million, self.output_cost_per_million, self.logical_cost_per_1k_tokens) < 0:
            raise ValueError("model costs cannot be negative")
        if not 0.0 <= self.reliability_score <= 1.0:
            raise ValueError("reliability_score must be between 0 and 1")
        if any(not 0.0 <= score <= 1.0 for score in self.benchmark_scores.values()):
            raise ValueError("benchmark scores must be between 0 and 1")
        if not isinstance(self.latency_class, LatencyClass):
            raise ValueError("latency_class must use LatencyClass")

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model_id}"


@dataclass(frozen=True, slots=True)
class RoutingRequest:
    role: str
    task_description: str
    task_type: str
    complexity: Complexity
    risk: RiskLevel
    repository: RepositoryMetadata = RepositoryMetadata()
    required_tools: frozenset[str] = frozenset()
    required_capabilities: frozenset[str] = frozenset()
    latency_requirement: LatencyClass = LatencyClass.STANDARD
    budgets: RoutingBudgets = RoutingBudgets()
    expected_prompt_tokens: int = 2_000
    retrieved_context_tokens: int = 0
    expected_response_tokens: int = 2_000
    expected_tool_calls: int = 0
    multi_candidate_permitted: bool = False
    previous_verification_failed: bool = False
    explicit_competition: bool = False
    request_id: str = ""
    task_id: str = ""
    parent_task_id: str | None = None
    session_id: str = ""
    workspace_id: str = ""
    repository_id: str = ""
    execution_lane: str = ""
    context_requirements: tuple[str, ...] = ()
    expected_output_type: str = "text"
    subagents_allowed: bool = False
    parallel_execution_allowed: bool = False
    main_model_requested_multi_agent: bool = False
    main_model_requested_parallel: bool = False
    isolation_available: bool = False
    independent_verifier_available: bool = False
    active_task_conflict: bool = False
    historical_parallel_benefit: float = 0.0
    historical_result_variance: float = 0.0
    similar_task_failures: int = 0
    plausible_strategy_count: int = 1
    maximum_concurrency: int = 1
    task_tree_depth: int = 0
    provider_health: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.role.strip() or not self.task_description.strip() or not self.task_type.strip():
            raise ValueError("routing request requires role, task_description, and task_type")
        if not isinstance(self.complexity, Complexity) or not isinstance(self.risk, RiskLevel):
            raise ValueError("complexity and risk must use the typed routing enums")
        if not isinstance(self.latency_requirement, LatencyClass):
            raise ValueError("latency_requirement must use LatencyClass")
        if min(self.expected_prompt_tokens, self.retrieved_context_tokens, self.expected_response_tokens, self.expected_tool_calls) < 0:
            raise ValueError("routing token and tool-call estimates cannot be negative")
        if not 0.0 <= self.historical_parallel_benefit <= 1.0 or not 0.0 <= self.historical_result_variance <= 1.0:
            raise ValueError("parallel evidence scores must be between 0 and 1")
        if min(self.similar_task_failures, self.plausible_strategy_count, self.maximum_concurrency, self.task_tree_depth) < 0:
            raise ValueError("routing orchestration counters cannot be negative")


@dataclass(frozen=True, slots=True)
class CandidateRejection:
    model: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    selected_model: str
    provider: str
    model_configuration: dict[str, Any]
    selected_role: str
    routing_score: float
    confidence: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float
    expected_latency_class: LatencyClass
    selection_reasons: tuple[str, ...]
    rejected_candidates: tuple[CandidateRejection, ...]
    candidate_competition: bool
    competition_candidates: tuple[str, ...]
    verifier_model: str | None
    verifier_independent: bool
    applicable_budgets: RoutingBudgets
    decision_id: str = ""
    request_id: str = ""
    task_id: str = ""
    routing_mode: RoutingMode = RoutingMode.SINGLE
    required_verification_level: str = "standard"
    parallel_execution_permitted: bool = False
    multi_agent_execution_permitted: bool = False
    applicable_limits: dict[str, Any] = field(default_factory=dict)
    deadline_seconds: int | None = None
    orchestration_reasons: tuple[str, ...] = ()

    def concise(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.selected_model,
            "role": self.selected_role,
            "score": self.routing_score,
            "confidence": self.confidence,
            "estimated_cost": self.estimated_cost,
            "competition": self.candidate_competition,
            "routing_mode": self.routing_mode.value,
            "decision_id": self.decision_id,
            "task_id": self.task_id,
            "parallel_execution_permitted": self.parallel_execution_permitted,
            "multi_agent_execution_permitted": self.multi_agent_execution_permitted,
            "verifier_model": self.verifier_model,
            "reasons": list(self.selection_reasons),
        }


class RoutingFailure(RuntimeError):
    def __init__(self, message: str, *, rejected: tuple[CandidateRejection, ...] = ()) -> None:
        super().__init__(message)
        self.rejected = rejected

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "rejected_candidates": [asdict(item) for item in self.rejected],
            "fallback_executed": False,
        }


@dataclass(frozen=True, slots=True)
class RoutingOutcome:
    provider: str
    model_id: str
    model_configuration: dict[str, Any]
    task_category: str
    repository_languages: tuple[str, ...]
    repository_frameworks: tuple[str, ...]
    complexity: str
    risk: str
    routing_score: float
    selection_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    actual_cost: float | None = None
    latency_seconds: float = 0.0
    tool_failures: int = 0
    verification_passed: bool | None = None
    retry_count: int = 0
    accepted: bool | None = None
    competition_result: str = "not_run"
    failure_kind: str = ""
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def safe_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["occurred_at"] = self.occurred_at.isoformat()
        data["model_configuration"] = sanitize_configuration(self.model_configuration)
        return data


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    enabled: bool = True
    minimum_confidence: float = 0.55
    competition_complexity_threshold: Complexity = Complexity.HIGH
    competition_risk_threshold: RiskLevel = RiskLevel.HIGH
    maximum_candidate_count: int = 2
    circuit_breaker_failures: int = 3
    circuit_breaker_window_seconds: int = 900
    reliability_decay_seconds: int = 3600
    model_failure_penalty_weight: float = 0.08
    provider_failure_penalty_weight: float = 0.04
    evidence_retention_days: int = 90
    simple_routing_default: bool = True
    multi_agent_enabled: bool = False
    parallel_execution_enabled: bool = False
    minimum_parallel_evidence: float = 0.65
    maximum_task_tree_depth: int = 3
    maximum_concurrency: int = 4
    task_timeout_seconds: int = 1800
    weights: dict[str, float] = field(default_factory=lambda: {
        "capability": 0.22,
        "quality": 0.25,
        "history": 0.18,
        "language": 0.10,
        "cost": 0.15,
        "latency": 0.10,
    })

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if self.maximum_candidate_count < 1:
            raise ValueError("maximum_candidate_count must be positive")
        if min(self.circuit_breaker_failures, self.circuit_breaker_window_seconds, self.reliability_decay_seconds, self.evidence_retention_days) < 1:
            raise ValueError("routing windows, retention, and circuit threshold must be positive")
        if not 0.0 <= self.minimum_parallel_evidence <= 1.0:
            raise ValueError("minimum_parallel_evidence must be between 0 and 1")
        if min(self.maximum_task_tree_depth, self.maximum_concurrency, self.task_timeout_seconds) < 1:
            raise ValueError("routing orchestration limits must be positive")
        if min(self.model_failure_penalty_weight, self.provider_failure_penalty_weight, *self.weights.values()) < 0:
            raise ValueError("routing weights cannot be negative")
