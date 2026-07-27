from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.gateway.lanes import LaneId
from mana_agent.config.model_catalog import ModelCapability, normalize_capabilities
from mana_agent.multi_agent.routing.agent_decision import agent_tool_descriptions

from .ids import stable_hash
from .models import EvalTask, EvalVariant, ExpectedOutcome, RetryPolicy, SCHEMA_VERSION

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class EvalConfigurationError(ValueError):
    pass


class SuiteDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_backend: str = "local-worktree"
    trials: int = 1
    timeout_seconds: int = 900
    fail_on_provider_error: bool = True
    concurrency: int = 1
    provider_concurrency: dict[str, int] = Field(default_factory=dict)
    retain_workspaces: bool = False
    required_platforms: list[Literal["linux", "windows", "macos"]] = Field(default_factory=list)
    minimum_platform_coverage: int = Field(default=0, ge=0, le=3)
    runtime: dict[str, list[str]] = Field(default_factory=dict)
    fleet: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fleet_defaults(self) -> "SuiteDefaults":
        if self.workspace_backend == "fleet":
            if not self.required_platforms:
                raise ValueError("fleet Eval suites require explicit required_platforms")
            if self.minimum_platform_coverage > len(set(self.required_platforms)):
                raise ValueError("minimum_platform_coverage exceeds required platforms")
            allowed = {"labels", "maximum_workers", "per_worker_concurrency"}
            unknown = set(self.fleet) - allowed
            if unknown:
                raise ValueError(f"unknown fleet Eval settings: {', '.join(sorted(unknown))}")
        return self


class ScoreWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_completion: float = 35
    tests: float = 30
    reviewer_quality: float = 15
    routing_correctness: float = 10
    policy_compliance: float = 5
    efficiency: float = 5
    reproducibility: float = 0
    verifier: float = 0
    patch_quality: float = 0
    safety: float = 0

    @model_validator(mode="after")
    def valid_weights(self) -> "ScoreWeights":
        values = self.model_dump().values()
        if any(value < 0 for value in values):
            raise ValueError("score weights cannot be negative")
        if sum(values) <= 0:
            raise ValueError("at least one score weight must be positive")
        return self


class GateThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum_task_count: int = 1
    maximum_pass_rate_drop: float = 0.02
    maximum_mean_score_drop: float = 1.0
    maximum_routing_accuracy_drop: float = 0.0
    maximum_cost_increase_ratio: float = 0.20
    maximum_latency_increase_ratio: float = 0.25
    allow_new_policy_violations: bool = False
    fail_on_missing_candidate: bool = True
    fail_on_incomplete_run: bool = True
    fail_on_provider_error: bool = True
    require_known_cost: bool = False


class EvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = SCHEMA_VERSION
    name: str
    version: str
    defaults: SuiteDefaults = Field(default_factory=SuiteDefaults)
    tasks: list[EvalTask]
    variants: list[EvalVariant]
    scoring: ScoreWeights = Field(default_factory=ScoreWeights)
    gate: GateThresholds = Field(default_factory=GateThresholds)
    extra_redaction_patterns: list[str] = Field(default_factory=list)
    pricing_overrides: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> "EvalSuite":
        task_ids = [item.task_id for item in self.tasks]
        variant_ids = [item.variant_id for item in self.variants]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("duplicate task IDs are not allowed")
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("duplicate variant IDs are not allowed")
        return self

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ or not os.environ[name].strip():
            raise EvalConfigurationError(f"required environment variable is missing: {name}")
        return os.environ[name]

    return _ENV.sub(replace, value)


def load_suite(path: str | Path, *, validate_runtime: bool = True) -> EvalSuite:
    path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise EvalConfigurationError("PyYAML is required to read evaluation suites") from exc
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvalConfigurationError(f"cannot read evaluation suite {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EvalConfigurationError("evaluation suite must be a YAML mapping")
    raw = _resolve_env(raw)
    suite = _normalize_suite(raw, path.parent)
    if validate_runtime:
        validate_suite_runtime(suite)
    return suite


def _normalize_suite(raw: dict[str, Any], base_dir: Path) -> EvalSuite:
    name = str(raw.get("name") or "").strip()
    version = str(raw.get("version") or "").strip()
    defaults = dict(raw.get("defaults") or {})
    task_rows: list[dict[str, Any]] = []
    for task in raw.get("tasks") or []:
        row = dict(task)
        repository = str(row.pop("repository", "."))
        repo_path = Path(repository)
        if not repo_path.is_absolute():
            candidate = (base_dir / repo_path).resolve()
            repository = str(candidate if candidate.exists() else Path.cwd().resolve())
        task_rows.append(
            {
                "task_id": row.pop("id", row.pop("task_id", "")),
                "suite_name": name,
                "suite_version": version,
                "description": row.pop("prompt", row.pop("description", "")),
                "repository": repository,
                "repository_commit": str(row.pop("commit", row.pop("repository_commit", "HEAD"))),
                "timeout_seconds": int(row.pop("timeout_seconds", defaults.get("timeout_seconds", 900))),
                "expected": ExpectedOutcome.model_validate(row.pop("expected", {})),
                **row,
            }
        )
    variants: list[dict[str, Any]] = []
    for variant in raw.get("variants") or []:
        row = dict(variant)
        models = dict(row.pop("models", {}))
        main = str(models.get("main") or models.get("router") or row.pop("main_model", ""))
        variants.append(
            {
                "variant_id": row.pop("id", row.pop("variant_id", "")),
                "display_name": row.pop("display_name", "") or str(variant.get("id", "")),
                "main_model": main,
                "router_model": str(models.get("router") or main),
                "coding_model": str(models.get("coding") or main),
                "reviewer_model": str(models.get("reviewer") or main),
                "verifier_model": str(models.get("verifier") or models.get("reviewer") or main),
                "workspace_backend": row.pop("workspace_backend", defaults.get("workspace_backend", "local-worktree")),
                "lane_configuration": row.pop("lane_config", row.pop("lane_configuration", {})) if isinstance(row.get("lane_config", row.get("lane_configuration", {})), dict) else {"name": row.pop("lane_config", "default")},
                "retry_policy": RetryPolicy.model_validate(row.pop("retry_policy", {})),
                **row,
            }
        )
    return EvalSuite.model_validate(
        {
            "schema_version": raw.get("schema_version", SCHEMA_VERSION),
            "name": name,
            "version": version,
            "defaults": defaults,
            "tasks": task_rows,
            "variants": variants,
            "scoring": raw.get("scoring", {}),
            "gate": raw.get("gate", {}),
            "extra_redaction_patterns": raw.get("extra_redaction_patterns", []),
            "pricing_overrides": raw.get("pricing_overrides", []),
        }
    )


def validate_suite_runtime(suite: EvalSuite) -> None:
    known_tools = {str(item["name"]) for item in agent_tool_descriptions()}
    known_lanes = {item.value for item in LaneId}
    for variant in suite.variants:
        if variant.workspace_backend not in {"local-worktree", "execution-fabric", "fleet"}:
            raise EvalConfigurationError(f"unsupported workspace backend: {variant.workspace_backend}")
        unknown_tools = (set(variant.tool_allowlist) | set(variant.tool_denylist)) - known_tools
        if unknown_tools:
            raise EvalConfigurationError(f"unknown tools in variant {variant.variant_id}: {', '.join(sorted(unknown_tools))}")
        configured_lanes = set(variant.lane_configuration) - {"name"}
        unknown_lanes = configured_lanes - known_lanes
        if unknown_lanes:
            raise EvalConfigurationError(f"unknown lanes in variant {variant.variant_id}: {', '.join(sorted(unknown_lanes))}")
        for model in {
            variant.main_model,
            variant.router_model,
            variant.coding_model,
            variant.reviewer_model,
            variant.verifier_model,
        }:
            provider, model_id = (model.split("/", 1) if "/" in model and not model.startswith("gpt-") else ("openai", model))
            if ModelCapability.TEXT_GENERATION not in normalize_capabilities(provider, model_id):
                raise EvalConfigurationError(
                    f"unknown or non-text model in variant {variant.variant_id}: {model}"
                )
    for task in suite.tasks:
        repository = Path(task.repository)
        result = subprocess.run(["git", "rev-parse", "--verify", f"{task.repository_commit}^{{commit}}"], cwd=repository, capture_output=True, text=True)
        if result.returncode != 0:
            raise EvalConfigurationError(f"invalid commit {task.repository_commit!r} for task {task.task_id}")
        if (task.expected.required_changed_files or not task.expected.no_repository_mutation) and not task.test_commands:
            raise EvalConfigurationError(f"task {task.task_id} requires at least one test command")
