"""Validated resilient-execution settings."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.config.settings import Settings, mana_home


class ExecutionSupervisorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    root: Path = Field(default_factory=lambda: mana_home() / "execution")
    lease_seconds: int = Field(default=60, ge=5)
    heartbeat_seconds: int = Field(default=15, ge=1)
    checkpoint_interval_seconds: int = Field(default=60, ge=1)
    default_retry_budget: int = Field(default=3, ge=0)
    max_replans: int = Field(default=2, ge=0)
    max_child_depth: int = Field(default=5, ge=0)
    max_children_per_task: int = Field(default=20, ge=0)
    max_total_subtasks: int = Field(default=100, ge=0)
    max_concurrent_children: int = Field(default=4, ge=1)
    default_task_deadline_seconds: int = Field(default=1800, ge=1)
    startup_recovery: bool = True
    verify_completion_artifacts: bool = True
    allow_unknown_side_effect_retry: bool = False
    max_backoff_seconds: float = Field(default=300.0, gt=0)
    base_backoff_seconds: float = Field(default=1.0, gt=0)
    event_retention: int = Field(default=100_000, ge=100)

    @model_validator(mode="after")
    def validate_intervals(self) -> "ExecutionSupervisorConfig":
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("execution supervisor heartbeat must be shorter than its lease")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> "ExecutionSupervisorConfig":
        return cls(
            enabled=settings.mana_execution_supervisor_enabled,
            root=mana_home() / "execution",
            lease_seconds=settings.mana_execution_supervisor_lease_seconds,
            heartbeat_seconds=settings.mana_execution_supervisor_heartbeat_seconds,
            checkpoint_interval_seconds=settings.mana_execution_supervisor_checkpoint_seconds,
            default_retry_budget=settings.mana_execution_supervisor_retry_budget,
            max_replans=settings.mana_execution_supervisor_max_replans,
            max_child_depth=settings.mana_execution_supervisor_max_child_depth,
            max_children_per_task=settings.mana_execution_supervisor_max_children,
            max_total_subtasks=settings.mana_execution_supervisor_max_total_subtasks,
            max_concurrent_children=settings.mana_execution_supervisor_max_concurrent_children,
            default_task_deadline_seconds=settings.mana_routing_task_timeout_seconds,
            startup_recovery=settings.mana_execution_supervisor_startup_recovery,
            verify_completion_artifacts=settings.mana_execution_supervisor_verify_artifacts,
            allow_unknown_side_effect_retry=settings.mana_execution_supervisor_allow_unknown_retry,
        )
