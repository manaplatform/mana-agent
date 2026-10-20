"""Data models and enums for the official OpenAI Batch API integration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BatchLifecycleStatus(str, Enum):
    """Complete lifecycle state machine for OpenAI Batch API jobs."""

    CREATED = "created"
    UPLOADING = "uploading"
    SUBMITTED = "submitted"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    RESULTS_IMPORTED = "results_imported"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class BatchExecutionMode(str, Enum):
    """Operational routing mode for model requests."""

    AUTO = "auto"
    BATCH = "batch"
    REALTIME = "realtime"


@dataclass
class BatchConfig:
    """Configuration settings for OpenAI Batch API execution."""

    enabled: bool = True
    mode: str = "auto"
    endpoint: str = "/v1/responses"
    completion_window: str = "24h"
    min_requests: int = 1
    always_enable_if_supported: bool = True
    max_requests_per_batch: int = 50000
    max_file_size_mb: int = 200
    auto_resume: bool = True
    retry_failed_requests: bool = True
    cost_optimization: bool = True

    @classmethod
    def from_settings(cls, settings: Any = None) -> BatchConfig:
        if settings is None:
            try:
                from mana_agent.config.settings import Settings

                settings = Settings()
            except Exception:
                return cls()
        return cls(
            enabled=bool(getattr(settings, "openai_batch_enabled", True)),
            mode=str(getattr(settings, "openai_batch_mode", "auto")).lower(),
            endpoint=str(getattr(settings, "openai_batch_endpoint", "/v1/responses")),
            completion_window=str(getattr(settings, "openai_batch_completion_window", "24h")),
            min_requests=int(getattr(settings, "openai_batch_min_requests", 1)),
            always_enable_if_supported=bool(getattr(settings, "openai_batch_always_enable_if_supported", True)),
            max_requests_per_batch=int(getattr(settings, "openai_batch_max_requests_per_batch", 50000)),
            max_file_size_mb=int(getattr(settings, "openai_batch_max_file_size_mb", 200)),
            auto_resume=bool(getattr(settings, "openai_batch_auto_resume", True)),
            retry_failed_requests=bool(getattr(settings, "openai_batch_retry_failed_requests", True)),
            cost_optimization=bool(getattr(settings, "openai_batch_cost_optimization", True)),
        )


@dataclass
class BatchItemRequest:
    """An individual model request destined for a JSONL batch payload."""

    custom_id: str
    body: dict[str, Any]
    method: str = "POST"
    url: str = "/v1/responses"
    origin_task_id: str = ""
    origin_run_id: str = ""
    created_at: float = field(default_factory=time.time)
    target_revision: str = ""
    file_path: str = ""
    expected_sha256: str = ""

    def to_jsonl_row(self) -> dict[str, Any]:
        """Convert to OpenAI Batch API JSONL line dictionary."""
        return {
            "custom_id": self.custom_id,
            "method": self.method,
            "url": self.url,
            "body": self.body,
        }


@dataclass
class BatchItemResult:
    """Structured result of an individual batch request parsed from output JSONL."""

    custom_id: str
    status_code: int = 200
    response_body: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    duration_ms: float = 0.0
    response: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.response is not None and not self.response_body:
            self.response_body = self.response
        elif self.response_body and self.response is None:
            self.response = self.response_body

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and self.error is None


    @classmethod
    def from_jsonl_row(cls, data: dict[str, Any]) -> BatchItemResult:
        if not isinstance(data, dict):
            raise ValueError("Batch result row must be an object")
        custom_id = str(data.get("custom_id") or "")
        if not custom_id.strip():
            raise ValueError("Batch result is missing custom_id")
        err = data.get("error")
        resp = data.get("response") or {}
        if not isinstance(resp, dict):
            raise ValueError("Batch result response must be an object")
        if err is None and (not isinstance(resp, dict) or "status_code" not in resp or not isinstance(resp.get("body"), dict)):
            raise ValueError("Batch result is missing a valid response or error")
        if err is not None and not isinstance(err, dict):
            raise ValueError("Batch result error must be an object")
        status_code = resp.get("status_code", 200 if not err else 400)
        body = resp.get("body") or {}
        return cls(
            custom_id=custom_id,
            status_code=int(status_code),
            response_body=body if isinstance(body, dict) else {},
            error=err if isinstance(err, dict) else None,
        )


@dataclass
class BatchJobRecord:
    """Durable state record representing an OpenAI Batch API job."""

    batch_id: str
    status: str = BatchLifecycleStatus.CREATED.value
    endpoint: str = "/v1/responses"
    model: str = ""
    input_file_id: str = ""
    output_file_id: str = ""
    error_file_id: str = ""
    completion_window: str = "24h"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    cancelled_at: float | None = None
    expires_at: float | None = None
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    task_id: str = ""
    run_id: str = ""
    idempotency_key: str = ""
    git_commit_sha: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    cost_estimate_usd: float = 0.0
    error_message: str = ""

    def is_terminal(self) -> bool:
        return self.status in {
            BatchLifecycleStatus.COMPLETED.value,
            BatchLifecycleStatus.RESULTS_IMPORTED.value,
            BatchLifecycleStatus.VERIFIED.value,
            BatchLifecycleStatus.FAILED.value,
            BatchLifecycleStatus.EXPIRED.value,
            BatchLifecycleStatus.CANCELLED.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "endpoint": self.endpoint,
            "model": self.model,
            "input_file_id": self.input_file_id,
            "output_file_id": self.output_file_id,
            "error_file_id": self.error_file_id,
            "completion_window": self.completion_window,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "expires_at": self.expires_at,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "idempotency_key": self.idempotency_key,
            "git_commit_sha": self.git_commit_sha,
            "metadata": self.metadata,
            "usage": self.usage,
            "cost_estimate_usd": self.cost_estimate_usd,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJobRecord:
        kwargs = dict(data)
        return cls(**kwargs)
