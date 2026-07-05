from __future__ import annotations

import importlib
import json
import logging
import time
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel

from mana_agent.multi_agent.runtime.tool_worker_process import (
    ToolRunRequest,
    ToolWorkerClient,
    ToolWorkerProcessError,
)

logger = logging.getLogger(__name__)
_FALLBACK_WARNINGS_EMITTED: set[str] = set()


STANDARD_ERROR_CODES = {
    "worker_unavailable",
    "enqueue_failed",
    "job_timeout",
    "job_failed",
    "result_decode_failed",
    "tools_only_violation",
}


def normalize_error_code(raw_code: str) -> str:
    code = str(raw_code or "").strip()
    if code in STANDARD_ERROR_CODES:
        return code
    if code in {"worker_dead", "worker_io_error", "not_initialized", "init_failed"}:
        return "worker_unavailable"
    if code == "tools_only_violation":
        return "tools_only_violation"
    return "job_failed"


class ToolsExecutionConfig(BaseModel):
    backend: Literal["local", "redis"] = "local"
    redis_url: str = "redis://127.0.0.1:6379/0"
    queue_name: str = "mana-tools"
    parallel_requests: int = 3
    ttl_seconds: int = 86_400


class BatchToolRequest(BaseModel):
    request_index: int
    request: ToolRunRequest


class BatchExecutionResult(BaseModel):
    request_index: int
    ok: bool = False
    response: dict[str, Any] | None = None
    error_code: str = ""
    error_message: str = ""
    duration_ms: float = 0.0
    backend: str = ""
    queue_wait_ms: float = 0.0


class ToolsExecutor:
    def run_batch(
        self,
        *,
        run_id: str,
        requests: Sequence[BatchToolRequest],
        on_event: Callable[[Any], None] | None = None,
    ) -> list[BatchExecutionResult]:
        _ = (run_id, on_event)
        out: list[BatchExecutionResult] = []
        for item in requests:
            out.append(
                BatchExecutionResult(
                    request_index=int(item.request_index),
                    ok=False,
                    response=None,
                    error_code="worker_unavailable",
                    error_message="ToolsExecutor has no execution backend configured.",
                    backend="base",
                )
            )
        return out


class LocalToolsExecutor(ToolsExecutor):
    def __init__(self, *, worker_client: ToolWorkerClient) -> None:
        self.worker_client = worker_client

    def run_batch(
        self,
        *,
        run_id: str,
        requests: Sequence[BatchToolRequest],
        on_event: Callable[[Any], None] | None = None,
    ) -> list[BatchExecutionResult]:
        _ = run_id
        out: list[BatchExecutionResult] = []
        for item in requests:
            t0 = time.perf_counter()
            try:
                response = self.worker_client.run_tools(item.request, on_event=on_event)
                out.append(
                    BatchExecutionResult(
                        request_index=int(item.request_index),
                        ok=True,
                        response=response.model_dump(),
                        duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                        backend="local",
                    )
                )
            except ToolWorkerProcessError as exc:
                out.append(
                    BatchExecutionResult(
                        request_index=int(item.request_index),
                        ok=False,
                        response=None,
                        error_code=normalize_error_code(exc.code),
                        error_message=str(exc),
                        duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                        backend="local",
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                out.append(
                    BatchExecutionResult(
                        request_index=int(item.request_index),
                        ok=False,
                        response=None,
                        error_code="job_failed",
                        error_message=str(exc),
                        duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                        backend="local",
                    )
                )
        return out


class RedisRQToolsExecutor(ToolsExecutor):
    def __init__(
        self,
        *,
        worker_init_payload: dict[str, Any],
        config: ToolsExecutionConfig,
    ) -> None:
        self.config = config
        self.worker_init_payload = dict(worker_init_payload)

        try:
            redis_mod = importlib.import_module("redis")
            rq_mod = importlib.import_module("rq")
            self._redis = redis_mod.Redis.from_url(config.redis_url)
            self._queue = rq_mod.Queue(config.queue_name, connection=self._redis)
        except Exception as exc:  # pragma: no cover - import/runtime guard
            raise RuntimeError(f"redis executor unavailable: {exc}") from exc

    def _key(self, run_id: str, suffix: str) -> str:
        return f"mana:tools:run:{run_id}:{suffix}"

    def _set_json(self, key: str, payload: dict[str, Any]) -> None:
        try:
            self._redis.set(key, json.dumps(payload, sort_keys=True))
            self._redis.expire(key, int(self.config.ttl_seconds))
        except Exception:
            return None

    def _append_event(self, key: str, payload: dict[str, Any]) -> None:
        try:
            self._redis.rpush(key, json.dumps(payload, sort_keys=True))
            self._redis.expire(key, int(self.config.ttl_seconds))
        except Exception:
            return None

    def _failure_result(
        self,
        *,
        request_index: int,
        code: str,
        message: str,
        t0: float,
        queue_wait_ms: float = 0.0,
    ) -> BatchExecutionResult:
        return BatchExecutionResult(
            request_index=int(request_index),
            ok=False,
            response=None,
            error_code=normalize_error_code(code),
            error_message=str(message or ""),
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            queue_wait_ms=round(float(queue_wait_ms), 3),
            backend="redis",
        )

    def run_batch(
        self,
        *,
        run_id: str,
        requests: Sequence[BatchToolRequest],
        on_event: Callable[[Any], None] | None = None,
    ) -> list[BatchExecutionResult]:
        _ = on_event  # redis workers are detached; events are persisted in redis keys.
        if not requests:
            return []

        from mana_agent.multi_agent.runtime.redis_tool_tasks import run_tool_request_task

        started_at = time.time()
        run_status_key = self._key(run_id, "status")
        self._set_json(
            run_status_key,
            {
                "run_id": run_id,
                "backend": "redis",
                "status": "running",
                "started_at": started_at,
                "requests_total": len(requests),
                "requests_done": 0,
            },
        )

        pending = list(sorted(requests, key=lambda x: int(x.request_index)))
        active: dict[int, tuple[Any, float, str, float]] = {}
        # request_index -> (job, enqueued_ts, last_status, started_perf_counter)
        results: dict[int, BatchExecutionResult] = {}
        parallel_limit = max(1, int(self.config.parallel_requests))

        while pending or active:
            while pending and len(active) < parallel_limit:
                req_item = pending.pop(0)
                idx = int(req_item.request_index)
                now = time.time()
                req_status_key = self._key(run_id, f"req:{idx}:status")
                req_events_key = self._key(run_id, f"req:{idx}:events")
                self._set_json(req_status_key, {"status": "queued", "request_index": idx, "at": now})
                self._append_event(req_events_key, {"event": "queued", "at": now})
                try:
                    timeout_seconds = max(int(req_item.request.timeout_seconds) + 5, 30)
                    job = self._queue.enqueue(
                        run_tool_request_task,
                        self.worker_init_payload,
                        req_item.request.model_dump(),
                        {
                            "run_id": run_id,
                            "request_index": idx,
                            "enqueued_at": now,
                        },
                        job_timeout=timeout_seconds,
                        result_ttl=int(self.config.ttl_seconds),
                        failure_ttl=int(self.config.ttl_seconds),
                        ttl=int(self.config.ttl_seconds),
                    )
                except Exception as exc:
                    fail_result = self._failure_result(
                        request_index=idx,
                        code="enqueue_failed",
                        message=str(exc),
                        t0=time.perf_counter(),
                    )
                    results[idx] = fail_result
                    self._set_json(
                        self._key(run_id, f"req:{idx}:result"),
                        fail_result.model_dump(),
                    )
                    self._append_event(req_events_key, {"event": "enqueue_failed", "at": time.time(), "error": str(exc)})
                    continue
                active[idx] = (job, now, "queued", time.perf_counter())

            if not active:
                continue

            progressed = False
            for idx in list(active.keys()):
                job, enqueued_ts, last_status, started_perf = active[idx]
                req_status_key = self._key(run_id, f"req:{idx}:status")
                req_events_key = self._key(run_id, f"req:{idx}:events")
                try:
                    job.refresh()
                except Exception as exc:
                    fail_result = self._failure_result(
                        request_index=idx,
                        code="job_failed",
                        message=str(exc),
                        t0=started_perf,
                    )
                    results[idx] = fail_result
                    self._set_json(self._key(run_id, f"req:{idx}:result"), fail_result.model_dump())
                    self._append_event(req_events_key, {"event": "job_refresh_failed", "at": time.time(), "error": str(exc)})
                    del active[idx]
                    progressed = True
                    continue

                try:
                    status = str(job.get_status(refresh=False) or "")
                except Exception:
                    status = last_status
                if status and status != last_status:
                    self._set_json(req_status_key, {"status": status, "request_index": idx, "at": time.time()})
                    self._append_event(req_events_key, {"event": status, "at": time.time()})
                    active[idx] = (job, enqueued_ts, status, started_perf)
                    progressed = True

                if getattr(job, "is_finished", False):
                    payload = getattr(job, "result", None)
                    if not isinstance(payload, dict):
                        fail_result = self._failure_result(
                            request_index=idx,
                            code="result_decode_failed",
                            message="redis task payload is not an object",
                            t0=started_perf,
                        )
                        results[idx] = fail_result
                        self._set_json(self._key(run_id, f"req:{idx}:result"), fail_result.model_dump())
                        self._append_event(req_events_key, {"event": "result_decode_failed", "at": time.time()})
                        del active[idx]
                        progressed = True
                        continue

                    ok = bool(payload.get("ok"))
                    if ok and isinstance(payload.get("response"), dict):
                        success_result = BatchExecutionResult(
                            request_index=idx,
                            ok=True,
                            response=payload.get("response"),
                            duration_ms=round(float(payload.get("duration_ms", 0.0) or 0.0), 3),
                            queue_wait_ms=round(float(payload.get("queue_wait_ms", 0.0) or 0.0), 3),
                            backend="redis",
                        )
                        results[idx] = success_result
                        self._set_json(self._key(run_id, f"req:{idx}:result"), success_result.model_dump())
                        self._append_event(req_events_key, {"event": "finished", "at": time.time()})
                    else:
                        fail_result = self._failure_result(
                            request_index=idx,
                            code=str(payload.get("error_code", "job_failed") or "job_failed"),
                            message=str(payload.get("error_message", "redis task failed") or "redis task failed"),
                            t0=started_perf,
                            queue_wait_ms=float(payload.get("queue_wait_ms", 0.0) or 0.0),
                        )
                        results[idx] = fail_result
                        self._set_json(self._key(run_id, f"req:{idx}:result"), fail_result.model_dump())
                        self._append_event(
                            req_events_key,
                            {
                                "event": "failed",
                                "at": time.time(),
                                "error_code": fail_result.error_code,
                                "error_message": fail_result.error_message,
                            },
                        )
                    del active[idx]
                    progressed = True
                    continue

                if getattr(job, "is_failed", False):
                    exc_info = str(getattr(job, "exc_info", "") or "")
                    fail_code = "job_timeout" if "JobTimeoutException" in exc_info else "job_failed"
                    fail_result = self._failure_result(
                        request_index=idx,
                        code=fail_code,
                        message=exc_info.strip().splitlines()[-1] if exc_info.strip() else "redis job failed",
                        t0=started_perf,
                    )
                    results[idx] = fail_result
                    self._set_json(self._key(run_id, f"req:{idx}:result"), fail_result.model_dump())
                    self._append_event(
                        req_events_key,
                        {"event": "failed", "at": time.time(), "error_code": fail_result.error_code},
                    )
                    del active[idx]
                    progressed = True

            requests_done = len(results)
            self._set_json(
                run_status_key,
                {
                    "run_id": run_id,
                    "backend": "redis",
                    "status": "running" if (pending or active) else "completed",
                    "started_at": started_at,
                    "requests_total": len(requests),
                    "requests_done": requests_done,
                    "updated_at": time.time(),
                },
            )

            if not progressed:
                time.sleep(0.05)

        ordered = [results[idx] for idx in sorted(results)]
        self._set_json(
            run_status_key,
            {
                "run_id": run_id,
                "backend": "redis",
                "status": "completed",
                "started_at": started_at,
                "finished_at": time.time(),
                "requests_total": len(requests),
                "requests_done": len(ordered),
                "requests_ok": len([x for x in ordered if x.ok]),
                "requests_failed": len([x for x in ordered if not x.ok]),
            },
        )
        return ordered


def build_tools_executor_with_fallback(
    *,
    worker_client: ToolWorkerClient,
    config: ToolsExecutionConfig,
    worker_init_payload: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    warning_key: str = "redis_executor_unavailable",
    local_executor_cls: type[ToolsExecutor] = LocalToolsExecutor,
    redis_executor_cls: type[ToolsExecutor] = RedisRQToolsExecutor,
) -> ToolsExecutor:
    """Select the requested backend, falling back to local with one warning per key."""
    if config.backend != "redis":
        return local_executor_cls(worker_client=worker_client)  # type: ignore[call-arg]
    try:
        return redis_executor_cls(
            worker_init_payload=worker_init_payload or {},
            config=config,
        )
    except Exception as exc:
        message = f"redis executor unavailable; falling back to local backend: {exc}"
        if warning_key not in _FALLBACK_WARNINGS_EMITTED:
            _FALLBACK_WARNINGS_EMITTED.add(warning_key)
            logger.warning(message)
            if warnings is not None:
                warnings.append(message)
        return local_executor_cls(worker_client=worker_client)  # type: ignore[call-arg]


__all__ = [
    "BatchExecutionResult",
    "BatchToolRequest",
    "LocalToolsExecutor",
    "RedisRQToolsExecutor",
    "ToolsExecutionConfig",
    "ToolsExecutor",
    "build_tools_executor_with_fallback",
    "normalize_error_code",
]
