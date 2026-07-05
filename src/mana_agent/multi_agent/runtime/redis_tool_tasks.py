from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from mana_agent.multi_agent.runtime.tool_worker_process import (
    ToolRunRequest,
    ToolWorkerProcessError,
    WorkerInitPayload,
    run_tool_request_once,
)
from mana_agent.multi_agent.runtime.tools_executor import normalize_error_code


def run_tool_request_task(
    init_payload_dict: dict[str, Any],
    tool_request_dict: dict[str, Any],
    runtime_meta_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_meta = dict(runtime_meta_dict or {})
    started = time.time()
    enqueued_at = float(runtime_meta.get("enqueued_at", started) or started)
    queue_wait_ms = max(0.0, (started - enqueued_at) * 1000.0)

    try:
        init_payload = WorkerInitPayload.model_validate(init_payload_dict)
        tool_request = ToolRunRequest.model_validate(tool_request_dict)
    except ValidationError as exc:
        return {
            "ok": False,
            "response": None,
            "error_code": "result_decode_failed",
            "error_message": f"request validation failed: {exc}",
            "duration_ms": round((time.time() - started) * 1000.0, 3),
            "queue_wait_ms": round(queue_wait_ms, 3),
            "events": [
                {"event": "started", "at": started},
                {"event": "validation_failed", "at": time.time(), "error": str(exc)},
            ],
            "runtime_meta": runtime_meta,
        }

    events: list[dict[str, Any]] = [{"event": "started", "at": started}]
    try:
        response = run_tool_request_once(init_payload=init_payload, request=tool_request)
        events.append({"event": "finished", "at": time.time()})
        return {
            "ok": True,
            "response": response.model_dump(),
            "error_code": "",
            "error_message": "",
            "duration_ms": round((time.time() - started) * 1000.0, 3),
            "queue_wait_ms": round(queue_wait_ms, 3),
            "events": events,
            "runtime_meta": runtime_meta,
        }
    except ToolWorkerProcessError as exc:
        code = normalize_error_code(exc.code)
        events.append(
            {
                "event": "failed",
                "at": time.time(),
                "error_code": code,
                "error_message": str(exc),
            }
        )
        return {
            "ok": False,
            "response": None,
            "error_code": code,
            "error_message": str(exc),
            "duration_ms": round((time.time() - started) * 1000.0, 3),
            "queue_wait_ms": round(queue_wait_ms, 3),
            "events": events,
            "runtime_meta": runtime_meta,
        }
    except Exception as exc:  # pragma: no cover - defensive
        events.append(
            {
                "event": "failed",
                "at": time.time(),
                "error_code": "job_failed",
                "error_message": str(exc),
            }
        )
        return {
            "ok": False,
            "response": None,
            "error_code": "job_failed",
            "error_message": str(exc),
            "duration_ms": round((time.time() - started) * 1000.0, 3),
            "queue_wait_ms": round(queue_wait_ms, 3),
            "events": events,
            "runtime_meta": runtime_meta,
        }


__all__ = ["run_tool_request_task"]

