"""Normalization helpers for structured tool result payloads."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import Any
from uuid import UUID

from pydantic import BaseModel


def json_safe_tool_payload(value: Any) -> Any:
    """Recursively sanitize any Python object into a JSON-serializable structure.

    - Converts sets and frozensets to lists (sorted if elements are comparable).
    - Converts tuples to lists.
    - Recursively normalizes dicts/mappings with string keys.
    - Serializes Pydantic models, dataclasses, enums, dates, times, UUIDs, Decimals,
      Paths, and exceptions to standard JSON types.
    - Preserves existing JSON types (str, int, float, bool, None, dict, list).
    - Never throws on non-serializable objects (falls back to str).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (set, frozenset)):
        normalized_set = [json_safe_tool_payload(item) for item in value]
        try:
            return sorted(normalized_set)
        except TypeError:
            return normalized_set
    if isinstance(value, (list, tuple)):
        return [json_safe_tool_payload(item) for item in value]
    if isinstance(value, BaseModel):
        try:
            return json_safe_tool_payload(value.model_dump())
        except Exception:
            try:
                return json_safe_tool_payload(value.model_dump(mode="json"))
            except Exception:
                pass
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe_tool_payload(asdict(value))
    if isinstance(value, Enum):
        return json_safe_tool_payload(value.value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (UUID, Path, PurePath)):
        return str(value)
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe_tool_payload(v) for k, v in value.items()}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return json_safe_tool_payload(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return json_safe_tool_payload(value.model_dump())
        except Exception:
            pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe_tool_payload(item) for item in value]
    return str(value)


def json_safe_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize any value to a JSON string after normalizing to JSON-safe types."""
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(json_safe_tool_payload(value), **kwargs)


def structured_tool_error_detail(output: Any) -> str | None:
    """Return a failure description when a tool completed with an error payload.

    A tool callback's normal completion only means its function returned. Tools
    commonly encode domain failures in JSON, so callers must inspect that
    payload before presenting the invocation as successful.
    """
    payload = output if isinstance(output, dict) else _parse_payload(output)
    if not isinstance(payload, dict):
        return None
    failed = payload.get("ok") is False or payload.get("is_error") is True
    status = str(payload.get("status", "")).strip().lower()
    failed = failed or status in {"error", "failed", "failure", "blocked"}
    error = payload.get("error")
    if not failed and error and payload.get("ok") is not True:
        failed = True
    if not failed:
        return None
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Tool failed.")
    return str(error or payload.get("message") or payload.get("error_code") or "Tool failed.")


def _parse_payload(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, str):
        return None
    text = output.strip()
    if not text:
        return None
    candidates = (text, text.split("\n", 1)[-1].strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            return json_safe_tool_payload(payload)
    return None
