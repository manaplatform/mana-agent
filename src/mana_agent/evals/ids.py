from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: Any, *, length: int = 20) -> str:
    return f"{prefix}_{stable_hash(value)[:length]}"


def execution_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
