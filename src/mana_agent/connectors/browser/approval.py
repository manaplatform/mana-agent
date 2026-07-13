from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from pydantic import BaseModel


class BrowserApprovalBinding(BaseModel):
    session_id: str
    page_version: int
    origin: str
    action: str
    target: str
    arguments: dict[str, Any]

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


class BrowserApproval(BaseModel):
    token: str
    binding_hash: str
    expires_at: datetime

    def valid_for(self, binding: BrowserApprovalBinding) -> bool:
        return self.binding_hash == binding.digest() and self.expires_at > datetime.now(timezone.utc)


def issue_approval(binding: BrowserApprovalBinding, *, ttl_minutes: int = 10) -> BrowserApproval:
    digest = binding.digest()
    token = hashlib.sha256(f"{digest}:{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:24]
    return BrowserApproval(token=token, binding_hash=digest, expires_at=datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes))

