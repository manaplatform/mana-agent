from __future__ import annotations
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from mana_agent.connectors.email.exceptions import ApprovalRequired

class ApprovalBinding(BaseModel):
    account_id: str; provider: str; action: str
    sender: str | None = None; recipients: list[str] = []
    cc: list[str] = []; bcc: list[str] = []; subject: str | None = None
    body_hash: str | None = None; attachment_hashes: list[str] = []
    source_message_id: str | None = None; affected_message_ids: list[str] = []
    def digest(self) -> str: return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()

class EmailApproval(BaseModel):
    id: str; binding_hash: str; expires_at: datetime
    def assert_valid_for(self, binding: ApprovalBinding, now: datetime | None = None) -> None:
        if self.binding_hash != binding.digest() or self.expires_at <= (now or datetime.now(timezone.utc)):
            raise ApprovalRequired("Email approval is missing, expired, or does not match this exact action.")

def approval_for(binding: ApprovalBinding, approval_id: str, ttl_minutes: int = 10) -> EmailApproval:
    return EmailApproval(id=approval_id, binding_hash=binding.digest(), expires_at=datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes))
