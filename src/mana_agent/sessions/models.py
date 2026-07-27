from __future__ import annotations

from pydantic import BaseModel



class SessionSummary(BaseModel):
    session_id: str
    short_id: str
    title: str
    current: bool = False
    status: str
    repository: str
    workspace_id: str
    created_at: str
    updated_at: str
    message_count: int = 0
    has_background_processes: bool = False


class SessionActivation(BaseModel):
    session: SessionSummary
    messages: list[dict]
