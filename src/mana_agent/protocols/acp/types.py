"""Internal ACP adapter state; wire models come exclusively from the ACP SDK."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AcpSessionState:
    acp_session_id: str
    mana_session_id: str
    cwd: str
    additional_directories: tuple[str, ...] = ()
    mode: str = "ask"
    read_only: bool = False
    mcp_overrides: list[str] = field(default_factory=list, repr=False)
    closed: bool = False
