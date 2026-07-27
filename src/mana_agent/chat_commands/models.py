from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    status: Literal["success", "error", "confirmation_required", "input_required"]
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    next_prompt: dict[str, Any] | None = None
    background_process_id: str | None = None


class CommandInvocation(BaseModel):
    name: str
    arguments: list[str] = Field(default_factory=list)
    confirmed: bool = False


class CommandContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    frontend: str
    session_id: str = ""
    workspace_id: str = ""
    repository_id: str = ""
    capabilities: set[str] = Field(default_factory=set)
    gateway: Any | None = None
    sessions: Any | None = None
    processes: Any | None = None
    connectors: Any | None = None
    frontend_data: dict[str, Any] = Field(default_factory=dict)


CommandHandler = Callable[[CommandContext, list[str]], CommandResult]


class CommandArguments(BaseModel):
    """Validated token list passed to a command-specific parser/handler."""

    values: list[str] = Field(default_factory=list, max_length=64)


class CommandDefinition(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    canonical_name: str
    aliases: tuple[str, ...] = ()
    description: str
    argument_schema: str = ""
    argument_model: type[BaseModel] = CommandArguments
    required_capability: str = "chat"
    frontends: frozenset[str] = frozenset({"cli", "tui", "dashboard", "api", "telegram"})
    confirmation_required: bool = False
    confirmation_actions: frozenset[str] = frozenset()
    accepts_secrets: bool = False
    execution_mode: Literal["inline", "task", "background"] = "inline"
    renderer: str = "text"
    handler: CommandHandler
