from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from mana_agent.config.settings import mana_home


class McpConfigError(ValueError):
    """Raised for invalid MCP configuration; callers must not guess a fallback."""


class McpServerConfig(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    transport: Literal["stdio", "streamable_http", "sse"]
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    token_env: str = ""
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("server id is required")
        return value

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "McpServerConfig":
        if self.transport == "stdio" and not self.command.strip():
            raise ValueError("stdio MCP servers require command")
        if self.transport != "stdio":
            try:
                HttpUrl(self.url)
            except Exception as exc:
                raise ValueError("HTTP MCP servers require a valid url") from exc
        return self

    @property
    def namespace(self) -> str:
        return f"mcp.{self.id}"

    def resolved_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        token = str(os.getenv(self.token_env) or "").strip() if self.token_env else load_mcp_token(self.id)
        if token:
            if self.id.casefold() == "context7":
                headers.setdefault("CONTEXT7_API_KEY", token)
            else:
                headers.setdefault("Authorization", f"Bearer {token}")
        return headers

    def resolved_env(self) -> dict[str, str] | None:
        """Return stdio environment overrides without exposing secrets in config."""
        values = dict(self.env)
        token = load_mcp_token(self.id)
        if token and self.id.casefold() == "context7":
            values.setdefault("CONTEXT7_API_KEY", token)
        if not values:
            return None
        return values


def default_mcp_config_path() -> Path:
    configured = str(os.getenv("MANA_MCP_CONFIG_PATH") or "").strip()
    return Path(configured).expanduser().resolve() if configured else mana_home() / "mcp.toml"


def mcp_secrets_path() -> Path:
    return mana_home() / "mcp_secrets.toml"


def load_mcp_token(server_id: str) -> str:
    path = mcp_secrets_path()
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        tokens = values.get("tokens") if isinstance(values, dict) else {}
        return str(tokens.get(server_id) or "") if isinstance(tokens, dict) else ""
    except (OSError, tomllib.TOMLDecodeError):
        return ""


def save_mcp_token(server_id: str, token: str) -> None:
    if not str(token).strip():
        raise McpConfigError("MCP token cannot be empty")
    path = mcp_secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open("rb") as handle:
            existing = tomllib.load(handle)
    tokens = dict(existing.get("tokens") or {})
    tokens[str(server_id)] = str(token)
    lines = ["[tokens]"] + [f"{key} = {json.dumps(value)}" for key, value in sorted(tokens.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def save_mcp_server(server: McpServerConfig, *, replace: bool = False) -> Path:
    path = default_mcp_config_path()
    servers = load_mcp_servers(path)
    if any(item.id == server.id for item in servers) and not replace:
        raise McpConfigError(f"MCP server already exists: {server.id}")
    if replace:
        servers = [item for item in servers if item.id != server.id]
    path.parent.mkdir(parents=True, exist_ok=True)
    servers.append(server)
    lines: list[str] = []
    for item in servers:
        lines.extend(["[[servers]]", f"id = {json.dumps(item.id)}", f"transport = {json.dumps(item.transport)}"])
        if item.command:
            lines.append(f"command = {json.dumps(item.command)}")
        if item.args:
            lines.append(f"args = {json.dumps(item.args)}")
        if item.url:
            lines.append(f"url = {json.dumps(item.url)}")
        if item.token_env:
            lines.append(f"token_env = {json.dumps(item.token_env)}")
        if item.headers:
            header_values = ", ".join(f"{key} = {json.dumps(value)}" for key, value in sorted(item.headers.items()))
            lines.append("headers = { " + header_values + " }")
        lines.append(f"timeout_seconds = {item.timeout_seconds}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def parse_mcp_server_json(value: str) -> McpServerConfig:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"invalid --mcp-server-json: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise McpConfigError("--mcp-server-json must be an object")
    try:
        return McpServerConfig.model_validate(raw)
    except Exception as exc:
        raise McpConfigError(f"invalid MCP server definition: {exc}") from exc


def load_mcp_servers(path: str | Path | None = None, overrides: list[str] | None = None) -> list[McpServerConfig]:
    config_path = Path(path).expanduser().resolve() if path else default_mcp_config_path()
    rows: list[Any] = []
    if config_path.exists():
        try:
            with config_path.open("rb") as handle:
                data = tomllib.load(handle)
            rows = data.get("servers", []) if isinstance(data, dict) else []
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise McpConfigError(f"could not load MCP config {config_path}: {exc}") from exc
    if not isinstance(rows, list):
        raise McpConfigError("mcp.toml [servers] must be an array of tables")
    servers: list[McpServerConfig] = []
    for row in rows:
        try:
            servers.append(McpServerConfig.model_validate(row))
        except Exception as exc:
            raise McpConfigError(f"invalid MCP server in {config_path}: {exc}") from exc
    servers.extend(parse_mcp_server_json(value) for value in (overrides or []))
    ids = [server.id for server in servers]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise McpConfigError("duplicate MCP server id(s): " + ", ".join(duplicates))
    return servers
