from __future__ import annotations

import shutil
import subprocess
import os
import asyncio
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static, Switch, TabbedContent, TabPane

from mana_agent.config.catalog_service import ModelCatalogService, ProviderValidationError
from mana_agent.config.model_catalog import ModelPurpose, filter_models
from mana_agent.config.provider_registry import PROVIDERS
from mana_agent.config.session import ConfigurationDraft
from mana_agent.config.user_config import migrate_legacy_config
from mana_agent.search.registry import SEARCH_PROVIDERS


def validate_memory_connection(values: dict[str, Any]) -> None:
    from mana_agent.memory import MemoryConfig, MemoryService

    config = MemoryConfig.load(values)

    async def check() -> None:
        service = MemoryService(config=config)
        try:
            health = await service.healthcheck()
            if not health.healthy:
                raise ValueError(health.detail or health.status.value)
        finally:
            await service.close()

    asyncio.run(check())


def validate_search_connection(values: dict[str, Any]) -> None:
    from mana_agent.search.web_provider import ConfiguredWebSearchProvider

    provider = str(values.get("MANA_WEB_SEARCH_PROVIDER") or "")
    if not provider:
        raise ValueError("Select a web search provider first.")
    client = ConfiguredWebSearchProvider(
        provider=provider,
        api_key=str(values.get("MANA_WEB_SEARCH_API_KEY") or ""),
        endpoint=str(values.get("MANA_WEB_SEARCH_ENDPOINT") or ""),
        engine_id=str(values.get("MANA_WEB_SEARCH_ENGINE_ID") or ""),
        timeout_seconds=int(values.get("MANA_SEARCH_TIMEOUT_SECONDS") or 15),
    )
    client.search_sync("Mana-Agent connection test", max_results=1)


def validate_github_connection(values: dict[str, Any]) -> str:
    from mana_agent.search.github_provider import GitHubSearchProvider

    source = str(values.get("MANA_GITHUB_CREDENTIAL_SOURCE") or "disabled")
    if source == "disabled":
        raise ValueError("Select a GitHub credential source first.")
    if source == "gh_cli":
        client = GitHubSearchProvider(credential_source="gh_cli")
    else:
        token = str(values.get("MANA_GITHUB_TOKEN") or "")
        if source == "environment":
            reference = str(values.get("MANA_GITHUB_SECRET_REF") or "")
            token = str(os.getenv(reference) or "") if reference else ""
        if not token:
            raise ValueError("GitHub authentication is not configured.")
        client = GitHubSearchProvider(token=token, credential_source="token")
    payload = client._get_json("https://api.github.com/user")
    username = str(payload.get("login") or "").strip()
    if not username:
        raise ValueError("GitHub authentication succeeded without an account name.")
    return username


class DiscardChangesScreen(ModalScreen[bool]):
    def compose(self) -> ComposeResult:
        with Vertical(id="discard-dialog"):
            yield Label("Discard unsaved configuration changes?")
            with Horizontal(classes="actions"):
                yield Button("Keep editing", id="keep-editing", variant="primary")
                yield Button("Discard", id="discard", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "discard")


class ManaConfigurationApp(App[bool]):
    """Full-screen configuration editor; credentials never enter widget output."""

    TITLE = "Mana-Agent Configuration"
    CSS = """
    Screen { background: #0f1117; color: #e5e7eb; }
    Header, Footer { background: #1a1d27; color: #a5b4fc; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 1 2; overflow-y: auto; }
    .section-title { text-style: bold; color: #a5b4fc; margin-bottom: 1; }
    .hint { color: #94a3b8; margin-bottom: 1; }
    .status { min-height: 1; color: #86efac; margin: 1 0; }
    Input, Select { margin-bottom: 1; }
    .actions { height: 3; dock: bottom; align-horizontal: right; }
    .actions Button { margin-left: 1; }
    #discard-dialog { width: 60; height: 9; padding: 1 2; border: round #6366f1; background: #1a1d27; }
    #overview-grid { padding: 1; border: round #334155; }
    """
    BINDINGS = [Binding("escape", "cancel", "Cancel"), Binding("ctrl+s", "save", "Save")]

    def __init__(self, *, draft: ConfigurationDraft | None = None, catalog_service: ModelCatalogService | None = None) -> None:
        super().__init__()
        migrate_legacy_config()
        self.draft = draft or ConfigurationDraft.load()
        self.catalog_service = catalog_service or ModelCatalogService()
        self.saved = False
        self._models: list[Any] = []
        self._provider_validated = bool(
            self._provider_api_key(self.draft.original, str(self.draft.original.get("MANA_AI_PROVIDER") or "openai"))
            and self._provider_base_url(self.draft.original, str(self.draft.original.get("MANA_AI_PROVIDER") or "openai"))
            and self.draft.original.get("OPENAI_CHAT_MODEL")
        )
        self._search_validated = bool(
            not self.draft.original.get("MANA_SEARCH_ENABLE_WEB")
            or (
                self.draft.original.get("MANA_WEB_SEARCH_PROVIDER")
                and self.draft.original.get("MANA_WEB_SEARCH_API_KEY")
            )
        )
        self._github_validated = bool(
            not self.draft.original.get("MANA_SEARCH_ENABLE_GITHUB")
            or self.draft.original.get("MANA_GITHUB_CREDENTIAL_SOURCE") not in (None, "", "disabled")
        )
        self._memory_validated = str(self.draft.original.get("MANA_MEMORY_MODE") or "internal") == "internal"

    def compose(self) -> ComposeResult:
        values = self.draft.values
        provider_id = str(values.get("MANA_AI_PROVIDER") or "openai")
        provider_options = [(item.display_name, item.id) for item in PROVIDERS.all()]
        search_options = [("Disabled", "disabled"), *((item.display_name, item.id) for item in SEARCH_PROVIDERS)]
        github_source = str(values.get("MANA_GITHUB_CREDENTIAL_SOURCE") or "disabled")
        yield Header()
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"):
                yield Static(self._overview_text(), id="overview-grid")
                yield Static("Use Continue or the tabs to review every section. Secrets are stored separately and never displayed.", classes="hint")
            with TabPane("AI providers", id="providers"):
                yield Label("Inference provider", classes="section-title")
                yield Select(provider_options, value=provider_id, id="provider-select", allow_blank=False)
                yield Input(value=self._provider_base_url(values, provider_id), placeholder="https://api.example.com/v1", id="provider-base-url")
                yield Input(password=True, placeholder=self._secret_placeholder(self._provider_secret_name(provider_id)), id="provider-api-key")
                yield Input(value=str(values.get("MANA_PROVIDER_DISPLAY_NAME") or ""), placeholder="Custom provider display name (optional)", id="provider-display-name")
                with Horizontal(classes="actions-inline"):
                    yield Button("Test", id="test-provider", variant="primary")
                    yield Button("Remove credential", id="remove-provider-secret", variant="warning")
                yield Static("Not tested", id="provider-status", classes="status")
            with TabPane("Model routing", id="models"):
                yield Label("Recommended logical levels", classes="section-title")
                yield Static("Agent roles use High reasoning, Coding, and Fast/tool. Only text-generation models appear here.", classes="hint")
                yield Select(self._initial_model_options("OPENAI_CHAT_MODEL"), id="high-model", allow_blank=False)
                yield Select(self._initial_model_options("OPENAI_CODING_PLANNER_MODEL"), id="coding-model", allow_blank=False)
                yield Select(self._initial_model_options("OPENAI_TOOL_WORKER_MODEL"), id="fast-model", allow_blank=False)
                yield Static(self._role_mapping_text(), id="role-mapping")
                yield Label("Advanced role mappings", classes="section-title")
                yield Static("Assign a logical level or a direct configured model to each role.", classes="hint")
                yield Select(self._role_options("MANA_MODEL_MAIN"), id="role-main", allow_blank=False)
                yield Select(self._role_options("MANA_MODEL_HEAD_DECISION"), id="role-head", allow_blank=False)
                yield Select(self._role_options("MANA_MODEL_PLANNER"), id="role-planner", allow_blank=False)
                yield Select(self._role_options("MANA_MODEL_CODING"), id="role-coding", allow_blank=False)
                yield Select(self._role_options("MANA_MODEL_VERIFIER"), id="role-verifier", allow_blank=False)
                yield Select(self._role_options("MANA_MODEL_REVIEWER"), id="role-reviewer", allow_blank=False)
                yield Select(self._role_options("MANA_MODEL_TOOL"), id="role-tool", allow_blank=False)
                yield Select(self._role_options("MANA_MODEL_SUMMARIZER"), id="role-summarizer", allow_blank=False)
            with TabPane("Embeddings", id="embeddings"):
                yield Label("Embedding model", classes="section-title")
                yield Static("Only embedding-capable models are offered after catalog refresh.", classes="hint")
                yield Select(self._initial_model_options("OPENAI_EMBED_MODEL", embedding=True), id="embedding-model", allow_blank=False)
            with TabPane("Coding runtime", id="coding-runtime"):
                yield Label("Coding backend", classes="section-title")
                yield Select(
                    [("Codex app-server", "codex"), ("Mana-Agent internal", "internal")],
                    value=str(values.get("MANA_CODING_BACKEND") or ("codex" if values.get("MANA_CODEX_ENABLED", True) else "internal")),
                    id="coding-backend",
                    allow_blank=False,
                )
                yield Switch(value=bool(values.get("MANA_CODEX_ENABLED", True)), id="codex-enabled")
                yield Label("Enable the Codex integration", classes="hint")
                yield Static(
                    "Backend selection is fixed before each coding turn. Codex failures are never retried through the internal backend.",
                    classes="hint",
                )
            with TabPane("Memory", id="memory"):
                yield Label("Memory mode", classes="section-title")
                yield Select(
                    [("Internal", "internal"), ("External", "external")],
                    value=str(values.get("MANA_MEMORY_MODE") or "internal"),
                    id="memory-mode",
                    allow_blank=False,
                )
                yield Static("Internal memory is locally managed by Mana-Agent and remains the default.", id="memory-hint", classes="hint")
                yield Select([("Mem0", "mem0")], value="mem0", id="memory-provider", allow_blank=False)
                yield Input(password=True, placeholder="Mem0 API key (stored in the OS keyring)", id="mem0-api-key")
                yield Input(value=str(values.get("MEM0_ORG_ID") or ""), placeholder="Organization ID (optional)", id="mem0-org-id")
                yield Input(value=str(values.get("MEM0_PROJECT_ID") or ""), placeholder="Project ID (optional)", id="mem0-project-id")
                yield Input(value=str(values.get("MEM0_BASE_URL") or ""), placeholder="Custom base URL (optional)", id="mem0-base-url")
                yield Button("Test memory", id="test-memory", variant="primary")
                yield Static("Local memory ready", id="memory-status", classes="status")
            with TabPane("Web search", id="search"):
                yield Switch(value=bool(values.get("MANA_SEARCH_ENABLE_WEB", False)), id="search-enabled")
                yield Label("Enable web search", classes="hint")
                yield Select(search_options, value=str(values.get("MANA_WEB_SEARCH_PROVIDER") or "disabled"), id="search-provider", allow_blank=False)
                yield Input(password=True, placeholder=self._secret_placeholder("MANA_WEB_SEARCH_API_KEY"), id="search-api-key")
                yield Input(value=str(values.get("MANA_WEB_SEARCH_ENGINE_ID") or ""), placeholder="Engine ID (Google CSE only)", id="search-engine-id")
                yield Input(value=str(values.get("MANA_WEB_SEARCH_ENDPOINT") or ""), placeholder="Custom endpoint (custom provider only)", id="search-endpoint")
                with Horizontal(classes="actions-inline"):
                    yield Button("Test", id="test-search", variant="primary")
                    yield Button("Remove credential", id="remove-search-secret", variant="warning")
                yield Static("Not tested", id="search-status", classes="status")
            with TabPane("GitHub", id="github"):
                yield Switch(value=bool(values.get("MANA_SEARCH_ENABLE_GITHUB", False)), id="github-search-enabled")
                yield Label("Enable GitHub search", classes="hint")
                yield Switch(value=bool(values.get("MANA_GITHUB_METADATA_ENABLED", False)), id="github-metadata-enabled")
                yield Label("Enable repository metadata access", classes="hint")
                yield Select(
                    [("Disabled", "disabled"), ("GitHub CLI authentication", "gh_cli"), ("Manual token", "token"), ("Environment secret reference", "environment")],
                    value=github_source,
                    id="github-source",
                    allow_blank=False,
                )
                yield Input(password=True, placeholder=self._secret_placeholder("MANA_GITHUB_TOKEN"), id="github-token")
                yield Input(value=str(values.get("MANA_GITHUB_SECRET_REF") or ""), placeholder="Environment variable name", id="github-secret-ref")
                with Horizontal(classes="actions-inline"):
                    yield Button("Test", id="test-github", variant="primary")
                    yield Button("Remove credential", id="remove-github-secret", variant="warning")
                yield Static(self._github_cli_status(), id="github-status", classes="status")
            with TabPane("Protocols", id="protocols"):
                yield Label("Agent Client Protocol", classes="section-title")
                yield Switch(value=bool(values.get("MANA_ACP_ENABLED", True)), id="acp-enabled")
                yield Label("Enable ACP stdio support", classes="hint")
                yield Input(value=str(values.get("MANA_ACP_ALLOWED_ROOTS") or ""), placeholder="Additional allowed roots (comma-separated)", id="acp-roots")
                yield Switch(value=bool(values.get("MANA_ACP_MCP_FORWARDING", True)), id="acp-mcp-forwarding")
                yield Label("Allow per-session MCP forwarding", classes="hint")
                yield Label("Agent2Agent 1.0", classes="section-title")
                yield Switch(value=bool(values.get("MANA_A2A_SERVER_ENABLED", False)), id="a2a-server-enabled")
                yield Label("Enable authenticated A2A server", classes="hint")
                yield Input(value=str(values.get("MANA_A2A_HOST") or "127.0.0.1"), placeholder="Bind host", id="a2a-host")
                yield Input(value=str(values.get("MANA_A2A_PORT") or 8766), placeholder="Port", id="a2a-port")
                yield Input(value=str(values.get("MANA_A2A_PUBLIC_BASE_URL") or ""), placeholder="https://agent.example", id="a2a-public-url")
                yield Input(password=True, placeholder=self._secret_placeholder("MANA_A2A_SERVER_TOKEN"), id="a2a-token")
                yield Input(value=str(values.get("MANA_A2A_ENABLED_SKILLS") or ""), placeholder="Enabled skill IDs (comma-separated)", id="a2a-skills")
                yield Input(value=str(values.get("MANA_A2A_MAX_CONCURRENT_TASKS") or 4), placeholder="Maximum concurrent tasks", id="a2a-concurrency")
                yield Switch(value=bool(values.get("MANA_A2A_DELEGATION_ENABLED", False)), id="a2a-delegation-enabled")
                yield Label("Enable explicitly authorized remote delegation", classes="hint")
                yield Input(value=str(values.get("MANA_A2A_MAX_DELEGATION_DEPTH") or 3), placeholder="Maximum delegation depth", id="a2a-depth")
            with TabPane("Computer control", id="computer-control"):
                computer = values.get("computer_control") if isinstance(values.get("computer_control"), dict) else {}
                yield Label("Local desktop automation", classes="section-title")
                yield Switch(value=bool(computer.get("enabled", False)), id="computer-enabled")
                yield Label("Enable scoped computer-control tools (disabled by default)", classes="hint")
                yield Switch(value=bool(computer.get("allow_remote_control", False)), id="computer-remote")
                yield Label("Allow authenticated remote clients (private-data scopes remain separately disabled)", classes="hint")
                yield Switch(value=bool(computer.get("require_local_confirmation_for_high_risk", True)), id="computer-local-confirmation")
                yield Label("Require trusted local confirmation for high-risk remote actions", classes="hint")
                yield Switch(value=bool(computer.get("audit_enabled", True)), id="computer-audit")
                yield Label("Keep sanitized action audit records (never note/page/clipboard/screenshot content)", classes="hint")
                yield Input(
                    value=",".join(str(item) for item in computer.get("allowed_paths", [])),
                    placeholder="Allowed filesystem roots (comma-separated absolute paths)",
                    id="computer-paths",
                )
                yield Static(
                    "Fine-grained permission decisions and the live supported/unsupported capability matrix are available in Dashboard → Computer Control.",
                    classes="hint",
                )
            with TabPane("Review and save", id="review"):
                yield Label("Review", classes="section-title")
                yield Static(self._overview_text(), id="review-summary")
                yield Static("Ctrl+S saves atomically. Existing credentials are preserved when masked fields are left empty.", classes="hint")
        with Horizontal(classes="actions"):
            yield Button("Back", id="back")
            yield Button("Continue", id="continue", variant="primary")
            yield Button("Cancel", id="cancel")
            yield Button("Save", id="save", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#provider-select", Select).focus()
        self._update_memory_fields()

    def _secret_placeholder(self, key: str) -> str:
        return "Configured •••••••• (leave blank to preserve)" if self.draft.values.get(key) else "Not configured"

    @staticmethod
    def _provider_secret_name(provider: str) -> str:
        return "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"

    @staticmethod
    def _provider_base_key(provider: str) -> str:
        return "OPENROUTER_BASE_URL" if provider == "openrouter" else "OPENAI_BASE_URL"

    def _provider_base_url(self, values: dict[str, Any], provider: str) -> str:
        return str(values.get(self._provider_base_key(provider)) or PROVIDERS.get(provider).default_base_url)

    def _provider_api_key(self, values: dict[str, Any], provider: str) -> str:
        return str(values.get(self._provider_secret_name(provider)) or "")

    def _initial_model_options(self, key: str, *, embedding: bool = False) -> list[tuple[str, str]]:
        current = str(self.draft.values.get(key) or "").strip()
        fallback = "text-embedding-3-small" if embedding else "gpt-4.1-mini"
        value = current or fallback
        return [(f"{value}  · manual/current", value)]

    def _role_options(self, key: str) -> list[tuple[str, str]]:
        current = str(self.draft.values.get(key) or "MODEL_LEVEL_3_HIGH_REASONING")
        levels = [
            ("High reasoning", "MODEL_LEVEL_3_HIGH_REASONING"),
            ("Coding", "MODEL_LEVEL_2_CODING"),
            ("Fast/tool", "MODEL_LEVEL_1_FAST_TOOL"),
        ]
        if current not in {value for _, value in levels}:
            levels.append((f"Direct: {current}", current))
        return levels

    def _overview_text(self) -> str:
        v = self.draft.values
        provider = str(v.get("MANA_AI_PROVIDER") or "openai")
        key_status = "Configured" if self._provider_api_key(v, provider) else "Not configured"
        search = str(v.get("MANA_WEB_SEARCH_PROVIDER") or "Disabled") if v.get("MANA_SEARCH_ENABLE_WEB") else "Disabled"
        github = str(v.get("MANA_GITHUB_CREDENTIAL_SOURCE") or "Disabled")
        memory = f"{v.get('MANA_MEMORY_MODE', 'internal')} / {v.get('MANA_MEMORY_PROVIDER', 'mana')}"
        return (
            f"AI provider       {v.get('MANA_AI_PROVIDER', 'openai')} ({key_status})\n"
            f"Primary model     {v.get('MANA_PRIMARY_MODEL') or v.get('OPENAI_CHAT_MODEL', 'Not selected')}\n"
            f"Embedding model   {v.get('MANA_EMBEDDING_MODEL') or v.get('OPENAI_EMBED_MODEL', 'Not selected')}\n"
            f"Web search        {search}\n"
            f"GitHub            {github}\n"
            f"Memory            {memory}"
            f"\nCoding backend    {v.get('MANA_CODING_BACKEND') or ('codex' if v.get('MANA_CODEX_ENABLED', True) else 'internal')}"
        )

    def _role_mapping_text(self) -> str:
        return "Main / Head decision / Planner / Reviewer → High reasoning\nCoding / Verifier → Coding\nTool / Summarizer → Fast/tool"

    @staticmethod
    def _github_cli_status() -> str:
        if shutil.which("gh") is None:
            return "GitHub CLI not installed"
        try:
            completed = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "GitHub CLI status unavailable"
        return "GitHub CLI authenticated" if completed.returncode == 0 else "GitHub CLI installed but not authenticated"

    def _collect(self) -> None:
        provider = str(self.query_one("#provider-select", Select).value)
        base_key = self._provider_base_key(provider)
        self.draft.values.update(
            {
                "MANA_AI_PROVIDER": provider,
                base_key: self.query_one("#provider-base-url", Input).value.strip() or PROVIDERS.get(provider).default_base_url,
                "MANA_PROVIDER_DISPLAY_NAME": self.query_one("#provider-display-name", Input).value.strip(),
                "MANA_SEARCH_ENABLE_WEB": self.query_one("#search-enabled", Switch).value,
                "MANA_WEB_SEARCH_PROVIDER": "" if str(self.query_one("#search-provider", Select).value) == "disabled" else str(self.query_one("#search-provider", Select).value),
                "MANA_WEB_SEARCH_ENGINE_ID": self.query_one("#search-engine-id", Input).value.strip(),
                "MANA_WEB_SEARCH_ENDPOINT": self.query_one("#search-endpoint", Input).value.strip(),
                "MANA_SEARCH_ENABLE_GITHUB": self.query_one("#github-search-enabled", Switch).value,
                "MANA_GITHUB_METADATA_ENABLED": self.query_one("#github-metadata-enabled", Switch).value,
                "MANA_GITHUB_CREDENTIAL_SOURCE": str(self.query_one("#github-source", Select).value),
                "MANA_GITHUB_SECRET_REF": self.query_one("#github-secret-ref", Input).value.strip(),
                "MANA_MEMORY_MODE": str(self.query_one("#memory-mode", Select).value),
                "MANA_MEMORY_PROVIDER": "mana" if str(self.query_one("#memory-mode", Select).value) == "internal" else str(self.query_one("#memory-provider", Select).value),
                "MANA_MEMORY_FALLBACK_TO_INTERNAL": False,
                "MANA_CODING_BACKEND": str(self.query_one("#coding-backend", Select).value),
                "MANA_CODEX_ENABLED": self.query_one("#codex-enabled", Switch).value,
                "MEM0_ORG_ID": self.query_one("#mem0-org-id", Input).value.strip(),
                "MEM0_PROJECT_ID": self.query_one("#mem0-project-id", Input).value.strip(),
                "MEM0_BASE_URL": self.query_one("#mem0-base-url", Input).value.strip(),
                "MANA_MODEL_MAIN": str(self.query_one("#role-main", Select).value),
                "MANA_MODEL_HEAD_DECISION": str(self.query_one("#role-head", Select).value),
                "MANA_MODEL_PLANNER": str(self.query_one("#role-planner", Select).value),
                "MANA_MODEL_CODING": str(self.query_one("#role-coding", Select).value),
                "MANA_MODEL_VERIFIER": str(self.query_one("#role-verifier", Select).value),
                "MANA_MODEL_REVIEWER": str(self.query_one("#role-reviewer", Select).value),
                "MANA_MODEL_TOOL": str(self.query_one("#role-tool", Select).value),
                "MANA_MODEL_SUMMARIZER": str(self.query_one("#role-summarizer", Select).value),
                "MANA_ACP_ENABLED": self.query_one("#acp-enabled", Switch).value,
                "MANA_ACP_ALLOWED_ROOTS": self.query_one("#acp-roots", Input).value.strip(),
                "MANA_ACP_MCP_FORWARDING": self.query_one("#acp-mcp-forwarding", Switch).value,
                "MANA_A2A_SERVER_ENABLED": self.query_one("#a2a-server-enabled", Switch).value,
                "MANA_A2A_HOST": self.query_one("#a2a-host", Input).value.strip(),
                "MANA_A2A_PORT": int(self.query_one("#a2a-port", Input).value),
                "MANA_A2A_PUBLIC_BASE_URL": self.query_one("#a2a-public-url", Input).value.strip(),
                "MANA_A2A_ENABLED_SKILLS": self.query_one("#a2a-skills", Input).value.strip(),
                "MANA_A2A_MAX_CONCURRENT_TASKS": int(self.query_one("#a2a-concurrency", Input).value),
                "MANA_A2A_DELEGATION_ENABLED": self.query_one("#a2a-delegation-enabled", Switch).value,
                "MANA_A2A_MAX_DELEGATION_DEPTH": int(self.query_one("#a2a-depth", Input).value),
                "MANA_COMPUTER_CONTROL_ENABLED": self.query_one("#computer-enabled", Switch).value,
            }
        )
        existing_computer = self.draft.values.get("computer_control")
        computer = dict(existing_computer) if isinstance(existing_computer, dict) else {}
        computer.update({
            "enabled": self.query_one("#computer-enabled", Switch).value,
            "allow_remote_control": self.query_one("#computer-remote", Switch).value,
            "require_local_confirmation_for_high_risk": self.query_one("#computer-local-confirmation", Switch).value,
            "audit_enabled": self.query_one("#computer-audit", Switch).value,
            "allowed_paths": [
                item.strip()
                for item in self.query_one("#computer-paths", Input).value.split(",")
                if item.strip()
            ],
        })
        self.draft.values["computer_control"] = computer
        self.draft.set_secret(self._provider_secret_name(provider), self.query_one("#provider-api-key", Input).value)
        self.draft.set_secret("MANA_WEB_SEARCH_API_KEY", self.query_one("#search-api-key", Input).value)
        self.draft.set_secret("MANA_GITHUB_TOKEN", self.query_one("#github-token", Input).value)
        self.draft.set_secret("MANA_A2A_SERVER_TOKEN", self.query_one("#a2a-token", Input).value)
        mem0_key = self.query_one("#mem0-api-key", Input).value.strip()
        if mem0_key:
            self.draft.values["MEM0_API_KEY"] = mem0_key
        self.draft.set_models(
            provider=provider,
            high=str(self.query_one("#high-model", Select).value),
            coding=str(self.query_one("#coding-model", Select).value),
            fast=str(self.query_one("#fast-model", Select).value),
            embedding=str(self.query_one("#embedding-model", Select).value),
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "test-provider":
            self._collect()
            status = self.query_one("#provider-status", Static)
            status.update("Testing…")
            try:
                models = await self.run_worker(
                    lambda: self.catalog_service.refresh(
                        provider=str(self.draft.values["MANA_AI_PROVIDER"]),
                        base_url=self._provider_base_url(self.draft.values, str(self.draft.values["MANA_AI_PROVIDER"])),
                        api_key=self._provider_api_key(self.draft.values, str(self.draft.values["MANA_AI_PROVIDER"])),
                        timeout_seconds=int(self.draft.values.get("MANA_SEARCH_TIMEOUT_SECONDS") or 15),
                    ),
                    thread=True,
                    exclusive=True,
                ).wait()
            except ProviderValidationError as exc:
                status.update(f"Validation failed: {exc}")
                return
            self._models = models
            self._provider_validated = True
            text_models = filter_models(models, ModelPurpose.AGENT)
            embedding_models = filter_models(models, ModelPurpose.EMBEDDING)
            for selector in ("#high-model", "#coding-model", "#fast-model"):
                widget = self.query_one(selector, Select)
                current = str(widget.value)
                options = [(self._model_label(item), item.id) for item in text_models]
                if current and current not in {value for _, value in options}:
                    options.append((f"{current} · manual", current))
                widget.set_options(options)
                widget.value = current if current in {value for _, value in options} else (options[0][1] if options else Select.BLANK)
            embed = self.query_one("#embedding-model", Select)
            current_embed = str(embed.value)
            embed_options = [(self._model_label(item), item.id) for item in embedding_models]
            if current_embed and current_embed not in {value for _, value in embed_options}:
                embed_options.append((f"{current_embed} · manual", current_embed))
            embed.set_options(embed_options)
            embed.value = current_embed if current_embed in {value for _, value in embed_options} else (embed_options[0][1] if embed_options else Select.BLANK)
            status.update(f"Connected · {len(text_models)} agent model(s), {len(embedding_models)} embedding model(s)")
            return
        if button_id == "test-memory":
            self._collect()
            status = self.query_one("#memory-status", Static)
            if self.draft.values.get("MANA_MEMORY_MODE") == "internal":
                self._memory_validated = True
                status.update("Healthy · locally managed by Mana-Agent")
                return
            status.update("Testing…")
            try:
                await self.run_worker(lambda: validate_memory_connection(self.draft.values), thread=True, exclusive=True).wait()
            except Exception as exc:
                self._memory_validated = False
                status.update(f"Validation failed: {exc}")
                return
            self._memory_validated = True
            status.update("Connected to Mem0")
            return
        if button_id in {"remove-provider-secret", "remove-search-secret", "remove-github-secret"}:
            key = {
                "remove-provider-secret": self._provider_secret_name(str(self.draft.values.get("MANA_AI_PROVIDER") or "openai")),
                "remove-search-secret": "MANA_WEB_SEARCH_API_KEY",
                "remove-github-secret": "MANA_GITHUB_TOKEN",
            }[button_id]
            self.draft.remove_secret(key)
            event.button.label = "Credential will be removed"
            return
        if button_id == "test-search":
            self._collect()
            status = self.query_one("#search-status", Static)
            status.update("Testing…")
            try:
                await self.run_worker(lambda: validate_search_connection(self.draft.values), thread=True, exclusive=True).wait()
            except Exception as exc:
                status.update(f"Validation failed: {exc}")
                self._search_validated = False
                return
            self._search_validated = True
            status.update("Connected and ready to use")
            return
        if button_id == "test-github":
            self._collect()
            status = self.query_one("#github-status", Static)
            status.update("Testing…")
            try:
                username = await self.run_worker(lambda: validate_github_connection(self.draft.values), thread=True, exclusive=True).wait()
            except Exception as exc:
                status.update(f"Validation failed: {exc}")
                self._github_validated = False
                return
            self._github_validated = True
            status.update(f"Authenticated as {username}")
            return
        if button_id == "save":
            self.action_save()
        elif button_id == "cancel":
            self.action_cancel()
        elif button_id in {"continue", "back"}:
            tabs = self.query_one(TabbedContent)
            order = ["overview", "providers", "models", "embeddings", "coding-runtime", "memory", "search", "github", "protocols", "computer-control", "review"]
            current = order.index(tabs.active)
            tabs.active = order[min(len(order) - 1, current + (1 if button_id == "continue" else -1))]

    @staticmethod
    def _model_label(model: Any) -> str:
        badges = ", ".join(sorted(capability.value.replace("_", " ").title() for capability in model.capabilities))
        return f"{model.id}  · {badges or 'Unknown'}"

    def action_save(self) -> None:
        try:
            self._collect()
            if not self._provider_validated:
                raise ValueError("Test the selected inference provider before saving.")
            if self.draft.values.get("MANA_SEARCH_ENABLE_WEB") and not self._search_validated:
                raise ValueError("Test the selected web search provider before saving.")
            if self.draft.values.get("MANA_SEARCH_ENABLE_GITHUB") and not self._github_validated:
                raise ValueError("Test the selected GitHub authentication before saving.")
            if not self._memory_validated:
                raise ValueError("Test the selected external memory provider before saving.")
            from mana_agent.protocols.common.config import validate_protocol_configuration
            from types import SimpleNamespace
            from mana_agent.coding.selection import resolve_coding_backend

            validate_protocol_configuration(self.draft.values)
            resolve_coding_backend(SimpleNamespace(
                mana_coding_backend=self.draft.values.get("MANA_CODING_BACKEND"),
                mana_codex_enabled=self.draft.values.get("MANA_CODEX_ENABLED"),
            ))
            self.draft.save()
        except Exception as exc:
            self.notify(str(exc), title="Configuration not saved", severity="error")
            return
        self.saved = True
        self.exit(True)

    def action_cancel(self) -> None:
        if not self.draft.dirty:
            self.exit(False)
            return
        self.push_screen(DiscardChangesScreen(), self._finish_cancel)

    def _finish_cancel(self, discard: bool | None) -> None:
        if discard:
            self.exit(False)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider-select":
            self._provider_validated = False
            provider = str(event.value)
            self.query_one("#provider-base-url", Input).value = self._provider_base_url(self.draft.values, provider)
            self.query_one("#provider-api-key", Input).placeholder = self._secret_placeholder(self._provider_secret_name(provider))
        elif event.select.id == "search-provider":
            self._search_validated = False
        elif event.select.id == "github-source":
            self._github_validated = False
        elif event.select.id in {"memory-mode", "memory-provider"}:
            self._memory_validated = str(self.query_one("#memory-mode", Select).value) == "internal"
            self._update_memory_fields()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in {"provider-api-key", "provider-base-url"} and event.value:
            self._provider_validated = False
        elif event.input.id in {"search-api-key", "search-engine-id", "search-endpoint"} and event.value:
            self._search_validated = False
        elif event.input.id in {"github-token", "github-secret-ref"} and event.value:
            self._github_validated = False
        elif event.input.id in {"mem0-api-key", "mem0-org-id", "mem0-project-id", "mem0-base-url"} and event.value:
            self._memory_validated = False

    def _update_memory_fields(self) -> None:
        external = str(self.query_one("#memory-mode", Select).value) == "external"
        for selector in ("#memory-provider", "#mem0-api-key", "#mem0-org-id", "#mem0-project-id", "#mem0-base-url", "#test-memory"):
            self.query_one(selector).display = external
        self.query_one("#memory-hint", Static).update(
            "Mem0 stores selected memory with an external provider. Review its privacy policy before enabling."
            if external else "Internal memory is locally managed by Mana-Agent and remains the default."
        )


def run_configuration_tui() -> bool:
    return bool(ManaConfigurationApp().run())
