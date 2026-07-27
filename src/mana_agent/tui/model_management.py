from __future__ import annotations

from dataclasses import dataclass
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from mana_agent.config.catalog_service import ModelCatalogService
from mana_agent.config.model_catalog import ModelDescriptor, ModelPurpose, filter_models
from mana_agent.config.provider_registry import split_qualified_model_id
from mana_agent.config.provider_registry import PROVIDERS
from mana_agent.config.user_config import load_effective_settings, save_user_config


@dataclass(frozen=True, slots=True)
class ModelSelection:
    provider: str
    model_id: str
    persist: bool = False

    @property
    def qualified_id(self) -> str:
        return f"{self.provider}/{self.model_id}"


def configured_agent_models(*, service: ModelCatalogService | None = None) -> list[ModelDescriptor]:
    values = load_effective_settings(include_env=False)
    provider = str(values.get("MANA_AI_PROVIDER") or "openai")
    configured = set(values.get("MANA_CONFIGURED_PROVIDERS") or [provider])
    if provider not in configured:
        return []
    catalog = (service or ModelCatalogService()).cached(
        provider=provider,
        base_url=str(values.get("OPENROUTER_BASE_URL") if provider == "openrouter" else values.get("OPENAI_BASE_URL") or PROVIDERS.get(provider).default_base_url),
    )
    models = filter_models(catalog, ModelPurpose.AGENT)
    current = str(values.get("OPENAI_CHAT_MODEL") or "").strip()
    if current and current not in {model.id for model in models}:
        from mana_agent.config.model_catalog import ModelCapability

        models.append(
            ModelDescriptor(
                provider=provider,
                id=current,
                capabilities=frozenset({ModelCapability.TEXT_GENERATION}),
                source="manual",
            )
        )
    return sorted(models, key=lambda item: item.qualified_id)


def save_default_model(selection: ModelSelection) -> None:
    save_user_config(
        {
            "MANA_AI_PROVIDER": selection.provider,
            "MANA_PRIMARY_MODEL": selection.qualified_id,
            "OPENAI_CHAT_MODEL": selection.model_id,
            "LLM_MODEL": selection.model_id,
            "MODEL_LEVEL_3_HIGH_REASONING": selection.model_id,
        },
        merge=True,
    )


class ModelManagementScreen(ModalScreen[ModelSelection | None]):
    """Credential-free in-chat model picker for already configured providers."""

    CSS = """
    ModelManagementScreen { align: center middle; }
    #model-dialog { width: 82; height: 85%; padding: 1 2; border: round #6366f1; background: #111827; }
    #model-list { height: 1fr; border: round #334155; }
    #model-details { min-height: 5; color: #cbd5e1; }
    .actions { height: 3; align-horizontal: right; }
    .actions Button { margin-left: 1; }
    """

    def __init__(self, *, current_model: str, catalog_service: ModelCatalogService | None = None) -> None:
        super().__init__()
        self.current_model = current_model
        self.catalog_service = catalog_service or ModelCatalogService()
        self.models: list[ModelDescriptor] = []
        self.visible_models: list[ModelDescriptor] = []

    def compose(self) -> ComposeResult:
        values = load_effective_settings(include_env=False)
        provider = str(values.get("MANA_AI_PROVIDER") or "openai")
        with Vertical(id="model-dialog"):
            yield Label("Models", classes="title")
            yield Static(f"Active provider: {provider}\nActive model: {self.current_model}\nConfigured providers: {', '.join(values.get('MANA_CONFIGURED_PROVIDERS') or [provider])}")
            yield Input(placeholder="Filter compatible models…", id="model-filter")
            yield ListView(id="model-list")
            yield Static("Select a model to view capabilities and catalog status.", id="model-details")
            with Horizontal(classes="actions"):
                yield Button("Refresh catalog", id="refresh-models")
                yield Button("Use for this session", id="use-session", variant="primary")
                yield Button("Save as default", id="save-default", variant="success")
                yield Button("Close", id="close-models")

    def on_mount(self) -> None:
        self._load_cached()
        self.query_one("#model-filter", Input).focus()

    def _load_cached(self) -> None:
        self.models = configured_agent_models(service=self.catalog_service)
        self._render_models(self.models)
        if not self.models:
            self.query_one("#model-details", Static).update(
                "Provider authentication is not configured or no compatible catalog is cached.\n"
                "Exit chat and run: mana-agent --configure"
            )

    def _render_models(self, models: list[ModelDescriptor]) -> None:
        self.visible_models = list(models)
        view = self.query_one("#model-list", ListView)
        view.clear()
        for model in models:
            active = "● " if model.id == self.current_model else "  "
            badges = ", ".join(capability.value.replace("_", " ") for capability in sorted(model.capabilities, key=str))
            view.append(ListItem(Label(f"{active}{model.qualified_id}  [{badges}]  · {model.source}")))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "model-filter":
            return
        needle = event.value.strip().lower()
        self._render_models([model for model in self.models if needle in model.qualified_id.lower()])

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        view = self.query_one("#model-list", ListView)
        if view.index is None or view.index >= len(self.visible_models):
            return
        model = self.visible_models[view.index]
        capabilities = ", ".join(capability.value for capability in sorted(model.capabilities, key=str)) or "unknown"
        availability = "available" if model.available else "currently unavailable"
        self.query_one("#model-details", Static).update(
            f"{model.qualified_id}\nCapabilities: {capabilities}\nCatalog: {model.source} · {availability}"
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-models":
            self.dismiss(None)
            return
        if event.button.id == "refresh-models":
            values = load_effective_settings(include_env=False)
            api_key = str(values.get("OPENROUTER_API_KEY") if provider == "openrouter" else values.get("OPENAI_API_KEY") or "")
            base_url = str(values.get("OPENROUTER_BASE_URL") if provider == "openrouter" else values.get("OPENAI_BASE_URL") or PROVIDERS.get(provider).default_base_url)
            if not api_key:
                self.query_one("#model-details", Static).update(
                    "Provider authentication is not configured.\nExit chat and run: mana-agent --configure"
                )
                return
            try:
                await self.run_worker(
                    lambda: self.catalog_service.refresh(
                        provider=str(values.get("MANA_AI_PROVIDER") or "openai"),
                        base_url=base_url,
                        api_key=api_key,
                        timeout_seconds=int(values.get("MANA_SEARCH_TIMEOUT_SECONDS") or 15),
                    ),
                    thread=True,
                    exclusive=True,
                ).wait()
            except Exception as exc:
                self.query_one("#model-details", Static).update(f"Catalog refresh failed: {exc}")
                return
            self._load_cached()
            return
        if event.button.id in {"use-session", "save-default"}:
            view = self.query_one("#model-list", ListView)
            if view.index is None or view.index >= len(self.visible_models):
                self.notify("Select a compatible model first.", severity="warning")
                return
            model = self.visible_models[view.index]
            selection = ModelSelection(model.provider, model.id, persist=event.button.id == "save-default")
            if selection.persist:
                save_default_model(selection)
            self.dismiss(selection)


def plain_models_command(command: str, *, current_model: str, catalog_service: ModelCatalogService | None = None) -> tuple[str, ModelSelection | None]:
    """Plain-terminal fallback for `/models current|refresh|set provider/model`."""
    values = load_effective_settings(include_env=False)
    provider = str(values.get("MANA_AI_PROVIDER") or "openai")
    parts = str(command or "").strip().split()
    action = parts[1].lower() if len(parts) > 1 else "current"
    if action == "current":
        return f"Active model: {provider}/{current_model}", None
    if action == "refresh":
        api_key = str(values.get("OPENROUTER_API_KEY") if provider == "openrouter" else values.get("OPENAI_API_KEY") or "")
        base_url = str(values.get("OPENROUTER_BASE_URL") if provider == "openrouter" else values.get("OPENAI_BASE_URL") or PROVIDERS.get(provider).default_base_url)
        if not api_key:
            return "Provider authentication is not configured. Exit chat and run: mana-agent --configure", None
        models = (catalog_service or ModelCatalogService()).refresh(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=int(values.get("MANA_SEARCH_TIMEOUT_SECONDS") or 15),
        )
        compatible = filter_models(models, ModelPurpose.AGENT)
        return f"Refreshed catalog: {len(compatible)} compatible agent model(s).", None
    if action == "set" and len(parts) == 3:
        selected_provider, model_id = split_qualified_model_id(parts[2], default_provider=provider)
        configured = set(values.get("MANA_CONFIGURED_PROVIDERS") or [provider])
        compatible = {model.qualified_id for model in configured_agent_models(service=catalog_service)}
        qualified = f"{selected_provider}/{model_id}"
        if selected_provider not in configured or qualified not in compatible:
            return "That provider/model is not configured or is not a compatible agent model.", None
        return f"Session model changed to {qualified}.", ModelSelection(selected_provider, model_id, persist=False)
    return "Use /models current, /models refresh, or /models set <provider/model>.", None
