"""Validated execution-fabric configuration and built-in provider assembly."""

from __future__ import annotations

import json
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mana_agent.config.settings import Settings
from mana_agent.execution.errors import ProviderConfigurationError
from mana_agent.execution.providers.custom_http import CustomHTTPProvider
from mana_agent.execution.providers.kubernetes import KubernetesProvider
from mana_agent.execution.providers.local_docker import LocalDockerProvider
from mana_agent.execution.providers.local_process import LocalProcessProvider
from mana_agent.execution.providers.modal import ModalProvider
from mana_agent.execution.providers.remote_ssh import RemoteSSHProvider
from mana_agent.execution.registry import ProviderRegistry


BUILTIN_PROVIDERS = (
    "local-process", "local-docker", "remote-ssh", "kubernetes", "modal", "custom-http-runtime",
)


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    concurrency_limit: int = Field(default=4, gt=0)


class RoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    high_risk_provider: str = "local-docker"
    expensive_provider: str = "kubernetes"
    gpu_provider: str = "modal"
    deny_silent_fallback: bool = True


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_provider: str = "local-process"
    allowed_providers: list[str] = Field(default_factory=lambda: list(BUILTIN_PROVIDERS))
    cleanup_on_exit: bool = True
    default_idle_timeout_seconds: int = Field(default=900, gt=0)
    default_max_lifetime_seconds: int = Field(default=7200, gt=0)
    global_concurrency_limit: int = Field(default=16, gt=0)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @field_validator("allowed_providers")
    @classmethod
    def validate_names(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(BUILTIN_PROVIDERS))
        if unknown:
            raise ValueError(f"unknown execution providers: {', '.join(unknown)}")
        return value

    def model_post_init(self, __context: object) -> None:
        _ = __context
        if not self.providers:
            self.providers = {name: ProviderConfig(enabled=name == "local-process") for name in BUILTIN_PROVIDERS}
        else:
            for name in BUILTIN_PROVIDERS:
                self.providers.setdefault(name, ProviderConfig(enabled=name == "local-process"))
        if self.default_provider not in self.allowed_providers:
            raise ValueError("execution default_provider must be allowed")
        if not self.providers[self.default_provider].enabled:
            raise ValueError("execution default_provider must be enabled")

    @classmethod
    def from_settings(cls, settings: Settings) -> "ExecutionConfig":
        raw_allowed = getattr(settings, "mana_execution_allowed_providers", list(BUILTIN_PROVIDERS))
        if isinstance(raw_allowed, str):
            raw_allowed = [item.strip() for item in raw_allowed.split(",") if item.strip()]
        raw_routing = getattr(settings, "mana_execution_routing", {}) or {}
        raw_providers = getattr(settings, "mana_execution_providers", {}) or {}
        if isinstance(raw_routing, str):
            raw_routing = json.loads(raw_routing)
        if isinstance(raw_providers, str):
            raw_providers = json.loads(raw_providers)
        return cls(
            default_provider=getattr(settings, "mana_execution_default_provider", "local-process"),
            allowed_providers=raw_allowed,
            cleanup_on_exit=getattr(settings, "mana_execution_cleanup_on_exit", True),
            default_idle_timeout_seconds=getattr(settings, "mana_execution_idle_timeout_seconds", 900),
            default_max_lifetime_seconds=getattr(settings, "mana_execution_max_lifetime_seconds", 7200),
            global_concurrency_limit=getattr(settings, "mana_execution_global_concurrency", 16),
            routing=raw_routing,
            providers=raw_providers,
        )


def build_provider_registry(config: ExecutionConfig) -> ProviderRegistry:
    registry = ProviderRegistry()
    for name in config.allowed_providers:
        provider_config = config.providers[name]
        options = provider_config.model_dump(exclude={"enabled", "concurrency_limit"})
        if name == "local-process":
            provider = LocalProcessProvider()
        elif name == "local-docker":
            provider = LocalDockerProvider(default_image=str(options.get("default_image") or "python:3.12"))
        elif name == "remote-ssh":
            provider = RemoteSSHProvider(hosts=list(options.get("hosts") or []))
        elif name == "kubernetes":
            provider = KubernetesProvider(namespace=str(options.get("namespace") or "mana-runtimes"), client=options.get("client"))
        elif name == "modal":
            provider = ModalProvider(client=options.get("client"))
        elif name == "custom-http-runtime":
            provider = CustomHTTPProvider(
                base_url=str(options.get("base_url") or ""), credential_ref=str(options.get("credential_ref") or ""),
                signing_secret_ref=str(options.get("signing_secret_ref") or ""),
            )
        else:  # validated above; defensive against future incomplete additions
            raise ProviderConfigurationError(f"no factory is defined for provider: {name}")
        registry.register(provider)
    return registry
