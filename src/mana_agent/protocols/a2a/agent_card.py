"""Runtime-derived official A2A 1.0 Agent Card."""

from __future__ import annotations

from mana_agent._version import get_version


DEFAULT_SKILLS = {
    "conversation": ("General conversation", "Answer questions through the shared Mana gateway."),
    "repository-analysis": ("Repository analysis", "Inspect and explain an authorized repository."),
    "coding": ("Coding changes", "Plan and make policy-approved repository changes."),
    "code-review": ("Code review", "Review changes and report actionable findings."),
    "verification": ("Verification", "Run model-selected checks and report results."),
    "documentation": ("Documentation", "Create or update repository documentation."),
}


def build_agent_card(
    *,
    public_base_url: str,
    enabled_skills: set[str] | None = None,
    authentication: str = "bearer",
    canvas_enabled: bool | None = None,
) -> object:
    from a2a.types.a2a_pb2 import (
        AgentCapabilities,
        AgentCard,
        AgentInterface,
        AgentExtension,
        AgentProvider,
        AgentSkill,
        HTTPAuthSecurityScheme,
        SecurityRequirement,
        SecurityScheme,
        StringList,
    )

    selected = enabled_skills if enabled_skills is not None else set(DEFAULT_SKILLS)
    skills = [
        AgentSkill(
            id=skill_id,
            name=DEFAULT_SKILLS[skill_id][0],
            description=DEFAULT_SKILLS[skill_id][1],
            tags=["mana-agent", "repository"],
            input_modes=["text/plain", "text/markdown"],
            output_modes=["text/plain", "text/markdown", "application/json", "text/x-diff"],
        )
        for skill_id in sorted(selected & set(DEFAULT_SKILLS))
    ]
    schemes = {}
    requirements = []
    if authentication == "bearer":
        schemes["bearer"] = SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                description="Mana-Agent A2A bearer token.",
                scheme="bearer",
                bearer_format="opaque",
            )
        )
        requirements = [SecurityRequirement(schemes={"bearer": StringList(list=[])})]
    base = public_base_url.rstrip("/")
    if canvas_enabled is None:
        from mana_agent.canvas.config import CanvasConfig
        from mana_agent.config.settings import Settings

        canvas_config = CanvasConfig.from_settings(Settings())
        canvas_enabled = canvas_config.enabled
    else:
        from mana_agent.canvas.config import CanvasConfig

        canvas_config = CanvasConfig(enabled=canvas_enabled)
    extensions = []
    output_modes = ["text/plain", "text/markdown", "application/json", "text/x-diff"]
    if canvas_enabled:
        extensions.append(AgentExtension(
            uri="https://a2ui.org/a2a-extension/a2ui/v0.8",
            description="Optional A2UI v0.9.1 declarative surfaces (wire version v0.9).",
            required=False,
            params={
                "supportedProtocolVersions": list(canvas_config.protocol_versions),
                "supportedCatalogIds": list(canvas_config.allowed_catalogs),
                "acceptsInlineCatalogs": canvas_config.accept_inline_catalogs,
                "mimeType": "application/a2ui+json",
            },
        ))
        output_modes.append("application/a2ui+json")
        for skill in skills:
            skill.output_modes.append("application/a2ui+json")
    return AgentCard(
        name="mana-agent",
        description="Repository-aware coding and analysis agent backed by Mana-Agent's shared gateway.",
        version=get_version(),
        supported_interfaces=[
            AgentInterface(url=f"{base}/a2a", protocol_binding="JSONRPC", protocol_version="1.0"),
            AgentInterface(url=base, protocol_binding="HTTP+JSON", protocol_version="1.0"),
        ],
        provider=AgentProvider(organization="Mana-Agent", url="https://github.com/ahmadiehsan/mana-agent"),
        capabilities=AgentCapabilities(
            streaming=True, push_notifications=False, extended_agent_card=False,
            extensions=extensions,
        ),
        security_schemes=schemes,
        security_requirements=requirements,
        default_input_modes=["text/plain", "text/markdown"],
        default_output_modes=output_modes,
        skills=skills,
    )
