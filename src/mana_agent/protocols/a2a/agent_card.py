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
) -> object:
    from a2a.types.a2a_pb2 import (
        AgentCapabilities,
        AgentCard,
        AgentInterface,
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
    return AgentCard(
        name="mana-agent",
        description="Repository-aware coding and analysis agent backed by Mana-Agent's shared gateway.",
        version=get_version(),
        supported_interfaces=[
            AgentInterface(url=f"{base}/a2a", protocol_binding="JSONRPC", protocol_version="1.0"),
            AgentInterface(url=base, protocol_binding="HTTP+JSON", protocol_version="1.0"),
        ],
        provider=AgentProvider(organization="Mana-Agent", url="https://github.com/ahmadiehsan/mana-agent"),
        capabilities=AgentCapabilities(streaming=True, push_notifications=False, extended_agent_card=False),
        security_schemes=schemes,
        security_requirements=requirements,
        default_input_modes=["text/plain", "text/markdown"],
        default_output_modes=["text/plain", "text/markdown", "application/json", "text/x-diff"],
        skills=skills,
    )
