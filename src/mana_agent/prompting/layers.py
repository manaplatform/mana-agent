from __future__ import annotations

from dataclasses import dataclass


PROMPT_TEMPLATE_VERSION = "stable-ephemeral-v1"


@dataclass(frozen=True, slots=True)
class PromptLayer:
    name: str
    content: str


@dataclass(frozen=True, slots=True)
class StablePromptState:
    identity: str
    tool_rules: str
    behavior_rules: str
    skill_index: str
    repo_rules: str
    verification_rules: str
    cache_key: str
    version: str


@dataclass(frozen=True, slots=True)
class EphemeralPromptContext:
    current_task: str
    mode: str
    retrieved_files: list[str]
    tool_results: list[str]
    recent_summary: str | None
    temporary_constraints: list[str]


STABLE_PROMPT_LAYER_ORDER: tuple[str, ...] = (
    "core_identity",
    "tool_rules",
    "agent_behavior_rules",
    "skills_index",
    "repo_rules",
    "verification_rules",
)


PROMPT_LAYER_ORDER = STABLE_PROMPT_LAYER_ORDER


def compose_layers(layers: list[PromptLayer], *, expected_order: tuple[str, ...] = STABLE_PROMPT_LAYER_ORDER) -> str:
    names = tuple(layer.name for layer in layers)
    if names != expected_order:
        expected = " -> ".join(expected_order)
        actual = " -> ".join(names)
        raise ValueError(f"prompt layers must follow stable order: {expected}; got: {actual}")
    return "\n\n".join(layer.content.strip() for layer in layers if layer.content.strip()).strip()
