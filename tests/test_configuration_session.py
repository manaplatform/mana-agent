from __future__ import annotations

from mana_agent.config.session import ConfigurationDraft


def test_set_models_allows_empty_selectors_while_provider_is_being_tested() -> None:
    previous_models = {
        "MANA_PRIMARY_MODEL": "openai/previous-chat",
        "MANA_EMBEDDING_MODEL": "openai/previous-embedding",
        "OPENAI_CHAT_MODEL": "previous-chat",
        "OPENAI_EMBED_MODEL": "previous-embedding",
    }
    draft = ConfigurationDraft(
        original=dict(previous_models),
        values=dict(previous_models),
    )

    draft.set_models(provider="nvidia", high="", coding="", fast="", embedding="")

    assert draft.values["MANA_AI_PROVIDER"] == "nvidia"
    assert draft.values["MANA_CONFIGURED_PROVIDERS"] == ["nvidia"]
    assert {key: draft.values[key] for key in previous_models} == previous_models


def test_set_models_does_not_partially_replace_models_when_required_selector_is_empty() -> None:
    previous_models = {
        "MANA_PRIMARY_MODEL": "openai/previous-chat",
        "MANA_EMBEDDING_MODEL": "openai/previous-embedding",
        "OPENAI_CHAT_MODEL": "previous-chat",
        "OPENAI_EMBED_MODEL": "previous-embedding",
    }
    draft = ConfigurationDraft(
        original=dict(previous_models),
        values=dict(previous_models),
    )

    draft.set_models(
        provider="nvidia",
        high="nvidia/nemotron-chat",
        coding="",
        fast="",
        embedding="",
    )

    assert {key: draft.values[key] for key in previous_models} == previous_models


def test_set_models_qualifies_selected_models() -> None:
    draft = ConfigurationDraft(original={}, values={})

    draft.set_models(
        provider="nvidia",
        high="meta/llama-chat",
        coding="meta/llama-code",
        fast="meta/llama-fast",
        embedding="nvidia/nv-embed",
    )

    assert draft.values["MANA_PRIMARY_MODEL"] == "nvidia/meta/llama-chat"
    assert draft.values["MANA_EMBEDDING_MODEL"] == "nvidia/nvidia/nv-embed"
    assert draft.values["OPENAI_CODING_PLANNER_MODEL"] == "meta/llama-code"
    assert draft.values["OPENAI_TOOL_WORKER_MODEL"] == "meta/llama-fast"
