from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mana_agent.config.settings import Settings
from mana_agent.config.user_config import validate_config_values
from mana_agent.integrations.codex.prompt_builder import build_codex_prompt
from mana_agent.multi_agent.core.types import AgentRole, ExecutionContext
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.runtime.coding_agent_prompt import CODING_SYSTEM_PROMPT
from mana_agent.multi_agent.runtime.prompts import (
    ASK_AGENT_SYSTEM_PROMPT,
    CONVERSATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from mana_agent.multi_agent.runtime.qna_chain import QnAChain
from mana_agent.prompting.builder import (
    PromptCache,
    build_coding_system_prompt,
    build_ephemeral_context,
    compose,
    get_or_build_stable_prompt,
)
from mana_agent.prompting.layers import PROMPT_LAYER_ORDER
from mana_agent.spirit.adapter import apply_spirit_instruction
from mana_agent.spirit.compiler import (
    SPIRIT_MAX_COMPILED_TOKENS,
    compile_spirit_instruction,
    compile_spirit_semantics,
    estimate_spirit_tokens,
    spirit_prompt_marker,
    strip_spirit_instruction,
)
from mana_agent.spirit.errors import SpiritResolutionError
from mana_agent.spirit.registry import default_mana_spirit, resolve_spirit
from mana_agent.spirit.schema import Spirit, SpiritSettings
from mana_agent.spirit.self_model import compose_runtime_self
from mana_agent.coding.models import CodingTask, WorkspaceContext


def test_default_mana_spirit_resolves_curious_bold_calm() -> None:
    spirit = resolve_spirit()
    default = default_mana_spirit()

    assert spirit.id == "mana"
    assert spirit.version == 1
    assert spirit.identity.name == "Mana"
    assert spirit.identity.product == "Mana-Agent"
    assert spirit.ref() == default.ref()
    assert "understand" in spirit.temperament.curious.meaning
    assert "assumptions" in spirit.temperament.curious.meaning
    assert "decisively" in spirit.temperament.bold.meaning
    assert "authority" in spirit.temperament.bold.meaning
    assert "deliberate" in spirit.temperament.calm.meaning
    assert "uncertainty" in spirit.temperament.calm.meaning or "failure" in spirit.temperament.calm.meaning


def test_runtime_self_contains_mana_identity_and_runtime_model() -> None:
    current = compose_runtime_self(
        agent_name="coding-agent",
        agent_role="coding",
        provider="openai",
        model="gpt-4.1-mini",
        purpose="inspect the prompt builder",
    )

    assert current.spirit.id == "mana"
    assert current.spirit.version == 1
    assert current.agent.name == "coding-agent"
    assert current.agent.role == "coding"
    assert current.runtime.provider == "openai"
    assert current.runtime.model == "gpt-4.1-mini"
    assert current.purpose.task == "inspect the prompt builder"


def test_changing_runtime_model_does_not_change_spirit() -> None:
    first = compose_runtime_self(provider="openai", model="gpt-4.1-mini", agent_role="main")
    second = compose_runtime_self(provider="anthropic", model="claude-sonnet-4", agent_role="main")

    assert first.spirit == second.spirit
    assert first.runtime.model != second.runtime.model
    assert first.runtime.provider != second.runtime.provider


def test_changing_roles_does_not_change_spirit() -> None:
    coding = compose_runtime_self(agent_name="coding-agent", agent_role="coding", model="gpt-4.1-mini")
    planner = compose_runtime_self(agent_name="planner-agent", agent_role="planner", model="gpt-4.1-mini")

    assert coding.spirit == planner.spirit
    assert coding.agent.role != planner.agent.role
    compiled_coding = compile_spirit_instruction(coding)
    compiled_planner = compile_spirit_instruction(planner)
    assert spirit_prompt_marker(coding.spirit) in compiled_coding
    assert compiled_coding == compiled_planner


def test_spirit_prompt_is_injected_exactly_once(tmp_path) -> None:
    current = compose_runtime_self(
        agent_name="coding-agent",
        agent_role="coding",
        provider="test",
        model="test-model",
    )
    prompt = build_coding_system_prompt(
        base_prompt="Core Identity",
        request="Inspect the prompt builder",
        repo_root=tmp_path,
        prompt_cache=PromptCache(),
        runtime_self=current,
    )
    marker = spirit_prompt_marker(current.spirit)
    assert prompt.count(marker) == 1
    assert prompt.index(marker) < prompt.index("Core Identity")

    messages = compose(
        get_or_build_stable_prompt(
            base_prompt="Core Identity",
            repo_root=tmp_path,
            cache=PromptCache(),
            runtime_self=current,
        ),
        build_ephemeral_context("Inspect the prompt builder"),
        "Inspect the prompt builder",
    )
    system_text = messages[0]["content"]
    assert system_text.count(marker) == 1
    assert all(item["content"].count(marker) == 0 for item in messages[1:])

    doubled = apply_spirit_instruction(prompt, current)
    assert doubled.count(marker) == 1


def test_policy_and_repo_instructions_retain_precedence(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "Repository coding contract: keep patches small and never skip verification.\n",
        encoding="utf-8",
    )
    current = compose_runtime_self(agent_name="coding-agent", agent_role="coding", model="test-model")
    prompt = build_coding_system_prompt(
        base_prompt="You are mana-agent's expert Coding Orchestrator Agent.\nUse apply_patch for multi-file edits.",
        request="Fix a bug",
        repo_root=tmp_path,
        prompt_cache=PromptCache(),
        runtime_self=current,
        full_auto_mode=True,
    )
    marker = spirit_prompt_marker(current.spirit)
    assert prompt.index(marker) < prompt.index("Safety and Verification Rules")
    assert prompt.index(marker) < prompt.index("Repository Rules")
    assert prompt.index(marker) < prompt.index("Language-aware tooling")
    assert "Do not claim edits, fixes, or verification without tool or command evidence." in prompt
    assert "Repository coding contract: keep patches small" in prompt
    assert "apply_patch" in prompt
    assert "Full-auto execution mode is enabled." in prompt


def test_coding_and_repository_instructions_are_not_moved_into_spirit() -> None:
    spirit = default_mana_spirit()
    dumped = json.dumps(spirit.model_dump(), sort_keys=True)
    compiled = compile_spirit_instruction(compose_runtime_self(model="test-model"))
    forbidden = (
        "apply_patch",
        "AGENTS.md",
        "permission",
        "jailbreak",
        "approval",
        "memory capsule",
        "policy",
        "tool_choice",
        "when asked who",
        "i say i am",
        "you are not",
        "ignore previous",
        "developer mode",
        "chatgpt",
        "pretend",
        "vendor chatbot",
    )
    for token in forbidden:
        assert token not in dumped.lower()
        assert token not in compiled.lower()
    schema = Spirit.model_json_schema()
    assert set(schema["properties"]) == {"id", "version", "identity", "temperament"}
    with pytest.raises(ValidationError):
        Spirit.model_validate({**spirit.model_dump(), "policy": "deny-all"})
    with pytest.raises(ValidationError):
        SpiritSettings.model_validate({"id": "mana", "version": 1, "memory": {}})


def test_legacy_configuration_without_spirit_remains_functional() -> None:
    cleaned = validate_config_values({"MANA_AI_PROVIDER": "openai"})
    assert "spirit" not in cleaned
    spirit = resolve_spirit(settings=Settings())
    assert spirit.id == "mana"
    assert spirit.version == 1
    empty = resolve_spirit(settings=type("SettingsStub", (), {"spirit": {}})())
    assert empty.ref() == spirit.ref()


def test_unknown_spirit_fails_closed() -> None:
    with pytest.raises(SpiritResolutionError, match="No fallback spirit was selected"):
        resolve_spirit(spirit_id="not-mana", spirit_version=1)
    with pytest.raises(SpiritResolutionError):
        validate_config_values({"spirit": {"id": "not-mana", "version": 1}})


def test_prompt_constants_do_not_embed_spirit() -> None:
    marker = "Mana's Spirit"
    assert marker not in SYSTEM_PROMPT
    assert marker not in ASK_AGENT_SYSTEM_PROMPT
    assert marker not in CODING_SYSTEM_PROMPT
    assert marker not in CONVERSATION_SYSTEM_PROMPT
    assert "when asked who" not in CONVERSATION_SYSTEM_PROMPT.lower()
    assert "chatgpt" not in CONVERSATION_SYSTEM_PROMPT.lower()


def test_conversation_executor_binds_routed_spirit_after_model_selection() -> None:
    captured: list[object] = []

    class _LLM:
        def invoke(self, messages):
            captured.extend(messages)
            return SimpleNamespace(content="ok")

    chain = QnAChain.__new__(QnAChain)
    chain.llm = _LLM()
    chain.model = "gpt-5.6-luna"
    chain.provider = "openai"
    current = compose_runtime_self(
        agent_name="conversation-agent",
        agent_role="conversation",
        provider="openai",
        model="gpt-5.6-luna",
        purpose="who are you?",
    )
    answer = chain.chat("who are you?", runtime_self=current)
    system = str(getattr(captured[0], "content", ""))
    assert answer == "ok"
    assert system.startswith(
        "You are Mana-Agent, currently instantiated through openai/gpt-5.6-luna."
    )
    assert "active session history" in system
    assert system.count("Mana's Spirit (mana/1)") == 1
    assert "when asked who" not in system.lower()
    assert "chatgpt" not in system.lower()
    assert "you are not" not in system.lower()


def test_spirit_semantics_are_model_free() -> None:
    text = compile_spirit_semantics()
    assert text.startswith("Mana's Spirit (mana/1) is curious, bold, and calm:")
    assert "instantiated" not in text.lower()
    assert "openai" not in text.lower()
    assert "gpt" not in text.lower()
    assert "model" not in text.lower()


def test_spirit_announces_product_and_inference_model() -> None:
    openai_self = compose_runtime_self(
        agent_name="main-agent",
        agent_role="main",
        provider="openai",
        model="gpt-4.1-mini",
    )
    compiled = compile_spirit_instruction(openai_self)
    assert compiled.startswith(
        "You are Mana-Agent, currently instantiated through openai/gpt-4.1-mini."
    )
    assert compile_spirit_semantics(openai_self.spirit) in compiled
    assert "not a separate persona you must imitate" in compiled
    lowered = compiled.lower()
    assert "when asked who" not in lowered
    assert "i say i am" not in lowered
    assert "chatgpt" not in lowered
    assert "you are not" not in lowered

    other = compile_spirit_instruction(
        compose_runtime_self(provider="anthropic", model="claude-sonnet-4", agent_role="main")
    )
    assert other.startswith(
        "You are Mana-Agent, currently instantiated through anthropic/claude-sonnet-4."
    )
    assert compile_spirit_semantics(openai_self.spirit) in other
    assert openai_self.spirit == compose_runtime_self(
        provider="anthropic", model="claude-sonnet-4", agent_role="main"
    ).spirit


def test_spirit_compiled_footprint_stays_small() -> None:
    current = compose_runtime_self(
        agent_name="coding-agent",
        agent_role="coding",
        provider="openai",
        model="openai/gpt-4.1-mini",
    )
    compiled = compile_spirit_instruction(current)
    tokens = estimate_spirit_tokens(compiled)
    assert 40 <= tokens <= SPIRIT_MAX_COMPILED_TOKENS
    assert compiled.startswith(
        "You are Mana-Agent, currently instantiated through openai/gpt-4.1-mini."
    )
    assert "curious" in compiled and "bold" in compiled and "calm" in compiled
    assert "As a curious, bold, and calm" not in compiled
    remainder = strip_spirit_instruction(f"{compiled}\n\nCore Identity")
    assert remainder == "Core Identity"
    assert "Mana-Agent" not in remainder


def test_subagents_inherit_mana_spirit() -> None:
    registry = AgentRegistry()
    main = registry.find_by_role(AgentRole.MAIN)
    coding = registry.create_subagent(AgentRole.CODING, main.agent_id, ["edit"])
    reviewer = registry.create_subagent(AgentRole.REVIEWER, main.agent_id, ["review"])

    assert main.spirit_id == "mana"
    assert main.spirit_version == 1
    assert coding.spirit_id == main.spirit_id
    assert coding.spirit_version == main.spirit_version
    assert reviewer.spirit_id == main.spirit_id
    assert reviewer.role is AgentRole.REVIEWER
    assert reviewer.role is not main.role

    parent_self = compose_runtime_self(
        execution_context=ExecutionContext(
            agent_id=main.agent_id,
            agent_role=main.role.value,
            spirit_id=main.spirit_id,
            spirit_version=main.spirit_version,
            resolved_model="gpt-4.1-mini",
        )
    )
    child_self = compose_runtime_self(
        execution_context=ExecutionContext(
            agent_id=coding.agent_id,
            agent_role=coding.role.value,
            spirit_id=coding.spirit_id,
            spirit_version=coding.spirit_version,
            resolved_model="codex-mini",
        )
    )
    assert parent_self.spirit == child_self.spirit
    assert child_self.runtime.model == "codex-mini"


def test_checkpoint_serialization_stores_spirit_ref_not_prompt_text() -> None:
    current = compose_runtime_self(
        agent_name="coding-agent",
        agent_role="coding",
        model="gpt-4.1-mini",
        purpose="finish the remaining edit",
    )
    context = ExecutionContext(
        agent_id="subagent_coding_0001",
        agent_role="coding",
        resolved_model="gpt-4.1-mini",
        spirit_id=current.spirit.id,
        spirit_version=current.spirit.version,
    )
    payload = {
        "self": current.durable_ref(),
        "execution_context": context.as_dict(),
    }
    serialized = json.dumps(payload)
    assert payload["self"] == {"id": "mana", "version": 1}
    assert payload["execution_context"]["spirit_id"] == "mana"
    assert payload["execution_context"]["spirit_version"] == 1
    assert "curious" not in serialized
    assert "understand before assuming" not in serialized
    assert compile_spirit_instruction(current) not in serialized

    restored = ExecutionContext.from_mapping(payload["execution_context"])
    assert restored.spirit_id == "mana"
    assert restored.spirit_version == 1
    assert compose_runtime_self(execution_context=restored).spirit == current.spirit


def test_legacy_execution_context_without_spirit_reconstructs_default() -> None:
    restored = ExecutionContext.from_mapping(
        {"agent_id": "agent_coding_1", "agent_role": "coding", "resolved_model": "fast-model"}
    )
    current = compose_runtime_self(execution_context=restored)
    assert restored.spirit_id is None
    assert current.spirit.id == "mana"
    assert current.runtime.model == "fast-model"


def test_user_facing_prompts_keep_existing_contracts(tmp_path) -> None:
    current = compose_runtime_self(agent_name="ask-agent", agent_role="tool", model="test-model")
    ask_prompt = apply_spirit_instruction(ASK_AGENT_SYSTEM_PROMPT, current)
    qna_prompt = apply_spirit_instruction(SYSTEM_PROMPT, current)
    assert ask_prompt.count(spirit_prompt_marker(current.spirit)) == 1
    assert "Never guess behavior" in ask_prompt or "repository evidence" in ask_prompt.lower()
    assert "ACCURACY" in qna_prompt
    assert "apply_patch" in qna_prompt

    codex = build_codex_prompt(
        CodingTask(task_id="task-1", goal="Inspect a file", requires_repository_write=False),
        WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="readOnly"),
    )
    assert codex.count(spirit_prompt_marker(current.spirit)) == 1
    assert "Never invoke `ssh` from this Codex process" in codex
    assert "Repository instructions" in codex


def test_stable_layer_order_starts_with_spirit() -> None:
    assert PROMPT_LAYER_ORDER[0] == "spirit"
    assert "core_identity" in PROMPT_LAYER_ORDER
    assert PROMPT_LAYER_ORDER.index("spirit") < PROMPT_LAYER_ORDER.index("repo_rules")
    assert PROMPT_LAYER_ORDER.index("spirit") < PROMPT_LAYER_ORDER.index("verification_rules")
