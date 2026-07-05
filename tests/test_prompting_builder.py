from pathlib import Path

from mana_agent.agent.flow import FLOW_ORDER, build_agent_flow
from mana_agent.agent.selection import AgentPhase
from mana_agent.prompting.builder import (
    PromptCache,
    build_coding_system_prompt,
    build_ephemeral_context,
    compose,
    get_or_build_stable_prompt,
    render_ephemeral_context,
)
from mana_agent.prompting.layers import PROMPT_LAYER_ORDER, PromptLayer, compose_layers


def test_build_agent_flow_connects_selection_context_and_verification(tmp_path: Path) -> None:
    flow = build_agent_flow(
        "Fix prompt builder flow in src/mana_agent/multi_agent/runtime/coding_agent.py",
        repo_root=tmp_path,
        candidate_files=("src/mana_agent/multi_agent/runtime/coding_agent.py",),
    )

    assert FLOW_ORDER[0] is AgentPhase.DISCOVER
    assert flow.context.mode == "edit"
    assert flow.context.phase is AgentPhase.READ
    assert flow.context.repo_root == tmp_path.resolve()
    assert "src/mana_agent/multi_agent/runtime/coding_agent.py" in flow.context.candidate_files
    assert "src/mana_agent/multi_agent/runtime/coding_agent.py" in flow.context.candidate_search_terms
    assert flow.context.done_criteria
    assert flow.verification.commands or flow.verification.notes


def test_coding_prompt_builder_composes_stable_layers(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "testing").mkdir()
    (tmp_path / "skills" / "testing" / "SKILL.md").write_text(
        "---\n"
        "name: testing\n"
        "description: Focused verification skill.\n"
        "trigger: Use when tests or pytest are mentioned.\n"
        "---\n\n"
        "# Testing Skill\n\n"
        "SECRET FULL BODY SHOULD ONLY BE EPHEMERAL.\n",
        encoding="utf-8",
    )
    (tmp_path / ".mana").mkdir()
    (tmp_path / ".mana" / "memory.md").write_text("Known command: pytest -q\n", encoding="utf-8")

    prompt = build_coding_system_prompt(
        base_prompt="Core Identity",
        request="Add pytest coverage for prompt builder",
        repo_root=tmp_path,
        full_auto_mode=True,
        include_edit_rules=True,
        flow_context="Flow ID: abc123",
    )

    assert prompt.index("Core Identity") < prompt.index("Language-aware tooling")
    assert prompt.index("Language-aware tooling") < prompt.index("Agent Behavior Rules")
    assert prompt.index("Agent Behavior Rules") < prompt.index("Available skills:")
    assert prompt.index("Available skills:") < prompt.index("Repository Rules")
    assert prompt.index("Repository Rules") < prompt.index("Safety and Verification Rules")
    assert prompt.index("Safety and Verification Rules") < prompt.index("Ephemeral Prompt Context")
    assert "description: Focused verification skill." in prompt
    assert "trigger: Use when tests or pytest are mentioned." in prompt
    assert "Matched Skill Content" in prompt
    assert "SECRET FULL BODY SHOULD ONLY BE EPHEMERAL." in prompt
    assert "Project Memory Snapshot" in prompt
    assert "Known command: pytest -q" in prompt
    assert "Current Task Context" in prompt
    assert "explicit_requirements" in prompt
    assert "done_criteria" in prompt
    assert "Output Contract" in prompt
    assert "Flow ID: abc123" in prompt
    assert prompt.count("Core Identity") == 1


def test_compose_layers_rejects_unstable_order() -> None:
    layers = [PromptLayer(name, name) for name in reversed(PROMPT_LAYER_ORDER)]

    try:
        compose_layers(layers)
    except ValueError as exc:
        assert "stable order" in str(exc)
    else:
        raise AssertionError("compose_layers should reject prompts outside the stable order")


def test_stable_prompt_cache_reuses_across_turns(tmp_path: Path) -> None:
    cache = PromptCache()
    kwargs = {
        "base_prompt": "Core Identity",
        "repo_root": tmp_path,
        "full_auto_mode": False,
        "enabled_tools": ("read_file", "apply_patch"),
        "model_profile": {"model": "test-model", "provider": "test"},
        "cache": cache,
    }

    first = get_or_build_stable_prompt(**kwargs)
    assert cache.last_hit is False
    second = get_or_build_stable_prompt(**kwargs)

    assert cache.last_hit is True
    assert second is first
    assert second.cache_key == first.cache_key


def test_current_task_does_not_affect_stable_cache_key(tmp_path: Path) -> None:
    cache = PromptCache()
    first = build_coding_system_prompt(
        base_prompt="Core Identity",
        request="Fix docs",
        repo_root=tmp_path,
        prompt_cache=cache,
        enabled_tools=("read_file",),
        model_profile={"model": "test"},
    )
    first_key = cache._state.cache_key  # noqa: SLF001
    second = build_coding_system_prompt(
        base_prompt="Core Identity",
        request="Implement a different feature",
        repo_root=tmp_path,
        prompt_cache=cache,
        enabled_tools=("read_file",),
        model_profile={"model": "test"},
    )

    assert cache.last_hit is True
    assert cache._state.cache_key == first_key  # noqa: SLF001
    assert "Fix docs" in first
    assert "Implement a different feature" in second
    assert "Implement a different feature" not in cache._state.identity  # noqa: SLF001


def test_retrieved_files_do_not_affect_stable_cache_key(tmp_path: Path) -> None:
    cache = PromptCache()
    stable = get_or_build_stable_prompt(
        base_prompt="Core Identity",
        repo_root=tmp_path,
        enabled_tools=("read_file",),
        model_profile={"model": "test"},
        cache=cache,
    )
    first_key = stable.cache_key

    compose(stable, build_ephemeral_context("Task", retrieved_files=["src/a.py snippet"]), "Task")
    compose(stable, build_ephemeral_context("Task", retrieved_files=["src/b.py snippet"]), "Task")

    assert cache._state.cache_key == first_key  # noqa: SLF001


def test_compose_adds_tool_messages_only_when_needed(tmp_path: Path) -> None:
    stable = get_or_build_stable_prompt(base_prompt="Core Identity", repo_root=tmp_path, cache=PromptCache())
    without_tools = compose(stable, build_ephemeral_context("Task"), "Task")
    with_tools = compose(stable, build_ephemeral_context("Task", tool_results=["pytest passed"]), "Task")

    assert [item["role"] for item in without_tools] == ["system", "developer", "user"]
    assert [item["role"] for item in with_tools] == ["system", "developer", "user", "tool"]
    assert with_tools[-1]["content"] == "pytest passed"


def test_skill_change_invalidates_stable_cache(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "testing").mkdir()
    skill = tmp_path / "skills" / "testing" / "SKILL.md"
    skill.write_text(
        "---\nname: testing\ndescription: Focused checks.\ntrigger: Use when tests are mentioned.\n---\n\nBody v1\n",
        encoding="utf-8",
    )
    cache = PromptCache()
    first = get_or_build_stable_prompt(
        base_prompt="Core Identity",
        repo_root=tmp_path,
        enabled_tools=("read_file",),
        model_profile={"model": "test"},
        cache=cache,
    )

    skill.write_text(
        "---\nname: testing\ndescription: Focused checks v2.\ntrigger: Use when tests are mentioned.\n---\n\nBody v2\n",
        encoding="utf-8",
    )
    second = get_or_build_stable_prompt(
        base_prompt="Core Identity",
        repo_root=tmp_path,
        enabled_tools=("read_file",),
        model_profile={"model": "test"},
        cache=cache,
    )

    assert cache.last_hit is False
    assert second.cache_key != first.cache_key


def test_stable_prompt_skill_index_excludes_full_skill_body_until_matched(tmp_path: Path) -> None:
    (tmp_path / "skills" / "django").mkdir(parents=True)
    (tmp_path / "skills" / "django" / "SKILL.md").write_text(
        "---\n"
        "name: django\n"
        "description: Django development skill.\n"
        "trigger: Use when Django models or migrations are mentioned.\n"
        "---\n\n"
        "# Django Skill\n\n"
        "FULL DJANGO BODY SHOULD NOT BE IN STABLE PROMPT.\n",
        encoding="utf-8",
    )

    stable = get_or_build_stable_prompt(base_prompt="Core Identity", repo_root=tmp_path, cache=PromptCache())
    stable_text = stable.skill_index

    assert "Django development skill." in stable_text
    assert "Use when Django models or migrations are mentioned." in stable_text
    assert "FULL DJANGO BODY SHOULD NOT BE IN STABLE PROMPT." not in stable_text


def test_tool_registry_change_invalidates_stable_cache(tmp_path: Path) -> None:
    cache = PromptCache()
    first = get_or_build_stable_prompt(
        base_prompt="Core Identity",
        repo_root=tmp_path,
        enabled_tools=("read_file",),
        model_profile={"model": "test"},
        cache=cache,
    )
    second = get_or_build_stable_prompt(
        base_prompt="Core Identity",
        repo_root=tmp_path,
        enabled_tools=("read_file", "apply_patch"),
        model_profile={"model": "test"},
        cache=cache,
    )

    assert cache.last_hit is False
    assert second.cache_key != first.cache_key


def test_ephemeral_prompt_changes_every_turn() -> None:
    first = render_ephemeral_context(build_ephemeral_context("Fix A", mode="edit", tool_results=["pytest failed"]))
    second = render_ephemeral_context(build_ephemeral_context("Explain B", mode="analyze", tool_results=["search found B"]))

    assert first != second
    assert "Fix A" in first
    assert "Explain B" in second


def test_prompt_size_does_not_grow_unbounded_across_turns(tmp_path: Path) -> None:
    cache = PromptCache()
    prompts = [
        build_coding_system_prompt(
            base_prompt="Core Identity",
            request=f"Turn {idx}: update docs with {'x' * 2000}",
            repo_root=tmp_path,
            prompt_cache=cache,
            enabled_tools=("read_file", "apply_patch"),
            model_profile={"model": "test"},
        )
        for idx in range(5)
    ]

    lengths = [len(item) for item in prompts]
    assert max(lengths) - min(lengths) < 600
    assert prompts[-1].count("Core Identity") == 1
    assert "Turn 0" not in prompts[-1]
