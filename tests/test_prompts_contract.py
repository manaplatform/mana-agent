from importlib import import_module

from mana_agent.multi_agent.runtime import prompts


_REQUIRED_PROMPTS = [
    "SYSTEM_PROMPT",
    "HUMAN_TEMPLATE",
    "ANALYZE_SYSTEM_PROMPT",
    "ANALYZE_HUMAN_TEMPLATE",
    "ASK_AGENT_SYSTEM_PROMPT",
    "TOOL_FIRST",
    "DEEP_FLOW_SYSTEM_PROMPT",
    "DEEP_FLOW_HUMAN_TEMPLATE",
]


def test_prompt_constants_exist_and_non_empty() -> None:
    for name in _REQUIRED_PROMPTS:
        value = getattr(prompts, name, None)
        assert isinstance(value, str)
        assert value.strip()


def test_prompt_import_smoke_for_dependent_modules() -> None:
    for module_name in [
        "mana_agent.multi_agent.runtime.ask_agent",
        "mana_agent.multi_agent.runtime.qna_chain",
        "mana_agent.multi_agent.runtime.repo_chain",
        "mana_agent.services.search_service",
    ]:
        import_module(module_name)


def test_toolsmanager_prompt_mentions_parallel_independence_and_ordering() -> None:
    text = str(getattr(prompts, "TOOLSMANAGER_PROMPT", "") or "").lower()
    assert "independent" in text
    assert "parallel" in text
    assert "input order" in text or "original input order" in text
    assert "changed_files" in text or "file-change evidence" in text
    assert "true blockers" in text


def test_coding_prompts_enforce_noop_retry_flow() -> None:
    system_text = str(getattr(prompts, "SYSTEM_PROMPT", "") or "").lower()
    recog_text = str(getattr(prompts, "CODING_AGENT_RECOGNITION_PROMPT", "") or "").lower()
    assert "no-op" in system_text
    assert "files changed" in system_text or "file-change evidence" in system_text
    assert "do not finalize on no-op" in system_text
    assert "if you want me to proceed" in system_text
    assert "no-op" in recog_text
    assert "edit_file" in recog_text and "multi_edit_file" in recog_text
    assert "apply_patch" in recog_text and "create_file" in recog_text and "write_file" in recog_text and "delete_file" in recog_text
    assert "execute the edit in the same turn" in recog_text
    assert "codex patch" in system_text
    assert "codex patch" in recog_text
    assert "old_start" not in system_text
    assert "old_start" not in recog_text


def test_coding_language_tooling_prompt_covers_python_node_and_ignores() -> None:
    text = str(getattr(prompts, "CODING_AGENT_LANGUAGE_TOOLING_PROMPT", "") or "").lower()
    assert ".venv" in text and "venv" in text
    assert "pytest -q" in text
    assert "node_modules" in text
    assert "npm install" in text
    assert "npm test" in text
