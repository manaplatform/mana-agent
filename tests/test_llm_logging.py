from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mana_agent.analysis.models import SearchHit
from mana_agent.context_cost.governor import ContextCostGovernor
from mana_agent.multi_agent.runtime.ask_agent import AskAgent
from mana_agent.multi_agent.runtime.qna_chain import QnAChain
import pytest

from mana_agent.multi_agent.runtime.run_logger import LlmRunLogger


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeInvoker:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _payload: dict) -> _FakeResponse:
        return _FakeResponse(content=self._content)


class _FakePrompt:
    def __init__(self, content: str) -> None:
        self._content = content

    def __or__(self, _other: object) -> _FakeInvoker:
        return _FakeInvoker(content=self._content)


class _FakeSearchService:
    def search(self, index_dir: Path, query: str, k: int) -> list[SearchHit]:
        assert index_dir
        assert query
        assert k > 0
        return [SearchHit(0.9, "/tmp/a.py", 1, 2, "x", "def x(): pass")]


class _FakeAIMessage:
    def __init__(self, content: str, tool_calls: list[dict] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeBoundModel:
    def __init__(self, responses: list[_FakeAIMessage]) -> None:
        self._responses = responses
        self._idx = 0

    def invoke(
        self, _messages: list[object], config: object | None = None
    ) -> _FakeAIMessage:
        del config
        value = self._responses[self._idx]
        self._idx += 1
        return value


class _FakeLLM:
    def __init__(self, responses: list[_FakeAIMessage]) -> None:
        self._responses = responses

    def bind_tools(self, _tools: list[object]) -> _FakeBoundModel:
        return _FakeBoundModel(self._responses)


def _read_rows(log_file: Path) -> list[dict]:
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_llm_run_logger_survives_deleted_cwd(tmp_path: Path, monkeypatch) -> None:
    """SWE-bench worktree thrash must not crash gateway init via Path.cwd()."""
    import os
    import shutil

    work = tmp_path / "cwd"
    work.mkdir()
    mana = tmp_path / "mana-home"
    mana.mkdir()
    monkeypatch.setenv("MANA_HOME", str(mana))
    monkeypatch.chdir(work)
    shutil.rmtree(work)
    with pytest.raises(FileNotFoundError):
        os.getcwd()

    logger = LlmRunLogger()
    assert logger.log_file.is_absolute()
    assert logger.log_file.name.endswith(".jsonl")
    # Project name should come from the safe fallback (MANA_HOME), not crash.
    assert "mana-home" in logger.log_file.name or "mana-home" in str(logger.log_file)


def test_qna_chain_logs_each_run(tmp_path: Path) -> None:
    log_file = tmp_path / "llm.jsonl"
    chain = QnAChain.__new__(QnAChain)
    chain.prompt = _FakePrompt("answer")
    chain.llm = object()
    chain.model = "fake-qna"
    chain.run_logger = LlmRunLogger(log_file)
    result = chain.run("q?", "ctx")
    assert result == "answer"
    rows = _read_rows(log_file)
    assert len(rows) == 1
    assert rows[0]["flow"] == "qna"


def test_ask_agent_logs_each_run(tmp_path: Path) -> None:
    log_file = tmp_path / "llm.jsonl"
    governor = ContextCostGovernor(
        session_id="test-llm-logging",
        settings=SimpleNamespace(
            mana_context_governor_enabled=False,
            mana_context_cost_log_enabled=False,
        ),
    )
    agent = AskAgent(
        api_key="test",
        model="fake-agent",
        search_service=_FakeSearchService(),
        project_root=tmp_path,
        context_cost_governor=governor,
    )
    agent._resolved_index = tmp_path / ".mana/index"
    agent.run_logger = LlmRunLogger(log_file)
    agent.llm = _FakeLLM([_FakeAIMessage("done /tmp/a.py:1-2", tool_calls=[])])

    result = agent.run("where?", tmp_path / ".mana/index", 3, max_steps=2, timeout_seconds=2)
    assert "done" in result.answer
    rows = _read_rows(log_file)
    assert len(rows) == 1
    assert rows[0]["flow"] == "ask-agent"
