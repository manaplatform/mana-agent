from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mana_agent.multi_agent.runtime.ask_agent import AskAgent


class _FakeBoundLLM:
    """LLM stub that replays a scripted list of AI messages, then stops."""

    def __init__(self, scripted: list[SimpleNamespace]) -> None:
        self._scripted = list(scripted)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages, config=None):  # noqa: ARG002
        if self._scripted:
            return self._scripted.pop(0)
        return SimpleNamespace(content="done", tool_calls=[])


class _CapturingLogger:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def log(self, payload: dict) -> None:
        self.records.append(payload)


def _make_agent(tmp_path: Path) -> AskAgent:
    agent = AskAgent(
        api_key="x",
        model="fake-model",
        search_service=SimpleNamespace(),  # type: ignore[arg-type]
        project_root=tmp_path,
    )
    return agent


def _read_call(path: str, mode: str = "full", call_id: str = "c") -> SimpleNamespace:
    return SimpleNamespace(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": path, "mode": mode}, "id": call_id}],
    )


def test_read_file_missing_returns_structured_error(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    tools, _traces, _sources, _warnings = agent._build_tools(k_default=1, timeout_seconds=5)
    read_tool = {tool.name: tool for tool in tools}["read_file"]

    raw = read_tool.invoke({"path": "agent.md", "mode": "full"})
    payload = json.loads(raw)
    assert payload["ok"] is False
    assert payload["error_code"] == "file_not_found"
    assert payload["tool"] == "read_file"
    assert payload["path"] == "agent.md"
    assert payload["resolved_path"].endswith("agent.md")


def test_repeated_failed_reads_stop_after_limit(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    logger = _CapturingLogger()
    agent.run_logger = logger  # type: ignore[assignment]

    # Model insists on reading a missing file four times in a row.
    agent.llm = _FakeBoundLLM(  # type: ignore[assignment]
        [
            _read_call("agent.md", call_id="c1"),
            _read_call("agent.md", call_id="c2"),
            _read_call("agent.md", call_id="c3"),
            _read_call("agent.md", call_id="c4"),
        ]
    )

    result = agent.run(
        question="read agent.md",
        index_dir=tmp_path / ".mana" / "index",
        k=1,
        max_steps=10,
    )

    # read_file is only actually executed twice; further attempts are blocked.
    read_traces = [t for t in result.trace if t.tool_name == "read_file"]
    assert len(read_traces) == 2

    record = logger.records[-1]
    assert record["tool_calls_blocked_by_policy"] >= 2
    assert record["tool_calls_failed"] == 2
    codes = {entry.get("error_code") for entry in record["tool_errors"]}
    assert "file_not_found" in codes
    # The missing file is mentioned in the final answer (no crash).
    assert isinstance(result.answer, str) and result.answer


def test_metrics_count_blocked_vs_failed(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    logger = _CapturingLogger()
    agent.run_logger = logger  # type: ignore[assignment]

    # One real (failing) read, then a blocked unknown tool, then finish.
    agent.llm = _FakeBoundLLM(  # type: ignore[assignment]
        [
            _read_call("missing.txt", call_id="r1"),
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "made_up_tool", "args": {}, "id": "u1"}],
            ),
        ]
    )

    agent.run(
        question="inspect",
        index_dir=tmp_path / ".mana" / "index",
        k=1,
        max_steps=6,
    )

    record = logger.records[-1]
    assert record["tool_calls_attempted"] >= 2
    assert record["tool_calls_failed"] >= 1
    # unknown tool is counted as failed (error), not hidden.
    assert any(e["tool"] == "read_file" for e in record["tool_errors"])
