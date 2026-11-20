from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from mana_agent.coding.evidence_tools import CodingEvidenceTools, INTEGRATE, INVESTIGATE
from mana_agent.integrations.codex.backend import CodexCodingBackend
from mana_agent.services.activity_context import ActivityContext, activity_scope


@pytest.fixture
def scoped_reader(tmp_path):
    task = SimpleNamespace(mode_workflow={}, coding_assignment={"allowed_scope": ["src"]},
                           primary_repository_id="repo", allowed_tools=[], forbidden_tools=[])
    calls = []
    refs = {}
    remaining = [1]

    def dispatch(name, args, *, call_id):
        calls.append((name, args, call_id))
        ref = {"evidence_id": f"read-{len(calls)}", "revision": 1}
        refs[ref["evidence_id"]] = SimpleNamespace(freshness="current", validation="supported", sources=[
            SimpleNamespace(path=args.get("path"), excerpt="observed", start_line=1, end_line=1),
        ])
        return {"ok": True, "repository_evidence_refs": [ref]}

    knowledge = SimpleNamespace(get=lambda ref: refs[ref.evidence_id], query=lambda query: query.model_dump())
    board = SimpleNamespace(get_task=lambda _: task, evidence_board=lambda **_: knowledge,
                            update_task=lambda _, **fields: vars(task).update(fields))
    tools = CodingEvidenceTools(board, "coding", tmp_path, dispatch=dispatch,
        remaining=lambda: remaining[0], charge=lambda: remaining.__setitem__(0, remaining[0] - 1))
    with activity_scope(ActivityContext(conversation_id="session", execution_id="turn", task_id="coding")):
        yield tools, task, calls, refs, remaining


def investigate(tools, *requests, call_id="select"):
    return tools.invoke(INVESTIGATE, {"objective": "Confirm handler behavior", "paths": ["src"],
                                    "requests": list(requests)}, call_id=call_id)


def test_batch_budget_is_atomic_and_cached_excerpt_survives_a_new_invocation(scoped_reader):
    tools, task, calls, refs, remaining = scoped_reader
    handler = {"tool": "read_file", "path": "src/handler.py"}
    search = {"tool": "repo_search", "paths": ["src"], "query": "handler"}
    failed = investigate(tools, handler, search)
    assert failed["error_code"] == "BUDGET_EXCEEDED"
    assert failed["remaining"] == 1 and remaining == [1] and calls == []

    assert investigate(tools, handler, call_id="narrow")["ok"]
    reopened = CodingEvidenceTools(tools.taskboard, tools.task_id, tools.root, dispatch=tools.dispatch,
                                   remaining=tools.remaining, charge=tools.charge)
    reused = investigate(reopened, handler, call_id="resume")
    assert reused["ok"] and reused["remaining"] == 0
    assert reused["observations"][0]["result"]["content"] == "observed"
    assert reused["observations"][0]["reused"]
    assert task.mode_workflow["coding_investigation"]["objective"] == "Confirm handler behavior"
    assert len(calls) == 1

    refs["read-1"].freshness = "stale"
    assert investigate(reopened, handler)["error_code"] == "BUDGET_EXCEEDED"
    assert len(calls) == 1


@pytest.mark.parametrize("path", [".", "src/.env", "src/../README.md", "README.md"])
def test_invalid_target_rejects_whole_batch(scoped_reader, path):
    tools, _, calls, _, remaining = scoped_reader
    with pytest.raises((ValueError, PermissionError)):
        investigate(tools, {"tool": "read_file", "path": "src/handler.py"}, {"tool": "read_file", "path": path})
    assert remaining == [1] and calls == []


def test_integration_queries_require_current_task_or_assigned_paths(scoped_reader):
    tools, _, calls, _, _ = scoped_reader
    result = tools.invoke(INTEGRATE, {"include_stale": True}, call_id="relevant")
    assert result["task_id"] == "coding" and not result["include_stale"]
    with pytest.raises(PermissionError):
        tools.invoke(INTEGRATE, {"task_id": "other-task"}, call_id="unrelated")
    assert calls == []


def test_codex_dynamic_tool_dispatch_is_turn_bound_and_deduplicated(scoped_reader):
    tools, _, calls, _, remaining = scoped_reader
    replies = []

    async def reply(request_id, *, success, text):
        replies.append((request_id, success, json.loads(text)))

    backend = SimpleNamespace(evidence_tools=tools, _client=SimpleNamespace(respond_to_dynamic_tool=reply))
    params = {"threadId": "thread", "turnId": "turn", "callId": "read", "tool": INVESTIGATE, "arguments": {
        "objective": "Confirm handler behavior", "paths": ["src"],
        "requests": [{"tool": "read_file", "path": "src/handler.py"}],
    }}

    async def dispatch_requests():
        results = {}
        for request_id, turn, args in ((1, "retired", params["arguments"]), (2, "turn", params["arguments"]),
                                       (3, "turn", params["arguments"]), (4, "turn", {})):
            await CodexCodingBackend._dispatch_dynamic_tool(backend,
                {"id": request_id, "params": {**params, "turnId": turn, "arguments": args}},
                thread_id="thread", turn_id="turn", results=results)

    asyncio.run(dispatch_requests())
    assert [success for _, success, _ in replies] == [False, True, True, False]
    assert replies[1][2] == replies[2][2]
    assert len(calls) == 1 and remaining == [0]
