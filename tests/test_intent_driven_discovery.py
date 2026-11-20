"""Instrument filesystem calls across the native repository discovery paths."""

import os
from pathlib import Path

import pytest

from mana_agent.agent.plan_service import EvidenceRequest, PlanService
from mana_agent.coding.programmatic import ToolExecutionDispatcher
from mana_agent.tools import repository
from mana_agent.multi_agent.agents import tool_runner_agent


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "project"
    expected = {
        "runtime/command.py": "def run_command():\n    return authorize()\n",
        "runtime/approval.py": "def authorize():\n    return True\n",
        "tests/test_command.py": "def test_run_command():\n    assert authorize()\n",
        "connectors/email.py": "# run_command and authorize are irrelevant here\n",
        "auth/providers.py": "# authorize\n",
        "docs/unrelated.md": "run_command\n",
        ".env": "run_command=fake\n",
    }
    for path, text in expected.items():
        target = root / path
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_text(text, encoding="utf-8")
    opens, walks = [], []
    original_open, original_walk = Path.open, os.walk

    def record_open(path, *args, **kwargs):
        if path.is_relative_to(root):
            opens.append(path.relative_to(root).as_posix())
        return original_open(path, *args, **kwargs)

    def record_walk(path, *args, **kwargs):
        if Path(path).is_relative_to(root):
            walks.append(Path(path).relative_to(root).as_posix())
        return original_walk(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(repository.os, "walk", record_walk)
    return root, opens, walks


@pytest.mark.parametrize("mode", ["plan", "debug", "coding"])
def test_sudo_feature_discovery_visits_only_command_approval_and_tests(checkout, mode):
    root, opens, walks = checkout
    goal = "Add sudo support to command execution."
    if mode == "coding":
        worker = ToolExecutionDispatcher.for_coding_task(repo_root=root, task_id="sudo-discovery")
        def inspect(tool, **args):
            return worker.dispatch(tool, args)
    else:
        service = PlanService(root, llm=None)
        def inspect(tool, *, reason, **args):
            if "path" in args:
                args["paths"] = [args.pop("path")]
            return service.inspect(EvidenceRequest(tool=tool, level=1, reason=reason, **args))

    def call(tool, **args):
        return inspect(tool, **({"reason": goal} if mode != "coding" else {}), **args)

    search = call("repo_search", query="run_command", glob="runtime/**/*.py")
    assert [item["file"] for item in search["matches"]] == ["runtime/command.py"]
    snippet = call("read_file", path=search["matches"][0]["file"], end_line=2)
    assert "authorize()" in snippet["content"]
    dependencies = call("repo_search", query="authorize", paths=["runtime/approval.py", "tests/test_command.py"])
    assert {item["file"] for item in dependencies["matches"]} == {
        "runtime/approval.py", "tests/test_command.py",
    }
    assert set(opens) == {"runtime/command.py", "runtime/approval.py", "tests/test_command.py"}
    assert walks == ["runtime"]


def test_localized_read_and_repeat_reuse_file_version_without_scanning(checkout):
    root, opens, walks = checkout
    worker = ToolExecutionDispatcher.for_coding_task(repo_root=root, task_id="known-path")
    args = {"path": "runtime/command.py", "start_line": 2, "end_line": 2}
    assert "authorize" in worker.dispatch("read_file", args, call_id="first")["content"]
    second = worker.dispatch("read_file", args, call_id="second")
    assert second["cache_hit"] is True
    assert opens == ["runtime/command.py"]
    (root / args["path"]).write_text("def run_command():\n    return False\n", encoding="utf-8")
    opens.clear()
    latest = worker.dispatch("read_file", args, call_id="third")
    assert "False" in latest["content"]
    assert not latest.get("cache_hit")
    assert opens == ["runtime/command.py"]
    assert not walks


def test_unscoped_content_search_requires_a_model_selection_then_can_expand(checkout):
    root, opens, walks = checkout
    worker = ToolExecutionDispatcher.for_coding_task(repo_root=root, task_id="expansion")
    rejected = worker.dispatch("repo_search", {"query": "authorize", "glob": "**/*.py"})
    assert rejected["error_code"] == "INVALID_DISCOVERY_SELECTION"
    assert not opens and not walks

    first = worker.dispatch("repo_search", {"query": "authorize", "glob": "runtime/*executor.py"})
    assert first["ok"] and not first["matches"]
    assert not opens
    second = worker.dispatch("repo_search", {"query": "authorize", "glob": "runtime/*approval.py"})
    assert {match["file"] for match in second["matches"]} == {"runtime/approval.py"}
    assert set(opens) == {"runtime/approval.py"}
    assert walks == ["runtime", "runtime"]

    audit = worker.dispatch("repo_search", {
        "query": "authorize", "repository_reason": "The user requests an audit of approval call sites across the repository",
    })
    assert {"auth/providers.py", "connectors/email.py"} <= {match["file"] for match in audit["matches"]}
    assert walks[-1] == "."
    assert ".env" not in opens


def test_batch_search_shares_traversal_and_content(checkout):
    root, opens, walks = checkout
    worker = ToolExecutionDispatcher.for_coding_task(repo_root=root, task_id="batch-search")
    result = worker.dispatch("repo_batch_search", {"patterns": [
        {"query": "run_command", "paths": ["runtime/command.py"]},
        {"query": "authorize", "paths": ["runtime/command.py"]},
    ]})
    assert result["ok"] and all(row["matches"] for row in result["results"])
    assert opens == ["runtime/command.py", "runtime/command.py"]  # Binary probe plus shared text read.
    assert not walks


def test_plain_shell_operation_does_not_take_repository_snapshots(checkout, monkeypatch):
    root, opens, walks = checkout
    worker = ToolExecutionDispatcher.for_coding_task(repo_root=root, task_id="shell-no-discovery")
    calls = []

    def observe(root, arguments, limit, *, source_candidates=False):
        calls.append((arguments["cmd"], source_candidates))
        return tool_runner_agent.FileSnapshot()

    monkeypatch.setattr(tool_runner_agent, "_snapshot", observe)
    result = worker.dispatch("run_command", {"cmd": "pwd"})
    if result.get("permission_required"):
        from mana_agent.transactional_actions.runtime import approve_action

        approval = approve_action(root, result["action_id"], approved_by="pytest-local-user")
        result = worker.dispatch("run_command", {"cmd": "pwd", "action_approval_id": approval})
    assert result["ok"], result
    assert not calls and not opens and not walks
    # Potential writes still enter the existing observation and approval path.
    command = "echo output > result.txt"
    result = worker.dispatch("run_command", {"cmd": command})
    assert calls == [(command, False), (command, False)]
    assert result.get("permission_required") or not result["ok"]
    assert not (root / "result.txt").exists()


def test_bounded_parent_cannot_delegate_repository_wide_search(checkout):
    import json

    from mana_agent.llm.tools import ToolBinding
    from mana_agent.multi_agent.runtime.tool_call_gate import ToolCallGate

    root, opens, walks = checkout
    delivered = []

    def send(request):
        delivered.append(request)
        return repository.execute_repository_discovery(root, request.tool_name, request.arguments)

    gate = ToolCallGate(
        root=root, task_id="bounded-parent",
        tools={"repo_search": ToolBinding.from_callable(lambda **kwargs: None, name="repo_search")},
        transport=send, repository_search_allowed=False,
    )
    blocked = json.loads(gate.invoke("repo_search", {
        "query": "authorize", "repository_reason": "Audit all packages",
    }, call_id="outside-parent-scope"))
    assert not blocked["ok"] and not delivered and not walks

    selected = json.loads(gate.invoke(
        "repo_search", {"query": "authorize", "paths": ["runtime/approval.py"]}, call_id="within-parent-scope",
    ))
    assert selected["ok"]
    assert len(delivered) == 1
    assert set(opens) == {"runtime/approval.py"} and not walks


def test_chat_session_does_not_load_saved_repository_analysis(checkout, monkeypatch):
    from types import SimpleNamespace

    from mana_agent.gateway import chat_gateway

    root, opens, walks = checkout
    gateway = chat_gateway.AgentChatGateway.__new__(chat_gateway.AgentChatGateway)
    gateway.root = root
    gateway._history_store = SimpleNamespace(list=lambda _: [])
    gateway._default_flow_id = None
    gateway._coding_agent = None

    def unexpected_load(_):
        pytest.fail("Opening a chat loaded the saved repository map")

    monkeypatch.setattr(chat_gateway, "load_analysis_context", unexpected_load)
    assert gateway._new_session_state("new-session")["history"] == []
    assert not opens and not walks
