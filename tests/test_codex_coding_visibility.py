"""Regression suite: coding visibility boundary + bridge tool conversion.

Protocol/state based — safety must not depend on regexes for unknown junk.
Evidence: DeepSeek/NVIDIA responses_bridge write turns that emitted hundreds of
assistant.delta events with zero mutations (session_238f484f39e54225897b).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mana_agent.coding.event_visibility import (
    EventSemanticKind,
    EventVisibility,
    classify_coding_event,
    is_user_publishable,
)
from mana_agent.coding.models import AgentEvent, CodingTask, CodingTaskResult, WorkspaceContext
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.event_adapter import adapt_codex_event
from mana_agent.integrations.codex.result_parser import parse_codex_result
from mana_agent.integrations.codex.responses_bridge.models import BridgeUpstreamConfig
from mana_agent.integrations.codex.responses_bridge.request_adapter import (
    convert_responses_request_to_chat,
)
from mana_agent.integrations.codex.responses_bridge.stream_adapter import (
    ChatToResponsesStreamAdapter,
)
from mana_agent.integrations.codex.runtime_config import (
    CodexRuntimeConfigBuilder,
    _mana_model_capability_bridge,
)
from mana_agent.integrations.codex.terminal_summary import build_coding_terminal_answer
from mana_agent.integrations.codex.tool_conversion import (
    BridgeToolCompatibilityError,
    convert_responses_tools,
)


FIXTURES = Path(__file__).parent / "fixtures" / "codex"
TOOLS_CATALOG = FIXTURES / "codex_app_server_tools_catalog.json"


def _git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True)


def _workspace(tmp_path: Path) -> WorkspaceContext:
    repo = tmp_path / "repo"
    _git_repo(repo)
    return WorkspaceContext(
        repository_path=repo,
        worktree_path=repo,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )


def _task(*, write: bool = True, goal: str = "bump version to v0.1.6") -> CodingTask:
    return CodingTask(
        task_id="task-vis-1",
        goal=goal,
        requires_repository_write=write,
    )


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, *args: Any) -> None:
        if len(args) == 1 and isinstance(args[0], dict):
            et = str(args[0].get("event_type") or "")
            self.events.append((et, args[0]))
            return
        if len(args) >= 2:
            self.events.append((str(args[0]), dict(args[1]) if isinstance(args[1], dict) else {}))


class _ScriptedBackend:
    """Yield scripted AgentEvents and install a CodingTaskResult."""

    def __init__(self, script: list[dict[str, Any]], result: CodingTaskResult) -> None:
        self.script = script
        self._result = result
        self.tasks: list[CodingTask] = []
        self.closed = False

    async def stream(self, task: CodingTask, workspace: WorkspaceContext):
        self.tasks.append(task)
        for note in self.script:
            yield adapt_codex_event(
                task.task_id,
                note,
                requires_repository_write=task.requires_repository_write,
            )

    def result_for(self, task_id: str) -> CodingTaskResult:
        return self._result.model_copy(update={"task_id": task_id})

    async def close(self) -> None:
        self.closed = True


def _deltas(text: str, *, thread: str = "thread-1") -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {"method": "turn/started", "params": {"threadId": thread}},
    ]
    for ch in text:
        events.append(
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": thread,
                    "delta": ch,
                    "item": {"type": "agentMessage"},
                },
            }
        )
    events.append(
        {
            "method": "item/completed",
            "params": {
                "threadId": thread,
                "item": {"type": "agentMessage", "text": text},
            },
        }
    )
    events.append(
        {"method": "turn/completed", "params": {"threadId": thread, "turn": {"status": "completed"}}}
    )
    return events


# ---------------------------------------------------------------------------
# 1 + 11: unknown junk never reaches user sink (no production regexes for it)
# ---------------------------------------------------------------------------


def test_unknown_malformed_assistant_stream_never_reaches_user_sink(tmp_path: Path) -> None:
    junk = (
        "I'll inspect the repo first.\n"
        "<NEVER_SEEN_PROTOCOL_XYZ foo=1>\n"
        "||SYNTH_TOOL_CALL::{not-real-syntax}||\n"
        "random dsml-ish soup WITHOUT matching any production pattern: "
        "{{{gironk zzpuct}}} and <qwerty:zorch_calls>\n"
    )
    # 200+ deltas, no tools, no diff — write required.
    script = _deltas(junk * 3)
    result = parse_codex_result(
        task=_task(write=True),
        workspace=_workspace(tmp_path),
        worker_id="w",
        thread_id="thread-1",
        turn_id="turn-1",
        notifications=script,
        changed_files=[],
    )
    assert result.status == "failed"
    assert result.errors == ["mutation_required_but_no_mutation_tool_attempted"]

    sink = _RecordingSink()
    backend = _ScriptedBackend(script, result)
    shim = CodexCodingAgentShim(
        repo_root=tmp_path / "repo",
        codex_settings=CodexSettings(enabled=True),
        backend_factory=lambda: backend,
        event_sink=sink,
    )
    # Ensure workspace exists for shim path
    if not (tmp_path / "repo").exists():
        _git_repo(tmp_path / "repo")

    # Route layer needs authority — use direct _result_payload + emit path instead
    # when full routing is heavy; exercise emit via public payload builder + sink.
    payload = CodexCodingAgentShim._result_payload(
        result,
        events=[
            adapt_codex_event("t", n, requires_repository_write=True) for n in script
        ],
        workspace_path=str(tmp_path / "repo"),
        requires_repository_write=True,
    )
    answer = payload["answer"]
    assert "NEVER_SEEN_PROTOCOL" not in answer
    assert "SYNTH_TOOL_CALL" not in answer
    assert "gironk" not in answer
    assert "I'll inspect" not in answer
    assert "No mutation tool was executed" in answer

    # Visibility classification does not depend on knowing junk syntax.
    for n in script:
        if n.get("method") == "item/agentMessage/delta":
            ev = adapt_codex_event("t", n)
            assert ev.visibility == EventVisibility.INTERNAL.value
            assert ev.semantic_kind == EventSemanticKind.ASSISTANT_GENERATION.value
            assert not is_user_publishable(ev.visibility)


# ---------------------------------------------------------------------------
# 2: hundreds of assistant.delta, no tools, no diff
# ---------------------------------------------------------------------------


def test_write_turn_assistant_deltas_internal_only_concise_failure(tmp_path: Path) -> None:
    draft = "Planning chatter... " + ("x" * 50)
    # Simulate hundreds of deltas
    script: list[dict[str, Any]] = [
        {"method": "turn/started", "params": {"threadId": "t1"}},
    ]
    for i in range(300):
        script.append(
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "t1",
                    "delta": draft[i % len(draft)],
                    "item": {"type": "agentMessage"},
                },
            }
        )
    script.append(
        {
            "method": "item/completed",
            "params": {
                "item": {"type": "agentMessage", "text": draft * 5},
            },
        }
    )
    script.append({"method": "turn/completed", "params": {"turn": {"status": "completed"}}})

    events = [adapt_codex_event("task-1", n) for n in script]
    assert sum(1 for e in events if e.event_type == "assistant.delta") == 300
    assert all(
        e.visibility == EventVisibility.INTERNAL.value
        for e in events
        if e.event_type == "assistant.delta"
    )

    result = parse_codex_result(
        task=_task(write=True),
        workspace=_workspace(tmp_path),
        worker_id="w",
        thread_id="t1",
        turn_id="turn-1",
        notifications=script,
        changed_files=[],
    )
    assert result.status == "failed"
    assert result.errors == ["mutation_required_but_no_mutation_tool_attempted"]
    # Failed write: do not keep model draft as summary
    assert "Planning chatter" not in (result.summary or "")

    payload = CodexCodingAgentShim._result_payload(
        result,
        events=events,
        workspace_path="/tmp/wt",
        requires_repository_write=True,
    )
    # Internal trace retains deltas
    assert sum(1 for e in payload["trace"] if e["event_type"] == "assistant.delta") == 300
    assert payload["answer"] == (
        "Codex did not complete the requested repository mutation. "
        "No mutation tool was executed and no files were changed."
    )
    assert "Planning chatter" not in payload["answer"]
    assert payload["auto_execute_terminal_reason"] == (
        "mutation_required_but_no_mutation_tool_attempted"
    )


# ---------------------------------------------------------------------------
# 3: successful write — one terminal summary
# ---------------------------------------------------------------------------


def test_successful_write_one_terminal_summary(tmp_path: Path) -> None:
    script = [
        {"method": "turn/started", "params": {"threadId": "t1"}},
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "command": "git status --short",
                    "status": "completed",
                    "exitCode": 0,
                }
            },
        },
        {
            "method": "item/completed",
            "params": {"item": {"type": "applyPatch", "status": "completed"}},
        },
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "command": "pytest -q",
                    "status": "completed",
                    "exitCode": 0,
                }
            },
        },
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "agentMessage",
                    "text": "Bumped version and verified.",
                }
            },
        },
        {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
    ]
    result = parse_codex_result(
        task=_task(write=True),
        workspace=_workspace(tmp_path),
        worker_id="w",
        thread_id="t1",
        turn_id="turn-1",
        notifications=script,
        changed_files=["pyproject.toml"],
    )
    assert result.status == "completed"
    payload = CodexCodingAgentShim._result_payload(
        result,
        events=[adapt_codex_event("t", n) for n in script],
        workspace_path="/tmp/wt",
        requires_repository_write=True,
    )
    answer = payload["answer"]
    assert "pyproject.toml" in answer
    assert "pytest -q" in answer or "Verification" in answer
    assert payload["auto_execute_terminal_reason"] == "completed"
    # Not a multi-draft stream
    assert answer.count("I'll inspect") == 0


# ---------------------------------------------------------------------------
# 4 + 5: recovery success / double failure — no draft leakage
# ---------------------------------------------------------------------------


class _EmptyThenMutateBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.tasks: list[CodingTask] = []
        self.results: dict[str, CodingTaskResult] = {}

    async def stream(self, task: CodingTask, workspace: WorkspaceContext):
        self.calls += 1
        self.tasks.append(task)
        draft = f"ATTEMPT{self.calls} RAW DRAFT <ZZZ_UNKNOWN_TAG> should not leak"
        yield adapt_codex_event(
            task.task_id,
            {
                "method": "item/agentMessage/delta",
                "params": {"delta": draft, "item": {"type": "agentMessage"}},
            },
        )
        if self.calls == 1:
            self.results[task.task_id] = parse_codex_result(
                task=task,
                workspace=workspace,
                worker_id="w",
                thread_id="th-1",
                turn_id="tu-1",
                changed_files=[],
                notifications=[
                    {
                        "method": "item/completed",
                        "params": {"item": {"type": "agentMessage", "text": draft}},
                    },
                    {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
                ],
            )
        else:
            self.results[task.task_id] = parse_codex_result(
                task=task,
                workspace=workspace,
                worker_id="w",
                thread_id="th-2",
                turn_id="tu-2",
                changed_files=["src/version.py"],
                notifications=[
                    {
                        "method": "item/completed",
                        "params": {"item": {"type": "applyPatch", "status": "completed"}},
                    },
                    {
                        "method": "item/completed",
                        "params": {
                            "item": {
                                "type": "agentMessage",
                                "text": "Version bumped.",
                            }
                        },
                    },
                    {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
                ],
            )

    def result_for(self, task_id: str) -> CodingTaskResult:
        return self.results[task_id]

    async def close(self) -> None:
        return None


class _AlwaysEmptyBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.results: dict[str, CodingTaskResult] = {}

    async def stream(self, task: CodingTask, workspace: WorkspaceContext):
        self.calls += 1
        draft = f"FAIL DRAFT {self.calls} <abc:unknown_tool>"
        yield adapt_codex_event(
            task.task_id,
            {
                "method": "item/agentMessage/delta",
                "params": {"delta": draft, "item": {"type": "agentMessage"}},
            },
        )
        self.results[task.task_id] = parse_codex_result(
            task=task,
            workspace=workspace,
            worker_id="w",
            thread_id=f"th-{self.calls}",
            turn_id=f"tu-{self.calls}",
            changed_files=[],
            notifications=[
                {
                    "method": "item/completed",
                    "params": {"item": {"type": "agentMessage", "text": draft}},
                },
                {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
            ],
        )

    def result_for(self, task_id: str) -> CodingTaskResult:
        return self.results[task_id]

    async def close(self) -> None:
        return None


def test_recovery_success_no_draft_leak(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    sink = _RecordingSink()
    backend = _EmptyThenMutateBackend()
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=CodexSettings(enabled=True),
        backend_factory=lambda: backend,
        event_sink=sink,
    )
    result = shim.generate_auto_execute("fix bug", auto_chat_mode="edit")
    assert backend.calls == 2
    assert result["status"] == "completed"
    assert result["changed_files"] == ["src/version.py"]
    assert result.get("mutation_recovery") is True
    assert "RAW DRAFT" not in result["answer"]
    assert "ZZZ_UNKNOWN" not in result["answer"]
    assert "src/version.py" in result["answer"]
    # No assistant.delta published to user sink
    assert not any(et == "assistant.delta" for et, _ in sink.events)


def test_recovery_fails_twice_one_concise_failure(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    sink = _RecordingSink()
    backend = _AlwaysEmptyBackend()
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=CodexSettings(enabled=True),
        backend_factory=lambda: backend,
        event_sink=sink,
    )
    result = shim.generate_auto_execute("fix bug", auto_chat_mode="edit")
    assert backend.calls == 2
    assert result["status"] == "failed"
    assert "FAIL DRAFT" not in result["answer"]
    assert "unknown_tool" not in result["answer"]
    assert "No mutation tool was executed" in result["answer"]
    assert not any(et == "assistant.delta" for et, _ in sink.events)


# ---------------------------------------------------------------------------
# 6: plan / read-only
# ---------------------------------------------------------------------------


def test_plan_mode_terminal_plan_returned(tmp_path: Path) -> None:
    plan = "1. Inspect auth module\n2. Introduce token refresh\n3. Add tests"
    script = [
        {"method": "turn/started", "params": {"threadId": "t1"}},
        {
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "text": plan}},
        },
        {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
    ]
    result = parse_codex_result(
        task=_task(write=False, goal="plan auth refactor"),
        workspace=_workspace(tmp_path),
        worker_id="w",
        thread_id="t1",
        turn_id="turn-1",
        notifications=script,
        changed_files=[],
    )
    assert result.status == "completed"
    assert result.errors == []
    payload = CodexCodingAgentShim._result_payload(
        result,
        events=[adapt_codex_event("t", n, requires_repository_write=False) for n in script],
        workspace_path="",
        requires_repository_write=False,
    )
    assert payload["auto_execute_terminal_reason"] == "completed"
    assert "token refresh" in payload["answer"]


# ---------------------------------------------------------------------------
# 7 + 8: tool conversion catalog + unsupported shapes
# ---------------------------------------------------------------------------


def test_codex_tool_catalog_survives_responses_to_chat_conversion() -> None:
    catalog = json.loads(TOOLS_CATALOG.read_text(encoding="utf-8"))
    tools = catalog["tools"]
    assert tools, "fixture must include tools"
    report = convert_responses_tools(
        tools,
        provider="nvidia",
        model="deepseek-ai/deepseek-v4-flash-0731",
        fail_on_unsupported=True,
    )
    assert report.original_count == len(tools)
    assert report.converted_count == len(tools)
    assert not report.unsupported
    names = {t["function"]["name"] for t in report.converted_tools}
    assert "shell" in names
    assert "apply_patch" in names
    assert "local_shell" in names
    assert "web_search" in names
    # Freeform custom apply_patch must be marked for custom_tool_call round-trip.
    assert report.tool_origins.get("apply_patch") == "custom"

    chat = convert_responses_request_to_chat(
        {
            "model": "deepseek-ai/deepseek-v4-flash-0731",
            "input": "edit files",
            "tools": tools,
            "stream": False,
        },
        upstream=BridgeUpstreamConfig(
            provider="nvidia",
            display_name="NVIDIA",
            api_key="nvapi-test",
            base_url="https://integrate.api.nvidia.com/v1",
            model="deepseek-ai/deepseek-v4-flash-0731",
        ),
    )
    assert len(chat["tools"]) == len(tools)
    assert all(t["type"] == "function" and "function" in t for t in chat["tools"])
    # Local-only origins metadata for stream adapters; never an upstream secret.
    meta = chat.get("_mana_bridge") or {}
    assert meta.get("tool_origins", {}).get("apply_patch") == "custom"


def test_unsupported_tool_shape_fails_explicitly() -> None:
    tools = [
        {"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {}}},
        {"type": "file_search", "vector_store_ids": ["vs_1"]},
        {"type": "computer_use_preview"},
    ]
    with pytest.raises(BridgeToolCompatibilityError) as exc_info:
        convert_responses_tools(
            tools,
            provider="nvidia",
            model="deepseek-ai/deepseek-v4-flash-0731",
            fail_on_unsupported=True,
        )
    diag = exc_info.value.diagnostics()
    assert diag["original_tool_count"] == 3
    assert diag["converted_tool_count"] == 1
    assert diag["unsupported_tool_count"] == 2
    assert diag["provider"] == "nvidia"
    # Error names the unsupported types so operators can act.
    assert "file_search" in str(exc_info.value)
    assert "api_key" not in json.dumps(diag)

    with pytest.raises(BridgeToolCompatibilityError):
        convert_responses_request_to_chat(
            {
                "model": "deepseek-ai/deepseek-v4-flash-0731",
                "input": "hi",
                "tools": tools,
            },
            upstream=BridgeUpstreamConfig(
                provider="nvidia",
                display_name="NVIDIA",
                api_key="nvapi-secret",
                base_url="https://integrate.api.nvidia.com/v1",
                model="deepseek-ai/deepseek-v4-flash-0731",
            ),
        )


def test_host_tools_ten_of_ten_match_production_failure_shape() -> None:
    """Regression for original=10 converted=8 unsupported=2 (freeform + host tools)."""
    tools = [
        {"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "list_dir", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "grep_files", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "read_file", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "update_plan", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "view_image", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "exec_command", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "write_stdin", "parameters": {"type": "object", "properties": {}}},
        # The two that previously failed conversion under fallback metadata:
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch",
            "format": {"type": "text"},
        },
        {"type": "web_search"},
    ]
    report = convert_responses_tools(
        tools,
        provider="nvidia",
        model="deepseek-ai/deepseek-v4-flash-0731",
        fail_on_unsupported=True,
    )
    assert report.original_count == 10
    assert report.converted_count == 10
    assert not report.unsupported
    assert report.tool_origins["apply_patch"] == "custom"
    assert report.tool_origins["web_search"].startswith("web_search")


def test_freeform_apply_patch_streams_as_custom_tool_call() -> None:
    adapter = ChatToResponsesStreamAdapter(
        model="deepseek-ai/deepseek-v4-flash-0731",
        tool_origins={"apply_patch": "custom"},
    )
    adapter.open_events()
    adapter.ingest_chat_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_patch",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": '{"input":"*** Begin Patch\\n*** End Patch"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    close = "".join(adapter.close_events())
    assert "custom_tool_call" in close
    assert "apply_patch" in close
    assert "*** Begin Patch" in close


# ---------------------------------------------------------------------------
# 9: fragmented tool-call arguments preserved
# ---------------------------------------------------------------------------


def test_streaming_tool_call_fragments_preserved() -> None:
    adapter = ChatToResponsesStreamAdapter(model="deepseek-ai/deepseek-v4-flash-0731")
    adapter.open_events()
    # Fragmented function arguments across chunks
    parts = ['{"com', 'mand":', '"ls', ' -la"}']
    for index, part in enumerate(parts):
        adapter.ingest_chat_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1" if index == 0 else None,
                                    "function": {
                                        "name": "shell" if index == 0 else None,
                                        "arguments": part,
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )
    close = adapter.close_events()
    joined = "".join(close)
    assert '"arguments": "{\\"command\\":\\"ls -la\\"}"' in joined or (
        adapter.tool_calls[0].arguments == '{"command":"ls -la"}'
    )
    assert adapter.tool_calls[0].name == "shell"


# ---------------------------------------------------------------------------
# 10: provider reasoning separate from assistant content
# ---------------------------------------------------------------------------


def test_provider_reasoning_kept_separate_from_assistant() -> None:
    adapter = ChatToResponsesStreamAdapter(model="deepseek-ai/deepseek-v4-flash-0731")
    adapter.open_events()
    adapter.ingest_chat_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "thinking step",
                        "content": "visible",
                    }
                }
            ]
        }
    )
    assert adapter.reasoning_parts == ["thinking step"]
    assert adapter.text_parts == ["visible"]
    events = adapter.ingest_chat_chunk(
        {"choices": [{"delta": {"reasoning_content": " more"}}]}
    )
    assert any("reasoning_summary_text.delta" in e for e in events)
    assert not any("output_text.delta" in e and "more" in e for e in events if "reasoning" in e)

    # Event adapter: reasoning is internal
    kind, vis = classify_coding_event("reasoning.update")
    assert kind is EventSemanticKind.REASONING
    assert vis is EventVisibility.INTERNAL


# ---------------------------------------------------------------------------
# 12: model metadata / capability bridge is explicit
# ---------------------------------------------------------------------------


def test_model_metadata_capability_bridge_explicit() -> None:
    window, compact, supports = _mana_model_capability_bridge(
        provider="nvidia",
        model="deepseek-ai/deepseek-v4-flash-0731",
    )
    assert window == 1_000_000
    assert compact is not None and compact < window
    assert supports is True

    # Unknown model: do not silently claim tool support
    window2, compact2, supports2 = _mana_model_capability_bridge(
        provider="nvidia",
        model="nvidia/totally-unknown-model-xyz-999",
    )
    # May have no maintained limits
    assert supports2 is not True or supports2 is False or supports2 is None

    # Runtime config TOML includes context window when known
    settings = CodexSettings(
        enabled=True,
        provider="openai",
        provider_display_name="OpenAI",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
        supports_responses_api=True,
        codex_transport=__import__(
            "mana_agent.config.provider_registry", fromlist=["CodexTransport"]
        ).CodexTransport.DIRECT_RESPONSES,
    )
    cfg = CodexRuntimeConfigBuilder.build(settings, sandbox_mode="workspace-write")
    toml = cfg.to_toml()
    if cfg.model_context_window:
        assert "model_context_window" in toml


def test_emit_event_suppresses_assistant_delta_from_user_sink(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    sink = _RecordingSink()
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=CodexSettings(enabled=True),
        event_sink=sink,
    )
    delta = adapt_codex_event(
        "task-1",
        {
            "method": "item/agentMessage/delta",
            "params": {
                "delta": "I will start by checking...",
                "item": {"type": "agentMessage"},
            },
        },
    )
    shim._emit_event(delta, requires_repository_write=True)
    assert sink.events == []

    progress = adapt_codex_event(
        "task-1",
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "command": "git status",
                    "status": "completed",
                }
            },
        },
    )
    shim._emit_event(progress, requires_repository_write=True)
    assert any(et.startswith("command.") for et, _ in sink.events)
    # No raw draft text
    for _, payload in sink.events:
        assert "I will start" not in json.dumps(payload)


def test_build_coding_terminal_answer_mutation_messages() -> None:
    result = CodingTaskResult(
        task_id="t",
        worker_id="w",
        backend="codex",
        status="failed",
        summary="huge draft that must not appear",
        errors=["mutation_required_but_no_mutation_tool_attempted"],
    )
    answer = build_coding_terminal_answer(result, requires_repository_write=True)
    assert "huge draft" not in answer
    assert "No mutation tool was executed" in answer
