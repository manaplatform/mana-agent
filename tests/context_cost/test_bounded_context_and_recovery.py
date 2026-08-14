"""Unit tests for bounded tool context, durable artifacts, API Manager, and recovery."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from mana_agent.api_manager.executor import ApiExecutionResult, ApiExecutor
from mana_agent.api_manager.models import OperationRiskLevel
from mana_agent.api_manager.request_builder import BuiltApiRequest
from mana_agent.api_manager.service import ApiManagerService
from mana_agent.context_cost.artifact_store import ContextArtifactStore
from mana_agent.context_cost.capabilities import build_core_tools
from mana_agent.context_cost.compression import (
    compress_tool_result,
    create_tool_result_envelope,
    render_envelope,
)
from mana_agent.context_cost.governor import ContextCostGovernor
from mana_agent.context_cost.models import (
    ContextManifest,
    ContextSegment,
    ToolResultEnvelope,
)
from mana_agent.gateway.followup_classifier import (
    FollowupClassification,
    FollowupClassificationOutput,
    FollowupClassifier,
    FollowupRetrievalAction,
)
from mana_agent.tools.context_retrieval import TurnRetrievalLedger, execute_memory_read


def test_tool_result_envelope_creation_and_store(tmp_path: Path) -> None:
    store = ContextArtifactStore(root=tmp_path / "artifacts")
    raw_data = [{"id": i, "name": f"user_{i}"} for i in range(100)]

    envelope = create_tool_result_envelope(
        raw_data,
        tool_name="database_query",
        tool_call_id="call_123",
        store=store,
        session_id="sess_abc",
        repository_id="repo_1",
        workspace_id="ws_1",
    )

    assert isinstance(envelope, ToolResultEnvelope)
    assert envelope.tool_name == "database_query"
    assert envelope.tool_call_id == "call_123"
    assert envelope.status == "success"
    assert envelope.artifact_ref.startswith("sha256:")
    assert envelope.content_hash
    assert envelope.original_tokens > 0
    assert envelope.projection_tokens > 0
    assert envelope.truncated is True
    assert envelope.more_available is True

    rendered = render_envelope(envelope)
    parsed = json.loads(rendered)
    assert parsed["artifact_ref"] == envelope.artifact_ref
    assert parsed["tool_name"] == "database_query"

    # Full data is retrievable from store
    retrieved = store.read(
        envelope.artifact_ref,
        session_id="sess_abc",
        repository_id="repo_1",
        workspace_id="ws_1",
    )
    assert json.loads(retrieved) == raw_data


def test_artifact_store_selectors(tmp_path: Path) -> None:
    store = ContextArtifactStore(root=tmp_path / "artifacts")

    # 1. JSON path selector
    json_data = {"meta": {"version": "1.0"}, "items": [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]}
    ref1 = store.put(json_data, session_id="s1", repository_id="r1", workspace_id="w1", content_type="json")
    val_json = store.read(ref1.artifact_id, session_id="s1", repository_id="r1", workspace_id="w1", json_path="items[1].val")
    assert val_json == "b"

    # 2. Record slice selector
    array_data = [{"idx": i} for i in range(20)]
    ref2 = store.put(array_data, session_id="s1", repository_id="r1", workspace_id="w1", content_type="json")
    val_records = store.read(ref2.artifact_id, session_id="s1", repository_id="r1", workspace_id="w1", record_start=5, record_count=3)
    assert len(val_records) == 3
    assert val_records[0]["idx"] == 5

    # 3. Markdown section selector
    doc_text = "# Introduction\nIntro text here.\n\n# Details\nDetailed content.\n\n# Summary\nEnd note."
    ref3 = store.put(doc_text, session_id="s1", repository_id="r1", workspace_id="w1", content_type="text")
    val_sec = store.read(ref3.artifact_id, session_id="s1", repository_id="r1", workspace_id="w1", section="Details")
    assert "Detailed content." in val_sec
    assert "Intro text here" not in val_sec

    # 4. Search query selector
    ref4 = store.put("line alpha\nline beta\nline gamma\nline beta two\n", session_id="s1", repository_id="r1", workspace_id="w1", content_type="text")
    val_search = store.read(ref4.artifact_id, session_id="s1", repository_id="r1", workspace_id="w1", query="beta")
    assert "line beta" in val_search
    assert "line beta two" in val_search
    assert "line alpha" not in val_search


def test_api_manager_inspect_and_import_artifact_first(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    doc_file = workspace / "openapi.json"
    doc_content = json.dumps({
        "openapi": "3.0.0",
        "info": {"title": "Sample API", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/status": {
                "get": {
                    "operationId": "getStatus",
                    "responses": {"200": {"description": "OK"}},
                }
            }
        }
    })
    doc_file.write_text(doc_content)

    manager = ApiManagerService(workspace_root=workspace)

    # 1. Inspect documentation -> produces documentation_ref and bounded preview
    inspect_result = manager.inspect_documentation(path=str(doc_file), session_id="sess_api")
    assert "documentation_ref" in inspect_result
    assert inspect_result["documentation_ref"].startswith("sha256:")
    assert "getStatus" in inspect_result["text"]

    # 2. Import using documentation_ref without re-passing text
    import_result = manager.import_documentation(
        name="sample-api",
        source_decision_id="dec_123",
        documentation_ref=inspect_result["documentation_ref"],
        session_id="sess_api",
        save=True,
    )
    assert import_result["saved"] is True
    assert import_result["operation_count"] == 1


def test_api_executor_response_artifact_and_projection(tmp_path: Path) -> None:
    executor = ApiExecutor(artifact_directory=tmp_path / "artifacts")
    request = BuiltApiRequest(
        integration_id="int_1",
        operation_id="largeOp",
        method="GET",
        url="https://api.example.com/data",
        headers={},
        timeout_seconds=30.0,
        risk_level=OperationRiskLevel.READ_ONLY,
        secret_values=(),
        session_id="sess_exec",
    )
    # Simulate large JSON response
    large_payload = json.dumps([{"id": i, "payload": "x" * 200} for i in range(100)]).encode("utf-8")
    raw_resp = SimpleNamespace(
        status=200,
        headers={"content-type": "application/json"},
        body=large_payload,
        redirects=(),
    )

    result = executor._result(request, raw_resp, latency_ms=15.0, attempts=1)
    assert isinstance(result, ApiExecutionResult)
    assert result.response_artifact_ref.startswith("sha256:")
    assert result.artifact_ref == result.response_artifact_ref
    # Bounded projection
    assert isinstance(result.json_body, list)
    assert len(result.json_body) <= 21  # 20 items + truncated note


def test_memory_read_status_codes() -> None:
    # 1. Unauthorized due to unauthenticated user
    res1 = json.loads(execute_memory_read(
        capsule_service=None,
        repository_id="repo-1",
        query="project config",
        session_id="s1",
        authenticated_user_id="",
        selected_memory_task_id="task-1",
        memory_task_candidates=({"task_id": "task-1"},),
    ))
    assert res1["status"] == "unauthorized"

    # 2. Unauthorized due to unoffered task
    res2 = json.loads(execute_memory_read(
        capsule_service=None,
        repository_id="repo-1",
        query="project config",
        session_id="s1",
        authenticated_user_id="user-1",
        selected_memory_task_id="task-unoffered",
        memory_task_candidates=({"task_id": "task-1"},),
    ))
    assert res2["status"] == "unauthorized"

    # 3. Budget exhausted
    res3 = json.loads(execute_memory_read(
        capsule_service=None,
        repository_id="repo-1",
        query="project config",
        session_id="s1",
        authenticated_user_id="user-1",
        selected_memory_task_id="task-1",
        memory_task_candidates=({"task_id": "task-1"},),
        max_tokens=0,
    ))
    assert res3["status"] == "retrieval_budget_exhausted"

    # 4. Not configured
    res4 = json.loads(execute_memory_read(
        capsule_service=None,
        repository_id="repo-1",
        query="project config",
        session_id="s1",
        authenticated_user_id="user-1",
        selected_memory_task_id="task-1",
        memory_task_candidates=({"task_id": "task-1"},),
    ))
    assert res4["status"] == "not_configured"


def test_followup_classifier_structured_query_failure() -> None:
    def failing_tool(query: str, max_turns: int = 1) -> str:
        raise ConnectionError("Network failure during conversation retrieval")

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.side_effect = [
        FollowupClassificationOutput(
            action="retrieve_context",
            category="new_task",
            related_task_id="",
            safe_to_continue=True,
            reason="Need more context",
            retrieval=FollowupRetrievalAction(query="previous test run", max_turns=1),
        ),
        FollowupClassificationOutput(
            action="classify",
            category="new_task",
            related_task_id="",
            safe_to_continue=True,
            reason="Classified as independent task despite retrieval failure",
        ),
    ]

    classifier = FollowupClassifier(llm=mock_llm)
    result = classifier.decide(
        message="Run the tests again",
        candidates=[],
        conversation_tool=failing_tool,
    )

    assert isinstance(result, FollowupClassification)
    assert result.category == "new_task"
    assert result.safe_to_continue is True


def test_context_manifest_component_breakdown(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        mana_context_governor_enabled=True,
        mana_context_governor_mode="soft",
        mana_context_tool_result_max_tokens=2000,
    )
    governor = ContextCostGovernor(
        settings=settings,
        session_id="sess_manifest",
        repository_id="repo_1",
        workspace_id="ws_1",
    )
    governor.artifacts = ContextArtifactStore(root=tmp_path / "artifacts")

    segments = [
        ContextSegment(kind="user", content="turn 1 text", source_id="turn-1", token_estimate=50),
        ContextSegment(kind="history", content="turn 0 text", source_id="turn-0", token_estimate=120),
        ContextSegment(kind="memory", content="mem text", source_id="mem-1", token_estimate=80),
        ContextSegment(kind="tool_result", content="call text", source_id="call-1", token_estimate=200),
        ContextSegment(kind="skill", content="skill text", source_id="skill-1", token_estimate=40),
    ]

    manifest = governor._persist_context_manifest("call-xyz", segments, identity={"task_id": "task-abc"})

    assert isinstance(manifest, ContextManifest)
    assert manifest.current_turn_tokens == 50
    assert manifest.conversation_tokens == 120
    assert manifest.memory_tokens == 80
    assert manifest.tool_tokens == 200
    assert manifest.skill_tokens == 40
    assert "mem-1" in manifest.memory_refs
    assert "skill-1" in manifest.skill_refs
