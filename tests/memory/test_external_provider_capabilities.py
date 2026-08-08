"""External AI memory must not disable local system-state stores.

These regression tests cover the capability boundary that previously crashed
review, plan, verification, and repository agent routes when
``MANA_MEMORY_MODE=external``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mana_agent.memory import (
    EvidenceMemory,
    MemoryCapabilities,
    MemoryConfig,
    MemoryConfigurationError,
    MemoryService,
)


def _install_fake_mem0(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    writes: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def add(self, text: str, **kwargs):
            writes.append({"text": text, **kwargs})
            return {"results": [{"id": f"mem-{len(writes)}", "memory": text}]}

        def search(self, _query: str, **kwargs):
            return {"results": []}

        def get_all(self, **_kwargs):
            return {"results": []}

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(MemoryClient=FakeClient))
    return writes


def _external_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs) -> MemoryService:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    _install_fake_mem0(monkeypatch)
    return MemoryService(
        root=tmp_path,
        config=MemoryConfig(mode="external", provider="mem0", api_key="test-key"),
        session_id="session-external-cap",
        workspace_id="workspace-1",
        repository_id="repository-1",
        **kwargs,
    )


def test_memory_capabilities_contract_defaults() -> None:
    caps = MemoryCapabilities()
    assert caps.evidence is True
    assert caps.supports("semantic_search") is False
    assert caps.as_dict()["task_state"] is True


def test_external_mode_declares_system_and_semantic_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _external_service(tmp_path, monkeypatch)
    caps = service.capabilities
    assert caps.conversation is True
    assert caps.semantic_search is True
    assert caps.evidence is True
    assert caps.checkpoints is True
    assert caps.coding_flow is True
    assert caps.task_state is True
    assert caps.multi_agent_runtime is True
    service.require_capability("evidence")
    service.require_capability("coding_flow")


def test_external_mode_keeps_run_evidence_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review / repository / verification routes use EvidenceMemory during agent turns."""
    service = _external_service(tmp_path, monkeypatch)
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('hello')\n", encoding="utf-8")

    evidence = service.evidence_memory(run_id="run-review-1")
    assert evidence.enabled()
    evidence.store(
        original_path="src/example.py",
        resolved=target,
        mode="full",
        start_line=1,
        end_line=1,
        line_count=1,
        content="print('hello')\n",
        summary="example",
    )
    hit, invalidated = evidence.lookup(
        resolved=target,
        mode="full",
        start_line=1,
        end_line=1,
    )
    assert invalidated is False
    assert hit is not None
    assert hit["cache_hit"] is True
    assert "print('hello')" in hit["content"]


def test_evidence_memory_facade_does_not_require_external_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AskAgent constructs EvidenceMemory(repo_root=..., run_id=...) directly."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MANA_MEMORY_MODE", "external")
    monkeypatch.setenv("MANA_MEMORY_PROVIDER", "mem0")
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    _install_fake_mem0(monkeypatch)

    target = tmp_path / "readme.md"
    target.write_text("# title\n", encoding="utf-8")
    # Must not raise MemoryConfigurationError about unmapped evidence.
    evidence = EvidenceMemory(repo_root=tmp_path, run_id="run-ask-agent")
    evidence.store(
        original_path="readme.md",
        resolved=target,
        mode="full",
        start_line=1,
        end_line=1,
        line_count=1,
        content="# title\n",
        summary="readme",
    )
    read_files = evidence.read_files()
    assert read_files
    assert any(path.endswith("readme.md") or path == "readme.md" for path in read_files)


def test_external_mode_keeps_coding_flow_and_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan continuation and verification need local coding-flow continuity."""
    service = _external_service(tmp_path, monkeypatch)
    flow_id = service.ensure_flow(flow_id=None, request="plan-only continuation")
    assert flow_id
    summary = service.get_flow_summary(flow_id)
    assert summary is not None
    assert "plan-only" in summary.objective
    service.checkpoint(flow_id, snapshot={"phase": "planned"})
    # Coding property remains the local system store in external mode.
    assert service.coding is not None
    assert service.coding.ensure_flow(flow_id=flow_id, request="plan-only continuation") == flow_id


def test_external_mode_duplicate_task_prevention_uses_runtime_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = _install_fake_mem0(monkeypatch)
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = MemoryService(
        root=tmp_path,
        config=MemoryConfig(mode="external", provider="mem0", api_key="test-key"),
        workspace_id="workspace-1",
        repository_id="repository-1",
        session_id="session-dup",
    )
    normalized, fingerprint = service.normalize_task(goal="review the same change twice")
    first = service.register_task(
        task_id="task-1",
        normalized_goal=normalized,
        fingerprint=fingerprint,
        assigned_agent_id="main",
        repository_ids=["repository-1"],
    )
    duplicate = service.find_duplicate_task(fingerprint)
    assert first.task_id == "task-1"
    assert duplicate is not None
    assert duplicate.task_id == "task-1"
    assert any(row.get("metadata", {}).get("mana_kind") == "task" for row in writes)


def test_external_mode_multi_agent_runtime_and_session_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _external_service(
        tmp_path,
        monkeypatch,
    )
    # Capsules are enabled by default; broad session payload is intentionally closed.
    with pytest.raises(MemoryConfigurationError, match="scoped capsules"):
        service.session_payload()
    runtime = service.multi_agent
    runtime.record_decision(
        agent_id="main",
        task_id="task-review",
        decision_type="route",
        input_summary="review-only",
        memory_used=[],
        decision="review",
        reason="validated model decision",
    )
    assert runtime.agent_decisions


def test_external_mode_does_not_write_semantic_ai_memory_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted semantic memory stays external; system stores are a separate domain."""
    writes = _install_fake_mem0(monkeypatch)
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = MemoryService(
        root=tmp_path,
        config=MemoryConfig(mode="external", provider="mem0", api_key="test-key"),
        workspace_id="workspace-1",
        repository_id="repository-1",
        session_id="session-1",
    )
    service.remember_repository_fact(f"Repository root: {tmp_path}")
    assert any(row.get("metadata", {}).get("mana_kind") == "repository_fact" for row in writes)
    # Local provider_memory.json must not be created as an AI-memory fallback.
    assert not list((tmp_path / "home").rglob("provider_memory.json"))


def test_require_capability_fails_closed_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    _install_fake_mem0(monkeypatch)
    service = MemoryService(
        root=tmp_path,
        config=MemoryConfig(mode="external", provider="mem0", api_key="test-key"),
        enable_compatibility=False,
    )
    assert service.capabilities.semantic_search is True
    assert service.capabilities.coding_flow is False
    with pytest.raises(MemoryConfigurationError, match="coding_flow"):
        service.require_capability("coding_flow")
    # Property access fails at the capability gate before the store lookup.
    with pytest.raises(MemoryConfigurationError, match="coding_flow.*unavailable|no safe backend"):
        _ = service.coding
