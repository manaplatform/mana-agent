"""Public activity contracts: privacy, correlation, ordering, and bounded projection.

Run explicitly; repository policy leaves verification execution to the user.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from mana_agent.chat.activity import (
    ActivityMetadata, ProviderSummaryAssembler, bound_operations, delivery_key,
    is_mutation, operation_key, project_activity, visible_activity,
)
from mana_agent.chat.events import ActivityEvent, CodingActivityEvent, ToolCallEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.chat.execution_trace import ExecutionTrace
from mana_agent.services.execution_event_hub import ExecutionEventHub


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setattr("mana_agent.services.execution_event_hub.conversation_events_path",
                        lambda repo, conversation: tmp_path / repo / conversation / "events.jsonl")
    instance = ExecutionEventHub()
    yield instance
    instance.close()


def event(event_type="tool.started", **changes):
    return {"event_id": "event-a", "type": event_type, "status": "running",
            "conversation_id": "session_activity", "execution_id": "turn_activity",
            "repository_id": "repo_activity", "task_id": "task-a", "agent_id": "agent-a",
            "attempt_id": "attempt-a", "attempt_generation": 1, "operation_id": "operation-a", **changes}


@pytest.mark.parametrize("kind", ["agent.routing", "agent.planning", "loop.iteration.started",
    "task.updated", "agent.handoff", "verification.completed", "steering.applied", "recovery.started"])
def test_general_activity_preserves_identity_and_metadata(kind):
    projected = project_activity(event(kind, metadata={"parent_agent_id": "main", "phase": "plan"}))
    assert projected is not None
    assert projected["type"] == projected["event_type"] == kind
    assert projected["attempt_id"] == "attempt-a"
    assert projected["agent_id"] == "agent-a"
    assert projected["parent_agent_id"] == "main"
    assert projected["metadata"]["phase"] == "plan"


@pytest.mark.parametrize("kind", ["reasoning.update", "reasoning.thought", "assistant.delta", "subagent.delta"])
def test_private_content_never_enters_chat_history(kind):
    history = ChatHistory()
    history.add(ActivityEvent(activity=event(kind, summary="private material"), turn_id="turn_activity"))
    assert history.get_events() == []
    assert CodingActivityEvent is ActivityEvent


def test_semantic_privacy_cannot_be_overridden_by_tool_label():
    assert project_activity(event("tool.call.completed", semantic_kind="reasoning", summary="private")) is None


def test_native_summary_assembly_redacts_split_credentials():
    assembler = ProviderSummaryAssembler(provider="openai", transport="direct_responses", supported=True)
    assembler.append("item-a", 0, "Checking compatibility. access_token=first")
    assembler.append("item-a", 0, "second")
    result = assembler.finish("item-a", 0)
    assert result is not None
    projected = project_activity(result)
    assert projected is not None
    assert "firstsecond" not in json.dumps(projected)
    assert "first" not in json.dumps(projected)
    assert projected["summary_source"] == "provider"
    assert projected["evidence_status"] == "unverified"
    assert assembler.finish("item-a", 0) is None


@pytest.mark.parametrize("provider,transport,supported", [
    ("openai", "responses_bridge", True), ("nvidia", "responses_bridge", True),
    ("unknown", "direct_responses", True), ("openai", "direct_responses", False),
])
def test_bridge_and_unknown_capabilities_remain_private(provider, transport, supported):
    assembler = ProviderSummaryAssembler(provider=provider, transport=transport, supported=supported)
    assembler.append("item", 0, "private raw reasoning")
    assert assembler.finish("item", 0) is None


def test_projection_omits_raw_arguments_prompts_and_protected_text():
    source = event(summary="Reading confidential source. Bearer abc.def", metadata={
        "args": {"password": "password-value"}, "system_prompt": "secret instructions",
        "lease_token": "worker-secret", "phase": "exploration", "command": "inspect",
    })
    projected = project_activity(source, protected=("confidential source",))
    encoded = json.dumps(projected)
    for secret in ("abc.def", "password-value", "secret instructions", "worker-secret", "confidential source"):
        assert secret not in encoded
    assert projected["metadata"]["phase"] == "exploration"
    assert source["metadata"]["args"]["password"] == "password-value"


def test_invalid_identity_is_not_published():
    assert project_activity(event(attempt_generation=-1)) is None
    assert ActivityMetadata().evidence_status == "unverified"


def test_mutation_requires_success_and_act_mode():
    assert not is_mutation(event("file.read", path="source.py", status="success"))
    assert not is_mutation(event("file.changed", path="source.py", status="failed"))
    assert not is_mutation(event("file.changed", path="source.py", status="success", execution_mode="plan"))
    assert is_mutation(event("file.changed", path="source.py", status="success"))


def test_deliveries_and_operations_have_separate_identities():
    started = event()
    completed = event("tool.finished", status="success")
    assert operation_key(started) == operation_key(completed)
    assert delivery_key(started) != delivery_key(completed)
    assert delivery_key(started) == delivery_key({**started, "sequence": 100, "timestamp": "later"})


def test_exact_duplicates_and_lifecycle_revisions(hub):
    received = []
    hub.subscribe("session_activity", received.append)
    first = hub.publish(event())
    duplicate = hub.publish(event())
    finished = hub.publish(event("tool.finished", status="success"))
    assert duplicate["sequence"] == first["sequence"]
    assert [row["sequence"] for row in received] == [1, 2]
    assert finished["sequence"] == 2


def test_identical_fragments_with_distinct_producer_sequences_are_retained(hub):
    for sequence in (1, 2):
        hub.publish(event("command.output", output_preview="same text", producer_sequence=sequence))
    assert len(hub.history(conversation_id="session_activity")) == 2


def test_concurrent_publishers_deliver_in_sequence(hub):
    received = []
    hub.subscribe("session_activity", received.append)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: hub.publish(event(event_id=f"event-{index}", operation_id=f"op-{index}")), range(80)))
    assert [row["sequence"] for row in received] == list(range(1, 81))


def test_reentrant_subscriber_cannot_overtake_current_delivery(hub):
    received = []
    def publish_another(row):
        if row["event_id"] == "first":
            hub.publish(event(event_id="second"))
    hub.subscribe("session_activity", publish_another)
    hub.subscribe("session_activity", received.append)
    hub.publish(event(event_id="first"))
    assert [row["event_id"] for row in received] == ["first", "second"]


def test_replay_pages_start_after_cursor_instead_of_skipping_to_tail(hub):
    for index in range(15):
        hub.publish(event(event_id=f"event-{index}"))
    page = hub.history(conversation_id="session_activity", repository_id="repo_activity", after_sequence=2, limit=4)
    assert [row["sequence"] for row in page] == [3, 4, 5, 6]
    earlier = hub.history(conversation_id="session_activity", repository_id="repo_activity", before_sequence=8, limit=3)
    assert [row["sequence"] for row in earlier] == [5, 6, 7]


def test_parallel_agent_steps_require_their_own_completion():
    trace = ExecutionTrace("turn_activity")
    first = trace.apply_event(event())
    trace.apply_event(event("model.started", event_id="other", operation_id="other", agent_id="agent-b"))
    assert first.status == "running"
    assert len(trace.active_steps) == 2
    trace.apply_event(event("tool.finished", status="success"))
    assert len(trace.active_steps) == 1
    assert trace.active_steps[0].metadata["agent_id"] == "agent-b"
    assert not trace.is_completed


def test_history_is_bounded_and_does_not_mutate_invocation():
    history = ChatHistory()
    call = ToolCallEvent(tool_name="shell", args={"password": "private"})
    history.add(call)
    assert call.args == {"password": "private"}
    assert history.get_events()[0].args == {}
    for index in range(500):
        history.add(ActivityEvent(activity=event("tool.finished", status="success", event_id=f"e-{index}",
            operation_id=f"o-{index}"), turn_id="turn_activity"))
    assert sum(isinstance(row, ActivityEvent) for row in history.get_events()) == 200


def test_bounding_preserves_active_work_and_visibility_does_not_change_events():
    active = event()
    completed = [event("tool.finished", status="success", operation_id=str(i)) for i in range(300)]
    assert len(bound_operations([active, *completed])) == 201
    assert active in bound_operations([active, *completed])
    assert visible_activity(event("command.output"), "verbose")
    assert not visible_activity(event("command.output"), "normal")
    assert visible_activity(event("action.approval.required"), "minimal")


def test_shared_parallel_fixture_preserves_owners_and_mutation_evidence():
    from pathlib import Path
    rows = json.loads((Path(__file__).parent / "fixtures/activity/parallel.json").read_text())
    trace = ExecutionTrace("turn_fixture")
    for row in rows:
        projected = project_activity(row)
        if projected is not None:
            trace.apply_event(projected)
    assert trace.changed_files == ["src/example.py"]
    assert {row["agent_id"] for row in trace.coding_events if row.get("agent_id")} == {"CodingAgent"}
    assert any(row.get("agent_id") == "ResearchAgent" for row in trace.main_events)
    assert trace.is_completed


def test_legacy_journal_replay_redacts_at_delivery_boundary(hub, tmp_path):
    path = tmp_path / "repo_activity/session_activity/events.jsonl"
    path.parent.mkdir(parents=True)
    rows = [event("tool.started", sequence=1, metadata={"args": {"password": "old-secret"}}),
            event("reasoning.thought", sequence=2, summary="private legacy reasoning")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    public = hub.history(conversation_id="session_activity", repository_id="repo_activity")
    assert "old-secret" not in json.dumps(public)
    assert "private legacy reasoning" not in json.dumps(public)
    assert public[-1]["sequence"] == 2
    assert public[-1]["visibility"] == "internal"


def test_internal_visibility_cannot_be_promoted_by_a_public_event_name(hub):
    hidden = hub.publish(event("tool.finished", visibility="internal", summary="private continuation"))
    assert hidden["visibility"] == "internal"
    assert hub.history(conversation_id="session_activity") == []


def test_untrusted_legacy_summary_is_omitted_but_retains_replay_cursor(hub, tmp_path):
    path = tmp_path / "repo_activity/session_activity/events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(event("agent.summary", sequence=1, summary="private bridge text",
        metadata={"summary_source": "provider", "summary_trusted": True,
                  "summary_complete": True, "provider": "openai", "transport": "responses_bridge"})) + "\n")
    replay = hub.history(conversation_id="session_activity", repository_id="repo_activity")
    assert replay[0]["sequence"] == 1
    assert replay[0]["visibility"] == "internal"
    assert "private bridge text" not in json.dumps(replay)


def test_completed_partial_record_is_recovered_before_sequence_allocation(hub, tmp_path):
    path = tmp_path / "repo_activity/session_activity/events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(event(sequence=1)))  # interrupted before newline
    next_event = hub.publish(event(event_id="next"))
    assert next_event["sequence"] == 2
    assert [row["sequence"] for row in hub.history(conversation_id="session_activity")] == [1, 2]


def test_persistence_failure_reports_degradation_without_creating_cursor_holes(hub, monkeypatch):
    delivered = []
    hub.subscribe("session_activity", delivered.append)
    append = hub._append_durable
    def fail(*args):
        raise OSError("fixture disk failure")
    monkeypatch.setattr(hub, "_append_durable", fail)
    ephemeral = hub.publish(event(event_id="ephemeral"))
    assert ephemeral["sequence"] == 0
    assert ephemeral["metadata"]["replayable"] is False
    assert any(row.get("type") == "activity.degraded" for row in delivered)
    monkeypatch.setattr(hub, "_append_durable", append)
    durable = hub.publish(event(event_id="durable"))
    assert durable["sequence"] == 1


def test_callbacks_do_not_hold_sequence_or_journal_locks(hub):
    import threading
    finished = threading.Event()
    blocked = []
    def callback(row):
        if row["event_id"] != "first":
            return
        worker = threading.Thread(target=lambda: (hub.publish(event(event_id="second")), finished.set()))
        worker.start()
        blocked.append(not finished.wait(2))
        worker.join(timeout=2)
    hub.subscribe("session_activity", callback)
    hub.publish(event(event_id="first"))
    assert finished.is_set()
    assert blocked == [False]


def test_summary_sections_flush_all_indices_and_require_native_endpoint():
    from mana_agent.chat.activity import native_summary_endpoint
    assembler = ProviderSummaryAssembler(provider="openai", transport="direct_responses", supported=True)
    assembler.append("item", 0, "First public section")
    assembler.append("item", 1, "Second public section")
    assert [row["metadata"]["summary_index"] for row in assembler.finish_item("item")] == [0, 1]
    assert native_summary_endpoint("https://api.openai.com/v1")
    assert not native_summary_endpoint("http://127.0.0.1:9000/v1")
    assert not native_summary_endpoint("https://api.openai.com.evil.example/v1")


def test_history_and_trace_reject_late_running_revisions():
    history = ChatHistory()
    history.add(ActivityEvent(activity=event("tool.finished", status="success", sequence=2), turn_id="turn_activity"))
    history.add(ActivityEvent(activity=event("tool.started", status="running", sequence=1), turn_id="turn_activity"))
    assert history.get_events()[0].activity["status"] == "success"
    trace = ExecutionTrace("turn_activity")
    trace.apply_event(event("execution.failed", status="failed", operation_id="root", error="failed"))
    trace.apply_event(event("turn.finished", status="success", operation_id="root"))
    assert trace.is_failed and not trace.is_completed


def test_terminal_escapes_and_arbitrary_tool_text_are_removed():
    projected = project_activity(event(summary="\x1b[31mRead source\x1b[0m\x00", metadata={
        "text": "arbitrary tool body", "instruction": "untrusted prompt", "content": "private source text"}))
    encoded = json.dumps(projected)
    assert "\\u001b" not in encoded and "\\u0000" not in encoded
    assert "arbitrary tool body" not in encoded
    assert "untrusted prompt" not in encoded
    assert "private source text" not in encoded


def test_buffered_subscriber_recovers_every_record_after_overflow(hub):
    import threading
    first = threading.Event()
    release = threading.Event()
    complete = threading.Event()
    received = []
    def slow(row):
        if row.get("sequence") == 1:
            first.set()
            release.wait(timeout=5)
        received.append(row["sequence"])
        if row["sequence"] == 1050:
            complete.set()
    unsubscribe = hub.subscribe("session_activity", slow, repository_id="repo_activity", buffered=True)
    try:
        hub.publish(event(event_id="first"))
        assert first.wait(2)
        for index in range(2, 1051):
            hub.publish(event(event_id=f"record-{index}"))
        release.set()
        assert complete.wait(5)
        assert received == list(range(1, 1051))
    finally:
        release.set()
        unsubscribe()


def test_oversized_summary_section_is_discarded_instead_of_truncated():
    assembler = ProviderSummaryAssembler(provider="openai", transport="direct_responses", supported=True)
    assembler.append("item", 0, "x" * 7999)
    assembler.append("item", 0, "protected-token-fragment")
    assert assembler.finish("item", 0) is None


def test_protected_secret_is_redacted_before_summary_preview_truncation():
    secret = "opaque-private-value-without-a-credential-prefix"
    assembler = ProviderSummaryAssembler(provider="openai", transport="direct_responses", supported=True)
    assembler.append("item", 0, "x" * 225 + secret)
    completed = assembler.finish("item", 0)
    public = project_activity(completed, protected=(secret,))
    assert secret[:12] not in json.dumps(public)
    assert "REDACTED" in json.dumps(public)


def test_final_response_content_is_not_limited_to_activity_preview_size():
    content = "Public response. " * 500
    public = project_activity(event("turn.finished", status="success", metadata={"content": content}))
    assert public["metadata"]["content"] == content


def test_recorded_root_cancellation_does_not_invent_a_tool_result():
    trace = ExecutionTrace("turn_activity")
    operation = trace.apply_event(event("tool.started", tool_name="read_file"))
    trace.apply_event(event("task.cancelled", status="cancelled", task_id="task-a", root_task_id="task-a",
        operation_id="lifecycle", metadata={"execution_supervisor": True}))
    assert trace.is_cancelled
    assert not trace.is_completed
    assert operation.status == "unverified"


def test_ordered_dispatch_backlog_is_bounded_and_recovered(hub):
    import threading
    finished = threading.Event()
    received = []
    def consume(row):
        sequence = int(row.get("sequence") or 0)
        if sequence:
            received.append(sequence)
        if sequence == 1100:
            finished.set()
    unsubscribe = hub.subscribe("session_activity", consume, repository_id="repo_activity", buffered=True)
    try:
        # Hold the reducer lock to model a busy dispatch thread while the same
        # process records a burst of authoritative lifecycle observations.
        with hub._lock:
            for index in range(1100):
                hub.publish(event(event_id=f"burst-{index}"), authoritative=True)
            assert len(hub._pending_delivery) <= 1024
        assert finished.wait(5)
        assert received == list(range(1, 1101))
    finally:
        unsubscribe()


def test_authoritative_observations_still_apply_scoped_protected_text(hub):
    from mana_agent.services.activity_context import ActivityContext, activity_scope
    with activity_scope(ActivityContext(conversation_id="session_activity", execution_id="turn_activity",
        repository_id="repo_activity", protected_text=("opaque-private-credential",))):
        public = hub.publish(event("task.updated", summary="Reported opaque-private-credential"), authoritative=True)
    assert "opaque-private-credential" not in json.dumps(public)


def test_provider_semantics_cannot_be_smuggled_under_a_tool_name(hub):
    private = hub.publish(event("tool.finished", semantic_kind="provider", summary="raw provider body"))
    assert private["visibility"] == "internal"
    assert hub.history(conversation_id="session_activity") == []
