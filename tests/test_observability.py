from datetime import datetime, timedelta, timezone

from mana_agent.cli.events import make_event
from mana_agent.observability import ObservabilityConfig, ObservabilityStore
from mana_agent.observability.service import redact_summary


def _event(session_id: str, *, title: str = "repository search", status: str = "success", duration_ms: int = 2500):
    event = make_event("tool.finished", title=title, status=status, session_id=session_id, metadata={"args": {"api_key": "secret", "path": "src"}})
    event.duration_ms = duration_ms
    event.ended_at = event.started_at
    return event


def test_store_links_spans_redacts_and_reports_bottleneck(tmp_path):
    store = ObservabilityStore(tmp_path)
    for index in range(3):
        event = _event("trace-1", status="failed" if index == 0 else "success")
        event.event_id = f"span-{index}"
        event.token_usage = type("Usage", (), {"as_dict": lambda self: {"total_tokens": 12000, "estimated": False}})()
        store.record_event(event)

    spans = store.spans(trace_id="trace-1")
    assert len(spans) == 3
    assert "[REDACTED]" in spans[0]["input_summary"]
    overview = store.overview()
    assert overview["total_tokens"] == 36000
    assert overview["bottlenecks"]
    assert "cost" not in overview


def test_retention_removes_completed_old_spans(tmp_path):
    store = ObservabilityStore(tmp_path, ObservabilityConfig(retention_days=1))
    event = _event("old")
    event.event_id = "old-span"
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    event.started_at = old
    event.ended_at = old
    store.record_event(event)
    assert store.spans(trace_id="old") == []


def test_redaction_handles_values_and_secret_keys():
    text = redact_summary({"authorization": "Bearer secret-value", "note": "sk-abcdefghijklmnop"})
    assert "secret-value" not in text
    assert "abcdefghijklmnop" not in text
    assert text.count("[REDACTED]") == 2


def test_otlp_failure_is_recorded_locally(tmp_path):
    config = ObservabilityConfig(otlp_endpoint="http://127.0.0.1:1")
    store = ObservabilityStore(tmp_path, config)
    store.record_event(_event("trace-otlp"))
    assert store.spans(trace_id="trace-otlp")
    assert store.health()["otlp"]["status"] == "failed"
