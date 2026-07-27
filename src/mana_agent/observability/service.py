"""SQLite-backed tracing, metrics, and optional OTLP export.

The store is deliberately independent of dashboard dependencies so every CLI
runtime can emit the same trace data.  It is an observational side channel:
storage or export failures are recorded as health state and never alter a
model-driven execution decision.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from mana_agent.workspaces.paths import repository_dir, repository_id_for_path

_SECRET_KEY = re.compile(r"(?:api[_-]?key|token|password|secret|authorization|cookie|credential)", re.I)
_SECRET_VALUE = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+\S+|ghp_[A-Za-z0-9]{12,})", re.I)
_MAX_SUMMARY = 2_000


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    retention_days: int = 30
    max_storage_mb: int = 500
    otlp_endpoint: str = ""
    otlp_headers: dict[str, str] | None = None

    @classmethod
    def from_environment(cls) -> "ObservabilityConfig":
        def positive(name: str, default: int) -> int:
            try:
                return max(1, int(os.getenv(name, default)))
            except ValueError:
                return default

        raw_headers = os.getenv("MANA_OBSERVABILITY_OTLP_HEADERS", "{}")
        try:
            headers = json.loads(raw_headers)
        except json.JSONDecodeError:
            headers = {}
        return cls(
            retention_days=positive("MANA_OBSERVABILITY_RETENTION_DAYS", 30),
            max_storage_mb=positive("MANA_OBSERVABILITY_MAX_STORAGE_MB", 500),
            otlp_endpoint=os.getenv("MANA_OBSERVABILITY_OTLP_ENDPOINT", "").strip(),
            otlp_headers={str(k): str(v) for k, v in headers.items()} if isinstance(headers, dict) else {},
        )


def redact_summary(value: Any, *, limit: int = _MAX_SUMMARY) -> str:
    """Return a bounded structured summary without carrying known secrets."""
    def clean(item: Any, key: str = "") -> Any:
        if _SECRET_KEY.search(key):
            return "[REDACTED]"
        if isinstance(item, dict):
            return {str(k): clean(v, str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item[:50]]
        text = str(item)
        return _SECRET_VALUE.sub("[REDACTED]", text)

    try:
        rendered = json.dumps(clean(value), ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        rendered = _SECRET_VALUE.sub("[REDACTED]", str(value))
    return rendered[:limit] + ("…" if len(rendered) > limit else "")


def _redacted_json(value: Any, *, limit: int = _MAX_SUMMARY) -> str:
    """Serialize redacted structured data without truncating invalid JSON."""
    rendered = redact_summary(value, limit=limit)
    if len(rendered) <= limit:
        return rendered
    return json.dumps({"truncated": True, "summary": rendered}, ensure_ascii=False)


def _decode_json_object(value: str | None) -> dict[str, Any]:
    """Read stored JSON defensively so corrupt historical telemetry is queryable."""
    try:
        decoded = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _epoch_ns(value: str | None) -> int:
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1_000_000_000)
    except (TypeError, ValueError):
        return time.time_ns()


class ObservabilityStore:
    """Canonical per-repository trace store and dashboard query surface."""

    def __init__(self, root: Path | str, config: ObservabilityConfig | None = None) -> None:
        self.root = Path(root).resolve()
        self.config = config or ObservabilityConfig.from_environment()
        # Scope telemetry per repository so multi-repo sessions and tests do not
        # share or pollute a single global SQLite database.
        self.directory = repository_dir(repository_id_for_path(self.root)) / "observability"
        self.path = self.directory / "telemetry.sqlite"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, parent_span_id TEXT,
                    session_id TEXT, turn_id TEXT, task_id TEXT, agent_id TEXT, subagent_id TEXT,
                    step_id TEXT, kind TEXT NOT NULL, event_type TEXT NOT NULL, title TEXT,
                    status TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT,
                    duration_ms REAL, queue_wait_ms REAL NOT NULL DEFAULT 0,
                    input_summary TEXT, output_summary TEXT, error_summary TEXT,
                    token_usage_json TEXT NOT NULL DEFAULT '{}', attributes_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS spans_trace_idx ON spans(trace_id, started_at);
                CREATE INDEX IF NOT EXISTS spans_filter_idx ON spans(started_at, status, kind);
                CREATE TABLE IF NOT EXISTS health (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
                """
            )

    @staticmethod
    def _event_payload(event: Any) -> dict[str, Any]:
        return event.as_dict() if hasattr(event, "as_dict") else dict(event)

    def record_event(self, event: Any) -> None:
        payload = self._event_payload(event)
        metadata = dict(payload.get("metadata") or payload.get("details") or {})
        usage = dict(payload.get("token_usage") or {})
        span_id = str(payload.get("event_id") or payload.get("id") or "")
        if not span_id:
            return
        trace_id = str(payload.get("session_id") or metadata.get("trace_id") or span_id)
        queue_wait = metadata.get("queue_wait_ms", metadata.get("queue_wait_duration_ms", 0))
        try:
            queue_wait = float(queue_wait or 0)
        except (TypeError, ValueError):
            queue_wait = 0.0
        error = metadata.get("error") or (payload.get("summary") if payload.get("status") == "failed" else "")
        row = (
            span_id, trace_id, payload.get("parent_event_id") or payload.get("parent_id"),
            payload.get("session_id", ""), payload.get("turn_id", ""), metadata.get("task_id", ""),
            payload.get("agent_id", ""), payload.get("subagent_id", ""), payload.get("step_id", ""),
            payload.get("kind", metadata.get("kind", "reasoning")), payload.get("type", "step.updated"),
            payload.get("title", ""), payload.get("status", "running"), payload.get("started_at", ""),
            payload.get("ended_at"), payload.get("duration_ms"), queue_wait,
            redact_summary(metadata.get("input") or metadata.get("args") or ""),
            redact_summary(metadata.get("output") or metadata.get("result_summary") or payload.get("summary") or ""),
            redact_summary(error), json.dumps(usage, ensure_ascii=False, default=str),
            _redacted_json(metadata),
        )
        with self._connect() as db:
            db.execute(
                """INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(span_id) DO UPDATE SET parent_span_id=excluded.parent_span_id, status=excluded.status,
                ended_at=excluded.ended_at, duration_ms=excluded.duration_ms, queue_wait_ms=excluded.queue_wait_ms,
                output_summary=excluded.output_summary, error_summary=excluded.error_summary,
                token_usage_json=excluded.token_usage_json, attributes_json=excluded.attributes_json""",
                row,
            )
        self.prune()
        self._export_otlp(row)

    def prune(self) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.config.retention_days)).isoformat()
        pruned = False
        with self._connect() as db:
            pruned = bool(db.execute("DELETE FROM spans WHERE ended_at IS NOT NULL AND ended_at < ?", (cutoff,)).rowcount)
            while self.path.exists() and self.path.stat().st_size > self.config.max_storage_mb * 1024 * 1024:
                deleted = db.execute("DELETE FROM spans WHERE span_id IN (SELECT span_id FROM spans WHERE ended_at IS NOT NULL ORDER BY ended_at LIMIT 100)").rowcount
                if not deleted:
                    break
                pruned = True
        if pruned:
            with self._connect() as db:
                db.execute("VACUUM")

    def _health(self, key: str, value: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO health VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at", (key, json.dumps(value), datetime.now(timezone.utc).isoformat()))

    def _export_otlp(self, row: tuple[Any, ...]) -> None:
        if not self.config.otlp_endpoint:
            return
        try:
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
            from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
            request = ExportTraceServiceRequest()
            span = request.resource_spans.add().scope_spans.add().spans.add()
            span.name = str(row[11] or row[10])
            span.trace_id = hashlib.sha256(str(row[1]).encode()).digest()[:16]
            span.span_id = hashlib.sha256(str(row[0]).encode()).digest()[:8]
            if row[2]:
                span.parent_span_id = hashlib.sha256(str(row[2]).encode()).digest()[:8]
            span.start_time_unix_nano = _epoch_ns(row[13])
            span.end_time_unix_nano = _epoch_ns(row[14] or row[13])
            for key, value in (("mana.kind", row[9]), ("mana.status", row[12]), ("mana.input", row[17]), ("mana.output", row[18])):
                span.attributes.append(KeyValue(key=key, value=AnyValue(string_value=str(value))))
            endpoint = self.config.otlp_endpoint.rstrip("/")
            if not endpoint.endswith("/v1/traces"):
                endpoint += "/v1/traces"
            request_headers = {"Content-Type": "application/x-protobuf", **(self.config.otlp_headers or {})}
            with urlopen(Request(endpoint, data=request.SerializeToString(), headers=request_headers, method="POST"), timeout=2):
                pass
            self._health("otlp", {"status": "ok", "endpoint": endpoint})
        except Exception as exc:  # export must never interrupt local tracing
            self._health("otlp", {"status": "failed", "reason": str(exc)[:300]})

    def spans(self, *, limit: int = 200, trace_id: str = "", status: str = "", kind: str = "", agent: str = "", since: str = "") -> list[dict[str, Any]]:
        clauses, values = ["1=1"], []
        for column, value in (("trace_id", trace_id), ("status", status), ("kind", kind), ("agent_id", agent)):
            if value:
                clauses.append(f"{column} = ?")
                values.append(value)
        if since:
            clauses.append("started_at >= ?")
            values.append(since)
        values.append(max(1, min(limit, 1000)))
        with self._connect() as db:
            rows = db.execute(f"SELECT * FROM spans WHERE {' AND '.join(clauses)} ORDER BY started_at DESC LIMIT ?", values).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["token_usage"] = _decode_json_object(item.pop("token_usage_json"))
        item["attributes"] = _decode_json_object(item.pop("attributes_json"))
        return item

    def overview(self, *, since: str = "") -> dict[str, Any]:
        rows = self.spans(limit=1000, since=since)
        durations = sorted(float(row["duration_ms"] or 0) for row in rows if row["duration_ms"] is not None)
        def percentile(value: float) -> float:
            return durations[min(len(durations) - 1, max(0, int((len(durations) - 1) * value)))] if durations else 0.0
        tokens = sum(int(row["token_usage"].get("total_tokens") or 0) for row in rows)
        return {"span_count": len(rows), "trace_count": len({row["trace_id"] for row in rows}), "error_count": sum(row["status"] == "failed" for row in rows), "total_tokens": tokens, "p50_latency_ms": round(percentile(.5), 1), "p95_latency_ms": round(percentile(.95), 1), "by_kind": self._group(rows, "kind"), "by_agent": self._group(rows, "agent_id"), "bottlenecks": self.bottlenecks(rows)}

    @staticmethod
    def _group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            name = str(row.get(key) or "unknown")
            entry = grouped.setdefault(name, {key: name, "count": 0, "errors": 0, "tokens": 0, "duration_ms": []})
            entry["count"] += 1; entry["errors"] += row["status"] == "failed"; entry["tokens"] += int(row["token_usage"].get("total_tokens") or 0); entry["duration_ms"].append(float(row["duration_ms"] or 0))
        return [{key: name, "count": entry["count"], "errors": entry["errors"], "tokens": entry["tokens"], "avg_latency_ms": round(sum(entry["duration_ms"]) / entry["count"], 1)} for name, entry in sorted(grouped.items(), key=lambda item: item[1]["count"], reverse=True)]

    @staticmethod
    def bottlenecks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows: groups.setdefault((row["kind"], row["title"]), []).append(row)
        findings = []
        for (kind, title), items in groups.items():
            if len(items) < 3: continue
            durations = sorted(float(x["duration_ms"] or 0) for x in items); p95 = durations[int((len(durations) - 1) * .95)]
            errors = sum(x["status"] == "failed" for x in items) / len(items); queue = sum(float(x["queue_wait_ms"] or 0) for x in items) / len(items); tokens = sum(int(x["token_usage"].get("total_tokens") or 0) for x in items) / len(items)
            reasons = []
            if p95 >= 2_000: reasons.append(f"p95 latency {p95:.0f}ms ≥ 2000ms")
            if errors >= .1: reasons.append(f"error rate {errors:.0%} ≥ 10%")
            if queue >= 1_000: reasons.append(f"average queue wait {queue:.0f}ms ≥ 1000ms")
            if tokens >= 10_000: reasons.append(f"average token use {tokens:.0f} ≥ 10000")
            if reasons: findings.append({"kind": kind, "title": title, "sample_size": len(items), "reasons": reasons, "trace_id": items[0]["trace_id"], "p95_latency_ms": round(p95, 1), "error_rate": round(errors, 3), "avg_tokens": round(tokens, 1)})
        return sorted(findings, key=lambda item: (len(item["reasons"]), item["p95_latency_ms"]), reverse=True)

    def health(self) -> dict[str, Any]:
        with self._connect() as db:
            rows = db.execute("SELECT key, value, updated_at FROM health").fetchall()
        return {row["key"]: {**json.loads(row["value"]), "updated_at": row["updated_at"]} for row in rows}
