from __future__ import annotations

from typing import Any

from mana_agent.multi_agent.runtime.tool_worker_process import ToolRunRequest
from mana_agent.multi_agent.runtime.tools_executor import (
    BatchToolRequest,
    RedisRQToolsExecutor,
    ToolsExecutionConfig,
    ToolsExecutor,
)


class _FakeRedisConnection:
    def __init__(self) -> None:
        self.set_calls: list[tuple[str, str]] = []
        self.expire_calls: list[tuple[str, int]] = []
        self.rpush_calls: list[tuple[str, str]] = []

    def set(self, key: str, value: str) -> None:
        self.set_calls.append((key, value))

    def expire(self, key: str, ttl: int) -> None:
        self.expire_calls.append((key, ttl))

    def rpush(self, key: str, value: str) -> None:
        self.rpush_calls.append((key, value))


class _FakeJob:
    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        finish_after: int = 1,
        failed: bool = False,
        timeout: bool = False,
    ) -> None:
        self._payload = payload
        self._finish_after = max(1, int(finish_after))
        self._failed = bool(failed)
        self._timeout = bool(timeout)
        self._refresh_count = 0
        self._status = "queued"
        self.exc_info = "JobTimeoutException: exceeded" if self._timeout else "Traceback: failed"

    def refresh(self) -> None:
        self._refresh_count += 1
        if self._refresh_count >= self._finish_after:
            self._status = "failed" if self._failed else "finished"

    def get_status(self, refresh: bool = False) -> str:
        _ = refresh
        return self._status

    @property
    def is_finished(self) -> bool:
        return self._status == "finished"

    @property
    def is_failed(self) -> bool:
        return self._status == "failed"

    @property
    def result(self) -> dict[str, Any] | None:
        return self._payload


def _install_fake_redis_stack(monkeypatch, jobs: list[_FakeJob]):
    fake_redis = _FakeRedisConnection()

    class _RedisClass:
        @staticmethod
        def from_url(_url: str):
            return fake_redis

    class _QueueClass:
        def __init__(self, _name: str, connection) -> None:
            _ = connection

        def enqueue(self, _func, *_args, **_kwargs):
            return jobs.pop(0)

    def _fake_import(name: str):
        if name == "redis":
            return type("_RedisModule", (), {"Redis": _RedisClass})
        if name == "rq":
            return type("_RQModule", (), {"Queue": _QueueClass})
        raise RuntimeError(f"unexpected import: {name}")

    monkeypatch.setattr("mana_agent.multi_agent.runtime.tools_executor.importlib.import_module", _fake_import)
    return fake_redis


def test_base_tools_executor_returns_structured_failures() -> None:
    results = ToolsExecutor().run_batch(
        run_id="base-run",
        requests=[
            BatchToolRequest(request_index=2, request=ToolRunRequest(question="q2")),
            BatchToolRequest(request_index=4, request=ToolRunRequest(question="q4")),
        ],
    )

    assert [item.request_index for item in results] == [2, 4]
    assert all(item.ok is False for item in results)
    assert {item.error_code for item in results} == {"worker_unavailable"}
    assert {item.backend for item in results} == {"base"}
    assert all("no execution backend" in item.error_message for item in results)


def test_redis_executor_keeps_deterministic_input_order(monkeypatch) -> None:
    jobs = [
        _FakeJob(
            payload={
                "ok": True,
                "response": {
                    "answer": "first",
                    "sources": [],
                    "mode": "agent-tools",
                    "trace": [{"idx": 0, "tool_name": "read_file", "status": "ok"}],
                    "warnings": [],
                },
                "duration_ms": 10.0,
                "queue_wait_ms": 2.0,
            },
            finish_after=2,
        ),
        _FakeJob(
            payload={
                "ok": True,
                "response": {
                    "answer": "second",
                    "sources": [],
                    "mode": "agent-tools",
                    "trace": [{"idx": 1, "tool_name": "semantic_search", "status": "ok"}],
                    "warnings": [],
                },
                "duration_ms": 8.0,
                "queue_wait_ms": 1.0,
            },
            finish_after=1,
        ),
    ]
    fake_redis = _install_fake_redis_stack(monkeypatch, jobs)
    config = ToolsExecutionConfig(
        backend="redis",
        redis_url="redis://127.0.0.1:6379/0",
        queue_name="mana-tools",
        parallel_requests=2,
        ttl_seconds=3600,
    )
    executor = RedisRQToolsExecutor(
        worker_init_payload={
            "api_key": "x",
            "model": "fake-model",
            "project_root": "/tmp",
            "repo_root": "/tmp",
            "tools_only_strict": True,
        },
        config=config,
    )

    results = executor.run_batch(
        run_id="run-1",
        requests=[
            BatchToolRequest(request_index=0, request=ToolRunRequest(question="q0", index_dir="/tmp/.mana/index")),
            BatchToolRequest(request_index=1, request=ToolRunRequest(question="q1", index_dir="/tmp/.mana/index")),
        ],
    )

    assert len(results) == 2
    assert [item.request_index for item in results] == [0, 1]
    assert all(item.ok for item in results)
    assert any("mana:tools:run:run-1:status" in key for key, _ in fake_redis.set_calls)
    assert fake_redis.expire_calls


def test_redis_executor_maps_timeout_failures_to_job_timeout(monkeypatch) -> None:
    jobs = [_FakeJob(payload=None, finish_after=1, failed=True, timeout=True)]
    _install_fake_redis_stack(monkeypatch, jobs)

    config = ToolsExecutionConfig(
        backend="redis",
        redis_url="redis://127.0.0.1:6379/0",
        queue_name="mana-tools",
        parallel_requests=1,
        ttl_seconds=3600,
    )
    executor = RedisRQToolsExecutor(
        worker_init_payload={
            "api_key": "x",
            "model": "fake-model",
            "project_root": "/tmp",
            "repo_root": "/tmp",
            "tools_only_strict": True,
        },
        config=config,
    )

    results = executor.run_batch(
        run_id="run-timeout",
        requests=[
            BatchToolRequest(request_index=0, request=ToolRunRequest(question="q0", index_dir="/tmp/.mana/index"))
        ],
    )
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error_code == "job_timeout"
