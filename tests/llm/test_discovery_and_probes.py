"""Controlled transports only. Running this suite never contacts a paid endpoint."""
from __future__ import annotations
import asyncio
from types import SimpleNamespace
import httpx
import pytest
from mana_agent.llm.discovery import ModelDiscoveryService
from mana_agent.llm.registry import ModelCapabilityRegistry
from mana_agent.llm.capabilities import CapabilityState
from mana_agent.llm.probes import run_capability_probe
from mana_agent.llm.messages import ToolCall
from mana_agent.llm.types import ModelResponse, Usage


def test_discovery_paginates_and_checks_exact_detail_identity(tmp_path, monkeypatch):
    calls = []
    def handle(request):
        calls.append(request)
        if request.url.path == "/v1/models":
            if request.url.params.get("after") == "a":
                return httpx.Response(200, json={"data": [{"id": "b"}], "has_more": False})
            return httpx.Response(200, json={"data": [{"id": "a"}], "has_more": True, "last_id": "a"})
        model = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={"id": model, "context_window": 4096})
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            service = ModelDiscoveryService(registry=ModelCapabilityRegistry(tmp_path / "profiles.json"), client=client)
            return await service.fetch_records(provider="openai", base_url="https://provider.example/v1", api_key="test")
    records = asyncio.run(execute())
    assert [record["id"] for record in records] == ["a", "b"]
    assert all(record["context_window"] == 4096 for record in records)
    assert len(calls) == 4
    assert all(request.headers["authorization"] == "Bearer test" for request in calls)


def test_catalog_exact_provider_and_model_identity(tmp_path, monkeypatch):
    registry = ModelCapabilityRegistry(tmp_path / "profiles.json")
    service = ModelDiscoveryService(registry=registry)
    async def records(**kwargs):
        return [{"id": "org/model"}, {"id": "model"}]
    async def catalog():
        return {"openai": {"models": {"org/model": {"tool_call": True}}}, "custom": {"models": {"model": {"reasoning": True}}}}
    monkeypatch.setattr(service, "fetch_records", records)
    monkeypatch.setattr(service, "catalog", catalog)
    profiles = asyncio.run(service.discover(provider="openai", base_url="https://provider.example/v1", api_key="test"))
    assert profiles[0].supports("tool_calling")
    assert not profiles[1].supports("tool_calling")
    assert not profiles[1].supports("reasoning")


def test_repeated_discovery_keeps_endpoint_evidence_separate(tmp_path, monkeypatch):
    registry = ModelCapabilityRegistry(tmp_path / "profiles.json")
    discovery = ModelDiscoveryService(registry=registry)
    monkeypatch.setenv("MANA_HOME", str(tmp_path))

    async def records(*, base_url, **options):
        return [{"id": "same", "tool_calling": base_url.endswith("/first")}]

    async def catalog():
        return {}

    monkeypatch.setattr(discovery, "fetch_records", records)
    monkeypatch.setattr(discovery, "catalog", catalog)
    first, = asyncio.run(discovery.discover(provider="openai", base_url="https://provider.example/first", api_key="test"))
    second, = asyncio.run(discovery.discover(provider="openai", base_url="https://provider.example/second", api_key="test"))

    assert first.configuration_fingerprint != second.configuration_fingerprint
    assert registry.get("openai", "same", first.configuration_fingerprint).supports("tool_calling")
    assert registry.get("openai", "same", second.configuration_fingerprint).fact("tool_calling").state is CapabilityState.UNSUPPORTED


def test_discovery_failure_has_no_offline_model_default(tmp_path):
    def handle(request):
        return httpx.Response(401, json={"error": "unauthorized"})
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            service = ModelDiscoveryService(registry=ModelCapabilityRegistry(tmp_path / "profiles.json"), client=client)
            await service.fetch_records(provider="openai", base_url="https://provider.example/v1", api_key="test")
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(execute())


class ProbeAdapter:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []
    async def generate(self, request, **_native_options):
        self.calls.append(request)
        if self.error:
            raise self.error
        return self.result


def probe_engine(profile, adapter, governor):
    return SimpleNamespace(provider="openai", model="test-model", parameters={}, timeout=1, adapter=adapter,
        context_cost_governor=governor, _adapter=lambda: adapter, _reserve=lambda request: "reservation",
        profile=lambda *_args: profile)


class Governor:
    enabled = True
    def __init__(self):
        self.completed, self.released = [], []
    def record_model_call(self, identity, **kwargs):
        self.completed.append(identity)
    def release_reservation(self, identity, **kwargs):
        self.released.append(identity)


def test_tool_probe_one_request_never_executes_a_tool(native_profile):
    adapter = ProbeAdapter(ModelResponse(tool_calls=[ToolCall(id="synthetic", name="mana_capability_echo", args={"value": "mana-capability-ok"})], usage=Usage(total_tokens=4)))
    governor = Governor()
    result = asyncio.run(run_capability_probe(probe_engine(native_profile, adapter, governor), "tool_calling", user_initiated=True, consent=True))
    assert result.state is CapabilityState.SUPPORTED
    assert len(adapter.calls) == 1
    assert adapter.calls[0].tools[0]["name"] == "mana_capability_echo"
    assert governor.completed == ["reservation"]


@pytest.mark.parametrize("status,code,parameter,expected", [
    (400, "unsupported_parameter", "tools", CapabilityState.UNSUPPORTED),
    (401, "invalid_api_key", "tools", CapabilityState.UNKNOWN),
    (429, "rate_limit", "tools", CapabilityState.UNKNOWN),
    (503, "overloaded", "tools", CapabilityState.UNKNOWN),
    (400, "unsupported_parameter", "temperature", CapabilityState.UNKNOWN),
])
def test_probe_failures_do_not_guess_unsupported(native_profile, status, code, parameter, expected):
    error = RuntimeError("controlled provider error")
    error.status_code, error.body = status, {"error": {"code": code, "param": parameter}}
    adapter, governor = ProbeAdapter(error=error), Governor()
    result = asyncio.run(run_capability_probe(probe_engine(native_profile, adapter, governor), "tool_calling", user_initiated=True, consent=True))
    assert result.state is expected
    assert len(adapter.calls) == 1
    assert governor.released == ["reservation"]


def test_probe_cancellation_interrupts_without_replay(native_profile):
    class Slow(ProbeAdapter):
        cancelled = False
        async def generate(self, request, **_native_options):
            self.calls.append(request)
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
    adapter, governor = Slow(), Governor()
    async def execute():
        cancel = asyncio.Event()
        task = asyncio.create_task(run_capability_probe(probe_engine(native_profile, adapter, governor), "tool_calling", user_initiated=True, consent=True, cancel=cancel))
        await asyncio.sleep(0)
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(execute())
    assert len(adapter.calls) <= 1
    assert adapter.cancelled
    assert governor.released == ["reservation"]


def test_public_catalog_never_inherits_provider_auth_or_headers(monkeypatch, tmp_path):
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    async def execute():
        def handler(request):
            assert request.url.host == "models.dev"
            assert "authorization" not in request.headers
            assert "x-api-key" not in request.headers
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                headers={"Authorization": "Bearer provider-key", "X-API-Key": "provider-key"}) as client:
            assert await ModelDiscoveryService(client=client).catalog() == {}
    asyncio.run(execute())
