"""Comprehensive tests for dynamic model metadata, tool capabilities, Batch API, and tools integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mana_agent.batch.builder import BatchLimits, BatchRequestBuilder
from mana_agent.batch.execution_router import (
    ExecutionMode,
    ExecutionModeRouter,
    ExecutionRoutingDecision,
)
from mana_agent.batch.manager import BatchLifecycleManager
from mana_agent.batch.models import (
    BatchConfig,
    BatchItemRequest,
    BatchItemResult,
    BatchJobRecord,
    BatchLifecycleStatus,
)
from mana_agent.batch.processor import BatchResultProcessor, ProcessedBatchResults
from mana_agent.batch.service import BatchExecutionService
from mana_agent.batch.store import BatchJobStore
from mana_agent.batch.tools import (
    batch_cancel_job,
    batch_check_status,
    batch_create_job,
    batch_download_results,
    batch_get_status,
    batch_list_jobs,
    batch_retrieve_results,
    batch_submit_requests,
    build_batch_langchain_tools,
)
from mana_agent.coding.programmatic import ToolExecutionDispatcher
from mana_agent.config.model_discovery import (
    GLOBAL_CAPABILITY_REGISTRY,
    ModelCapabilityProfile,
    ModelCapabilityRegistry,
    ModelDiscoveryService,
    ModelMetadataResolver,
)
from mana_agent.config.tool_compatibility import (
    ModelConfigurationValidator,
    ToolCompatibilityResolver,
)
from mana_agent.tools.contracts import coding_tool_contracts


# --- 1. Model Discovery & Capability Registry ---


def test_capability_registry_luna_and_flagship_models() -> None:
    registry = ModelCapabilityRegistry()

    # Luna flagship models
    luna_prof = registry.get("openai", "gpt-6-luna")
    assert luna_prof is not None
    assert luna_prof.model_id == "gpt-6-luna"
    assert luna_prof.context_window == 1_050_000
    assert luna_prof.supports_batch is True
    assert luna_prof.supports_batch_display == "yes"
    assert "low" in luna_prof.reasoning_effort_levels
    assert luna_prof.function_calling is True
    assert luna_prof.verification_source == "maintained_registry"
    assert luna_prof.verified_at != ""

    # Bare luna alias
    bare_luna = registry.get("openai", "luna")
    assert bare_luna is not None
    assert bare_luna.supports_batch is True

    # GPT-5.6 Luna
    gpt56_luna = registry.get("openai", "gpt-5.6-luna")
    assert gpt56_luna is not None
    assert gpt56_luna.context_window == 400_000
    assert gpt56_luna.supports_batch is True

    # Unsupported batch model
    realtime_prof = registry.get("openai", "gpt-4o-realtime-preview")
    assert realtime_prof is not None
    assert realtime_prof.supports_batch is False
    assert realtime_prof.supports_batch_display == "no"
    assert len(realtime_prof.batch_restrictions) > 0


def test_metadata_resolver_dynamic_and_offline() -> None:
    resolver = ModelMetadataResolver()

    # Known model resolution
    profile = resolver.resolve("gpt-6-luna", provider="openai")
    assert profile.supports_batch is True
    assert profile.confidence == "high"

    # API record augmentation
    api_record = {
        "id": "gpt-6-luna",
        "created": 1700000000,
        "owned_by": "openai",
        "context_window": 1_050_000,
    }
    augmented = resolver.resolve("gpt-6-luna", provider="openai", raw_metadata=api_record)
    assert augmented.verification_source == "openai_api"
    assert augmented.raw_metadata.get("owned_by") == "openai"


def test_model_discovery_service_offline_fallback() -> None:
    discovery = ModelDiscoveryService()
    # When api_key is empty or offline, returns maintained registry profiles
    profiles = discovery.discover_models(provider="openai", api_key="")
    assert len(profiles) > 0
    ids = {p.model_id for p in profiles}
    assert "gpt-6-luna" in ids
    assert "gpt-4o" in ids
    assert "o3" in ids


# --- 2. Tool Compatibility & Configuration Validator ---


def test_tool_compatibility_resolver() -> None:
    resolver = ToolCompatibilityResolver()

    # Compatible tools for standard text model
    tools = ["read_file", "write_file", "batch_create_job", "shell"]
    report = resolver.resolve("gpt-6-luna", tools, execution_mode="realtime")
    assert report.all_compatible
    assert report.requires_batch is True
    assert len(report.incompatible_tools) == 0

    # In batch mode, interactive shell tools are restricted
    report_batch = resolver.resolve("gpt-6-luna", ["shell", "read_file"], execution_mode="batch")
    assert not report_batch.all_compatible
    assert "shell" in report_batch.batch_restricted_tools


def test_model_configuration_validator() -> None:
    validator = ModelConfigurationValidator()

    # Valid config
    res_ok = validator.validate("gpt-6-luna", mode="batch", requested_tokens=100_000)
    assert res_ok.ok
    assert res_ok.valid

    # Unsupported batch model in batch mode
    res_bad = validator.validate("gpt-4o-realtime-preview", mode="batch")
    assert not res_bad.ok
    assert any("unsupported" in err.lower() for err in res_bad.errors)

    # Token overflow
    res_overflow = validator.validate("gpt-5.6-luna", requested_tokens=900_000)
    assert not res_overflow.ok
    assert any("exceed" in err.lower() for err in res_overflow.errors)


# --- 3. Batch Request Builder ---


def test_batch_request_builder_and_grouping() -> None:
    builder = BatchRequestBuilder()
    limits = builder.limits
    assert limits.max_requests_per_batch == 50_000
    assert limits.max_file_size_mb == 200
    assert limits.completion_window == "24h"
    assert limits.upload_purpose == "batch"

    item1 = builder.build_item(
        custom_id="req-1",
        body={"input": "test prompt 1"},
        model="gpt-6-luna",
        url="/v1/responses",
    )
    item2 = builder.build_item(
        custom_id="req-2",
        body={"input": "test prompt 2"},
        model="gpt-6-luna",
        url="/v1/responses",
    )
    item3 = builder.build_item(
        custom_id="req-3",
        body={"input": "test prompt 3"},
        model="gpt-4o",
        url="/v1/chat/completions",
    )

    # Grouping
    groups = builder.group_by_model_and_endpoint([item1, item2, item3])
    assert ("gpt-6-luna", "/v1/responses") in groups
    assert len(groups[("gpt-6-luna", "/v1/responses")]) == 2
    assert ("gpt-4o", "/v1/chat/completions") in groups
    assert len(groups[("gpt-4o", "/v1/chat/completions")]) == 1

    # JSONL serialization
    raw_jsonl = builder.validate_and_serialize_jsonl([item1, item2])
    lines = raw_jsonl.decode("utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed_first = json.loads(lines[0])
    assert parsed_first["custom_id"] == "req-1"
    assert parsed_first["body"]["model"] == "gpt-6-luna"

    # Duplicate custom_id rejection
    dup_item = builder.build_item(
        custom_id="req-1",
        body={"model": "gpt-6-luna", "input": "dup"},
    )
    with pytest.raises(ValueError, match="Duplicate custom_id"):
        builder.validate_and_serialize_jsonl([item1, dup_item])


# --- 4. Batch Result Processor & Retries ---


def test_batch_result_processor_correlation_and_retry() -> None:
    processor = BatchResultProcessor()

    output_jsonl = """{"custom_id": "task-1", "response": {"status_code": 200, "body": {"output": "ok 1"}}}
{"custom_id": "task-2", "response": {"status_code": 200, "body": {"output": "ok 2"}}}
"""
    error_jsonl = """{"custom_id": "task-3", "response": {"status_code": 400}, "error": {"message": "Invalid prompt"}}
"""

    processed = processor.correlate_results(
        batch_id="batch_123",
        output_jsonl=output_jsonl,
        error_jsonl=error_jsonl,
    )

    assert processed.total_results == 3
    assert len(processed.successful_results) == 2
    assert len(processed.failed_results) == 1
    assert processed.results_by_custom_id["task-1"].ok
    assert not processed.results_by_custom_id["task-3"].ok

    # Retry extraction: only failed request is selected
    orig_items = [
        BatchItemRequest(custom_id="task-1", body={"model": "gpt-6-luna"}),
        BatchItemRequest(custom_id="task-2", body={"model": "gpt-6-luna"}),
        BatchItemRequest(custom_id="task-3", body={"model": "gpt-6-luna"}),
    ]
    retry_items = processor.extract_failed_items_for_retry(orig_items, processed)
    assert len(retry_items) == 1
    assert retry_items[0].custom_id == "task-3"


# --- 5. Execution Mode Router ---


def test_execution_mode_router() -> None:
    router = ExecutionModeRouter(BatchConfig(always_enable_if_supported=True))

    items = [
        BatchItemRequest(custom_id="1", body={"model": "gpt-6-luna", "input": "prompt"}, url="/v1/responses"),
    ]

    # Auto mode routes supported model to batch
    dec = router.route(items, model="gpt-6-luna")
    assert dec.is_batch
    assert dec.selected_mode == ExecutionMode.BATCH
    assert dec.discount_factor == 0.5

    # Latency-sensitive work is NEVER scheduled in batch
    dec_latency = router.route(items, model="gpt-6-luna", is_latency_sensitive=True)
    assert not dec_latency.is_batch
    assert dec_latency.selected_mode == ExecutionMode.REALTIME
    assert dec_latency.is_latency_sensitive

    # Streaming required -> Realtime
    dec_stream = router.route(items, model="gpt-6-luna", requires_streaming=True)
    assert dec_stream.selected_mode == ExecutionMode.REALTIME

    # Intra-dependencies -> Realtime
    dec_deps = router.route(items, model="gpt-6-luna", has_intra_dependencies=True)
    assert dec_deps.selected_mode == ExecutionMode.REALTIME


# --- 6. Tool Contracts & Dispatching ---


def test_batch_tool_contracts_presence() -> None:
    contracts = coding_tool_contracts()
    names = {c.name for c in contracts}
    assert "batch_create_job" in names
    assert "batch_get_status" in names
    assert "batch_list_jobs" in names
    assert "batch_cancel_job" in names
    assert "batch_download_results" in names
    assert "batch_submit_requests" in names
    assert "batch_check_status" in names
    assert "batch_retrieve_results" in names


def test_tool_dispatcher_executes_batch_tools() -> None:
    dispatcher = ToolExecutionDispatcher(repo_root=".")

    # Mock client and store for dispatcher testing
    mock_client = MagicMock()
    mock_client.upload_batch_file.return_value = "file-test123"
    mock_client.create_batch.return_value = BatchJobRecord(
        batch_id="batch_mock123",
        status=BatchLifecycleStatus.VALIDATING.value,
        input_file_id="file-test123",
        endpoint="/v1/responses",
        model="gpt-6-luna",
        total_requests=1,
    )
    mock_client.retrieve_batch.return_value = BatchJobRecord(
        batch_id="batch_mock123",
        status=BatchLifecycleStatus.IN_PROGRESS.value,
        input_file_id="file-test123",
        endpoint="/v1/responses",
        model="gpt-6-luna",
        total_requests=1,
        completed_requests=1,
        output_file_id="file-out123",
    )
    mock_client.cancel_batch.return_value = BatchJobRecord(
        batch_id="batch_mock123",
        status=BatchLifecycleStatus.CANCELLED.value,
    )
    mock_client.download_results.return_value = [
        BatchItemResult(custom_id="req-1", status_code=200, response={"text": "hello"}),
    ]

    store = BatchJobStore(":memory:")
    mgr = BatchLifecycleManager(client=mock_client, store=store)

    with patch("mana_agent.batch.tools.get_batch_manager", return_value=mgr):
        # 1. Create job
        res_create = dispatcher.execute_tool(
            "batch_create_job",
            {
                "requests": [{"custom_id": "req-1", "body": {"model": "gpt-6-luna", "input": "Hello"}}],
                "model": "gpt-6-luna",
            },
        )
        assert res_create["ok"]
        assert res_create["batch_id"] == "batch_mock123"

        # 2. Get status
        res_status = dispatcher.execute_tool(
            "batch_get_status",
            {"batch_id": "batch_mock123"},
        )
        assert res_status["ok"]
        assert res_status["batch_id"] == "batch_mock123"

        # 3. List jobs
        res_list = dispatcher.execute_tool(
            "batch_list_jobs",
            {"limit": 10},
        )
        assert res_list["ok"]
        assert res_list["total"] >= 1

        # 4. Download results
        res_dl = dispatcher.execute_tool(
            "batch_download_results",
            {"batch_id": "batch_mock123"},
        )
        assert res_dl["ok"]
        assert res_dl["total_results"] == 1

        # 5. Cancel job
        res_cancel = dispatcher.execute_tool(
            "batch_cancel_job",
            {"batch_id": "batch_mock123"},
        )
        assert res_cancel["ok"]
        assert res_cancel["status"] == BatchLifecycleStatus.CANCELLED.value

        # 6. Test aliases: batch_submit_requests, batch_check_status, batch_retrieve_results
        res_sub = dispatcher.execute_tool(
            "batch_submit_requests",
            {
                "requests": [{"custom_id": "req-2", "body": {"model": "gpt-6-luna", "input": "World"}}],
                "model": "gpt-6-luna",
            },
        )
        assert res_sub["ok"]

        res_check = dispatcher.execute_tool(
            "batch_check_status",
            {"batch_id": "batch_mock123"},
        )
        assert res_check["ok"]

        res_ret = dispatcher.execute_tool(
            "batch_retrieve_results",
            {"batch_id": "batch_mock123"},
        )
        assert res_ret["ok"]


def test_build_batch_langchain_tools() -> None:
    tools = build_batch_langchain_tools()
    assert len(tools) == 8
    names = {t.name for t in tools}
    assert "batch_create_job" in names
    assert "batch_get_status" in names
    assert "batch_list_jobs" in names
    assert "batch_cancel_job" in names
    assert "batch_download_results" in names
    assert "batch_submit_requests" in names
    assert "batch_check_status" in names
    assert "batch_retrieve_results" in names


def test_batch_tools_internal_mode_execution() -> None:
    """Test mode 2: Internal batch engine for non-OpenAI models or explicit internal mode."""
    store = BatchJobStore(":memory:")
    # OpenAIBatchClient mock will fail if called, ensuring internal mode does NOT call OpenAI
    mock_client = MagicMock()
    mock_client.upload_batch_file.side_effect = AssertionError("OpenAI upload should not be called in internal mode")
    mock_client.create_batch.side_effect = AssertionError("OpenAI create should not be called in internal mode")

    def mock_internal_handler(item: BatchItemRequest) -> BatchItemResult:
        return BatchItemResult(
            custom_id=item.custom_id,
            status_code=200,
            response={
                "id": f"resp_{item.custom_id}",
                "custom_id": item.custom_id,
                "response": {
                    "body": {
                        "choices": [{"message": {"role": "assistant", "content": f"Result for {item.custom_id}"}}]
                    }
                },
            },
        )

    mgr = BatchLifecycleManager(client=mock_client, store=store, internal_handler=mock_internal_handler)

    with patch("mana_agent.batch.tools.get_batch_manager", return_value=mgr):
        # 1. Non-OpenAI model (e.g. claude-3-5-sonnet) automatically routes to internal mode
        res_create = batch_create_job(
            requests=[
                {"custom_id": "req-c1", "body": {"input": "Analyze A"}},
                {"custom_id": "req-c2", "body": {"input": "Analyze B"}},
            ],
            model="claude-3-5-sonnet",
            provider="anthropic",
        )
        assert res_create["ok"]
        assert res_create["mode"] == "internal"
        batch_id = res_create["batch_id"]
        assert batch_id.startswith("batch_int_")
        assert res_create["total_requests"] == 2

        # 2. Status check retrieves from local SQLite store
        res_status = batch_get_status(batch_id=batch_id)
        assert res_status["ok"]
        assert res_status["batch_id"] == batch_id
        assert res_status["completed_requests"] == 2

        # 3. Retrieve results loads parsed results from store
        res_dl = batch_download_results(batch_id=batch_id)
        assert res_dl["ok"]
        assert res_dl["total_results"] == 2
        results_map = {r["custom_id"]: r for r in res_dl["results"]}
        assert "req-c1" in results_map
        assert "req-c2" in results_map

        # 4. List jobs contains the internal job
        res_list = batch_list_jobs()
        assert res_list["ok"]
        assert any(j["batch_id"] == batch_id for j in res_list["jobs"])


def test_batch_tools_mode_selection_auto() -> None:
    """Test auto mode correctly dispatches: OpenAI model -> OpenAI Batch API, non-OpenAI -> Internal."""
    from mana_agent.batch.router import is_openai_batch_supported

    # OpenAI supported models
    assert is_openai_batch_supported("gpt-4o")
    assert is_openai_batch_supported("gpt-6-luna")
    assert is_openai_batch_supported("o3-mini")

    # Non-OpenAI models / providers -> False (routes to internal mode)
    assert not is_openai_batch_supported("claude-3-5-sonnet", provider="anthropic")
    assert not is_openai_batch_supported("anthropic/claude-3-5-sonnet")
    assert not is_openai_batch_supported("ollama/llama3")
    assert not is_openai_batch_supported("google/gemini-2.5-pro")
    assert not is_openai_batch_supported("gpt-4o-realtime-preview")

