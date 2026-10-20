"""Validation of a selected Batch action, without semantic routing fallbacks."""

from __future__ import annotations

from collections.abc import Sequence

from mana_agent.batch.models import BatchConfig, BatchItemRequest
from mana_agent.config.model_discovery import GLOBAL_CAPABILITY_REGISTRY


def validate_batch_selection(
    config: BatchConfig, *, model: str, provider: str, endpoint: str,
    items: Sequence[BatchItemRequest] = (),
) -> None:
    if not config.enabled:
        raise ValueError("OpenAI Batch is disabled; no alternate execution was selected")
    if not items:
        raise ValueError("OpenAI Batch requires at least one explicitly selected request")
    if provider != "openai":
        raise ValueError("This OpenAI Batch transport requires provider=openai")
    if not model:
        raise ValueError("OpenAI Batch requires an explicitly selected model")
    profile = GLOBAL_CAPABILITY_REGISTRY.get(provider, model)
    if profile is None or profile.supports_batch is not True:
        raise ValueError(f"Batch capability for model '{model}' is unsupported or unknown")
    if endpoint not in profile.supported_batch_endpoints:
        raise ValueError(f"Batch endpoint '{endpoint}' is unsupported for model '{model}'")
    if config.completion_window not in profile.supported_completion_windows:
        raise ValueError("Unsupported Batch completion window")
    for item in items:
        if item.method != "POST" or item.url != endpoint or item.body.get("model") != model:
            raise ValueError("Batch items must share the selected model and endpoint and use POST")
        if item.body.get("stream"):
            raise ValueError("OpenAI Batch does not support streaming requests")
        if item.body.get("tools") and profile.function_calling is not True:
            raise ValueError("Selected Batch model does not support registered function tools")
        structured = item.body.get("response_format") or (item.body.get("text") or {}).get("format")
        if structured and profile.structured_output_supported is not True:
            raise ValueError("Selected Batch model does not support the required structured output contract")
    from mana_agent.batch.builder import BatchLimits, BatchRequestBuilder
    BatchRequestBuilder(BatchLimits(max_requests_per_batch=config.max_requests_per_batch,
                                   max_file_size_mb=config.max_file_size_mb)).validate_and_serialize_jsonl(items)
