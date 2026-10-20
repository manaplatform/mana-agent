"""One durable model turn through the official OpenAI Batch API."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mana_agent.batch.manager import BatchLifecycleManager
from mana_agent.batch.models import BatchItemRequest
from mana_agent.batch.operations import BatchOperationStore
from mana_agent.batch.policy import BatchPending, PendingBatchOperation
from mana_agent.batch.validation import validate_batch_selection


def response_message(body: dict[str, Any]) -> dict[str, Any]:
    """Decode only public assistant output and function calls, never reasoning."""
    if body.get("error") or body.get("status") in {"failed", "incomplete", "cancelled"}:
        raise ValueError("Batch model response failed or was incomplete")
    if "choices" in body:
        choices = body["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "tool_calls"}:
            raise ValueError("Batch response must contain one complete assistant choice")
        message = dict(choices[0].get("message") or {})
        if message.get("role") != "assistant":
            raise ValueError("Batch completion did not return an assistant message")
        if message.get("refusal"):
            raise ValueError("Batch model refused the request")
        if isinstance(message.get("content"), list):
            parts = message["content"]
            if any(part.get("type") != "text" for part in parts):
                raise ValueError("Batch completion returned unsupported assistant content")
            message["content"] = "".join(str(part.get("text") or "") for part in parts)
    else:
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        protocol_items: list[dict[str, Any]] = []
        for item in body.get("output") or []:
            if item.get("type") == "message":
                protocol_items.append(item)
                for part in item.get("content") or []:
                    if part.get("type") == "refusal":
                        raise ValueError("Batch model refused the request")
                    if part.get("type") == "output_text":
                        texts.append(str(part.get("text") or ""))
            elif item.get("type") == "function_call":
                protocol_items.append(item)
                calls.append({"id": item.get("call_id"), "type": "function", "function": {
                    "name": item.get("name"), "arguments": item.get("arguments"),
                }})
            elif item.get("type") == "reasoning":
                if not item.get("encrypted_content"):
                    raise ValueError("Batch reasoning continuation requires its opaque encrypted receipt")
                # Keep the provider's opaque continuation, never raw reasoning
                # text. It stays in protocol checkpoints, outside visible history.
                protocol_items.append({key: value for key, value in item.items() if key != "content"})
            else:
                raise ValueError("Batch response returned an unsupported output item")
        message = {"role": "assistant", "content": "".join(texts) or body.get("output_text") or ""}
        if calls:
            message["tool_calls"] = calls
        if protocol_items:
            message["_response_items"] = protocol_items
    calls = message.get("tool_calls") or []
    ids: set[str] = set()
    for call in calls:
        if call.get("type") != "function" or not call.get("id") or call["id"] in ids or not call.get("function", {}).get("name"):
            raise ValueError("Batch response contains invalid or duplicate tool calls")
        ids.add(call["id"])
        args = json.loads(call["function"]["arguments"])
        if not isinstance(args, dict):
            raise ValueError("Batch function arguments must be an object")
    if not calls and not str(message.get("content") or "").strip():
        raise ValueError("Empty Batch model output is not completion evidence")
    return message


def request_body(model: str, endpoint: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], schema: dict[str, Any] | None, *, strict_schema: bool = True) -> dict[str, Any]:
    if schema and strict_schema:
        def strict(value: Any) -> Any:
            if isinstance(value, list):
                return [strict(item) for item in value]
            if not isinstance(value, dict):
                return value
            result = {key: strict(item) for key, item in value.items() if key != "default"}
            if result.get("type") == "object" or "properties" in result:
                result["additionalProperties"] = False
                result["required"] = list(result.get("properties", {}))
            return result
        schema = strict(schema)
    if endpoint == "/v1/chat/completions":
        body: dict[str, Any] = {"model": model, "messages": [
            {key: value for key, value in message.items() if not key.startswith("_")} for message in messages
        ]}
        if tools:
            body["tools"] = tools
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "strict": strict_schema, "schema": schema}}
        return body
    if endpoint != "/v1/responses":
        raise ValueError("Task model turns require Responses or Chat Completions")
    inputs: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] == "tool":
            inputs.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]})
            continue
        if message.get("_response_items"):
            inputs.extend(message["_response_items"])
            continue
        if message.get("content"):
            inputs.append({"role": message["role"], "content": message["content"]})
        for call in message.get("tool_calls") or []:
            inputs.append({"type": "function_call", "call_id": call["id"], **call["function"]})
    body = {"model": model, "input": inputs, "store": False, "include": ["reasoning.encrypted_content"]}
    if tools:
        body["tools"] = [{"type": "function", **tool["function"]} for tool in tools]
    if schema:
        body["text"] = {"format": {"type": "json_schema", "name": "decision", "strict": strict_schema, "schema": schema}}
    return body


class BatchModelTransport:
    def __init__(self, manager: BatchLifecycleManager) -> None:
        self.manager = manager
        self.operations = BatchOperationStore(manager.store)

    def invoke(
        self, *, task_id: str, phase: str, step: int, model: str, provider: str,
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        max_output_tokens: int = 4096,
        strict_schema: bool = True,
    ) -> dict[str, Any]:
        endpoint = self.manager.config.endpoint
        body = request_body(model, endpoint, messages, tools or [], schema, strict_schema=strict_schema)
        body["max_output_tokens" if endpoint == "/v1/responses" else "max_completion_tokens"] = max_output_tokens
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        operation_id = f"batch:{task_id}:{phase}:{step}"
        operation = self.operations.get(operation_id)
        if operation:
            saved_items = json.loads(operation["payload"])["items"]
            token_key = "max_output_tokens" if endpoint == "/v1/responses" else "max_completion_tokens"
            # Already submitted output limits are part of the authorized
            # receipt; later accounting must not mutate a replayed request.
            body[token_key] = saved_items[0]["body"][token_key]
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        item = BatchItemRequest(custom_id=operation_id, body=body, url=endpoint, origin_task_id=task_id)
        validate_batch_selection(self.manager.config, model=model, provider=provider, endpoint=endpoint, items=[item])
        # submit_batch persists intent and reconciles the same operation on retry.
        if not operation or not operation["batch_id"]:
            job = self.manager.submit_batch([item], model=model, provider=provider, endpoint=endpoint,
                                            task_id=task_id, idempotency_key=operation_id, mode="openai")
        else:
            if operation["fingerprint"] != digest:
                raise ValueError("Saved Batch request does not match the current model input")
            job = self.manager.store.get_job(operation["batch_id"])
            if job is None:
                raise ValueError("Batch operation has no durable job receipt")
        if job.status in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"OpenAI Batch {job.batch_id} ended with status {job.status}: {job.error_message}")
        if job.status not in {"completed", "results_imported", "verified"}:
            raise BatchPending(PendingBatchOperation(operation_id=operation_id, batch_id=job.batch_id, task_id=task_id, phase=phase))
        _, results = self.manager.finalize_batch(job.batch_id)
        if len(results) != 1 or results[0].custom_id != operation_id or not results[0].ok:
            raise ValueError("Batch model operation did not return its one successful correlated result")
        return response_message(results[0].response_body)
