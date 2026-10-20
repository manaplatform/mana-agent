"""Batch-backed structured decisions for post-coding integration gates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.messages import AIMessage

from mana_agent.batch.transport import BatchModelTransport
from mana_agent.batch.policy import BatchPending


class BatchDecisionModel:
    def __init__(self, manager: Any, *, task_id: str, model: str, provider: str,
                 supervisor: Any, attempt_id: str, lease_token: str, schema: Any = None) -> None:
        self.manager, self.task_id, self.model, self.provider = manager, task_id, model, provider
        self.supervisor, self.attempt_id, self.lease_token = supervisor, attempt_id, lease_token
        self.schema = schema

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "BatchDecisionModel":
        return BatchDecisionModel(self.manager, task_id=self.task_id, model=self.model, provider=self.provider,
            supervisor=self.supervisor, attempt_id=self.attempt_id, lease_token=self.lease_token, schema=schema)

    def invoke(self, messages: Any) -> Any:
        from mana_agent.execution.batch_loop import BatchTaskLoop, batch_output_limit
        transport = BatchModelTransport(self.manager)
        serialized = [{"role": {"human": "user", "ai": "assistant"}.get(message.type, message.type),
                       "content": message.content} for message in messages]
        schema = self.schema.model_json_schema() if self.schema else None
        digest = hashlib.sha256(json.dumps({"messages": serialized, "schema": schema}, sort_keys=True).encode()).hexdigest()[:20]
        phase = "integration-" + digest
        loop = BatchTaskLoop(self.manager, supervisor=self.supervisor,
            attempt_id=self.attempt_id, lease_token=self.lease_token)
        loop._validate_authority(self.task_id)
        saved_run = loop.store.load_run(self.task_id)
        pending = (saved_run or {}).get("gateway_pending_operation")
        if pending and pending["operation_id"] != f"batch:{self.task_id}:{phase}:0":
            raise RuntimeError("Saved Batch integration context changed; reconcile the pending decision before submitting another")
        if saved_run and saved_run["model_turns"] >= saved_run.get("limits", {}).get("max_model_turns", 120):
            raise RuntimeError("Batch task model-turn budget exhausted")
        output_limit = batch_output_limit(self.supervisor, self.task_id, self.provider, self.model, serialized, [], schema)
        try:
            response = transport.invoke(task_id=self.task_id, phase=phase, step=0, model=self.model,
                provider=self.provider, messages=serialized, schema=schema, strict_schema=False, max_output_tokens=output_limit)
        except BatchPending:
            raise
        except Exception:
            if saved_run:
                loop._account_failed_result(saved_run, phase, operation_step=0)
            raise
        run = loop.store.load_run(self.task_id)
        if run:
            operation_id = f"batch:{self.task_id}:{phase}:0"
            if operation_id not in run["accounted_operations"]:
                saved_step = run["step"]
                run["step"] = 0
                loop._account_result(run, phase)
                run["step"] = saved_step
                run["model_turns"] += 1
            if run.get("gateway_pending_operation", {}).get("operation_id") == operation_id:
                run.pop("gateway_pending_operation", None)
            loop.store.save_run(self.task_id, run)
        if self.schema:
            return self.schema.model_validate_json(response["content"])
        return AIMessage(content=response["content"])
