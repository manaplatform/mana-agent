from __future__ import annotations

import json
import logging
import re
import uuid
from time import perf_counter
from typing import Any, Sequence

from mana_agent.llm.messages import AssistantMessage, UserMessage, SystemMessage, ToolResultMessage
from mana_agent.llm.prompts import MessageTemplate
from mana_agent.llm.engine import create_model

from mana_agent.multi_agent.runtime.prompts import (
    CONVERSATION_SYSTEM_PROMPT,
    HUMAN_TEMPLATE,
    SYSTEM_PROMPT,
)
from mana_agent.multi_agent.runtime.run_logger import LlmRunLogger
from mana_agent.spirit.adapter import apply_spirit_instruction
from mana_agent.spirit.self_model import compose_runtime_self
from mana_agent.utils.text import extract_model_text

logger = logging.getLogger(__name__)


class QnAChain:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        *,
        provider: str | None = None,
        default_headers: dict[str, str] | None = None,
        model_configuration: dict[str, Any] | None = None,
    ) -> None:
        logger.debug("Initializing QnA chain with model=%s", model)
        self.llm = create_model(
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            default_headers=default_headers,
            model_configuration=model_configuration,
        )
        self.model = model
        self.provider = str(provider or getattr(self.llm, "selected_provider", "") or "unknown")
        self.prompt = self._build_prompt()
        self.run_logger = LlmRunLogger()

    def _build_prompt(self) -> MessageTemplate:
        return MessageTemplate.from_messages(
            [
                (
                    "system",
                    apply_spirit_instruction(
                        SYSTEM_PROMPT,
                        compose_runtime_self(
                            agent_name="qna-agent",
                            agent_role="research",
                            provider=self.provider,
                            model=self.model,
                        ),
                    ),
                ),
                ("human", HUMAN_TEMPLATE),
            ]
        )

    def update_model_assignment(self, provider: str, model: str, *, settings: Any | None = None,
                                model_configuration: dict[str, Any] | None = None) -> None:
        from mana_agent.config.inference_provider import resolve_inference_connection
        from mana_agent.config.settings import Settings

        governor = getattr(self.llm, "context_cost_governor", None)
        connection = resolve_inference_connection(settings or Settings(), provider=provider)
        self.llm = create_model(
            api_key=connection.api_key,
            model=model,
            base_url=connection.base_url,
            provider=connection.provider,
            default_headers=connection.headers,
            model_configuration=model_configuration,
        )
        self.llm.context_cost_governor = governor
        self.model = model
        self.provider = connection.provider
        self.prompt = self._build_prompt()

    def run(self, question: str, context: str) -> str:
        logger.info("Invoking QnA chain")
        logger.debug("Prompt sizes: question_chars=%d context_chars=%d", len(question), len(context))
        rendered_prompt = self.prompt
        started = perf_counter()
        response = self.llm.generate_sync(rendered_prompt.render({"question": question, "context": context}))
        elapsed_ms = (perf_counter() - started) * 1000
        extracted_res = extract_model_text(getattr(response, "content", response))
        run_logger = getattr(self, "run_logger", None)
        if run_logger is not None:
            run_logger.log(
                {
                    "flow": "qna",
                    "model": getattr(self, "model", "unknown"),
                    "question_chars": len(question),
                    "context_chars": len(context),
                    "question": question,
                    "context": context,
                    "duration_ms": round(elapsed_ms, 3),
                    "response": extracted_res,
                }
            )
        logger.info("QnA chain completed in %.2fms", elapsed_ms)
        return extracted_res

    def chat(
        self,
        question: str,
        *,
        runtime_self: Any | None = None,
        context_tools: Sequence[Any] | None = None,
        recent_history: Sequence[Any] | None = None,
        multimodal_content: Any | None = None,
    ) -> str:
        """Answer from the session transcript after the routed Self is bound, executing bounded retrieval if needed."""
        current = runtime_self or compose_runtime_self(
            agent_name="conversation-agent",
            agent_role="conversation",
            provider=self.provider,
            model=self.model,
        )
        tools = list(context_tools or [])
        tool_map: dict[str, Any] = {
            t.name: t for t in tools if hasattr(t, "name") and isinstance(t.name, str)
        }

        active_llm = self.llm.with_tools(list(tool_map.values())) if tool_map else self.llm
        tool_gate = None
        if tool_map:
            from mana_agent.services.activity_context import current_activity_context
            from mana_agent.multi_agent.runtime.tool_call_gate import ToolCallGate
            from pathlib import Path
            context = current_activity_context()
            if context is None:
                raise RuntimeError("Conversation context tools require an authorized task activity context")
            tool_gate = ToolCallGate(root=Path.cwd(), tools=tool_map, task_id=context.task_id, repository_read_only=True)

        messages: list[Any] = [
            SystemMessage(
                content=apply_spirit_instruction(CONVERSATION_SYSTEM_PROMPT, current)
            ),
        ]
        if recent_history:
            from mana_agent.llm.messages import messages_from_records
            messages.extend(messages_from_records(recent_history))
        user_content = multimodal_content if multimodal_content is not None else question
        messages.append(UserMessage(content=user_content))

        continuation = None
        max_retrieval_rounds = 2
        for _ in range(max_retrieval_rounds + 1):
            response = active_llm.generate_sync(messages, continuation=continuation)
            continuation = getattr(response, "continuation", None)

            # Check for structured tool calls first
            tool_calls = getattr(response, "tool_calls", None)
            if isinstance(tool_calls, list) and tool_calls and tool_map:
                messages.append(response.message)
                for tc in tool_calls:
                    fn_name = tc.get("name")
                    args = tc.get("args") or {}
                    call_id = tc["id"]
                    if fn_name in tool_map:
                        tool = tool_map[fn_name]
                        try:
                            tool_res = tool_gate.invoke(fn_name, args, call_id=call_id)
                        except Exception as exc:
                            tool_res = f'{{"error": "{exc}"}}'
                        governor = getattr(self.llm, "context_cost_governor", None)
                        if governor is not None and getattr(governor, "enabled", False):
                            rendered_res = governor.prepare_tool_result(
                                tool_res,
                                tool_name=fn_name,
                                tool_call_id=call_id,
                                turn_id="qna_chat",
                            )
                        else:
                            from mana_agent.context_cost.compression import normalize_permitted_result
                            norm = normalize_permitted_result(tool_res)
                            rendered_res = norm if isinstance(norm, str) else json.dumps(norm, ensure_ascii=False, default=str)
                        messages.append(ToolResultMessage(content=rendered_res, tool_call_id=call_id, caller=tc.get("caller")))
                continue

            content_text = extract_model_text(getattr(response, "content", response))

            return content_text.strip()

        return extract_model_text(getattr(response, "content", response)).strip()
