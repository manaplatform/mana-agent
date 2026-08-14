from __future__ import annotations

import json
import logging
import re
import uuid
from time import perf_counter
from typing import Any, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from mana_agent.multi_agent.runtime.compatibility import create_chat_model

from mana_agent.multi_agent.runtime.prompts import (
    CONVERSATION_SYSTEM_PROMPT,
    HUMAN_TEMPLATE,
    SYSTEM_PROMPT,
)
from mana_agent.multi_agent.runtime.run_logger import LlmRunLogger
from mana_agent.spirit.adapter import apply_spirit_instruction
from mana_agent.spirit.self_model import compose_runtime_self

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
    ) -> None:
        logger.debug("Initializing QnA chain with model=%s", model)
        self.llm = create_chat_model(
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            default_headers=default_headers,
        )
        self.model = model
        self.provider = str(provider or getattr(self.llm, "selected_provider", "") or "unknown")
        self.prompt = self._build_prompt()
        self.run_logger = LlmRunLogger()

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
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

    def update_model_assignment(self, provider: str, model: str, *, settings: Any | None = None) -> None:
        from mana_agent.config.inference_provider import resolve_inference_connection
        from mana_agent.config.settings import Settings

        governor = getattr(self.llm, "context_cost_governor", None)
        connection = resolve_inference_connection(settings or Settings(), provider=provider)
        self.llm = create_chat_model(
            api_key=connection.api_key,
            model=model,
            base_url=connection.base_url,
            provider=connection.provider,
            default_headers=connection.headers,
        )
        self.llm.context_cost_governor = governor
        self.model = model
        self.provider = connection.provider
        self.prompt = self._build_prompt()

    def run(self, question: str, context: str) -> str:
        logger.info("Invoking QnA chain")
        logger.debug("Prompt sizes: question_chars=%d context_chars=%d", len(question), len(context))
        chain = self.prompt | self.llm
        started = perf_counter()
        response = chain.invoke({"question": question, "context": context})
        elapsed_ms = (perf_counter() - started) * 1000
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
                    "response": str(response.content),
                }
            )
        logger.info("QnA chain completed in %.2fms", elapsed_ms)
        return str(response.content)

    def chat(
        self,
        question: str,
        *,
        runtime_self: Any | None = None,
        context_tools: Sequence[Any] | None = None,
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

        active_llm = self.llm
        if tool_map and hasattr(self.llm, "bind_tools") and callable(getattr(self.llm, "bind_tools", None)):
            try:
                active_llm = self.llm.bind_tools(list(tool_map.values()))
            except Exception:
                active_llm = self.llm

        messages: list[Any] = [
            SystemMessage(
                content=apply_spirit_instruction(CONVERSATION_SYSTEM_PROMPT, current)
            ),
            HumanMessage(content=question),
        ]

        max_retrieval_rounds = 2
        for _ in range(max_retrieval_rounds + 1):
            response = active_llm.invoke(messages)

            # Check for structured tool calls first
            tool_calls = getattr(response, "tool_calls", None)
            if isinstance(tool_calls, list) and tool_calls and tool_map:
                messages.append(response)
                for tc in tool_calls:
                    fn_name = tc.get("name")
                    args = tc.get("args") or {}
                    call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    if fn_name in tool_map:
                        tool = tool_map[fn_name]
                        try:
                            if hasattr(tool, "invoke") and callable(tool.invoke):
                                tool_res = tool.invoke(args)
                            elif callable(tool):
                                tool_res = tool(**args)
                            else:
                                tool_res = str(tool)
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
                        messages.append(ToolMessage(content=rendered_res, tool_call_id=call_id))
                continue

            # Check for text-based tool call strings emitted directly by model
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content_text = " ".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                content_text = str(content or "")

            match = re.search(r"\[Tool Call:\s*([a-zA-Z0-9_]+)\]", content_text, re.IGNORECASE)
            if match and tool_map:
                matched_tool_name = match.group(1)
                if matched_tool_name in tool_map:
                    tool = tool_map[matched_tool_name]
                    try:
                        if hasattr(tool, "invoke") and callable(tool.invoke):
                            tool_res = tool.invoke({"query": question, "max_turns": 3})
                        elif callable(tool):
                            tool_res = tool(query=question, max_turns=3)
                        else:
                            tool_res = str(tool)
                    except Exception as exc:
                        tool_res = f'{{"error": "{exc}"}}'
                    governor = getattr(self.llm, "context_cost_governor", None)
                    if governor is not None and getattr(governor, "enabled", False):
                        rendered_res = governor.prepare_tool_result(
                            tool_res,
                            tool_name=matched_tool_name,
                            tool_call_id=f"call_{uuid.uuid4().hex[:8]}",
                            turn_id="qna_chat",
                        )
                    else:
                        from mana_agent.context_cost.compression import normalize_permitted_result
                        norm = normalize_permitted_result(tool_res)
                        rendered_res = norm if isinstance(norm, str) else json.dumps(norm, ensure_ascii=False, default=str)
                    messages.append(AIMessage(content=content_text))
                    messages.append(
                        HumanMessage(
                            content=(
                                f"Retrieved context for `{matched_tool_name}`:\n{rendered_res}\n\n"
                                "Please answer the user's conversational request now using the context above."
                            )
                        )
                    )
                    continue

            return content_text.strip()

        return str(getattr(response, "content", response) or "").strip()
