from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
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

    def chat(self, question: str, *, runtime_self: Any | None = None) -> str:
        """Answer from the session transcript after the routed Self is bound."""
        current = runtime_self or compose_runtime_self(
            agent_name="conversation-agent",
            agent_role="conversation",
            provider=self.provider,
            model=self.model,
        )
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=apply_spirit_instruction(CONVERSATION_SYSTEM_PROMPT, current)
                ),
                HumanMessage(content=question),
            ]
        )
        return str(getattr(response, "content", response) or "").strip()
