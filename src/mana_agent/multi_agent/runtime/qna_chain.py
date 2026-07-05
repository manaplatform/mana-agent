from __future__ import annotations

import logging
from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from mana_agent.multi_agent.runtime.prompts import HUMAN_TEMPLATE, SYSTEM_PROMPT
from mana_agent.multi_agent.runtime.run_logger import LlmRunLogger

logger = logging.getLogger(__name__)


class QnAChain:
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        logger.debug("Initializing QnA chain with model=%s", model)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_TEMPLATE),
            ]
        )
        kwargs = {"api_key": api_key, "model": model}
        if base_url:
            kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**kwargs)
        self.model = model
        self.run_logger = LlmRunLogger()

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
