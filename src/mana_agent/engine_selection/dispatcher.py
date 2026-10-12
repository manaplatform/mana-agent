"""Central Smart Dispatcher with executeAuto / execute_auto engine routing."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

from mana_agent.evals.ids import stable_hash
from mana_agent.engine_selection.decisions_client import (
    DecisionsClient,
    DecisionsError,
)
from mana_agent.engine_selection.models import (
    DecisionsAnswer,
    EngineSelectionResult,
    EngineType,
    ExecutionResult,
    Operation,
    OperationKind,
    Question,
)
from mana_agent.engine_selection.selector import EngineSelector

logger = logging.getLogger(__name__)


class SmartDispatcher:
    """Central orchestration dispatcher that routes and executes operations via the optimal engine.

    Guarantees:
    - Decisions is optional and selected automatically only when natively supported.
    - Never calls Decisions merely to check eligibility.
    - Deterministic execution bypasses AI completely.
    - Shares decisions across callers to avoid duplicate AI requests.
    - Bounded fallback if a supported Decisions call fails.
    """

    def __init__(
        self,
        *,
        selector: EngineSelector | None = None,
        decisions_client: DecisionsClient | None = None,
        default_llm: Any = None,
    ) -> None:
        self.selector = selector or EngineSelector()
        self._decisions_client = decisions_client
        self.default_llm = default_llm
        self._decision_cache: dict[str, ExecutionResult] = {}
        self._cache_lock = threading.Lock()

    @property
    def decisions_client(self) -> DecisionsClient:
        if self._decisions_client is None:
            self._decisions_client = DecisionsClient()
        return self._decisions_client

    def execute_auto(
        self,
        operation: Operation,
        context: Any = None,
        *,
        llm: Any = None,
        decisions_client: DecisionsClient | None = None,
        force_refresh: bool = False,
    ) -> ExecutionResult:
        """Inspect, select engine, execute, and return normalized ExecutionResult."""
        effective_client = decisions_client or self._decisions_client or self.decisions_client
        effective_llm = llm if llm is not None else self.default_llm

        # Step 1: Check shared cache to prevent duplicate AI calls
        cache_key = self._cache_key(operation, context)
        if not force_refresh:
            with self._cache_lock:
                cached = self._decision_cache.get(cache_key)
                if cached is not None:
                    logger.debug("execute_auto.cache_hit key=%s engine=%s", cache_key, cached.engine_used)
                    return ExecutionResult(
                        success=cached.success,
                        engine_used=cached.engine_used,
                        result=cached.result,
                        confidence=cached.confidence,
                        reasoning=cached.reasoning,
                        fallback_occurred=cached.fallback_occurred,
                        fallback_reason=cached.fallback_reason,
                        cached=True,
                        metadata=dict(cached.metadata),
                    )

        # Step 2: Check whether deterministic execution is sufficient
        if operation.is_deterministic():
            logger.debug("execute_auto.deterministic_direct name=%s", operation.name)
            res = self._execute_deterministic(operation, context)
            exec_result = ExecutionResult(
                success=True,
                engine_used=EngineType.DETERMINISTIC,
                result=res,
                confidence=1.0,
                reasoning="Executed directly via native application code without AI.",
                metadata={"deterministic": True},
            )
            self._store_cache(cache_key, exec_result)
            return exec_result

        # Step 3 & 4: Capability detection and engine selection
        selection: EngineSelectionResult = self.selector.select_engine(operation, context)
        logger.debug(
            "execute_auto.engine_selected name=%s engine=%s eligible=%s reason=%s",
            operation.name,
            selection.selected_engine,
            selection.is_decisions_eligible,
            selection.reason,
        )

        # Step 5 & 6 & 7: Execute using chosen engine
        engine = selection.selected_engine
        exec_result: ExecutionResult

        if engine in {
            EngineType.DECISIONS_PREDICATE,
            EngineType.DECISIONS_CHOICE,
            EngineType.DECISIONS_SCORE,
            EngineType.DECISIONS_BATCH,
        }:
            exec_result = self._execute_decisions_with_fallback(
                operation,
                context,
                client=effective_client,
                llm=effective_llm,
                selected_engine=engine,
            )

        elif engine == EngineType.RESPONSES_STRUCTURED:
            exec_result = self._execute_structured(operation, context, llm=effective_llm)

        elif engine == EngineType.FUNCTION_CALLING:
            exec_result = self._execute_function_calling(operation, context, llm=effective_llm)

        elif engine == EngineType.REASONING_ENGINE:
            exec_result = self._execute_reasoning(operation, context, llm=effective_llm)

        elif engine == EngineType.GENERATION_ENGINE:
            exec_result = self._execute_generation(operation, context, llm=effective_llm)

        elif engine == EngineType.CONVERSATION_LLM:
            exec_result = self._execute_conversation(operation, context, llm=effective_llm)

        elif engine == EngineType.COMPATIBLE_FALLBACK:
            exec_result = self._execute_fallback_engine(
                operation,
                context,
                llm=effective_llm,
                fallback_reason=selection.reason,
            )

        else:
            exec_result = ExecutionResult(
                success=False,
                engine_used=engine,
                result=None,
                reasoning=f"Unsupported engine type: {engine}",
            )

        self._store_cache(cache_key, exec_result)
        return exec_result

    def executeAuto(
        self,
        operation: Operation,
        context: Any = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Alias for execute_auto (camelCase per specification)."""
        return self.execute_auto(operation, context, **kwargs)

    def share_decision(self, operation: Operation, result: ExecutionResult, context: Any = None) -> None:
        """Manually share a previously computed decision across downstream components."""
        key = self._cache_key(operation, context)
        self._store_cache(key, result)

    def clear_decision_cache(self) -> None:
        """Clear cached decisions."""
        with self._cache_lock:
            self._decision_cache.clear()

    # -------------------------------------------------------------------------
    # Internal execution methods
    # -------------------------------------------------------------------------

    def _execute_deterministic(self, operation: Operation, context: Any) -> Any:
        if operation.handler is not None:
            try:
                return operation.handler(operation.input, context=context)
            except TypeError:
                return operation.handler(operation.input)
        return operation.input

    def _execute_decisions_with_fallback(
        self,
        operation: Operation,
        context: Any,
        *,
        client: DecisionsClient,
        llm: Any,
        selected_engine: EngineType,
    ) -> ExecutionResult:
        try:
            if selected_engine == EngineType.DECISIONS_PREDICATE:
                ans: DecisionsAnswer = client.predicate(
                    input_data=operation.input,
                    instructions=operation.instructions,
                    name=operation.name or "predicate_eval",
                )
                return ExecutionResult(
                    success=True,
                    engine_used=EngineType.DECISIONS_PREDICATE,
                    result=ans.result,
                    confidence=ans.confidence,
                    metadata={"probability": ans.probability, "answer_name": ans.name},
                )

            if selected_engine == EngineType.DECISIONS_CHOICE:
                ans = client.choice(
                    input_data=operation.input,
                    instructions=operation.instructions,
                    choices=operation.choices,
                    name=operation.name or "choice_eval",
                )
                return ExecutionResult(
                    success=True,
                    engine_used=EngineType.DECISIONS_CHOICE,
                    result=ans.choice,
                    confidence=ans.confidence,
                    metadata={"probabilities": ans.probabilities, "answer_name": ans.name},
                )

            if selected_engine == EngineType.DECISIONS_SCORE:
                ans = client.score(
                    input_data=operation.input,
                    instructions=operation.instructions,
                    levels=operation.levels,
                    name=operation.name or "score_eval",
                )
                return ExecutionResult(
                    success=True,
                    engine_used=EngineType.DECISIONS_SCORE,
                    result=ans.score,
                    confidence=ans.confidence,
                    metadata={"probabilities": ans.probabilities, "answer_name": ans.name},
                )

            if selected_engine == EngineType.DECISIONS_BATCH:
                questions: list[Question] = []
                for idx, sub_op in enumerate(operation.sub_operations):
                    q_name = sub_op.name or f"sub_op_{idx}"
                    if sub_op.kind == OperationKind.PREDICATE:
                        questions.append(Question(type="predicate", name=q_name, instructions=sub_op.instructions))
                    elif sub_op.kind == OperationKind.CHOICE:
                        normalized_choices = self.selector._normalize_choices(sub_op.choices)
                        questions.append(
                            Question(type="choice", name=q_name, instructions=sub_op.instructions, choices=normalized_choices)
                        )
                    elif sub_op.kind == OperationKind.SCORE:
                        normalized_levels = self.selector._normalize_levels(sub_op.levels)
                        questions.append(
                            Question(type="score", name=q_name, instructions=sub_op.instructions, levels=normalized_levels)
                        )

                answers = client.batch(input_data=operation.input, questions=questions)
                batch_results = [
                    a.result if a.type == "predicate" else (a.choice if a.type == "choice" else a.score)
                    for a in answers
                ]
                return ExecutionResult(
                    success=True,
                    engine_used=EngineType.DECISIONS_BATCH,
                    result=batch_results,
                    confidence=sum(a.confidence for a in answers) / max(1, len(answers)),
                    metadata={"answers": [a.to_dict() for a in answers]},
                )

        except DecisionsError as exc:
            logger.warning(
                "decisions.execution_failed engine=%s error=%s. Triggering safe bounded fallback.",
                selected_engine,
                exc,
            )
            # Safe bounded fallback
            return self._execute_fallback_engine(
                operation,
                context,
                llm=llm,
                fallback_reason=f"Decisions API failure: {exc}",
            )
        except Exception as exc:
            logger.warning(
                "decisions.unexpected_error engine=%s error=%s. Triggering safe bounded fallback.",
                selected_engine,
                exc,
            )
            return self._execute_fallback_engine(
                operation,
                context,
                llm=llm,
                fallback_reason=f"Unexpected error in Decisions execution: {exc}",
            )

    def _execute_structured(self, operation: Operation, context: Any, *, llm: Any) -> ExecutionResult:
        schema = operation.output_schema
        if llm is None:
            return ExecutionResult(
                success=False,
                engine_used=EngineType.RESPONSES_STRUCTURED,
                result=None,
                reasoning="No LLM provided for structured output extraction.",
            )

        with_struct = getattr(llm, "with_structured_output", None)
        if callable(with_struct) and schema is not None:
            try:
                runner = with_struct(schema)
                res = runner.invoke(self._prompt_messages(operation))
                return ExecutionResult(
                    success=True,
                    engine_used=EngineType.RESPONSES_STRUCTURED,
                    result=res,
                    confidence=1.0,
                )
            except Exception as exc:
                logger.warning("structured_output.invoke_failed error=%s", exc)

        # Fallback to standard invoke
        return self._invoke_llm_json(operation, llm=llm, engine_used=EngineType.RESPONSES_STRUCTURED)

    def _execute_function_calling(self, operation: Operation, context: Any, *, llm: Any) -> ExecutionResult:
        if llm is None:
            return ExecutionResult(
                success=False,
                engine_used=EngineType.FUNCTION_CALLING,
                result=None,
                reasoning="No LLM provided for function calling.",
            )

        # Function calling via tools binding if supported
        bind_tools = getattr(llm, "bind_tools", None)
        if callable(bind_tools) and operation.tool_schema:
            try:
                tool_llm = bind_tools([operation.tool_schema])
                msg = tool_llm.invoke(self._prompt_messages(operation))
                tool_calls = getattr(msg, "tool_calls", None) or []
                return ExecutionResult(
                    success=True,
                    engine_used=EngineType.FUNCTION_CALLING,
                    result=tool_calls[0] if tool_calls else msg,
                    confidence=1.0,
                )
            except Exception as exc:
                logger.warning("function_calling.invoke_failed error=%s", exc)

        # Fallback to standard invoke
        return self._invoke_llm_json(operation, llm=llm, engine_used=EngineType.FUNCTION_CALLING)

    def _execute_reasoning(self, operation: Operation, context: Any, *, llm: Any) -> ExecutionResult:
        return self._invoke_llm_text(operation, llm=llm, engine_used=EngineType.REASONING_ENGINE)

    def _execute_generation(self, operation: Operation, context: Any, *, llm: Any) -> ExecutionResult:
        return self._invoke_llm_text(operation, llm=llm, engine_used=EngineType.GENERATION_ENGINE)

    def _execute_conversation(self, operation: Operation, context: Any, *, llm: Any) -> ExecutionResult:
        return self._invoke_llm_text(operation, llm=llm, engine_used=EngineType.CONVERSATION_LLM)

    def _execute_fallback_engine(
        self,
        operation: Operation,
        context: Any,
        *,
        llm: Any,
        fallback_reason: str,
    ) -> ExecutionResult:
        """Bounded fallback to existing compatible engine."""
        logger.info(
            "execute_auto.fallback operation=%s kind=%s reason=%s",
            operation.name,
            operation.kind,
            fallback_reason,
        )

        if llm is None:
            return ExecutionResult(
                success=False,
                engine_used=EngineType.COMPATIBLE_FALLBACK,
                result=None,
                fallback_occurred=True,
                fallback_reason=fallback_reason,
                reasoning="No compatible engine (LLM) available for fallback execution.",
            )

        # For predicate: prompt LLM for boolean evaluation
        if operation.kind == OperationKind.PREDICATE:
            prompt = (
                f"Evaluate the following question ({operation.name}) based on the evidence.\n"
                f"Question: {operation.instructions}\n"
                f"Evidence: {operation.input}\n"
                f"Answer with JSON only: {{\"result\": true|false, \"reason\": \"...\"}}"
            )
            try:
                res = self._invoke_llm_json_raw(prompt, llm=llm)
                bool_val = bool(res.get("result", False))
                return ExecutionResult(
                    success=True,
                    engine_used=EngineType.COMPATIBLE_FALLBACK,
                    result=bool_val,
                    fallback_occurred=True,
                    fallback_reason=fallback_reason,
                    confidence=0.8,
                    reasoning=str(res.get("reason", "")),
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    engine_used=EngineType.COMPATIBLE_FALLBACK,
                    result=None,
                    fallback_occurred=True,
                    fallback_reason=f"{fallback_reason} -> LLM fallback failed: {exc}",
                )

        # For choice: prompt LLM for option classification
        if operation.kind == OperationKind.CHOICE:
            normalized_choices = self.selector._normalize_choices(operation.choices)
            choice_values = [c.value for c in normalized_choices]
            prompt = (
                f"Choose the single best option ({operation.name}) for this input from the available choices.\n"
                f"Instructions: {operation.instructions}\n"
                f"Available Choices: {json.dumps(choice_values)}\n"
                f"Input: {operation.input}\n"
                f"Answer with JSON only: {{\"choice\": \"<selected_choice>\", \"reason\": \"...\"}}"
            )
            try:
                res = self._invoke_llm_json_raw(prompt, llm=llm)
                selected = str(res.get("choice") or "").strip()
                if selected not in choice_values and choice_values:
                    # Match nearest or first valid
                    for cv in choice_values:
                        if cv.lower() in selected.lower():
                            selected = cv
                            break
                    if selected not in choice_values:
                        selected = choice_values[0]
                return ExecutionResult(
                    success=True,
                    engine_used=EngineType.COMPATIBLE_FALLBACK,
                    result=selected,
                    fallback_occurred=True,
                    fallback_reason=fallback_reason,
                    confidence=0.8,
                    reasoning=str(res.get("reason", "")),
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    engine_used=EngineType.COMPATIBLE_FALLBACK,
                    result=None,
                    fallback_occurred=True,
                    fallback_reason=f"{fallback_reason} -> LLM fallback failed: {exc}",
                )

        # For score: prompt LLM for rubric scoring
        if operation.kind == OperationKind.SCORE:
            levels = self.selector._normalize_levels(operation.levels)
            labels = [l.label for l in levels]
            prompt = (
                f"Score the input ({operation.name}) against the following ordered levels.\n"
                f"Instructions: {operation.instructions}\n"
                f"Levels: {json.dumps([l.to_dict() for l in levels])}\n"
                f"Input: {operation.input}\n"
                f"Answer with JSON only: {{\"score\": <numeric_score_or_level_index>, \"level\": \"<label>\", \"reason\": \"...\"}}"
            )
            try:
                res = self._invoke_llm_json_raw(prompt, llm=llm)
                score_val = res.get("score")
                if score_val is None and "level" in res:
                    lvl_lbl = str(res["level"]).strip()
                    if lvl_lbl in labels:
                        score_val = float(labels.index(lvl_lbl))
                return ExecutionResult(
                    success=True,
                    engine_used=EngineType.COMPATIBLE_FALLBACK,
                    result=float(score_val if score_val is not None else 0.0),
                    fallback_occurred=True,
                    fallback_reason=fallback_reason,
                    confidence=0.8,
                    reasoning=str(res.get("reason", "")),
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    engine_used=EngineType.COMPATIBLE_FALLBACK,
                    result=None,
                    fallback_occurred=True,
                    fallback_reason=f"{fallback_reason} -> LLM fallback failed: {exc}",
                )

        # General text/structured invoke
        return self._invoke_llm_text(
            operation,
            llm=llm,
            engine_used=EngineType.COMPATIBLE_FALLBACK,
            fallback_occurred=True,
            fallback_reason=fallback_reason,
        )

    # -------------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------------

    def _prompt_messages(self, operation: Operation) -> list[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        instructions = operation.instructions or "Execute the requested operation."
        evidence = json.dumps(operation.input, ensure_ascii=False) if not isinstance(operation.input, str) else operation.input
        return [
            SystemMessage(content=instructions),
            HumanMessage(content=evidence),
        ]

    def _invoke_llm_text(
        self,
        operation: Operation,
        *,
        llm: Any,
        engine_used: EngineType,
        fallback_occurred: bool = False,
        fallback_reason: str = "",
    ) -> ExecutionResult:
        if llm is None:
            return ExecutionResult(
                success=False,
                engine_used=engine_used,
                result=None,
                reasoning="No LLM provided.",
            )
        try:
            from mana_agent.utils.text import extract_model_text

            msg = llm.invoke(self._prompt_messages(operation))
            text = extract_model_text(getattr(msg, "content", msg))
            return ExecutionResult(
                success=True,
                engine_used=engine_used,
                result=text,
                fallback_occurred=fallback_occurred,
                fallback_reason=fallback_reason,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                engine_used=engine_used,
                result=None,
                fallback_occurred=fallback_occurred,
                fallback_reason=f"{fallback_reason} -> error: {exc}" if fallback_reason else str(exc),
                reasoning=str(exc),
            )

    def _invoke_llm_json(self, operation: Operation, *, llm: Any, engine_used: EngineType) -> ExecutionResult:
        if llm is None:
            return ExecutionResult(
                success=False,
                engine_used=engine_used,
                result=None,
                reasoning="No LLM provided.",
            )
        try:
            from mana_agent.gateway.entry_routing import _extract_json
            from mana_agent.utils.text import extract_model_text

            msg = llm.invoke(self._prompt_messages(operation))
            raw = extract_model_text(getattr(msg, "content", msg))
            try:
                data = _extract_json(raw)
            except Exception:
                data = {"raw": raw, "tool_name": operation.tool_name or operation.name}
            return ExecutionResult(
                success=True,
                engine_used=engine_used,
                result=data,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                engine_used=engine_used,
                result=None,
                reasoning=str(exc),
            )

    def _invoke_llm_json_raw(self, prompt: str, *, llm: Any) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage
        from mana_agent.gateway.entry_routing import _extract_json
        from mana_agent.utils.text import extract_model_text

        msg = llm.invoke([HumanMessage(content=prompt)])
        raw = extract_model_text(getattr(msg, "content", msg))
        try:
            return _extract_json(raw)
        except Exception:
            raw_str = str(raw).strip()
            if raw_str.lower() in ("true", "yes", "1"):
                return {"result": True}
            if raw_str.lower() in ("false", "no", "0"):
                return {"result": False}
            return {"result": raw_str, "choice": raw_str, "text": raw_str}

    def _cache_key(self, operation: Operation, context: Any) -> str:
        ctx_val = getattr(context, "session_id", None) or getattr(context, "turn_id", None) or context
        payload = {
            "kind": operation.kind.value,
            "name": operation.name,
            "instructions": operation.instructions,
            "input": operation.input if isinstance(operation.input, (str, int, float, bool)) else stable_hash(operation.input),
            "choices": [c if isinstance(c, (str, int, float)) else str(c) for c in operation.choices],
            "context": str(ctx_val),
        }
        return stable_hash(payload)

    def _store_cache(self, key: str, result: ExecutionResult) -> None:
        with self._cache_lock:
            # Bound cache size to prevent runaway memory
            if len(self._decision_cache) > 2000:
                self._decision_cache.clear()
            self._decision_cache[key] = result


# Module-level singleton and functions for convenient usage across the repository
_GLOBAL_DISPATCHER = SmartDispatcher()


def execute_auto(
    operation: Operation,
    context: Any = None,
    *,
    llm: Any = None,
    decisions_client: DecisionsClient | None = None,
    force_refresh: bool = False,
) -> ExecutionResult:
    """Execute an operation using automatic capability-aware engine selection."""
    return _GLOBAL_DISPATCHER.execute_auto(
        operation,
        context,
        llm=llm,
        decisions_client=decisions_client,
        force_refresh=force_refresh,
    )


def executeAuto(
    operation: Operation,
    context: Any = None,
    **kwargs: Any,
) -> ExecutionResult:
    """Alias for execute_auto."""
    return execute_auto(operation, context, **kwargs)
