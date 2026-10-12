"""OpenAI Decisions API client for gpt-6-luna via /v1/decisions."""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

import httpx

from mana_agent.config.user_config import get_setting
from mana_agent.engine_selection.models import (
    DecisionsAnswer,
    DecisionsChoiceItem,
    DecisionsResult,
    DecisionsScoreLevel,
    Question,
)

logger = logging.getLogger(__name__)

DEFAULT_DECISIONS_MODEL = "gpt-6-luna"
DEFAULT_DECISIONS_TIMEOUT_SECONDS = 30.0


class DecisionsError(RuntimeError):
    """Base error for Decisions API operations."""


class DecisionsRefusalError(DecisionsError):
    """Raised when the model refuses to answer a Decisions question."""

    def __init__(self, message: str, *, name: str = "", refusal: str = "") -> None:
        super().__init__(message)
        self.name = name
        self.refusal = refusal


class DecisionsTimeoutError(DecisionsError):
    """Raised when a Decisions API call times out."""


class DecisionsAPIError(DecisionsError):
    """Raised when the Decisions API endpoint returns an HTTP error."""

    def __init__(self, message: str, *, status_code: int = 500, response_text: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class DecisionsMalformedResponseError(DecisionsError):
    """Raised when the Decisions API response structure cannot be parsed."""


class DecisionsClient:
    """Client for executing predicate, choice, and score operations on /v1/decisions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_DECISIONS_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = str(api_key or get_setting("OPENAI_API_KEY", "") or "").strip()
        raw_base = str(
            base_url
            or get_setting("MANA_DECISIONS_BASE_URL", "")
            or get_setting("OPENAI_BASE_URL", "https://api.openai.com/v1")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.base_url = raw_base
        self.endpoint = f"{raw_base}/decisions" if not raw_base.endswith("/decisions") else raw_base
        self.model = str(
            model or get_setting("MANA_DECISIONS_MODEL", DEFAULT_DECISIONS_MODEL) or DEFAULT_DECISIONS_MODEL
        ).strip()
        self.timeout = timeout
        self._external_client = http_client

    def decide(
        self,
        *,
        input_data: Any,
        questions: list[Question | dict[str, Any]],
        model: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> DecisionsResult:
        """Call /v1/decisions with input and questions."""
        target_model = model or self.model
        formatted_input = self._format_input(input_data)
        formatted_questions = [
            q.to_dict() if isinstance(q, Question) else q for q in questions
        ]

        payload = {
            "model": target_model,
            "input": formatted_input,
            "questions": formatted_questions,
        }

        if not self.api_key:
            raise DecisionsAPIError("OpenAI API key is required to call Decisions API.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if extra_headers:
            headers.update(extra_headers)

        started = time.perf_counter()
        logger.debug(
            "decisions.request endpoint=%s model=%s question_count=%d",
            self.endpoint,
            target_model,
            len(formatted_questions),
        )

        try:
            if self._external_client is not None:
                response = self._external_client.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.endpoint,
                        json=payload,
                        headers=headers,
                    )
        except httpx.TimeoutException as exc:
            logger.error("decisions.timeout endpoint=%s timeout=%.1fs", self.endpoint, self.timeout)
            raise DecisionsTimeoutError(f"Decisions API request timed out after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            logger.error("decisions.network_error error=%s", exc)
            raise DecisionsAPIError(f"Network error calling Decisions API: {exc}") from exc

        duration = time.perf_counter() - started

        if response.status_code >= 400:
            logger.error(
                "decisions.http_error status=%d response=%s",
                response.status_code,
                response.text[:200],
            )
            raise DecisionsAPIError(
                f"Decisions API returned HTTP {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
                response_text=response.text,
            )

        try:
            data = response.json()
        except Exception as exc:
            logger.error("decisions.malformed_json error=%s text=%s", exc, response.text[:200])
            raise DecisionsMalformedResponseError(
                f"Decisions API response is not valid JSON: {response.text[:200]}"
            ) from exc

        answers = self._parse_answers(data)
        logger.debug(
            "decisions.response latency=%.3fs answer_count=%d",
            duration,
            len(answers),
        )

        return DecisionsResult(
            model=target_model,
            answers=answers,
            raw_response=data,
            latency_seconds=duration,
        )

    def predicate(
        self,
        *,
        input_data: Any,
        instructions: str,
        name: str = "predicate_eval",
        **kwargs: Any,
    ) -> DecisionsAnswer:
        """Execute a single boolean predicate evaluation."""
        q = Question(type="predicate", name=name, instructions=instructions)
        result = self.decide(input_data=input_data, questions=[q], **kwargs)
        if not result.answers:
            raise DecisionsMalformedResponseError("No answer returned for predicate question")
        answer = result.answers[0]
        if answer.is_refusal:
            raise DecisionsRefusalError(
                f"Decisions API refused question '{name}': {answer.refusal}",
                name=name,
                refusal=str(answer.refusal or ""),
            )
        return answer

    def choice(
        self,
        *,
        input_data: Any,
        instructions: str,
        choices: list[str | dict[str, Any] | DecisionsChoiceItem],
        name: str = "choice_eval",
        **kwargs: Any,
    ) -> DecisionsAnswer:
        """Execute a single choice classification."""
        choice_items = [
            c if isinstance(c, DecisionsChoiceItem)
            else DecisionsChoiceItem(value=c) if isinstance(c, str)
            else DecisionsChoiceItem(
                value=str(c.get("value") or c.get("name") or c.get("label") or ""),
                label=str(c.get("label") or ""),
                description=str(c.get("description") or ""),
            )
            for c in choices
        ]
        q = Question(type="choice", name=name, instructions=instructions, choices=choice_items)
        result = self.decide(input_data=input_data, questions=[q], **kwargs)
        if not result.answers:
            raise DecisionsMalformedResponseError("No answer returned for choice question")
        answer = result.answers[0]
        if answer.is_refusal:
            raise DecisionsRefusalError(
                f"Decisions API refused choice '{name}': {answer.refusal}",
                name=name,
                refusal=str(answer.refusal or ""),
            )
        return answer

    def score(
        self,
        *,
        input_data: Any,
        instructions: str,
        levels: list[dict[str, Any] | DecisionsScoreLevel],
        name: str = "score_eval",
        **kwargs: Any,
    ) -> DecisionsAnswer:
        """Execute a single ordered scoring evaluation."""
        level_items = [
            l if isinstance(l, DecisionsScoreLevel)
            else DecisionsScoreLevel(
                label=str(l.get("label") or l.get("name") or ""),
                description=str(l.get("description") or ""),
                value=l.get("value"),
            )
            for l in levels
        ]
        q = Question(type="score", name=name, instructions=instructions, levels=level_items)
        result = self.decide(input_data=input_data, questions=[q], **kwargs)
        if not result.answers:
            raise DecisionsMalformedResponseError("No answer returned for score question")
        answer = result.answers[0]
        if answer.is_refusal:
            raise DecisionsRefusalError(
                f"Decisions API refused score '{name}': {answer.refusal}",
                name=name,
                refusal=str(answer.refusal or ""),
            )
        return answer

    def batch(
        self,
        *,
        input_data: Any,
        questions: list[Question | dict[str, Any]],
        **kwargs: Any,
    ) -> list[DecisionsAnswer]:
        """Execute multiple independent evaluations in a single batch request."""
        result = self.decide(input_data=input_data, questions=questions, **kwargs)
        return result.answers

    def _format_input(self, input_data: Any) -> Any:
        if isinstance(input_data, (str, list, dict)):
            return input_data
        return str(input_data or "")

    def _parse_answers(self, data: dict[str, Any]) -> list[DecisionsAnswer]:
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, list):
            raise DecisionsMalformedResponseError("Missing 'answers' array in Decisions API response")

        parsed: list[DecisionsAnswer] = []
        for item in raw_answers:
            if not isinstance(item, dict):
                continue
            ans_type = str(item.get("type") or "").strip().lower()
            name = str(item.get("name") or "")

            if ans_type == "refusal":
                parsed.append(
                    DecisionsAnswer(
                        type="refusal",
                        name=name,
                        refusal=str(item.get("refusal") or item.get("message") or "Refused"),
                    )
                )
            elif ans_type == "predicate":
                # boolean or probability
                prob = item.get("probability")
                res = item.get("result")
                if res is None and prob is not None:
                    res = float(prob) >= 0.5
                conf = float(item.get("confidence") or (prob if prob is not None else 1.0))
                parsed.append(
                    DecisionsAnswer(
                        type="predicate",
                        name=name,
                        result=bool(res),
                        probability=float(prob) if prob is not None else None,
                        confidence=conf,
                    )
                )
            elif ans_type == "choice":
                chosen = str(item.get("choice") or "").strip()
                conf = float(item.get("confidence", 1.0) or 1.0)
                probs = item.get("probabilities", [])
                parsed.append(
                    DecisionsAnswer(
                        type="choice",
                        name=name,
                        choice=chosen,
                        confidence=conf,
                        probabilities=list(probs) if isinstance(probs, list) else [],
                    )
                )
            elif ans_type == "score":
                score_val = item.get("score")
                conf = float(item.get("confidence", 1.0) or 1.0)
                probs = item.get("probabilities", [])
                parsed.append(
                    DecisionsAnswer(
                        type="score",
                        name=name,
                        score=float(score_val) if score_val is not None else None,
                        confidence=conf,
                        probabilities=list(probs) if isinstance(probs, list) else [],
                    )
                )
            else:
                # Generic fallback parsing
                parsed.append(
                    DecisionsAnswer(
                        type=ans_type or "unknown",
                        name=name,
                        result=item.get("result"),
                        choice=item.get("choice"),
                        score=item.get("score"),
                        confidence=float(item.get("confidence", 1.0) or 1.0),
                    )
                )

        return parsed
