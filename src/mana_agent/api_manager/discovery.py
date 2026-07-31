"""Operation retrieval plus strict model-selected routing validation."""

from __future__ import annotations

import re
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from mana_agent.api_manager.errors import AmbiguousOperationError, RequestValidationError
from mana_agent.api_manager.models import (
    ApiOperation,
    OperationRiskLevel,
    RoutingEvidence,
    StrictModel,
)
from mana_agent.api_manager.registry import ApiIntegrationRegistry
from mana_agent.config.settings import mana_home


class OperationCandidate(StrictModel):
    integration_id: str
    integration_name: str
    operation_id: str
    operation_name: str
    method: str
    path: str
    description: str
    tags: tuple[str, ...]
    risk_level: OperationRiskLevel
    retrieval_score: float = Field(ge=0)
    matched_terms: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    prior_success_count: int = 0


class ApiRouteDecision(StrictModel):
    source_decision_id: str = Field(min_length=1)
    task_intent: str = Field(min_length=1)
    workflow: str = Field(
        pattern=r"^(documentation_ingestion|integration_configuration|operation_search|request_preview|request_execution)$"
    )
    integration_id: str = ""
    operation_id: str = ""
    confidence: float = Field(ge=0, le=1)
    matched_terms: tuple[str, ...] = ()
    required_missing_inputs: tuple[str, ...] = ()
    reason: str = Field(min_length=1)
    safe_to_continue: bool
    complete: bool = False


class ApiOperationDiscovery:
    def __init__(
        self,
        registry: ApiIntegrationRegistry,
        *,
        evidence_path: Path | None = None,
    ) -> None:
        self.registry = registry
        self.evidence_path = evidence_path or (
            mana_home() / "api_manager" / "routing_evidence.jsonl"
        )
        self._lock = threading.RLock()

    def search(self, query: str, *, limit: int = 10) -> list[OperationCandidate]:
        """Retrieve candidates; this never chooses or executes an operation."""
        terms = tuple(
            sorted(
                {
                    token.lower()
                    for token in re.findall(r"[A-Za-z0-9_.:-]{2,}", query)
                    if len(token) >= 2
                }
            )
        )
        candidates: list[OperationCandidate] = []
        prior = self._matching_evidence(terms)
        for integration in self.registry.list(include_disabled=False):
            for operation in integration.operations:
                searchable = " ".join(
                    [
                        integration.name,
                        integration.description,
                        operation.operation_id,
                        operation.name,
                        operation.description,
                        operation.method.value,
                        operation.path,
                        *operation.tags,
                        *(parameter.name for parameter in operation.parameters),
                    ]
                ).lower()
                matched = tuple(term for term in terms if term in searchable)
                if terms and not matched:
                    continue
                required = tuple(
                    [
                        *(
                            f"{parameter.location.value}:{parameter.name}"
                            for parameter in operation.parameters
                            if parameter.required
                        ),
                        *(["body"] if operation.request_body and operation.request_body.required else []),
                    ]
                )
                prior_count = prior.get(
                    (integration.integration_id, operation.operation_id),
                    0,
                )
                score = len(matched) / max(1, len(terms)) + min(0.25, prior_count * 0.025)
                candidates.append(
                    OperationCandidate(
                        integration_id=integration.integration_id,
                        integration_name=integration.name,
                        operation_id=operation.operation_id,
                        operation_name=operation.name,
                        method=operation.method.value,
                        path=operation.path,
                        description=operation.description,
                        tags=operation.tags,
                        risk_level=operation.risk_level,
                        retrieval_score=score,
                        matched_terms=matched,
                        required_inputs=required,
                        prior_success_count=prior_count,
                    )
                )
        candidates.sort(
            key=lambda item: (
                -item.retrieval_score,
                item.integration_name.lower(),
                item.operation_id,
            )
        )
        return candidates[: max(1, min(limit, 50))]

    def record_success(
        self,
        *,
        task_intent: str,
        integration_id: str,
        operation_id: str,
    ) -> None:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_terms": sorted(
                set(re.findall(r"[A-Za-z0-9_.:-]{2,}", task_intent.lower()))
            )[:100],
            "integration_id": integration_id,
            "operation_id": operation_id,
            "outcome": "success",
        }
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock, self.evidence_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")

    def _matching_evidence(
        self,
        terms: tuple[str, ...],
    ) -> dict[tuple[str, str], int]:
        if not terms or not self.evidence_path.exists():
            return {}
        try:
            lines = self.evidence_path.read_text(encoding="utf-8").splitlines()[-1000:]
        except OSError:
            return {}
        counts: dict[tuple[str, str], int] = {}
        requested = set(terms)
        for line in lines:
            try:
                row = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if row.get("outcome") != "success" or not requested.intersection(row.get("task_terms") or ()):
                continue
            key = (str(row.get("integration_id") or ""), str(row.get("operation_id") or ""))
            if all(key):
                counts[key] = counts.get(key, 0) + 1
        return counts

    def validate_decision(
        self,
        decision: ApiRouteDecision | dict[str, Any],
        *,
        candidates: list[OperationCandidate],
        minimum_confidence: float = 0.65,
    ) -> RoutingEvidence:
        selected = ApiRouteDecision.model_validate(decision)
        if not selected.safe_to_continue:
            raise RequestValidationError(
                "The model API-routing decision marked the task unsafe. No API action was executed."
            )
        if not selected.integration_id or not selected.operation_id:
            raise RequestValidationError(
                "The model API-routing decision did not select an integration and operation. "
                "No fallback operation was selected."
            )
        candidate = next(
            (
                item
                for item in candidates
                if item.integration_id == selected.integration_id
                and item.operation_id == selected.operation_id
            ),
            None,
        )
        if candidate is None:
            raise RequestValidationError(
                "The model selected an operation outside the retrieved candidate set."
            )
        if selected.confidence < minimum_confidence:
            raise AmbiguousOperationError(
                "The model API-routing decision is below the execution confidence threshold.",
                details={
                    "confidence": selected.confidence,
                    "minimum_confidence": minimum_confidence,
                    "candidates": [
                        {
                            "integration_id": item.integration_id,
                            "operation_id": item.operation_id,
                            "name": item.operation_name,
                        }
                        for item in candidates[:5]
                    ],
                },
            )
        if selected.required_missing_inputs:
            raise RequestValidationError(
                "The model operation decision identifies missing required inputs.",
                details={"missing": list(selected.required_missing_inputs)},
            )
        return RoutingEvidence(
            integration_id=selected.integration_id,
            operation_id=selected.operation_id,
            confidence=selected.confidence,
            matched_terms=selected.matched_terms,
            required_missing_inputs=selected.required_missing_inputs,
            risk_classification=candidate.risk_level,
            reason=selected.reason,
            source_decision_id=selected.source_decision_id,
        )
