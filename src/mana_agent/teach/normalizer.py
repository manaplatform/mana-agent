"""Semantic event normalization and stable selector ranking."""

from __future__ import annotations

import json
from collections.abc import Iterable

from .models import RecordedEvent, SelectorCandidate


SELECTOR_STABILITY = {
    "playwright_role": 1.0,
    "automation_id": 0.96,
    "test_id": 0.94,
    "label": 0.9,
    "text": 0.76,
    "css": 0.7,
    "xpath": 0.45,
    "coordinate": 0.2,
}


def rank_selectors(candidates: list[SelectorCandidate]) -> list[SelectorCandidate]:
    for candidate in candidates:
        base = SELECTOR_STABILITY.get(candidate.type, 0.5)
        observations = candidate.successes + candidate.failures
        reliability = candidate.successes / observations if observations else candidate.confidence
        candidate.confidence = max(0, min(1, round(base * 0.6 + reliability * 0.4, 4)))
    return sorted(candidates, key=lambda item: (-item.confidence, item.type))


class SemanticNormalizer:
    """Reduces event noise without inventing actions."""

    def normalize(self, events: Iterable[RecordedEvent]) -> list[RecordedEvent]:
        normalized: list[RecordedEvent] = []
        previous_fingerprint = ""
        for event in sorted(events, key=lambda item: item.timestamp):
            fingerprint = json.dumps(
                {
                    "source": event.source.value,
                    "action": event.action,
                    "application": event.application.id,
                    "target": {
                        "role": event.target.role,
                        "name": event.target.name,
                        "label": event.target.label,
                        "automation_id": event.target.automation_id,
                        "selectors": sorted(
                            (
                                {"type": item.type, "value": item.value}
                                for item in event.target.selectors
                            ),
                            key=lambda item: json.dumps(item, sort_keys=True),
                        ),
                    },
                    "data": event.data,
                },
                sort_keys=True,
            )
            if fingerprint == previous_fingerprint:
                continue
            previous_fingerprint = fingerprint
            event.target.selectors = rank_selectors(event.target.selectors)
            normalized.append(event)
        return normalized
