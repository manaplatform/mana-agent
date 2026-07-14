"""Run-scoped evidence cache shared by parent agents and delegated workers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

from mana_agent.multi_agent.runtime.edit_scope import resolve_repo_path


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    evidence_id: str
    canonical_path: str
    fingerprint: str
    start_line: int | None
    end_line: int | None
    tool_name: str
    normalized_arguments: str
    purpose: str
    content: str
    created_at: float = field(default_factory=time.time)

    @property
    def full_file(self) -> bool:
        return self.start_line is None and self.end_line is None

    def covers(self, start_line: int | None, end_line: int | None) -> bool:
        if self.full_file:
            return True
        if start_line is None or end_line is None:
            return False
        return bool(
            self.start_line is not None
            and self.end_line is not None
            and self.start_line <= start_line
            and self.end_line >= end_line
        )


@dataclass(slots=True)
class RunMetrics:
    routing_model_calls: int = 0
    delegation_model_calls: int = 0
    unique_searches: int = 0
    unique_file_reads: int = 0
    cache_hits: int = 0
    deduplicated_jobs: int = 0
    mutations: int = 0
    mutation_retries: int = 0
    verification_commands: int = 0
    agent_messages: int = 0
    scope_escalations: int = 0
    queue_wait_ms: float = 0.0
    model_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    total_elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class EvidenceLedger:
    """Canonical, version-aware read and operation evidence for one run."""

    def __init__(self, repo_root: str | Path, *, metrics: RunMetrics | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.metrics = metrics or RunMetrics()
        self._by_path: dict[str, list[EvidenceReference]] = {}
        self._operations: dict[str, EvidenceReference] = {}

    def canonical_path(self, raw_path: str) -> str:
        resolution = resolve_repo_path(self.repo_root, raw_path)
        if not resolution.ok:
            raise FileNotFoundError(f"repository path cannot be resolved: {raw_path} ({resolution.reason})")
        return resolution.resolved_path

    def fingerprint(self, canonical_path: str) -> str:
        data = (self.repo_root / canonical_path).read_bytes()
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def normalize_arguments(arguments: dict[str, Any]) -> str:
        normalized: dict[str, Any] = {}
        for key, value in sorted(arguments.items()):
            if key in {"path", "file", "file_path"}:
                continue
            normalized[str(key)] = value
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)

    def get_read(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
        purpose: str = "",
    ) -> EvidenceReference | None:
        canonical = self.canonical_path(path)
        current = self.fingerprint(canonical)
        for item in reversed(self._by_path.get(canonical.casefold(), [])):
            if item.fingerprint == current and item.covers(start_line, end_line):
                self.metrics.cache_hits += 1
                return item
        return None

    def read_file(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
        purpose: str = "",
    ) -> tuple[EvidenceReference, bool]:
        cached = self.get_read(path, start_line=start_line, end_line=end_line, purpose=purpose)
        if cached is not None:
            return cached, True
        canonical = self.canonical_path(path)
        target = self.repo_root / canonical
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        if start_line is not None or end_line is not None:
            start = max(1, int(start_line or 1))
            end = max(start, int(end_line or len(lines)))
            content = "".join(lines[start - 1 : end])
            stored_start: int | None = start
            stored_end: int | None = min(end, len(lines))
        else:
            content = text
            stored_start = None
            stored_end = None
        fingerprint = self.fingerprint(canonical)
        args = self.normalize_arguments({"start_line": stored_start, "end_line": stored_end})
        evidence_id = "ev_" + hashlib.sha1(
            f"{canonical.casefold()}|{fingerprint}|{stored_start}|{stored_end}|read_file|{args}|{purpose}".encode()
        ).hexdigest()[:16]
        reference = EvidenceReference(
            evidence_id=evidence_id,
            canonical_path=canonical,
            fingerprint=fingerprint,
            start_line=stored_start,
            end_line=stored_end,
            tool_name="read_file",
            normalized_arguments=args,
            purpose=purpose,
            content=content,
        )
        self._by_path.setdefault(canonical.casefold(), []).append(reference)
        self._operations[evidence_id] = reference
        self.metrics.unique_file_reads += 1
        return reference, False

    def read_many(self, paths: Iterable[str], *, purpose: str = "") -> tuple[list[EvidenceReference], int]:
        references: list[EvidenceReference] = []
        cache_hits = 0
        for path in paths:
            reference, hit = self.read_file(path, purpose=purpose)
            references.append(reference)
            cache_hits += int(hit)
        return references, cache_hits

    def invalidate(self, changed_paths: Iterable[str]) -> None:
        """Invalidate only evidence for files affected by a mutation."""
        for raw in changed_paths:
            try:
                canonical = self.canonical_path(raw)
            except FileNotFoundError:
                canonical = str(raw).replace("\\", "/").lstrip("./")
            self._by_path.pop(canonical.casefold(), None)

    def references(self) -> list[EvidenceReference]:
        return list(self._operations.values())


__all__ = ["EvidenceLedger", "EvidenceReference", "RunMetrics"]
