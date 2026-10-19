"""Central authority for the Global Self-Learning Memory System."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any

from mana_agent.memory.learning.engine import MemoryLearningEngine
from mana_agent.memory.learning.models import (
    LearningCategory,
    LearningHistoryEntry,
    LearningMemory,
    LearningScope,
    LearningStatus,
    utc_now,
)
from mana_agent.memory.learning.store import LearningMemoryStore, default_learning_db_path

logger = logging.getLogger(__name__)

# Patterns for scrubbing credentials before memory persistence
SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{10,}", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|token|secret|password|passwd|client[_-]?secret)\s*[:=]\s*['\"]?)[A-Za-z0-9_\-\.]{8,}(['\"]?)", re.IGNORECASE),
    re.compile(r"(https?://[^:/@\s]+:)[^@\s/]+(@)", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    if not text:
        return ""
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(r"\1[REDACTED]\2" if r"\2" in pattern.pattern else r"\1[REDACTED]", redacted)
    return redacted


class GlobalMemory:
    """Central authority for learned mistakes, corrections, and reusable execution knowledge."""

    def __init__(
        self,
        store: LearningMemoryStore | None = None,
        engine: MemoryLearningEngine | None = None,
        *,
        default_project_id: str | None = None,
        default_environment_fingerprint: str | None = None,
    ) -> None:
        self.store = store or LearningMemoryStore()
        self.engine = engine or MemoryLearningEngine()
        self.default_project_id = default_project_id
        self.default_environment_fingerprint = default_environment_fingerprint

    def record_failure(
        self,
        *,
        attempted_action: str,
        error: str,
        category: LearningCategory | str = LearningCategory.GENERAL,
        context: str = "",
        scope: LearningScope | str = LearningScope.PROJECT,
        execution_id: str = "",
        project_id: str | None = None,
        environment_fingerprint: str | None = None,
        evidence_references: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LearningMemory | None:
        """Record an observed execution failure. If it represents a real mistake, create or update a learning record."""
        try:
            clean_action = redact_secrets(attempted_action).strip()
            clean_error = redact_secrets(error).strip()
            clean_context = redact_secrets(context).strip()

            analysis = self.engine.analyze_mistake(
                attempted_action=clean_action,
                error=clean_error,
                category=category,
                context=clean_context,
                metadata=metadata,
            )

            if not analysis.get("is_meaningful_mistake"):
                logger.debug("Skipping non-meaningful or transient error: %s", analysis.get("reason"))
                return None

            cat = category if isinstance(category, LearningCategory) else LearningCategory(category)
            scp = scope if isinstance(scope, LearningScope) else LearningScope(scope)
            proj_id = project_id or self.default_project_id
            env_fp = environment_fingerprint or self.default_environment_fingerprint
            sig = analysis.get("failure_signature")

            # Check if this failure was already recorded
            existing = self.store.find_by_signature(sig, project_id=proj_id) if sig else []
            if not existing:
                existing = self.store.find_by_action_exact(clean_action, project_id=proj_id, category=cat)

            if existing:
                # Merge duplicate failure
                memory = existing[0]
                prev_conf = memory.confidence
                memory.failure_count += 1
                memory.error = clean_error
                if clean_context and clean_context not in memory.context:
                    memory.context = f"{memory.context}\n{clean_context}".strip()
                if evidence_references:
                    for ref in evidence_references:
                        if ref not in memory.evidence_references:
                            memory.evidence_references.append(ref)
                memory.updated_at = utc_now()
                # If typo analysis found a definite correction, enrich existing if missing
                if not memory.correction and analysis.get("correction"):
                    memory.correction = analysis.get("correction")
                    memory.root_cause = analysis.get("root_cause")
                    memory.prevention_rule = analysis.get("prevention_rule")
                saved = self.store.save(memory)

                self.store.record_history(
                    LearningHistoryEntry(
                        id=uuid.uuid4().hex,
                        memory_id=saved.id,
                        action="failure_reoccurred",
                        execution_id=execution_id,
                        evidence=clean_error[:200],
                        previous_confidence=prev_conf,
                        new_confidence=saved.confidence,
                        details={"failure_count": saved.failure_count},
                    )
                )
                return saved

            # Create new learning record
            mem_id = uuid.uuid4().hex
            has_correction = bool(analysis.get("correction"))
            initial_status = LearningStatus.CANDIDATE
            initial_confidence = 0.8 if (has_correction and not analysis.get("is_hypothesis")) else 0.5

            memory = LearningMemory(
                id=mem_id,
                category=cat,
                scope=scp,
                context=clean_context,
                attempted_action=clean_action,
                failure_signature=sig,
                error=clean_error,
                root_cause=analysis.get("root_cause"),
                correction=analysis.get("correction"),
                prevention_rule=analysis.get("prevention_rule"),
                status=initial_status,
                confidence=initial_confidence,
                source_execution_id=execution_id,
                evidence_references=list(evidence_references or []),
                project_id=proj_id,
                environment_fingerprint=env_fp,
                is_hypothesis=analysis.get("is_hypothesis", False),
                metadata=dict(metadata or {}),
            )
            saved = self.store.save(memory)

            self.store.record_history(
                LearningHistoryEntry(
                    id=uuid.uuid4().hex,
                    memory_id=saved.id,
                    action="created",
                    execution_id=execution_id,
                    evidence=clean_error[:200],
                    previous_confidence=0.0,
                    new_confidence=saved.confidence,
                    details={"is_hypothesis": saved.is_hypothesis, "has_correction": has_correction},
                )
            )
            return saved

        except Exception as exc:
            logger.warning("GlobalMemory.record_failure failed safely: %s", exc)
            return None

    def record_correction(
        self,
        memory_id: str,
        correction: str,
        *,
        root_cause: str | None = None,
        prevention_rule: str | None = None,
        evidence: str = "",
        execution_id: str = "",
    ) -> LearningMemory | None:
        """Add or update the verified correction and prevention rule for an existing learning record."""
        try:
            memory = self.store.get(memory_id)
            if not memory:
                return None

            clean_corr = redact_secrets(correction).strip()
            clean_rc = redact_secrets(root_cause or "").strip() or memory.root_cause
            clean_rule = redact_secrets(prevention_rule or "").strip()
            if not clean_rule:
                clean_rule = f"Use '{clean_corr}' instead of '{memory.attempted_action}'."

            prev_conf = memory.confidence
            memory.correction = clean_corr
            memory.root_cause = clean_rc
            memory.prevention_rule = clean_rule
            memory.confidence = min(1.0, memory.confidence + 0.2)
            memory.updated_at = utc_now()
            if memory.is_hypothesis:
                memory.is_hypothesis = False

            saved = self.store.save(memory)
            self.store.record_history(
                LearningHistoryEntry(
                    id=uuid.uuid4().hex,
                    memory_id=saved.id,
                    action="correction_added",
                    execution_id=execution_id,
                    evidence=redact_secrets(evidence)[:200],
                    previous_confidence=prev_conf,
                    new_confidence=saved.confidence,
                    details={"correction": clean_corr, "rule": clean_rule},
                )
            )
            return saved
        except Exception as exc:
            logger.warning("GlobalMemory.record_correction failed safely: %s", exc)
            return None

    def verify_learning(
        self,
        memory_id: str,
        success: bool,
        *,
        evidence: str = "",
        execution_id: str = "",
    ) -> LearningMemory | None:
        """Reinforce or dispute a learning rule based on verified outcome evidence."""
        try:
            memory = self.store.get(memory_id)
            if not memory:
                return None

            prev_conf = memory.confidence
            now = utc_now()
            memory.last_applied_at = now
            memory.updated_at = now

            if success:
                memory.success_count += 1
                memory.confidence = min(1.0, memory.confidence + 0.2)
                if memory.status == LearningStatus.CANDIDATE:
                    memory.status = LearningStatus.VERIFIED
                action_name = "verified"

                # Check for promotion from PROJECT to GLOBAL
                if self.engine.evaluate_promotion(memory):
                    memory.scope = LearningScope.GLOBAL
                    action_name = "promoted_to_global"
                    logger.info("Promoted learning memory %s to GLOBAL scope", memory.id)
            else:
                memory.failure_count += 1
                memory.confidence = max(0.0, memory.confidence - 0.25)
                if memory.confidence < 0.35:
                    memory.status = LearningStatus.DISPUTED
                action_name = "disputed"

            saved = self.store.save(memory)
            self.store.record_history(
                LearningHistoryEntry(
                    id=uuid.uuid4().hex,
                    memory_id=saved.id,
                    action=action_name,
                    execution_id=execution_id,
                    evidence=redact_secrets(evidence)[:200],
                    previous_confidence=prev_conf,
                    new_confidence=saved.confidence,
                    details={
                        "success": success,
                        "status": saved.status.value,
                        "scope": saved.scope.value,
                    },
                )
            )
            return saved
        except Exception as exc:
            logger.warning("GlobalMemory.verify_learning failed safely: %s", exc)
            return None

    def retrieve_relevant(
        self,
        action_or_query: str,
        *,
        category: LearningCategory | str | None = None,
        scope: LearningScope | str | None = None,
        project_id: str | None = None,
        environment_fingerprint: str | None = None,
        limit: int = 5,
    ) -> list[LearningMemory]:
        """Retrieve applicable previous learnings before executing an action."""
        try:
            clean_query = redact_secrets(action_or_query).strip()
            if not clean_query:
                return []

            proj_id = project_id or self.default_project_id
            env_fp = environment_fingerprint or self.default_environment_fingerprint

            results: list[LearningMemory] = []
            seen_ids: set[str] = set()

            # 1. Exact action match
            exact_matches = self.store.find_by_action_exact(
                clean_query,
                project_id=proj_id,
                category=category,
            )
            for m in exact_matches:
                if m.is_applicable(proj_id, env_fp) and m.id not in seen_ids:
                    results.append(m)
                    seen_ids.add(m.id)

            # 2. Tokenized search
            search_results = self.store.search(
                clean_query,
                category=category,
                scope=scope,
                project_id=proj_id,
                environment_fingerprint=env_fp,
                active_only=True,
                limit=limit * 2,
            )
            for m in search_results:
                if m.is_applicable(proj_id, env_fp) and m.id not in seen_ids:
                    results.append(m)
                    seen_ids.add(m.id)

            return results[: max(1, limit)]
        except Exception as exc:
            logger.warning("GlobalMemory.retrieve_relevant failed safely: %s", exc)
            return []

    def search(
        self,
        query: str,
        *,
        category: LearningCategory | str | None = None,
        scope: LearningScope | str | None = None,
        project_id: str | None = None,
        limit: int = 10,
    ) -> list[LearningMemory]:
        try:
            return self.store.search(
                redact_secrets(query),
                category=category,
                scope=scope,
                project_id=project_id or self.default_project_id,
                active_only=False,
                limit=limit,
            )
        except Exception as exc:
            logger.warning("GlobalMemory.search failed safely: %s", exc)
            return []

    def invalidate(self, memory_id: str, reason: str = "") -> LearningMemory | None:
        """Mark a memory record as disputed/invalidated."""
        try:
            memory = self.store.get(memory_id)
            if not memory:
                return None
            prev_conf = memory.confidence
            memory.status = LearningStatus.DISPUTED
            memory.confidence = 0.0
            memory.updated_at = utc_now()
            saved = self.store.save(memory)
            self.store.record_history(
                LearningHistoryEntry(
                    id=uuid.uuid4().hex,
                    memory_id=saved.id,
                    action="invalidated",
                    evidence=redact_secrets(reason),
                    previous_confidence=prev_conf,
                    new_confidence=0.0,
                    details={"reason": reason},
                )
            )
            return saved
        except Exception as exc:
            logger.warning("GlobalMemory.invalidate failed safely: %s", exc)
            return None

    def supersede(
        self,
        old_memory_id: str,
        new_memory_id: str,
        reason: str = "",
    ) -> tuple[LearningMemory | None, LearningMemory | None]:
        """Replace an obsolete memory rule with a superseding rule."""
        try:
            old_mem = self.store.get(old_memory_id)
            new_mem = self.store.get(new_memory_id)
            if not old_mem or not new_mem:
                return (old_mem, new_mem)

            now = utc_now()
            old_mem.status = LearningStatus.SUPERSEDED
            old_mem.superseded_by = new_memory_id
            old_mem.updated_at = now

            new_mem.replaces_memory_id = old_memory_id
            new_mem.updated_at = now

            saved_old = self.store.save(old_mem)
            saved_new = self.store.save(new_mem)

            self.store.record_history(
                LearningHistoryEntry(
                    id=uuid.uuid4().hex,
                    memory_id=saved_old.id,
                    action="superseded",
                    evidence=redact_secrets(reason),
                    previous_confidence=old_mem.confidence,
                    new_confidence=old_mem.confidence,
                    details={"superseded_by": new_memory_id, "reason": reason},
                )
            )
            return (saved_old, saved_new)
        except Exception as exc:
            logger.warning("GlobalMemory.supersede failed safely: %s", exc)
            return (None, None)

    def get_learning_history(self, memory_id: str) -> list[dict[str, Any]]:
        try:
            return [e.to_dict() for e in self.store.get_history(memory_id)]
        except Exception as exc:
            logger.warning("GlobalMemory.get_learning_history failed safely: %s", exc)
            return []

    def format_retrieved_rules(self, memories: list[LearningMemory]) -> str:
        """Format retrieved learning memories into concise markdown instructions for agents."""
        if not memories:
            return ""
        lines = ["## Past Learnings & Prevention Rules (Avoid Repeating Past Mistakes):"]
        for m in memories:
            if m.prevention_rule:
                lines.append(f"- **[Rule - {m.category.value}]**: {m.prevention_rule} (Confidence: {m.confidence:.1f})")
            elif m.correction:
                lines.append(f"- **[Mistake Avoidance]**: When attempting '{m.attempted_action}', use '{m.correction}' instead.")
            elif m.error:
                lines.append(f"- **[Past Failure]**: Avoid '{m.attempted_action}' - failed with: {m.error[:120]}")
        return "\n".join(lines)


# Global singleton cache
_GLOBAL_MEMORY_INSTANCE: GlobalMemory | None = None


def get_global_memory(
    *,
    db_path: str | Path | None = None,
    project_id: str | None = None,
    environment_fingerprint: str | None = None,
) -> GlobalMemory:
    """Retrieve or initialize the centralized GlobalMemory authority."""
    global _GLOBAL_MEMORY_INSTANCE
    if _GLOBAL_MEMORY_INSTANCE is None or db_path is not None:
        store = LearningMemoryStore(db_path=db_path)
        _GLOBAL_MEMORY_INSTANCE = GlobalMemory(
            store=store,
            default_project_id=project_id,
            default_environment_fingerprint=environment_fingerprint,
        )
    if project_id and not _GLOBAL_MEMORY_INSTANCE.default_project_id:
        _GLOBAL_MEMORY_INSTANCE.default_project_id = project_id
    if environment_fingerprint and not _GLOBAL_MEMORY_INSTANCE.default_environment_fingerprint:
        _GLOBAL_MEMORY_INSTANCE.default_environment_fingerprint = environment_fingerprint
    return _GLOBAL_MEMORY_INSTANCE


def reset_global_memory() -> None:
    """Reset cached singleton instance (primarily for test isolation)."""
    global _GLOBAL_MEMORY_INSTANCE
    _GLOBAL_MEMORY_INSTANCE = None
