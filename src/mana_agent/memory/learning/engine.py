"""Learning and Correction Engine for mistake detection, root cause analysis, and rule formation."""

from __future__ import annotations

import difflib
import hashlib
import logging
import re
from typing import Any

from mana_agent.memory.learning.models import (
    LearningCategory,
    LearningMemory,
    LearningScope,
    LearningStatus,
)

logger = logging.getLogger(__name__)

# Transient error patterns that should NOT become persistent mistake rules
TRANSIENT_PATTERNS = [
    re.compile(r"\b(502|503|504)\s+(Bad Gateway|Service Unavailable|Gateway Timeout)\b", re.IGNORECASE),
    re.compile(r"\b(connection\s+reset|connection\s+refused|network\s+unreachable|temporary\s+failure)\b", re.IGNORECASE),
    re.compile(r"\b(rate\s+limit|too\s+many\s+requests|429)\b", re.IGNORECASE),
    re.compile(r"\b(operation\s+cancelled|task\s+was\s+cancelled|user_cancelled)\b", re.IGNORECASE),
    re.compile(r"\b(ETIMEDOUT|ECONNRESET|ECONNREFUSED)\b", re.IGNORECASE),
]

COMMON_CLI_TYPOS = {
    "bulid": "build",
    "biuld": "build",
    "buid": "build",
    "tset": "test",
    "tets": "test",
    "statuss": "status",
    "stauts": "status",
    "chechout": "checkout",
    "comit": "commit",
    "puush": "push",
    "pul": "pull",
    "isntall": "install",
    "instlal": "install",
    "intall": "install",
    "udpate": "update",
    "pyetst": "pytest",
    "pyest": "pytest",
}


class MemoryLearningEngine:
    """Analyzes execution outcomes, classifies mistakes, forms corrections, and detects evolution triggers."""

    def __init__(self, promotion_threshold: int = 3) -> None:
        self.promotion_threshold = promotion_threshold

    def is_transient_error(self, error: str) -> bool:
        """Identify transient failures (outages, network disconnects, rate limits, user cancellation)."""
        clean_error = error.strip()
        if not clean_error:
            return False
        return any(pattern.search(clean_error) for pattern in TRANSIENT_PATTERNS)

    def extract_failure_signature(
        self,
        *,
        category: LearningCategory | str,
        attempted_action: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Compute a deterministic signature to correlate identical or near-identical mistakes."""
        meta = metadata or {}
        cat_str = category.value if isinstance(category, LearningCategory) else str(category)

        if cat_str == LearningCategory.COMMAND.value:
            cmd = attempted_action.strip()
            # Normalize whitespace
            cmd_norm = " ".join(cmd.split())
            # Check for known exit code in meta or error
            exit_code = meta.get("exit_code")
            return f"cmd:{cmd_norm}" + (f":exit_{exit_code}" if exit_code is not None else "")

        if cat_str == LearningCategory.API.value:
            method = meta.get("method") or "HTTP"
            endpoint = meta.get("endpoint") or meta.get("url") or attempted_action.split("?")[0]
            status_code = meta.get("status_code", "")
            return f"api:{method}:{endpoint}:{status_code}"

        if cat_str == LearningCategory.TOOL.value:
            tool_name = meta.get("tool_name") or attempted_action.split()[0] if attempted_action else "tool"
            err_line = error.strip().splitlines()[0] if error.strip() else ""
            err_hash = hashlib.sha256(err_line.encode("utf-8")).hexdigest()[:8]
            return f"tool:{tool_name}:{err_hash}"

        # Generic hash signature
        norm = f"{cat_str}:{attempted_action}:{error[:100]}"
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

    def analyze_mistake(
        self,
        *,
        attempted_action: str,
        error: str,
        category: LearningCategory | str = LearningCategory.GENERAL,
        context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Examine a failure and generate candidate root causes, prevention rules, and signatures."""
        meta = metadata or {}
        cat = category if isinstance(category, LearningCategory) else LearningCategory(category)

        # 1. Check if transient
        if self.is_transient_error(error):
            return {
                "is_meaningful_mistake": False,
                "reason": "transient_error",
            }

        signature = self.extract_failure_signature(
            category=cat,
            attempted_action=attempted_action,
            error=error,
            metadata=meta,
        )

        root_cause: str | None = None
        correction: str | None = None
        prevention_rule: str | None = None
        is_hypothesis = False

        # 2. Check for command typos
        if cat in {LearningCategory.COMMAND, LearningCategory.GENERAL}:
            tokens = attempted_action.strip().split()
            fixed_tokens = list(tokens)
            typo_found = False
            for i, token in enumerate(tokens):
                low = token.lower()
                if low in COMMON_CLI_TYPOS:
                    fixed_tokens[i] = COMMON_CLI_TYPOS[low]
                    typo_found = True

            if typo_found:
                corrected_cmd = " ".join(fixed_tokens)
                root_cause = f"Typo detected in command '{attempted_action}'"
                correction = corrected_cmd
                prevention_rule = f"Use '{corrected_cmd}' instead of '{attempted_action}'."
            else:
                is_hypothesis = True
                root_cause = f"Command failed: {error.strip().splitlines()[-1] if error else 'non-zero exit'}"

        # 3. Check for API 4xx parameter or endpoint errors
        elif cat == LearningCategory.API:
            status_code = meta.get("status_code") or 0
            if status_code in {400, 422}:
                root_cause = f"Invalid API request payload or parameters (HTTP {status_code}): {error}"
                is_hypothesis = True
            elif status_code == 404:
                root_cause = f"API endpoint not found (HTTP 404): {attempted_action}"
                is_hypothesis = True
            elif status_code in {401, 403}:
                root_cause = f"API authentication/permission failure (HTTP {status_code}): {error}"
                is_hypothesis = True

        # 4. Tool failures
        elif cat == LearningCategory.TOOL:
            root_cause = f"Tool execution failed: {error}"
            is_hypothesis = True

        else:
            is_hypothesis = True
            root_cause = f"Execution error: {error}"

        return {
            "is_meaningful_mistake": True,
            "failure_signature": signature,
            "root_cause": root_cause,
            "correction": correction,
            "prevention_rule": prevention_rule,
            "is_hypothesis": is_hypothesis,
        }

    def correlate_and_correct(
        self,
        failed_memory: LearningMemory,
        successful_action: str,
        evidence: str = "",
    ) -> dict[str, Any]:
        """Correlate a subsequent successful operation with a previous failed attempt to formulate a verified rule."""
        failed_action = failed_memory.attempted_action.strip()
        success_action = successful_action.strip()

        if failed_action == success_action:
            # Re-running the exact same action successfully implies prior failure was transient or environment healed
            return {
                "correlated": True,
                "correction": success_action,
                "root_cause": failed_memory.root_cause or "Transient condition resolved on retry",
                "prevention_rule": f"Ensure environment prerequisites are met before executing '{success_action}'.",
                "confidence_delta": 0.2,
            }

        similarity = difflib.SequenceMatcher(None, failed_action, success_action).ratio()
        root_cause = f"Operation '{failed_action}' failed; succeeded with '{success_action}'."
        rule = f"When attempting '{failed_action}', use '{success_action}' instead."

        return {
            "correlated": True,
            "correction": success_action,
            "root_cause": root_cause,
            "prevention_rule": rule,
            "confidence_delta": 0.4 if similarity > 0.5 else 0.2,
        }

    def evaluate_promotion(self, memory: LearningMemory) -> bool:
        """Determine if a PROJECT memory has enough evidence to generalize into GLOBAL scope."""
        if memory.scope != LearningScope.PROJECT:
            return False
        if memory.status != LearningStatus.VERIFIED:
            return False
        if memory.success_count < self.promotion_threshold:
            return False

        # Reject if action or rule mentions absolute local paths or specific user dirs
        text = f"{memory.attempted_action} {memory.correction or ''} {memory.prevention_rule or ''}"
        if re.search(r"(/[a-zA-Z0-9_\-\.]+/){2,}", text) or re.search(r"([A-Za-z]:\\[a-zA-Z0-9_\-\.\\]+)", text):
            return False

        # CLI commands, standard API endpoints, and generic tools are eligible for global promotion
        return memory.category in {
            LearningCategory.COMMAND,
            LearningCategory.API,
            LearningCategory.TOOL,
            LearningCategory.GENERAL,
        }

    def detect_contradiction(self, active_memories: list[LearningMemory], proposed_rule: str) -> LearningMemory | None:
        """Check if a proposed prevention rule contradicts an existing verified rule."""
        clean_prop = proposed_rule.strip().lower()
        for mem in active_memories:
            if not mem.is_active() or not mem.prevention_rule:
                continue
            clean_exist = mem.prevention_rule.strip().lower()
            # Simple direct negation or inverted correction detection
            if (
                f"instead of '{mem.correction}'" in clean_prop
                or (mem.correction and f"use '{mem.attempted_action}'" in clean_prop)
            ):
                return mem
        return None
