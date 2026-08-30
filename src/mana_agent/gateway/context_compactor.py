"""Multi-tier context compactor and bounded routing capsule builder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from mana_agent.config.settings import Settings
from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.context_cost.models import AccountingSnapshot
from mana_agent.gateway.entry_routing import EntryRouteContext
from mana_agent.gateway.envelope import (
    ApprovalState,
    ConversationContextAvailability,
    ExecutionRecoveryState,
    IdentitySessionRelationship,
    MemoryAvailability,
    ModelCandidateCapacity,
    PreviousTurnPointers,
    RoutingExecutionEnvelope,
    build_routing_execution_envelope,
)
from mana_agent.workspaces.retention import RetentionPolicy


def default_accounting_snapshot(task_id: str = "", turn_id: str = "") -> AccountingSnapshot:
    """Create a safe default AccountingSnapshot when none is available."""
    return AccountingSnapshot(
        task_id=task_id,
        turn_id=turn_id,
        task_budget_tokens=100_000,
        task_consumed_tokens=0,
        task_reserved_tokens=0,
        task_remaining_tokens=100_000,
        turn_budget_tokens=None,
        turn_consumed_tokens=0,
        turn_remaining_tokens=None,
        verification_reserve_tokens=0,
        session_budget_tokens=None,
        session_consumed_tokens=0,
        session_remaining_tokens=None,
        cost_budget=None,
        cost_consumed=0.0,
        cost_remaining=None,
        active_reservations_count=0,
        status="ok",
    )


class ContextCategory(str, Enum):
    USER_REQUEST = "user_request"
    SYSTEM_PROMPT = "system_prompt"
    ROUTE_AVAILABILITY = "route_availability"
    EXECUTION_STATE = "execution_state"
    TASK_CANDIDATES = "task_candidates"
    ARTIFACT_EVIDENCE = "artifact_evidence"
    MEMORY_CANDIDATES = "memory_candidates"
    ACCOUNTING = "accounting"
    TOOLS_AND_CAPABILITIES = "tools_and_capabilities"
    LOGS_AND_TRACES = "logs_and_traces"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ContextComponentBreakdown:
    user_request: int = 0
    system_prompt: int = 0
    route_availability: int = 0
    execution_state: int = 0
    task_candidates: int = 0
    artifact_evidence: int = 0
    memory_candidates: int = 0
    accounting: int = 0
    tools_and_capabilities: int = 0
    logs_and_traces: int = 0
    other: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "user_request": self.user_request,
            "system_prompt": self.system_prompt,
            "route_availability": self.route_availability,
            "execution_state": self.execution_state,
            "task_candidates": self.task_candidates,
            "artifact_evidence": self.artifact_evidence,
            "memory_candidates": self.memory_candidates,
            "accounting": self.accounting,
            "tools_and_capabilities": self.tools_and_capabilities,
            "logs_and_traces": self.logs_and_traces,
            "other": self.other,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class CompactedRoutingContext:
    bounded_envelope: RoutingExecutionEnvelope
    bounded_context: EntryRouteContext
    raw_context_tokens: int
    compacted_context_tokens: int
    context_tokens_saved: int
    logs_excluded_tokens: int
    stale_records_pruned: int
    workspace_records_pruned: int
    repository_records_compacted: int
    routing_context_deficit_before_compaction: int
    routing_context_deficit_after_compaction: int
    breakdown: ContextComponentBreakdown
    attempted_compaction: bool
    remaining_oversized_categories: tuple[str, ...] = ()
    is_valid: bool = True
    deficit: int = 0
    diagnostic_details: dict[str, Any] = field(default_factory=dict)


class ContextCompactor:
    """Compacts routing context into a bounded capsule and calculates token budgets."""

    def __init__(
        self,
        settings: Settings | None = None,
        policy: RetentionPolicy | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.policy = policy or RetentionPolicy.from_settings(self.settings)

    def calculate_raw_breakdown(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        context: Any,
        envelope: Any | None = None,
        routes: Sequence[Mapping[str, Any]] = (),
    ) -> ContextComponentBreakdown:
        """Measure token counts for all context categories before compaction."""
        eff_envelope = envelope or getattr(context, "envelope", None)

        user_req_tokens = estimate_value_tokens(user_prompt)
        sys_prompt_tokens = estimate_value_tokens(system_prompt)
        routes_tokens = estimate_value_tokens(list(routes))

        exec_state_tokens = 0
        task_candidates_tokens = 0
        accounting_tokens = 0
        tools_tokens = 0
        logs_tokens = 0

        if eff_envelope is not None:
            exec_state = getattr(eff_envelope, "execution_state", None)
            if exec_state is not None:
                all_rec = getattr(exec_state, "all_recovery_candidates", ()) or ()
                rec_cand = getattr(exec_state, "recoverable_task_candidates", ()) or ()
                task_candidates_tokens += estimate_value_tokens(
                    list(all_rec) + list(rec_cand)
                )
                lane_dict = dict(getattr(exec_state, "lane_states", {}) or {})
                for lane_val in lane_dict.values():
                    if isinstance(lane_val, dict) and "logs" in lane_val:
                        logs_tokens += estimate_value_tokens(lane_val["logs"])
                if "logs" in lane_dict:
                    logs_tokens += estimate_value_tokens(lane_dict["logs"])

                exec_state_tokens = estimate_value_tokens({
                    "active_flow_id": getattr(exec_state, "active_flow_id", None),
                    "active_route": getattr(exec_state, "active_route", ""),
                    "lane_id": getattr(exec_state, "lane_id", ""),
                    "lane_states": lane_dict,
                    "pending_required_work": getattr(exec_state, "pending_required_work", False),
                    "pending_checkpoint_id": getattr(exec_state, "pending_checkpoint_id", None),
                })

            acct = getattr(eff_envelope, "accounting_snapshot", None)
            if acct is not None:
                accounting_tokens = estimate_value_tokens(
                    acct.as_dict() if hasattr(acct, "as_dict") else str(acct)
                )

        raw_art = getattr(context, "artifact_evidence", {}) or {}
        art_tokens = estimate_value_tokens(dict(raw_art) if isinstance(raw_art, dict) else raw_art)
        raw_mem = getattr(context, "memory_task_candidates", ()) or ()
        mem_tokens = estimate_value_tokens(list(raw_mem) if isinstance(raw_mem, (list, tuple)) else raw_mem)
        task_candidates_tokens += mem_tokens

        other_tokens = estimate_value_tokens({
            "session_id": getattr(context, "session_id", ""),
            "conversation_id": getattr(context, "conversation_id", ""),
            "turn_id": getattr(context, "turn_id", ""),
            "previous_route": getattr(context, "previous_route", ""),
            "authenticated_user_id": getattr(context, "authenticated_user_id", ""),
        })

        context_dict = (
            context.to_dict()
            if hasattr(context, "to_dict")
            else {
                "session_id": getattr(context, "session_id", ""),
                "conversation_id": getattr(context, "conversation_id", ""),
                "turn_id": getattr(context, "turn_id", ""),
                "previous_route": getattr(context, "previous_route", ""),
                "conversation_summary": getattr(context, "conversation_summary", ""),
                "artifact_evidence": raw_art,
                "memory_task_candidates": list(raw_mem) if isinstance(raw_mem, (list, tuple)) else raw_mem,
                "memory_capsules_enabled": getattr(context, "memory_capsules_enabled", False),
                "atomic_child": getattr(context, "atomic_child", False),
                "orchestration_parent_task_id": getattr(context, "orchestration_parent_task_id", ""),
                "authenticated_user_id": getattr(context, "authenticated_user_id", ""),
            }
        )

        sample_payload = {
            "user_prompt": str(user_prompt or "").strip(),
            "context": context_dict,
            "routes": list(routes),
            "routing_constraints": {
                "atomic_child": getattr(context, "atomic_child", False),
                "disallowed_routes": [],
                "orchestration_parent_task_id": str(getattr(context, "orchestration_parent_task_id", "")),
            },
        }

        total = sys_prompt_tokens + estimate_value_tokens(sample_payload) + logs_tokens

        return ContextComponentBreakdown(
            user_request=user_req_tokens,
            system_prompt=sys_prompt_tokens,
            route_availability=routes_tokens,
            execution_state=exec_state_tokens,
            task_candidates=task_candidates_tokens,
            artifact_evidence=art_tokens,
            memory_candidates=mem_tokens,
            accounting=accounting_tokens,
            tools_and_capabilities=tools_tokens,
            logs_and_traces=logs_tokens,
            other=other_tokens,
            total_tokens=total,
        )

    def compact_recovery_candidates(
        self,
        candidates: Sequence[Any],
        *,
        max_candidates: int = 8,
    ) -> list[dict[str, Any]]:
        """Compact recovery candidate records into concise pointers."""
        compacted: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()

        # Sort: active / waiting / blockers first, then newest completed
        def sort_key(item: Any) -> tuple[int, str]:
            if isinstance(item, dict):
                state = str(item.get("state") or "")
                has_checkpoint = bool(item.get("checkpoint_id"))
                updated = str(item.get("updated_at") or item.get("created_at") or "")
            else:
                state = str(getattr(item, "state", "") or "")
                has_checkpoint = bool(getattr(item, "checkpoint_id", None))
                updated = str(getattr(item, "updated_at", getattr(item, "created_at", "")) or "")
            is_active = state not in {"completed", "cancelled", "failed"}
            priority = 0 if is_active else (1 if has_checkpoint else 2)
            return (priority, updated)

        sorted_candidates = sorted(candidates, key=sort_key)

        for item in sorted_candidates:
            if isinstance(item, dict):
                task_id = str(item.get("task_id") or "")
                intent = str(item.get("normalized_intent") or item.get("intent") or "")[:120]
                state = str(item.get("state") or "")
                checkpoint_id = item.get("checkpoint_id")
                recovery_intervention_id = item.get("recovery_intervention_id")
                owning_lane = item.get("owning_lane")
                is_terminal = item.get("is_terminal")
            else:
                task_id = str(getattr(item, "task_id", "") or "")
                intent = str(getattr(item, "normalized_intent", getattr(item, "intent", "")) or "")[:120]
                state = str(getattr(item, "state", "") or "")
                checkpoint_id = getattr(item, "checkpoint_id", None)
                recovery_intervention_id = getattr(item, "recovery_intervention_id", None)
                owning_lane = getattr(item, "owning_lane", None)
                is_terminal = getattr(item, "is_terminal", None)

            dedup_key = (intent.strip().casefold(), state)

            # Deduplicate repeated identical intentions with identical state
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            compact_item: dict[str, Any] = {
                "task_id": task_id,
                "state": state,
                "normalized_intent": intent,
            }
            if checkpoint_id:
                compact_item["checkpoint_id"] = str(checkpoint_id)
            if recovery_intervention_id:
                compact_item["recovery_intervention_id"] = str(recovery_intervention_id)
            if owning_lane:
                compact_item["owning_lane"] = str(owning_lane)
            if is_terminal is not None:
                compact_item["is_terminal"] = bool(is_terminal)

            compacted.append(compact_item)
            if len(compacted) >= max_candidates:
                break

        return compacted

    def compact_lane_states(
        self,
        lane_states: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Strip operational logs and detailed event histories from lane states."""
        compact_states: dict[str, Any] = {}
        for lane_id, state in lane_states.items():
            if not isinstance(state, dict):
                compact_states[lane_id] = str(state)
                continue
            # Keep only concise status flags, exclude logs and internal event traces
            compact_states[lane_id] = {
                "state": str(state.get("state") or "idle"),
                "active_task_id": str(state.get("active_task_id") or ""),
                "active": bool(state.get("active", False)),
            }
            if state.get("blocker"):
                compact_states[lane_id]["blocker"] = str(state["blocker"])[:100]
        return compact_states

    def compact_tools_and_capabilities(
        self,
        tools: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """Compact tool representations to basic signatures without heavy JSON schemas."""
        compact_tools: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                name = str(tool.get("name") or "")
                desc = str(tool.get("description") or "")[:120]
            else:
                name = str(getattr(tool, "name", "") or "")
                desc = str(getattr(tool, "description", "") or "")[:120]
            compact_tools.append({
                "name": name,
                "description": desc,
            })
        return compact_tools

    def compact_artifact_evidence(
        self,
        evidence: Any,
        *,
        max_references: int = 6,
    ) -> dict[str, Any]:
        """Bound artifact references and summaries."""
        if not isinstance(evidence, dict):
            return {}
        refs = list(evidence.get("references") or [])[:max_references]
        compact_refs = []
        for r in refs:
            if isinstance(r, dict):
                compact_refs.append({
                    "filename": str(r.get("filename") or ""),
                    "family": str(r.get("family") or ""),
                    "provenance": str(r.get("provenance") or ""),
                })
        return {
            "references": compact_refs,
            "artifact_families": list(evidence.get("artifact_families") or []),
            "detected_extensions": list(evidence.get("detected_extensions") or []),
            "has_user_artifact": bool(evidence.get("has_user_artifact", False)),
        }

    def compact_routing_context(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        context: Any,
        envelope: Any | None = None,
        routes: Sequence[Mapping[str, Any]] = (),
        context_window: int | None = None,
        response_reserve_tokens: int = 512,
        stale_records_pruned: int = 0,
        workspace_records_pruned: int = 0,
        repository_records_compacted: int = 0,
    ) -> CompactedRoutingContext:
        """Run the full multi-tier context reduction pass and construct a bounded routing capsule."""
        eff_envelope = envelope or getattr(context, "envelope", None)

        if context_window is None:
            cand_windows = [
                int(getattr(c, "context_window", 0))
                for c in getattr(eff_envelope, "model_candidates", ())
                if getattr(c, "context_window", 0)
            ] if eff_envelope else []
            if cand_windows:
                resolved_window = max(cand_windows)
            else:
                resolved_window = int(
                    getattr(self.settings, "mana_context_unknown_model_context_window", 128_000)
                    or 128_000
                )
        else:
            resolved_window = int(context_window)

        raw_breakdown = self.calculate_raw_breakdown(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            context=context,
            envelope=envelope,
            routes=routes,
        )

        logs_excluded = raw_breakdown.logs_and_traces


        # Step 1 & 2: Relevance filtering, log separation, candidate compaction
        max_candidates = self.policy.max_recovery_candidates
        all_candidates = (
            getattr(eff_envelope.execution_state, "all_recovery_candidates", ())
            if (eff_envelope is not None and getattr(eff_envelope, "execution_state", None) is not None)
            else ()
        )
        rec_candidates = (
            getattr(eff_envelope.execution_state, "recoverable_task_candidates", ())
            if (eff_envelope is not None and getattr(eff_envelope, "execution_state", None) is not None)
            else ()
        )

        compact_all_rec = self.compact_recovery_candidates(
            all_candidates, max_candidates=max_candidates
        )
        compact_rec = self.compact_recovery_candidates(
            rec_candidates, max_candidates=max_candidates
        )

        compact_lanes = (
            self.compact_lane_states(getattr(eff_envelope.execution_state, "lane_states", {}))
            if (eff_envelope is not None and getattr(eff_envelope, "execution_state", None) is not None)
            else {}
        )
        compact_tools = (
            self.compact_tools_and_capabilities(getattr(eff_envelope, "capabilities_and_tools", ()))
            if eff_envelope is not None
            else []
        )
        raw_art = getattr(context, "artifact_evidence", {}) or {}
        compact_artifacts = self.compact_artifact_evidence(raw_art)
        raw_mem = getattr(context, "memory_task_candidates", ()) or ()
        compact_memory_candidates = list(raw_mem)[:max_candidates]

        # Step 3: Capacity reservation
        user_tokens = estimate_value_tokens(user_prompt)
        sys_tokens = estimate_value_tokens(system_prompt)
        reserved_budget = user_tokens + sys_tokens + response_reserve_tokens
        remaining_budget = max(0, resolved_window - reserved_budget)

        # Tiered reduction if still exceeds budget
        if estimate_value_tokens(compact_all_rec) > remaining_budget // 2 and remaining_budget > 0:
            compact_all_rec = self.compact_recovery_candidates(
                compact_all_rec, max_candidates=max(2, max_candidates // 2)
            )
            compact_rec = self.compact_recovery_candidates(
                compact_rec, max_candidates=max(2, max_candidates // 2)
            )

        # Build bounded execution state
        if eff_envelope is not None and getattr(eff_envelope, "execution_state", None) is not None:
            es = eff_envelope.execution_state
            bounded_exec_state = ExecutionRecoveryState(
                active_flow_id=getattr(es, "active_flow_id", None),
                active_route=getattr(es, "active_route", ""),
                lane_id=getattr(es, "lane_id", ""),
                lane_states=compact_lanes,
                recoverable_task_candidates=tuple(compact_rec),
                all_recovery_candidates=tuple(compact_all_rec),
                pending_required_work=getattr(es, "pending_required_work", False),
                pending_checkpoint_id=getattr(es, "pending_checkpoint_id", None),
            )
        else:
            bounded_exec_state = ExecutionRecoveryState(
                lane_states=compact_lanes,
                recoverable_task_candidates=tuple(compact_rec),
                all_recovery_candidates=tuple(compact_all_rec),
            )

        identity = (
            getattr(eff_envelope, "identity", None)
            if eff_envelope is not None
            else None
        ) or IdentitySessionRelationship(
            authenticated_user_id=getattr(context, "authenticated_user_id", ""),
            session_id=getattr(context, "session_id", ""),
            conversation_id=getattr(context, "conversation_id", ""),
            turn_id=getattr(context, "turn_id", ""),
        )

        acct_snap = (
            getattr(eff_envelope, "accounting_snapshot", None)
            if eff_envelope is not None
            else None
        ) or default_accounting_snapshot(
            task_id=getattr(context, "turn_id", ""),
            turn_id=getattr(context, "turn_id", ""),
        )

        bounded_envelope = build_routing_execution_envelope(
            user_request=user_prompt,
            identity=identity,
            execution_state=bounded_exec_state,
            accounting_snapshot=acct_snap,
            model_candidates=getattr(eff_envelope, "model_candidates", ()) if eff_envelope else (),
            route_availability=getattr(eff_envelope, "route_availability", ()) if eff_envelope else (),
            capabilities_and_tools=tuple(compact_tools),
            approval_state=getattr(eff_envelope, "approval_state", None) if eff_envelope else None,
            artifact_metadata=compact_artifacts,
            previous_turn_pointers=getattr(eff_envelope, "previous_turn_pointers", None) if eff_envelope else None,
            conversation_context_availability=getattr(eff_envelope, "conversation_context_availability", None) if eff_envelope else None,
            memory_availability=getattr(eff_envelope, "memory_availability", None) if eff_envelope else None,
        )

        if isinstance(context, EntryRouteContext):
            bounded_context = EntryRouteContext(
                session_id=context.session_id,
                conversation_id=context.conversation_id,
                turn_id=context.turn_id,
                previous_route=context.previous_route,
                conversation_summary=context.conversation_summary,
                artifact_evidence=compact_artifacts,
                memory_task_candidates=tuple(compact_memory_candidates),
                memory_capsules_enabled=context.memory_capsules_enabled,
                atomic_child=context.atomic_child,
                orchestration_parent_task_id=context.orchestration_parent_task_id,
                authenticated_user_id=context.authenticated_user_id,
                envelope=bounded_envelope,
            )
        else:
            bounded_context = EntryRouteContext(
                session_id=getattr(context, "session_id", ""),
                conversation_id=getattr(context, "conversation_id", ""),
                turn_id=getattr(context, "turn_id", ""),
                previous_route=getattr(context, "previous_route", ""),
                conversation_summary=getattr(context, "conversation_summary", ""),
                artifact_evidence=compact_artifacts,
                memory_task_candidates=tuple(compact_memory_candidates),
                memory_capsules_enabled=getattr(context, "memory_capsules_enabled", False),
                atomic_child=getattr(context, "atomic_child", False),
                orchestration_parent_task_id=getattr(context, "orchestration_parent_task_id", ""),
                authenticated_user_id=getattr(context, "authenticated_user_id", ""),
                envelope=bounded_envelope,
            )
            if hasattr(context, "__dict__"):
                setattr(context, "envelope", bounded_envelope)
                setattr(context, "artifact_evidence", compact_artifacts)
                setattr(context, "memory_task_candidates", tuple(compact_memory_candidates))

        # Calculate compacted breakdown
        compacted_breakdown = self.calculate_raw_breakdown(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            context=bounded_context,
            envelope=bounded_envelope,
            routes=routes,
        )

        raw_total = raw_breakdown.total_tokens
        compacted_total = compacted_breakdown.total_tokens
        tokens_saved = max(0, raw_total - compacted_total)

        deficit_before = max(0, raw_total - resolved_window)
        deficit_after = max(0, compacted_total - resolved_window)
        is_valid = deficit_after == 0

        remaining_oversized: list[str] = []
        if deficit_after > 0:
            if compacted_breakdown.user_request > resolved_window // 2:
                remaining_oversized.append("user_request")
            if compacted_breakdown.task_candidates > resolved_window // 4:
                remaining_oversized.append("task_candidates")
            if compacted_breakdown.artifact_evidence > resolved_window // 4:
                remaining_oversized.append("artifact_evidence")

        diagnostic_details = {
            "context_limit": resolved_window,
            "required_tokens": compacted_total,
            "deficit": deficit_after,
            "tokens_by_category": compacted_breakdown.to_dict(),
            "raw_tokens_by_category": raw_breakdown.to_dict(),
            "attempted_compaction": True,
            "remaining_oversized_categories": remaining_oversized,
            "phase": "entry_route",
            "provider_call_executed": False,
        }


        return CompactedRoutingContext(
            bounded_envelope=bounded_envelope,
            bounded_context=bounded_context,
            raw_context_tokens=raw_total,
            compacted_context_tokens=compacted_total,
            context_tokens_saved=tokens_saved,
            logs_excluded_tokens=logs_excluded,
            stale_records_pruned=stale_records_pruned,
            workspace_records_pruned=workspace_records_pruned,
            repository_records_compacted=repository_records_compacted,
            routing_context_deficit_before_compaction=deficit_before,
            routing_context_deficit_after_compaction=deficit_after,
            breakdown=compacted_breakdown,
            attempted_compaction=True,
            remaining_oversized_categories=tuple(remaining_oversized),
            is_valid=is_valid,
            deficit=deficit_after,
            diagnostic_details=diagnostic_details,
        )


__all__ = [
    "CompactedRoutingContext",
    "ContextCategory",
    "ContextCompactor",
    "ContextComponentBreakdown",
    "default_accounting_snapshot",
]
