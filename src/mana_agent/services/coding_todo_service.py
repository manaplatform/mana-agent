"""Durable todo ledger that connects the pre-execution plan to flow memory.

The coding agent produces a *prechecklist* (an ordered list of plan steps) before
it executes. ``TodoService`` turns that ephemeral preview into persistent todos
tied to the active flow, backed by :class:`CodingMemoryService`'s SQLite store.

This gives three things the deterministic ``QueueManager`` previously lacked:

* **Continuity** — todos survive across turns of the same flow, so a multi-turn
  request shows real progress instead of replanning from scratch each turn.
* **Accuracy** — step status is reconciled from authoritative run results
  (changed files + verification), not from natural-language guesses.
* **Low cost** — no extra LLM calls. Classification and reconciliation are
  deterministic, so connecting todos to the plan adds storage, not latency.
"""

from __future__ import annotations

from typing import Any

from mana_agent.memory import CodingMemoryService

# Tools whose presence in a step means the step performs a real mutation. A
# step requiring any of these is an "edit" step and is only ``done`` once a
# mutation has actually landed.
_MUTATION_TOOLS = frozenset({"apply_patch", "write_file", "create_file", "delete_file"})
_VERIFY_TOOLS = frozenset({"verify_project", "run_tests", "pytest"})

# Keyword fallbacks used only when a step lists no tools (deterministic
# fallback checklists often omit them).
_EDIT_HINTS = ("edit", "patch", "write", "create", "implement", "modify", "update", "add ")
_VERIFY_HINTS = ("verify", "test", "check", "validate", "lint")


class TodoService:
    """Maps the plan prechecklist to durable, status-tracked flow todos."""

    def __init__(self, *, memory: CodingMemoryService) -> None:
        self.memory = memory

    # -- classification -----------------------------------------------------

    @staticmethod
    def classify_step(step: dict[str, Any]) -> str:
        """Bucket a plan step into discover | read | edit | verify.

        Tool requirements win over title heuristics because they come straight
        from the planner's structured output and are unambiguous.
        """
        tools = {
            str(t).strip().lower()
            for t in (step.get("requires_tools") or step.get("allowed_tools") or [])
            if str(t).strip()
        }
        if tools & _MUTATION_TOOLS:
            return "edit"
        if tools & _VERIFY_TOOLS:
            return "verify"
        title = str(step.get("title") or "").lower()
        if any(hint in title for hint in _VERIFY_HINTS):
            return "verify"
        if any(hint in title for hint in _EDIT_HINTS):
            return "edit"
        if tools or "read" in title or "inspect" in title:
            return "read"
        return "discover"

    # -- preview -> todos ---------------------------------------------------

    def sync_from_preview(
        self,
        *,
        flow_id: str,
        prechecklist: dict[str, Any],
        source: str = "",
    ) -> list[dict[str, Any]]:
        """Persist a prechecklist's steps as flow todos and return the ledger."""
        steps: list[dict[str, Any]] = []
        raw_steps = prechecklist.get("steps") if isinstance(prechecklist, dict) else None
        for index, raw in enumerate(raw_steps or []):
            if not isinstance(raw, dict):
                continue
            step_id = str(raw.get("id") or "").strip() or f"step-{index + 1}"
            steps.append(
                {
                    "id": step_id,
                    "title": str(raw.get("title") or "").strip() or step_id,
                    "kind": self.classify_step(raw),
                    "status": str(raw.get("status") or "pending").strip() or "pending",
                    "requires_tools": list(raw.get("requires_tools") or raw.get("allowed_tools") or []),
                }
            )
        self.memory.sync_plan_steps(flow_id=flow_id, steps=steps, source=source)
        return self.memory.list_plan_steps(flow_id)

    def list(self, flow_id: str) -> list[dict[str, Any]]:
        """Return the current todo ledger for a flow."""
        return self.memory.list_plan_steps(flow_id)

    # -- run results -> todo status ----------------------------------------

    def reconcile_after_run(
        self,
        *,
        flow_id: str,
        changed_files: list[str],
        mutation_succeeded: bool,
        verification_passed: bool,
        run_blocked: bool,
    ) -> list[dict[str, Any]]:
        """Advance todo statuses from authoritative run results.

        Rules (deterministic, monotonic):

        * discover/read steps complete once the run has executed at all.
        * edit steps complete only when a mutation actually landed; if the run
          is blocked without changes they are marked ``blocked``.
        * verify steps complete when verification passed; ``blocked`` if it ran
          and failed (surfaced as ``run_blocked``).
        """
        had_change = bool(changed_files) or mutation_succeeded
        for step in self.memory.list_plan_steps(flow_id):
            kind = step.get("kind") or self.classify_step(step)
            step_id = str(step.get("id"))
            if kind in ("discover", "read"):
                status = "done"
            elif kind == "edit":
                status = "done" if had_change else ("blocked" if run_blocked else "in_progress")
            elif kind == "verify":
                if verification_passed:
                    status = "done"
                elif run_blocked:
                    status = "blocked"
                else:
                    status = "in_progress"
            else:
                status = "done" if not run_blocked else "in_progress"
            self.memory.update_plan_step_status(flow_id=flow_id, step_id=step_id, status=status)
        return self.memory.list_plan_steps(flow_id)
