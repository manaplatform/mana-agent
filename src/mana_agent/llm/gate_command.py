"""Authoritative gate command + policy/proof model for the coding agent.

Every worker runs strictly under a :class:`GateCommand` issued by the coding
agent.  The coding agent is the *only* authority that decides the current gate,
the allowed tools, the required proof, and whether a todo is done.  Workers and
the tools manager may never auto-advance gates, mark todos done, choose
unrelated tools, or produce a final answer without first satisfying the gate
policy enforced here.

This module is intentionally dependency-free (pure data + pure functions) so it
can be unit-tested in isolation and reused by both ``RunStateStore`` and the
coding-agent orchestrator without import cycles.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# Canonical gate order.  This is the single source of truth shared with
# ``RunStateStore.gates`` (kept identical on purpose).
GATE_ORDER: tuple[str, ...] = (
    "locate_candidates",
    "read_candidates",
    "classify_evidence",
    "plan_patch",
    "apply_changes",
    "verify_changes",
    "final_report",
)

GATE_TO_PHASE: dict[str, str] = {
    "locate_candidates": "DISCOVERY",
    "read_candidates": "READING",
    "classify_evidence": "EXTRACTION",
    "plan_patch": "PATCHING",
    "apply_changes": "PATCHING",
    "verify_changes": "VERIFYING",
    "final_report": "FINAL",
}

# Wrapper-level tool that does not itself perform repository work.  It only
# counts as progress when it reports a concrete inner tool that is allowed for
# the active gate.
WRAPPER_TOOLS: frozenset[str] = frozenset({"toolsmanager_request"})

MUTATION_TOOLS: frozenset[str] = frozenset({"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"})
VERIFICATION_TOOLS: frozenset[str] = frozenset({"run_command", "verify_project"})


@dataclass(frozen=True)
class GatePolicy:
    """Static, authoritative policy for a single gate."""

    gate: str
    phase: str
    allowed_tools: frozenset[str]
    forbidden_tools: frozenset[str]
    # Concrete tools that satisfy the gate's required-tool obligation.  Empty
    # means the gate has no hard required tool (e.g. planning / final report).
    required_tool_set: frozenset[str]
    # Proof keys, at least one of which must carry a positive signal for the
    # gate to count as progress / be completable.
    progress_keys: tuple[str, ...]

    def is_allowed(self, tool_name: str) -> bool:
        name = _norm(tool_name)
        if name in self.forbidden_tools:
            return False
        if not self.allowed_tools:
            return False
        return name in self.allowed_tools


# Authoritative per-gate policy table.  ``RunStateStore`` builds GateCommands
# from this; workers can never widen it.
GATE_POLICIES: dict[str, GatePolicy] = {
    "locate_candidates": GatePolicy(
        gate="locate_candidates",
        phase="DISCOVERY",
        allowed_tools=frozenset({"repo_search", "semantic_search", "list_files", "run_command"}),
        forbidden_tools=frozenset(),
        required_tool_set=frozenset(),
        progress_keys=("files_found", "files_discovered", "candidate_files", "no_candidates"),
    ),
    "read_candidates": GatePolicy(
        gate="read_candidates",
        phase="READING",
        allowed_tools=frozenset({"read_file"}),
        forbidden_tools=frozenset(),
        required_tool_set=frozenset({"read_file"}),
        progress_keys=("files_read",),
    ),
    "classify_evidence": GatePolicy(
        gate="classify_evidence",
        phase="EXTRACTION",
        allowed_tools=frozenset({"run_command"}),
        forbidden_tools=frozenset(),
        required_tool_set=frozenset(),
        progress_keys=("evidence", "classification", "files_read"),
    ),
    "plan_patch": GatePolicy(
        gate="plan_patch",
        phase="PATCHING",
        allowed_tools=frozenset({"run_command"}),
        forbidden_tools=frozenset(),
        required_tool_set=frozenset(),
        progress_keys=("pending_edits", "patch_plan", "mutation_payload"),
    ),
    "apply_changes": GatePolicy(
        gate="apply_changes",
        phase="PATCHING",
        allowed_tools=frozenset(MUTATION_TOOLS),
        forbidden_tools=frozenset({"read_file"}),
        required_tool_set=frozenset(MUTATION_TOOLS),
        progress_keys=("modified_files", "successful_patches"),
    ),
    "verify_changes": GatePolicy(
        gate="verify_changes",
        phase="VERIFYING",
        allowed_tools=frozenset(VERIFICATION_TOOLS),
        forbidden_tools=frozenset({"toolsmanager_request"}),
        required_tool_set=frozenset(VERIFICATION_TOOLS),
        progress_keys=("verification_commands", "verification_result"),
    ),
    "final_report": GatePolicy(
        gate="final_report",
        phase="FINAL",
        allowed_tools=frozenset(),
        forbidden_tools=frozenset(),
        required_tool_set=frozenset(),
        progress_keys=("final_answer",),
    ),
}


@dataclass(frozen=True)
class GateCommand:
    """A single, immutable command the coding agent issues to one worker.

    A worker may only act within the bounds of the command it received.  Every
    tool call a worker makes must carry this ``command_id`` and is validated
    against the embedded policy before execution.
    """

    run_id: str
    plan_id: str
    gate: str
    step_id: str
    todo_id: str
    allowed_tools: tuple[str, ...]
    required_tool_set: tuple[str, ...]
    target_files: tuple[str, ...]
    done_condition: str
    max_attempts: int
    proof_schema: tuple[str, ...]
    forbidden_tools: tuple[str, ...]
    parent_command_id: str = ""
    command_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    @property
    def phase(self) -> str:
        return GATE_TO_PHASE.get(self.gate, "DISCOVERY")

    def is_tool_allowed(self, tool_name: str) -> bool:
        name = _norm(tool_name)
        if name in {_norm(t) for t in self.forbidden_tools}:
            return False
        allowed = {_norm(t) for t in self.allowed_tools}
        if not allowed:
            return False
        return name in allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "gate": self.gate,
            "phase": self.phase,
            "step_id": self.step_id,
            "todo_id": self.todo_id,
            "allowed_tools": list(self.allowed_tools),
            "required_tool_set": list(self.required_tool_set),
            "target_files": list(self.target_files),
            "done_condition": self.done_condition,
            "max_attempts": self.max_attempts,
            "proof_schema": list(self.proof_schema),
            "forbidden_tools": list(self.forbidden_tools),
            "parent_command_id": self.parent_command_id,
            "command_id": self.command_id,
        }


def build_gate_command(
    *,
    gate: str,
    run_id: str,
    plan_id: str = "",
    step_id: str = "",
    todo_id: str = "",
    target_files: Sequence[str] = (),
    done_condition: str = "",
    max_attempts: int = 3,
    parent_command_id: str = "",
) -> GateCommand:
    """Construct a GateCommand from the authoritative gate policy.

    Worker-supplied ``allowed_tools`` are never used; the policy is the only
    source.  This guarantees gate-specific allowed tools cannot be widened by a
    global/default tool set.
    """

    policy = GATE_POLICIES.get(gate, GATE_POLICIES["final_report"])
    return GateCommand(
        run_id=str(run_id or ""),
        plan_id=str(plan_id or ""),
        gate=gate,
        step_id=str(step_id or gate),
        todo_id=str(todo_id or gate),
        allowed_tools=tuple(sorted(policy.allowed_tools)),
        required_tool_set=tuple(sorted(policy.required_tool_set)),
        target_files=tuple(str(p).strip() for p in target_files if str(p).strip()),
        done_condition=str(done_condition or ""),
        max_attempts=max(1, int(max_attempts or 1)),
        proof_schema=tuple(policy.progress_keys),
        forbidden_tools=tuple(sorted(policy.forbidden_tools)),
        parent_command_id=str(parent_command_id or ""),
    )


# --------------------------------------------------------------------------- #
# Gate pointer reconciliation (fix #1)
# --------------------------------------------------------------------------- #


def reconcile_gate_pointer(
    *,
    completed_gates: Sequence[str],
    pending_gates: Sequence[str],
    pending_files: int = 0,
) -> tuple[str, str]:
    """Return the authoritative ``(current_gate, current_phase)``.

    The current gate is the first *pending* gate, regardless of any stale
    ``current_gate`` text.  Phase is always derived from the gate, never kept as
    free-floating mutable text.  We never resume into ``locate_candidates`` once
    it is completed and there are no pending files.
    """

    completed = [g for g in (str(x).strip() for x in completed_gates) if g]
    pending = [g for g in (str(x).strip() for x in pending_gates) if g]
    completed_set = set(completed)

    # Drop any pending gate that is already completed (state can be inconsistent
    # on resume); a gate cannot be both completed and pending.
    pending = [g for g in pending if g not in completed_set]

    if not pending:
        # Everything completed -> point at the terminal gate.
        gate = "final_report"
        return gate, GATE_TO_PHASE.get(gate, "FINAL")

    gate = pending[0]

    # Never resume into a completed discovery gate when there is nothing left to
    # read/discover.  Advance to the first genuinely pending gate instead.
    if (
        gate == "locate_candidates"
        and "locate_candidates" in completed_set
        and int(pending_files or 0) <= 0
    ):
        for candidate in pending:
            if candidate != "locate_candidates":
                gate = candidate
                break

    return gate, GATE_TO_PHASE.get(gate, "DISCOVERY")


# --------------------------------------------------------------------------- #
# Preflight tool policy enforcement (fix #3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of preflight policy enforcement for one tool selection."""

    allowed: bool
    # One of: "ok", "policy_violation", "no_progress".
    outcome: str
    reason: str
    concrete_tool: str = ""

    @property
    def is_policy_violation(self) -> bool:
        return self.outcome == "policy_violation"


def _resolve_concrete_tool(
    tool_name: str,
    *,
    inner_tool: str = "",
) -> tuple[str, bool]:
    """Resolve the concrete tool a request actually executes.

    ``toolsmanager_request`` is a wrapper: it only counts as a concrete task
    tool when it reports a valid inner tool.  Returns ``(concrete, is_wrapper)``.
    """

    name = _norm(tool_name)
    if name in WRAPPER_TOOLS:
        return _norm(inner_tool), True
    return name, False


def preflight_tool_policy(
    command: GateCommand,
    *,
    tool_name: str,
    inner_tool: str = "",
) -> PolicyDecision:
    """Validate a worker's tool selection *before* execution.

    - If the (concrete) tool is not allowed for the gate -> blocked,
      ``policy_violation`` (must not be recorded as success or as no_progress).
    - If a wrapper tool reports no concrete inner tool -> blocked,
      ``no_progress`` (it did nothing).
    - Forbidden tools are always blocked as ``policy_violation``.
    """

    concrete, is_wrapper = _resolve_concrete_tool(tool_name, inner_tool=inner_tool)
    forbidden = {_norm(t) for t in command.forbidden_tools}

    # A forbidden tool is a policy violation regardless of any (missing) inner
    # tool — e.g. toolsmanager_request is forbidden outright in verify_changes.
    if _norm(tool_name) in forbidden:
        return PolicyDecision(
            allowed=False,
            outcome="policy_violation",
            reason=f"tool '{tool_name}' is forbidden for gate {command.gate}",
            concrete_tool=concrete,
        )

    if is_wrapper and not concrete:
        return PolicyDecision(
            allowed=False,
            outcome="no_progress",
            reason="toolsmanager_request reported no concrete inner tool call",
            concrete_tool="",
        )

    if is_wrapper and concrete in forbidden:
        return PolicyDecision(
            allowed=False,
            outcome="policy_violation",
            reason=f"tool '{concrete or tool_name}' is forbidden for gate {command.gate}",
            concrete_tool=concrete,
        )

    if not command.is_tool_allowed(concrete):
        return PolicyDecision(
            allowed=False,
            outcome="policy_violation",
            reason=(
                f"tool '{concrete or tool_name}' is not in allowed_tools "
                f"{sorted(command.allowed_tools)} for gate {command.gate}"
            ),
            concrete_tool=concrete,
        )

    return PolicyDecision(allowed=True, outcome="ok", reason="", concrete_tool=concrete)


# --------------------------------------------------------------------------- #
# Gate-specific proof validation (fix #4)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProofResult:
    valid: bool
    reason: str = ""


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple, set)) else []


def validate_gate_proof(gate: str, proof: Mapping[str, Any]) -> ProofResult:
    """Validate that ``proof`` actually satisfies the gate's done condition.

    A gate cannot be completed unless this validator passes.  Wrapper-only,
    answer-text-only proofs never pass.
    """

    proof = dict(proof or {})
    concrete, _ = _resolve_concrete_tool(
        str(proof.get("tool_name", "")),
        inner_tool=str(proof.get("inner_tool", "")),
    )

    # A wrapper that produced no concrete inner tool is never proof of work.
    if _norm(proof.get("tool_name")) in WRAPPER_TOOLS and not concrete:
        return ProofResult(False, "toolsmanager_request produced no concrete inner tool")

    files_read = len(_as_list(proof.get("files_read")))
    modified = [p for p in _as_list(proof.get("modified_files")) if str(p).strip()]
    successful_patches = _as_int(proof.get("successful_patches"))

    if gate in {"locate_candidates"}:
        if (
            _as_list(proof.get("files_found"))
            or _as_list(proof.get("files_discovered"))
            or _as_list(proof.get("candidate_files"))
            or bool(proof.get("no_candidates"))
        ):
            return ProofResult(True)
        return ProofResult(False, "discover proof requires files_found/files_discovered or no-candidates evidence")

    if gate == "read_candidates":
        if files_read > 0:
            return ProofResult(True)
        return ProofResult(False, "read proof requires files_read > 0 (read_file_no_files_read is not progress)")

    if gate == "classify_evidence":
        if proof.get("evidence") or proof.get("classification") or files_read > 0:
            return ProofResult(True)
        return ProofResult(False, "classify proof requires evidence/classification")

    if gate == "plan_patch":
        if _as_list(proof.get("pending_edits")) or proof.get("patch_plan") or proof.get("mutation_payload"):
            return ProofResult(True)
        return ProofResult(False, "plan proof requires pending_edits/patch_plan/mutation_payload")

    if gate == "apply_changes":
        if concrete and concrete not in MUTATION_TOOLS:
            return ProofResult(False, f"apply_changes requires a mutation tool, got '{concrete}'")
        if modified or successful_patches > 0:
            return ProofResult(True)
        return ProofResult(False, "apply_changes proof requires modified_files or successful_patches > 0")

    if gate == "verify_changes":
        if _norm(proof.get("tool_name")) in WRAPPER_TOOLS:
            return ProofResult(False, "toolsmanager_request is forbidden for verify_changes")
        if concrete and concrete not in VERIFICATION_TOOLS:
            return ProofResult(False, f"verify_changes requires a verification tool, got '{concrete}'")
        has_command = bool(proof.get("verification_command") or proof.get("command_output"))
        has_result = bool(proof.get("verification_result") or proof.get("command_output"))
        if has_command and has_result:
            return ProofResult(True)
        return ProofResult(False, "verify proof requires a verification command and result")

    if gate == "final_report":
        if proof.get("final_answer") or proof.get("answer"):
            return ProofResult(True)
        return ProofResult(False, "final_report requires a final answer")

    return ProofResult(False, f"unknown gate {gate}")


def can_run_verify(*, successful_patches: int, modified_files: Sequence[str]) -> bool:
    """verify_changes cannot run until at least one mutation has landed."""

    return _as_int(successful_patches) > 0 or any(str(p).strip() for p in modified_files)


def can_run_final_report(*, verify_completed: bool, verify_skip_reason: str = "") -> bool:
    """final_report cannot run until verify is done or explicitly skipped."""

    return bool(verify_completed) or bool(str(verify_skip_reason or "").strip())


# --------------------------------------------------------------------------- #
# No-progress / duplicate fingerprinting (fix #6)
# --------------------------------------------------------------------------- #


def tool_fingerprint(
    *,
    tool_name: str,
    args: Mapping[str, Any] | None,
    gate: str,
    target_file: str = "",
) -> str:
    """Fingerprint a concrete tool call: name + normalized args + gate + file."""

    payload = {
        "tool": _norm(tool_name),
        "gate": _norm(gate),
        "target": _norm(target_file),
        "args": _normalize_args(args or {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_args(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_args(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize_args(v) for v in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def progress_signal_present(gate: str, proof: Mapping[str, Any]) -> bool:
    """True only when the proof carries a real progress signal for the gate.

    answer_chars-only / answer-text-only proofs are never progress.
    """

    return validate_gate_proof(gate, proof).valid


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()
