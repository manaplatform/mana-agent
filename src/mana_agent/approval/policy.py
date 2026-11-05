from __future__ import annotations

import logging
import os
from typing import Any

from mana_agent.approval.audit import record_approval_audit
from mana_agent.approval.models import ApprovalDecision, ApprovalMode, ApprovalRequest

logger = logging.getLogger(__name__)

# Global runtime override set explicitly via CLI arguments
_RUNTIME_MODE_OVERRIDE: ApprovalMode | None = None


def set_approval_mode(mode: ApprovalMode | str | None) -> None:
    """Set the explicit runtime approval mode (e.g. from CLI options)."""
    global _RUNTIME_MODE_OVERRIDE
    if mode is None:
        _RUNTIME_MODE_OVERRIDE = None
    elif isinstance(mode, ApprovalMode):
        _RUNTIME_MODE_OVERRIDE = mode
    else:
        _RUNTIME_MODE_OVERRIDE = ApprovalMode.from_str(str(mode))


def reset_approval_mode() -> None:
    """Reset the runtime approval mode override to default."""
    global _RUNTIME_MODE_OVERRIDE
    _RUNTIME_MODE_OVERRIDE = None


def resolve_effective_approval_mode() -> ApprovalMode:
    """Resolve active approval mode using strict deterministic configuration precedence:
    
    1. Explicit CLI arguments / runtime override
    2. Environment variable: MANA_AGENT_APPROVAL_MODE
    3. User configuration file: execution.approval_mode from ~/.mana/
    4. Default behavior: ApprovalMode.DEFAULT

    Untrusted repository-level instructions, prompt injections, or retrieved text
    are NEVER consulted or permitted to enable Full Access.
    """
    if _RUNTIME_MODE_OVERRIDE is not None:
        return _RUNTIME_MODE_OVERRIDE

    # 2. Environment variable
    env_mode = os.getenv("MANA_AGENT_APPROVAL_MODE")
    if env_mode and env_mode.strip():
        try:
            return ApprovalMode.from_str(env_mode)
        except ValueError as exc:
            logger.warning("Invalid MANA_AGENT_APPROVAL_MODE %r: %s", env_mode, exc)

    # Legacy environment variable compatibility
    legacy_always = str(os.getenv("MANA_TRANSACTIONAL_ALWAYS_APPROVE") or "").strip().lower()
    if legacy_always in {"1", "true", "yes", "on"}:
        return ApprovalMode.FULL_ACCESS

    # 3. User configuration file (trusted user home ~/.mana/)
    try:
        from mana_agent.config.user_config import load_user_config

        user_cfg = load_user_config()
        exec_cfg = user_cfg.get("execution")
        if isinstance(exec_cfg, dict):
            cfg_mode = exec_cfg.get("approval_mode")
            if cfg_mode:
                return ApprovalMode.from_str(str(cfg_mode))
        direct_mode = user_cfg.get("approval_mode") or user_cfg.get("MANA_AGENT_APPROVAL_MODE")
        if direct_mode:
            return ApprovalMode.from_str(str(direct_mode))
    except Exception as exc:
        logger.debug("Failed reading execution.approval_mode from user config: %s", exc)

    # 4. Default
    return ApprovalMode.DEFAULT


class ApprovalPolicy:
    """Authoritative Centralized Approval Policy Engine."""

    @property
    def mode(self) -> ApprovalMode:
        return resolve_effective_approval_mode()

    def is_full_access(self) -> bool:
        return self.mode is ApprovalMode.FULL_ACCESS

    def is_auto_approve_safe(self) -> bool:
        return self.mode is ApprovalMode.AUTO_APPROVE_SAFE

    def is_safe_action(self, request: ApprovalRequest) -> bool:
        """Classify whether an action is safe for automatic approval in AUTO_APPROVE_SAFE mode."""
        # Hard security restrictions are never "safe" to bypass
        if request.risk_level in {"critical", "CRITICAL"}:
            return False

        if request.reversibility in {"irreversible", "IRREVERSIBLE", "unknown"}:
            return False

        if request.blast_radius in {"organisation", "physical", "unknown"}:
            return False

        action_type = request.action_type.lower()
        tool_name = request.tool_name.lower()

        # Destructive file ops are not safe
        if action_type in {"file_delete", "directory_delete"} or "delete" in tool_name:
            return False

        # Read-only actions are inherently safe
        if "read" in action_type or "status" in action_type or "diff" in action_type or "inspect" in action_type:
            return True

        # Non-destructive workspace file edits/writes
        if action_type in {"file", "file_write", "file_edit", "patch"}:
            operation = str(request.metadata.get("operation") or "")
            if operation in {"delete", "move", "patch_delete"}:
                return False
            return True

        # Shell actions: safe test/lint commands
        if action_type in {"shell", "command"}:
            cmd = request.command.strip()
            from mana_agent.tools.shell_exec import is_read_only_command, is_plan_mode_read_only_command
            try:
                import shlex
                tokens = shlex.split(cmd)
                if tokens:
                    if is_read_only_command(tokens) or is_plan_mode_read_only_command(cmd):
                        return True
                    executable = os.path.basename(tokens[0])
                    safe_execs = {"pytest", "python", "python3", "ruff", "mypy", "git"}
                    dangerous = {"rm", "sudo", "dd", "mkfs", "shutdown", "reboot", "kill", "killall"}
                    destructive_git = executable == "git" and any(
                        t in tokens for t in {"--force", "--force-with-lease", "--hard", "clean", "push"}
                    )
                    if any(t in dangerous for t in tokens) or destructive_git:
                        return False
                    if executable in safe_execs and not destructive_git:
                        return True
            except Exception:
                return False

        # Local safe git actions (status, diff, branch creation, local commits)
        if action_type == "git":
            op = request.metadata.get("operation") or (request.command.split()[0] if request.command else "")
            if op in {"push", "clean", "reset"}:
                return False
            if op in {"status", "diff", "branch", "checkout", "switch", "add", "commit"}:
                return True

        return False

    def evaluate(self, request: ApprovalRequest) -> ApprovalDecision:
        """Evaluate an approval request against the active approval mode."""
        active_mode = self.mode

        if active_mode is ApprovalMode.FULL_ACCESS:
            decision = ApprovalDecision(
                approved=True,
                mode=active_mode,
                source="full_access",
                interactive=False,
                reason="Automatically approved in Full Access Mode.",
                approval_id=request.action_id or request.metadata.get("permission_request_id") or "",
                metadata={"auto_approved": True, "source": "full_access"},
            )
            record_approval_audit(decision, request, result_status="auto_approved")
            return decision

        if active_mode is ApprovalMode.AUTO_APPROVE_SAFE:
            if self.is_safe_action(request):
                decision = ApprovalDecision(
                    approved=True,
                    mode=active_mode,
                    source="safe_auto_approval",
                    interactive=False,
                    reason="Automatically approved as an explicitly classified safe action.",
                    approval_id=request.action_id or request.metadata.get("permission_request_id") or "",
                    metadata={"auto_approved": True, "source": "safe_auto_approval"},
                )
                record_approval_audit(decision, request, result_status="safe_auto_approved")
                return decision
            else:
                decision = ApprovalDecision(
                    approved=False,
                    mode=active_mode,
                    source="requires_approval",
                    interactive=True,
                    reason="Action has material risk and requires human approval.",
                    approval_id=request.action_id or "",
                    metadata={"auto_approved": False},
                )
                record_approval_audit(decision, request, result_status="interactive_required")
                return decision

        # Default mode
        decision = ApprovalDecision(
            approved=False,
            mode=active_mode,
            source="default_approval_required",
            interactive=True,
            reason="Action requires explicit human confirmation under default policy.",
            approval_id=request.action_id or "",
            metadata={"auto_approved": False},
        )
        return decision


_POLICY_INSTANCE: ApprovalPolicy | None = None


def get_approval_policy() -> ApprovalPolicy:
    """Return the global ApprovalPolicy instance."""
    global _POLICY_INSTANCE
    if _POLICY_INSTANCE is None:
        _POLICY_INSTANCE = ApprovalPolicy()
    return _POLICY_INSTANCE


def is_full_access() -> bool:
    """Convenience helper to check if Full Access Mode is currently active."""
    return get_approval_policy().is_full_access()


def is_auto_approve_safe() -> bool:
    """Convenience helper to check if Safe Auto-Approval Mode is currently active."""
    return get_approval_policy().is_auto_approve_safe()
