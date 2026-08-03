from __future__ import annotations

import hashlib
import json
import shlex
import getpass
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field

from .models import (
    ActionIntent,
    ApprovalScope,
    BlastRadius,
    DataDisclosure,
    PolicyDecision,
    PolicyOutcome,
    Reversibility,
    StrictModel,
    utc_now,
)


class PolicyScope(StrictModel):
    denied_tools: tuple[str, ...] = ()
    approval_tools: tuple[str, ...] = ()
    denied_capabilities: tuple[str, ...] = ()


class PolicyConfig(StrictModel):
    version: str = "transactional-actions-v1"
    user_scope: PolicyScope = Field(default_factory=PolicyScope)
    project_scope: PolicyScope = Field(default_factory=PolicyScope)
    repository_scope: PolicyScope = Field(default_factory=PolicyScope)
    organisation_scope: PolicyScope = Field(default_factory=PolicyScope)
    workspace_roots: tuple[Path, ...] = ()
    allowed_http_hosts: tuple[str, ...] = ()
    allow_insecure_http: bool = False
    allow_safe_workspace_file_writes: bool = True
    allow_narrow_transaction_approval: bool = True
    safe_shell_executables: tuple[str, ...] = ("git", "python", "python3", "pytest", "ruff", "mypy")
    secret_argument_names: tuple[str, ...] = (
        "authorization", "api_key", "apikey", "password", "secret", "token", "credential"
    )
    approval_ttl_seconds: int = Field(default=300, ge=30, le=3600)
    approval_reviewer_type: str = "person"
    approval_reviewer_id: str = Field(default_factory=getpass.getuser)

    def fingerprint(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ActionPolicy:
    """Deterministic, fail-closed classification over normalized adapter data."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def evaluate(self, action: ActionIntent) -> PolicyDecision:
        now = utc_now()
        outcome, codes, explanation, rules = self._classify(action)
        return PolicyDecision(
            outcome=outcome,
            reason_codes=codes,
            explanation=explanation,
            matched_rules=rules,
            required_approval_scope=(
                ApprovalScope.TRANSACTION
                if outcome is PolicyOutcome.REQUIRE_APPROVAL
                and action.transaction_id
                and self.config.allow_narrow_transaction_approval
                else ApprovalScope.ACTION_ONCE
                if outcome is PolicyOutcome.REQUIRE_APPROVAL
                else None
            ),
            policy_fingerprint=self.config.fingerprint(),
            decided_at=now,
            expires_at=min(action.expires_at, now + timedelta(seconds=self.config.approval_ttl_seconds)),
            assigned_reviewer_type=self.config.approval_reviewer_type,
            assigned_reviewer_id=self.config.approval_reviewer_id,
        )

    def _classify(self, action: ActionIntent) -> tuple[PolicyOutcome, list[str], str, list[str]]:
        scope_decision = self._scoped_policy(action)
        if scope_decision is not None:
            return scope_decision
        if action.data_disclosure is DataDisclosure.SECRET:
            return PolicyOutcome.DENY, ["secret_disclosure"], "Secret-bearing actions are denied.", ["deny_secret_disclosure"]
        if action.blast_radius in {BlastRadius.ORGANISATION, BlastRadius.PHYSICAL, BlastRadius.UNKNOWN}:
            return PolicyOutcome.REQUIRE_APPROVAL, ["high_or_unknown_blast_radius"], "The action has a high or unknown blast radius.", ["approve_high_blast_radius"]
        if action.tool_name == "file":
            return self._file(action)
        if action.tool_name == "shell":
            return self._shell(action)
        if action.tool_name == "http":
            return self._http(action)
        if action.tool_name == "git":
            return self._git(action)
        if action.tool_name == "computer":
            return (
                PolicyOutcome.REQUIRE_APPROVAL,
                ["computer_control"],
                "Computer-control actions require exact approval.",
                ["approve_computer_control"],
            )
        if action.tool_name == "mcp":
            return (
                PolicyOutcome.REQUIRE_APPROVAL,
                ["external_mcp_operation"],
                "MCP provider operations require exact approval.",
                ["approve_external_mcp_operation"],
            )
        return PolicyOutcome.DENY, ["unclassified_tool"], "No policy rule safely classifies this tool.", ["default_deny"]

    def _scoped_policy(self, action: ActionIntent) -> tuple[PolicyOutcome, list[str], str, list[str]] | None:
        scopes = (
            ("user", self.config.user_scope),
            ("project", self.config.project_scope),
            ("repository", self.config.repository_scope),
            ("organisation", self.config.organisation_scope),
        )
        for name, scope in scopes:
            if action.tool_name in scope.denied_tools or any(
                capability in scope.denied_capabilities
                for capability in action.requested_capabilities
            ):
                return PolicyOutcome.DENY, [f"{name}_scope_deny"], f"The {name} policy scope denies this action.", [f"{name}_scope"]
        approvals = [name for name, scope in scopes if action.tool_name in scope.approval_tools]
        if approvals:
            return PolicyOutcome.REQUIRE_APPROVAL, ["scoped_approval"], "Configured policy scope requires exact approval.", [f"{name}_scope" for name in approvals]
        return None

    def _file(self, action: ActionIntent) -> tuple[PolicyOutcome, list[str], str, list[str]]:
        roots = tuple(root.expanduser().resolve() for root in self.config.workspace_roots)
        for raw in action.target_resources:
            target = Path(raw).expanduser().resolve()
            if not roots or not any(_is_within(target, root) for root in roots):
                return PolicyOutcome.DENY, ["outside_workspace"], "A target is outside the approved workspace.", ["workspace_boundary"]
        resources = action.preview.resources if action.preview else []
        if action.operation_name == "cleanup_generated_parts":
            directory = Path(str(action.normalized_arguments.get("parts_directory") or "")).resolve()
            final_target = Path(str(action.normalized_arguments.get("final_target") or "")).resolve()
            expected_directory = final_target.parent / f".{final_target.name}.parts"
            part_hashes = action.normalized_arguments.get("part_hashes")
            part_paths = [Path(str(path)).resolve() for path in part_hashes] if isinstance(part_hashes, dict) else []
            if not (
                action.normalized_arguments.get("ephemeral_generated") is True
                and action.normalized_arguments.get("aggregate_matches_target") is True
                and directory == expected_directory.resolve()
                and part_paths
                and all(path.parent == directory and path.suffix == ".part" for path in part_paths)
                and set(action.target_resources) == {str(path) for path in part_paths} | {str(directory)}
            ):
                return PolicyOutcome.DENY, ["unverified_cleanup"], "Generated-part cleanup lacks verified redundancy evidence.", ["cleanup_precondition"]
            return PolicyOutcome.ALLOW, ["verified_generated_cleanup"], "Verified redundant chunk artifacts may be removed after final-file commit.", ["allow_generated_cleanup"]
        if action.operation_name in {"patch", "patch_delete"}:
            for resource in resources:
                change = str(resource.get("change") or "")
                existed = str(resource.get("before_sha256") or "") != "missing"
                if change == "add" and existed:
                    return PolicyOutcome.DENY, ["target_exists"], "A patch add cannot overwrite an existing target.", ["patch_precondition"]
                if change in {"update", "delete"} and not existed:
                    return PolicyOutcome.DENY, ["source_missing"], "A patch update or delete requires an existing target.", ["patch_precondition"]
            if action.operation_name == "patch_delete":
                return PolicyOutcome.REQUIRE_APPROVAL, ["destructive_patch"], "A patch containing deletion requires exact approval.", ["approve_destructive_patch"]
            if self.config.allow_safe_workspace_file_writes:
                return PolicyOutcome.ALLOW, ["workspace_patch"], "A bounded repository patch with validated resources is allowed.", ["allow_workspace_patch"]
            return PolicyOutcome.REQUIRE_APPROVAL, ["patch_approval"], "Repository patches require exact approval by policy.", ["approve_patch"]
        primary = resources[0] if resources else {}
        if action.operation_name == "create" and bool(primary.get("exists")):
            return PolicyOutcome.DENY, ["target_exists"], "A create action cannot overwrite an existing target.", ["file_precondition"]
        if action.operation_name in {"edit", "delete", "move"} and primary.get("exists") is False:
            return PolicyOutcome.DENY, ["source_missing"], "The selected file source does not exist.", ["file_precondition"]
        if action.operation_name in {"delete", "move"} or action.reversibility in {Reversibility.IRREVERSIBLE, Reversibility.UNKNOWN}:
            return PolicyOutcome.REQUIRE_APPROVAL, ["destructive_file_action"], "Destructive or uncertain file actions require exact approval.", ["approve_destructive_file"]
        if self.config.allow_safe_workspace_file_writes:
            return PolicyOutcome.ALLOW, ["workspace_file_write"], "A bounded, reversible workspace file action is allowed.", ["allow_workspace_file_write"]
        return PolicyOutcome.REQUIRE_APPROVAL, ["file_write_approval"], "Workspace file writes require exact approval by policy.", ["approve_file_write"]

    def _shell(self, action: ActionIntent) -> tuple[PolicyOutcome, list[str], str, list[str]]:
        argv = action.normalized_arguments.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            return PolicyOutcome.DENY, ["invalid_shell_argv"], "Shell actions require a normalized argv list.", ["shell_argv_required"]
        context = action.normalized_arguments.get("policy_context")
        context = context if isinstance(context, dict) else {}
        executable = Path(argv[0]).name
        inspected = list(argv)
        if executable in {"sh", "bash", "zsh", "dash", "cmd.exe", "powershell", "pwsh"}:
            command_text = str(argv[-1]) if len(argv) >= 3 else ""
            try:
                lexer = shlex.shlex(command_text, posix=True, punctuation_chars=True)
                lexer.whitespace_split = True
                inspected = list(lexer)
            except ValueError:
                return PolicyOutcome.DENY, ["invalid_nested_shell"], "The nested shell command could not be parsed safely.", ["shell_parse"]
            if not inspected:
                return PolicyOutcome.DENY, ["empty_nested_shell"], "The nested shell command is empty.", ["shell_parse"]
            executable = Path(inspected[0]).name
        dangerous = {"rm", "sudo", "dd", "mkfs", "shutdown", "reboot", "kill", "killall"}
        destructive_git = executable == "git" and any(
            item in inspected for item in {"--force", "--force-with-lease", "--hard", "clean"}
        )
        if any(item in dangerous for item in inspected) or "sudo" in inspected or destructive_git or action.reversibility is Reversibility.IRREVERSIBLE:
            return PolicyOutcome.DENY, ["destructive_shell"], "The command is destructive or irreversible.", ["deny_destructive_shell"]
        if (
            context.get("kind") == "validated_verification"
            and action.actor == "tool_worker"
            and action.normalized_arguments.get("allow_command_result_verification") is True
        ):
            return PolicyOutcome.ALLOW, ["validated_verification_command"], "A bounded verification queue job may execute and commit from its command-result receipt.", ["allow_verification_queue"]
        if executable not in self.config.safe_shell_executables:
            return PolicyOutcome.REQUIRE_APPROVAL, ["shell_executable_not_allowlisted"], "This executable requires exact approval.", ["approve_unknown_shell"]
        return PolicyOutcome.REQUIRE_APPROVAL, ["shell_side_effect"], "Shell commands require exact approval because declared effects cannot be inferred from exit status alone.", ["approve_shell"]

    def _http(self, action: ActionIntent) -> tuple[PolicyOutcome, list[str], str, list[str]]:
        method = str(action.normalized_arguments.get("method") or "").upper()
        parsed = urlsplit(str(action.normalized_arguments.get("url") or ""))
        if method not in {"POST", "PUT", "PATCH", "DELETE"} or parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return PolicyOutcome.DENY, ["invalid_http_mutation"], "HTTP mutations require a normalized method and absolute HTTP(S) URL.", ["http_shape"]
        host = parsed.hostname.rstrip(".").lower()
        allowed = {item.rstrip(".").lower() for item in self.config.allowed_http_hosts}
        if not allowed:
            return PolicyOutcome.DENY, ["http_host_policy_missing"], "No approved HTTP destination host was supplied by the network policy.", ["http_host_allowlist"]
        if host not in allowed:
            return PolicyOutcome.DENY, ["http_host_not_allowed"], "The destination host is outside the approved policy allowlist.", ["http_host_allowlist"]
        if parsed.scheme == "http" and not self.config.allow_insecure_http:
            return PolicyOutcome.DENY, ["insecure_http"], "Unencrypted HTTP mutation is denied.", ["https_required"]
        if action.data_disclosure is DataDisclosure.UNKNOWN:
            return PolicyOutcome.DENY, ["unknown_disclosure"], "The data disclosure could not be classified.", ["classify_disclosure"]
        return PolicyOutcome.REQUIRE_APPROVAL, ["remote_mutation"], "Remote mutation requires exact approval.", ["approve_http_mutation"]

    def _git(self, action: ActionIntent) -> tuple[PolicyOutcome, list[str], str, list[str]]:
        risk = str(action.normalized_arguments.get("risk_level") or "")
        context = action.normalized_arguments.get("policy_context")
        context = context if isinstance(context, dict) else {}
        argv = action.normalized_arguments.get("argv")
        argv = argv if isinstance(argv, list) else []
        if context.get("kind") == "managed_workspace" and action.actor == "workspace_manager":
            operation = str(context.get("operation") or "")
            if operation == "create" and argv[1:3] == ["worktree", "add"]:
                return PolicyOutcome.ALLOW, ["managed_workspace_create"], "A bounded managed-worktree creation selected by the workspace coordinator is allowed.", ["allow_managed_workspace"]
            operation_matches = (
                (operation == "merge" and argv[1:2] == ["merge"])
                or (operation == "remove" and argv[1:3] == ["worktree", "remove"])
                or (operation == "delete_branch" and argv[1:2] == ["branch"] and "-d" in argv)
            )
            if operation_matches and context.get("explicit_user_intent") is True:
                return PolicyOutcome.ALLOW, ["managed_workspace_explicit_action"], "The workspace coordinator supplied validated explicit intent for this exact managed-workspace action.", ["allow_explicit_managed_workspace"]
            return PolicyOutcome.DENY, ["invalid_managed_workspace_context"], "Managed-workspace policy context was incomplete or did not match the Git operation.", ["managed_workspace_shape"]
        if context.get("kind") == "validated_queue_git" and action.actor == "tool_worker" and risk in {"LOCAL_SAFE_WRITE", "LOCAL_HISTORY_WRITE"}:
            return PolicyOutcome.ALLOW, ["validated_local_git_workflow"], "A validated task workflow may perform this exact local Git action.", ["allow_validated_local_git"]
        if risk in {"DESTRUCTIVE", "HISTORY_REWRITE"}:
            return PolicyOutcome.DENY, ["destructive_git"], "Destructive or history-rewriting Git actions are denied by the transactional policy.", ["deny_destructive_git"]
        if risk == "REMOTE_WRITE":
            return PolicyOutcome.REQUIRE_APPROVAL, ["remote_git_write"], "A remote Git mutation requires exact approval and observable post-execution evidence.", ["approve_remote_git"]
        if risk in {"LOCAL_SAFE_WRITE", "LOCAL_HISTORY_WRITE"}:
            return PolicyOutcome.REQUIRE_APPROVAL, ["local_git_write"], "A local Git mutation requires exact approval.", ["approve_git_write"]
        return PolicyOutcome.DENY, ["unclassified_git_risk"], "The Git mutation risk was not safely classified.", ["default_deny_git"]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
