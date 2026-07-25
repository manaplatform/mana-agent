"""Exact target and action approvals; arbitrary is never implicitly trusted."""

from __future__ import annotations

from enum import Enum

from mana_agent.remote_execution.models import RemoteExecutionRequest


class TargetPolicyMode(str, Enum):
    PROMPT_EACH_TIME = "prompt_each_time"
    APPROVED_TARGETS = "approved_targets"
    UNRESTRICTED = "unrestricted"


def target_identity(request: RemoteExecutionRequest) -> str:
    key = request.authentication.key_path or "agent"
    target = request.target
    return f"{request.worker_id}|{target.user}@{target.host}:{target.port}|{key}"


class TargetPolicy:
    def __init__(self, mode: TargetPolicyMode = TargetPolicyMode.PROMPT_EACH_TIME) -> None:
        self.mode = mode
        self._approved: set[str] = set()
        self._approved_actions: set[str] = set()

    def approve_target(self, request: RemoteExecutionRequest) -> None:
        self._approved.add(target_identity(request))

    def revoke_target(self, request: RemoteExecutionRequest) -> None:
        self._approved.discard(target_identity(request))

    def approve_action(self, request: RemoteExecutionRequest) -> None:
        self._approved_actions.add(request.exact_action_key())

    def requires_approval(self, request: RemoteExecutionRequest) -> bool:
        if self.mode is TargetPolicyMode.UNRESTRICTED:
            return False
        if request.exact_action_key() in self._approved_actions:
            return False
        return self.mode is TargetPolicyMode.PROMPT_EACH_TIME or target_identity(request) not in self._approved
