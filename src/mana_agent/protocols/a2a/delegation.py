"""Explicit remote-delegation authorization and loop prevention."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from mana_agent.multi_agent.core.types import HandoffRecord, TaskStatus
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.protocols.common.exceptions import ProtocolPolicyError

from .types import DelegationEnvelope, RemoteAgentRecord


@dataclass(frozen=True, slots=True)
class DelegationPolicy:
    enabled: bool = False
    max_depth: int = 3
    allowed_skills: frozenset[str] = frozenset()
    allowed_workspaces: frozenset[str] = frozenset()

    def authorize(self, remote: RemoteAgentRecord, envelope: DelegationEnvelope, *, workspace_id: str, authentication_available: bool) -> None:
        if not self.enabled:
            raise ProtocolPolicyError("Remote A2A delegation is disabled.")
        if envelope.hop_count >= self.max_depth:
            raise ProtocolPolicyError("Maximum A2A delegation depth reached.")
        if remote.agent_id in envelope.visited_agents or remote.agent_id in envelope.delegation_chain:
            raise ProtocolPolicyError("A2A delegation loop detected.")
        if envelope.task_fingerprint in envelope.delegation_chain:
            raise ProtocolPolicyError("Repeated A2A task fingerprint detected.")
        if envelope.selected_skill not in set(remote.allowed_skills):
            raise ProtocolPolicyError("Remote agent is not authorized for the selected skill.")
        if self.allowed_skills and envelope.selected_skill not in self.allowed_skills:
            raise ProtocolPolicyError("Remote skill is denied by local policy.")
        if self.allowed_workspaces and workspace_id not in self.allowed_workspaces:
            raise ProtocolPolicyError("Workspace is denied by remote-delegation policy.")
        if remote.allowed_workspaces and workspace_id not in remote.allowed_workspaces:
            raise ProtocolPolicyError("Remote agent is not authorized for this workspace.")
        if remote.auth_reference and not authentication_available:
            raise ProtocolPolicyError("Remote A2A authentication is required.")
        if not envelope.approved_context.strip():
            raise ProtocolPolicyError("Delegation requires an explicitly approved context package.")


class RemoteDelegationService:
    """Track an explicitly authorized remote invocation in the local task board."""

    def __init__(self, *, root: str | Path, client: object, policy: DelegationPolicy) -> None:
        self.taskboard = TaskBoard(root)
        self.client = client
        self.policy = policy

    async def delegate(self, remote: RemoteAgentRecord, *, task: str, skill: str, bearer_token: str = "") -> list[object]:
        correlation_id = f"corr_{uuid.uuid4().hex[:20]}"
        fingerprint = hashlib.sha256(task.strip().encode("utf-8")).hexdigest()
        envelope = DelegationEnvelope(
            origin_agent_id="mana-agent",
            correlation_id=correlation_id,
            task_fingerprint=fingerprint,
            delegation_chain=("mana-agent",),
            visited_agents=frozenset({"mana-agent"}),
            approved_context=task,
            selected_skill=skill,
        )
        self.policy.authorize(
            remote,
            envelope,
            workspace_id=self.taskboard.store.workspace_id,
            authentication_available=bool(bearer_token or not remote.auth_reference),
        )
        local = self.taskboard.create_task(
            title=f"Remote A2A delegation to {remote.name}",
            user_request=task,
            normalized_goal=task,
            action_type="a2a_remote",
            workspace_id=self.taskboard.store.workspace_id,
            primary_repository_id=self.taskboard.store.repository_id,
        )
        self.taskboard.update_status(local.task_id, TaskStatus.ROUTED)
        self.taskboard.add_handoff(
            local.task_id,
            HandoffRecord(
                from_agent_id="mana-agent",
                to_agent_id=remote.agent_id,
                task_id=local.task_id,
                reason=f"A2A skill={skill}; correlation_id={correlation_id}; approved_context_only=true",
            ),
        )
        self.taskboard.update_status(local.task_id, TaskStatus.IN_PROGRESS)
        try:
            events = await self.client.delegate(remote.agent_id, task, bearer_token=bearer_token)
        except Exception:
            self.taskboard.update_status(local.task_id, TaskStatus.FAILED, reason="Remote A2A invocation failed after acceptance.")
            raise
        terminal = self._record_events(local.task_id, remote, correlation_id, events)
        if terminal == "completed":
            self.taskboard.update_status(local.task_id, TaskStatus.DONE)
        elif terminal == "cancelled":
            self.taskboard.update_status(local.task_id, TaskStatus.CANCELLED)
        elif terminal == "rejected":
            self.taskboard.update_status(local.task_id, TaskStatus.SKIPPED)
        elif terminal in {"input_required", "auth_required"}:
            self.taskboard.update_status(local.task_id, TaskStatus.BLOCKED, reason=f"Remote task requires {terminal.replace('_', ' ')}.")
        else:
            self.taskboard.update_status(local.task_id, TaskStatus.FAILED, reason="Remote A2A task did not reach successful completion.")
        return events

    def _record_events(self, local_task_id: str, remote: RemoteAgentRecord, correlation_id: str, events: list[object]) -> str:
        try:
            from a2a.types.a2a_pb2 import TaskState
        except ImportError:
            return "failed"
        terminal = "failed"
        state_names = {
            TaskState.TASK_STATE_COMPLETED: "completed",
            TaskState.TASK_STATE_FAILED: "failed",
            TaskState.TASK_STATE_CANCELLED: "cancelled",
            TaskState.TASK_STATE_REJECTED: "rejected",
            TaskState.TASK_STATE_INPUT_REQUIRED: "input_required",
            TaskState.TASK_STATE_AUTH_REQUIRED: "auth_required",
        }
        for event in events:
            remote_task_id = ""
            remote_context_id = ""
            state = 0
            if event.HasField("task"):
                remote_task_id = event.task.id
                remote_context_id = event.task.context_id
                state = event.task.status.state
            elif event.HasField("status_update"):
                remote_task_id = event.status_update.task_id
                remote_context_id = event.status_update.context_id
                state = event.status_update.status.state
            elif event.HasField("artifact_update"):
                remote_task_id = event.artifact_update.task_id
                remote_context_id = event.artifact_update.context_id
            elif event.HasField("message"):
                remote_task_id = event.message.task_id
                remote_context_id = event.message.context_id
                terminal = "completed"
            self.taskboard.record_tool_event(
                local_task_id,
                {
                    "type": "a2a.remote_event",
                    "remote_agent_id": remote.agent_id,
                    "remote_task_id": remote_task_id,
                    "remote_context_id": remote_context_id,
                    "correlation_id": correlation_id,
                },
            )
            if state in state_names:
                terminal = state_names[state]
        return terminal
