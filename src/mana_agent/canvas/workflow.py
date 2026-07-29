"""Owner-bound Canvas facade for workflow and multi-agent nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mana_agent.canvas.models import (
    CanvasSource,
    OwnerRef,
    RendererAction,
    SurfaceSnapshot,
)
from mana_agent.canvas.service import CanvasService


@dataclass(frozen=True, slots=True)
class CanvasNodeContext:
    service: CanvasService
    session_id: str
    conversation_id: str
    correlation_id: str
    owner: OwnerRef

    def create(
        self, surface_id: str, *, retain_on_complete: bool = True
    ) -> SurfaceSnapshot:
        return self.service.create_surface(
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            surface_id=surface_id,
            owner=self.owner,
            correlation_id=self.correlation_id,
            source=CanvasSource.NODE,
            retain_on_complete=retain_on_complete,
            workflow_id=self.owner.workflow_id,
            node_id=self.owner.node_id,
            task_id=self.owner.task_id,
            agent_id=self.owner.agent_id,
        )

    def update_components(
        self, surface_id: str, components: list[dict[str, Any]]
    ) -> SurfaceSnapshot:
        return self.service.update_components(
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            surface_id=surface_id,
            components=components,
            correlation_id=self.correlation_id,
            source=CanvasSource.NODE,
            workflow_id=self.owner.workflow_id,
            node_id=self.owner.node_id,
            task_id=self.owner.task_id,
            agent_id=self.owner.agent_id,
        )

    def update_data(
        self, surface_id: str, value: dict[str, Any], *, path: str = "/"
    ) -> SurfaceSnapshot:
        return self.service.update_data(
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            surface_id=surface_id,
            value=value,
            path=path,
            correlation_id=self.correlation_id,
            source=CanvasSource.NODE,
            workflow_id=self.owner.workflow_id,
            node_id=self.owner.node_id,
            task_id=self.owner.task_id,
            agent_id=self.owner.agent_id,
        )

    def wait_for_action(
        self, surface_id: str, action_name: str, *, timeout: float | None = None
    ) -> RendererAction:
        return self.service.wait_for_action(
            session_id=self.session_id,
            surface_id=surface_id,
            action_name=action_name,
            timeout=timeout,
        )

    def complete(self, surface_id: str) -> SurfaceSnapshot:
        return self.service.complete_surface(
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            surface_id=surface_id,
            correlation_id=self.correlation_id,
            source=CanvasSource.NODE,
            workflow_id=self.owner.workflow_id,
            node_id=self.owner.node_id,
            task_id=self.owner.task_id,
            agent_id=self.owner.agent_id,
        )
