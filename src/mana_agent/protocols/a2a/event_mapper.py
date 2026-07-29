"""Map user-safe Mana progress into official A2A events."""

from __future__ import annotations

from typing import Any

from mana_agent.protocols.common.security import redact_protocol_value


class A2AEventMapper:
    A2UI_EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.8"

    def canvas_capabilities(self, context: Any) -> dict[str, Any] | None:
        """Return negotiated capabilities only when the A2UI extension is activated."""
        if self.A2UI_EXTENSION_URI not in set(getattr(context, "requested_extensions", set()) or set()):
            return None
        metadata = dict(getattr(context, "metadata", {}) or {})
        capabilities = metadata.get("a2uiClientCapabilities")
        if not isinstance(capabilities, dict):
            return None
        version = capabilities.get("v0.9", capabilities)
        return version if isinstance(version, dict) else None

    def canvas(
        self, *, task_id: str, context_id: str, event: Any,
        capabilities: dict[str, Any] | None,
    ) -> Any | None:
        if capabilities is None or not isinstance(event, dict):
            return None
        metadata = dict(event.get("metadata") or event.get("details") or {})
        envelope = metadata.get("canvas_event")
        if not isinstance(envelope, dict):
            return None
        from mana_agent.canvas.config import MANA_CATALOG_ID
        if MANA_CATALOG_ID not in set(capabilities.get("supportedCatalogIds") or []):
            return None
        from a2a.helpers.proto_helpers import new_data_artifact_update_event
        return new_data_artifact_update_event(
            task_id, context_id, "canvas", [envelope.get("payload")],
            media_type="application/a2ui+json", append=True, last_chunk=False,
            artifact_id=f"artifact-{task_id}-canvas",
        )

    def progress(self, *, task_id: str, context_id: str, event: Any) -> Any | None:
        from a2a.helpers.proto_helpers import new_text_status_update_event
        from a2a.types.a2a_pb2 import TaskState

        text = ""
        if isinstance(event, dict):
            text = str(event.get("summary") or event.get("message") or event.get("title") or "")
        else:
            text = str(getattr(event, "summary", "") or "")
        if not text:
            return None
        safe = str(redact_protocol_value(text))[:2000]
        return new_text_status_update_event(task_id, context_id, TaskState.TASK_STATE_WORKING, safe)
