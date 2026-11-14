"""Model-selectable Teach Mode tools used by normal chat and agents."""

from __future__ import annotations

import json
from typing import Any, Literal

from mana_agent.llm.tools import ToolBinding
from pydantic import BaseModel, Field

from .models import TeachError
from .service import TeachService


class _Decision(BaseModel):
    source_decision_id: str = Field(min_length=1, description="Validated model decision selecting this exact operation.")


class _Start(_Decision):
    task_name: str = Field(min_length=1, max_length=240)
    permissions: list[str] = Field(default_factory=list)
    desktop: bool | None = None


class _Session(_Decision):
    session_id: str | None = None


class _Explain(_Session):
    explanation: str = Field(min_length=1, max_length=4000)


class _Flow(_Decision):
    flow_id: str


class _Replay(_Flow):
    mode: Literal["dry_run", "guided", "normal"] = "dry_run"
    inputs: dict[str, Any] = Field(default_factory=dict)


class _Handoff(_Flow):
    version: int | None = Field(default=None, ge=1)


def _response(operation) -> str:
    try:
        value = operation()
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", by_alias=True)
        return json.dumps({"ok": True, "result": value}, default=str)
    except (TeachError, ValueError) as exc:
        return json.dumps(
            {
                "ok": False,
                "error_code": "teach_operation_failed",
                "message": str(exc),
                "corrective_action": "Inspect `mana-agent teach doctor` and the flow/session status.",
            }
        )


def build_teach_tools() -> list[Any]:
    """Build lazy local tools; no recorder starts until the model selects start."""
    return [
        ToolBinding.from_callable(
            name="teach_start",
            description="Start an explicit visible local Teach Mode recording for a user-named task.",
            input_model=_Start,
            func=lambda task_name, permissions, desktop, source_decision_id: _response(
                lambda: TeachService().start(task_name, permissions=permissions, desktop=desktop)
            ),
        ),
        ToolBinding.from_callable(
            name="teach_stop",
            description="Stop an active Teach recording and compile a draft flow requiring review.",
            input_model=_Session,
            func=lambda source_decision_id, session_id=None: _response(
                lambda: {
                    "session": (result := TeachService().stop(session_id))[0].model_dump(mode="json"),
                    "flow": result[1].model_dump(mode="json", by_alias=True),
                }
            ),
        ),
        ToolBinding.from_callable(
            name="teach_explain",
            description="Attach a typed parameterization hint to the active recording.",
            input_model=_Explain,
            func=lambda explanation, source_decision_id, session_id=None: _response(
                lambda: TeachService().explain(explanation, session_id)
            ),
        ),
        ToolBinding.from_callable(
            name="teach_list_flows",
            description="List local learned Mana Flows and activation status.",
            input_model=_Decision,
            func=lambda source_decision_id: _response(lambda: TeachService().storage.list_flows()),
        ),
        ToolBinding.from_callable(
            name="teach_replay",
            description="Dry-run, guide, or normally replay an explicitly selected learned flow; verification determines success.",
            input_model=_Replay,
            func=lambda flow_id, mode, inputs, source_decision_id: _response(
                lambda: TeachService().replay(flow_id, mode=mode, inputs=inputs)
            ),
        ),
        ToolBinding.from_callable(
            name="teach_automation_handoff",
            description=(
                "Return secret-free structured metadata for an exact reviewed and verified "
                "Teach flow version before creating an automation."
            ),
            input_model=_Handoff,
            func=lambda flow_id, source_decision_id, version=None: _response(
                lambda: TeachService().automation_handoff(flow_id, version=version)
            ),
        ),
    ]
