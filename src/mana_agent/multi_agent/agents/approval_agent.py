"""Approval agent for model-driven human approval and decision gating."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal
from pydantic import BaseModel, Field

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.utils.text import extract_model_text

logger = logging.getLogger(__name__)

APPROVAL_ALLOWED_TOOLS = [
    "approval_request",
    "request_user_approval",
    "wait_for_approval",
    "git_status",
    "git_diff",
    "run_command",
]


class ModelApprovalJudgment(BaseModel):
    """Advisory assessment, not a human decision or execution authorization."""

    judgment: Literal["accept", "approval", "deny"]
    reason: str
    suggested_command: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    risk_assessment: str = ""


def judge_approval_request(
    *,
    command: str = "",
    reason: str = "",
    goal: str = "",
    working_directory: str = "",
    risk_level: str = "medium",
    llm: Any = None,
    action_id: str = "",
    error: str = "",
) -> ModelApprovalJudgment:
    """Assess an action; deterministic policy and human approval remain authoritative."""
    clean_cmd = str(command or "").strip()

    if llm is None:
        try:
            from mana_agent.config.user_config import get_setting
            from mana_agent.multi_agent.runtime.compatibility import create_chat_model

            model = (
                get_setting("MANA_MODEL_APPROVAL", "")
                or get_setting("MANA_PRIMARY_MODEL", "")
                or get_setting("OPENAI_MODEL", "")
                or ""
            )
            api_key = get_setting("OPENAI_API_KEY", "") or get_setting("MANA_API_KEY", "") or ""
            if model and api_key:
                llm = create_chat_model(api_key=api_key, model=model.removeprefix("openai/"))
        except Exception as exc:
            logger.debug("Could not resolve default LLM for approval judgment: %s", exc)

    if llm is None:
        return ModelApprovalJudgment(
            judgment="approval",
            reason="No model judgment engine is available to evaluate the approval request; human approval is required.",
            suggested_command="",
            confidence=0.0,
            risk_assessment="unverified_model_unavailable",
        )

    system_prompt = (
        "You are Mana-Agent's advisory action risk assessor.\n"
        "Your assessment cannot authorize execution, replace a human decision, or override deterministic policy.\n"
        "Assess whether a proposed command appears acceptable, needs human approval, or should be denied.\n\n"
        "CRITICAL RULES:\n"
        "1. ACCEPT (judgment='accept'): Safe, non-destructive, or verified commands needed to inspect, build, test, verify, or retry/advance the user's task goal.\n"
        "2. REQUIRE HUMAN APPROVAL (judgment='approval'): Consequential, state-modifying, or sensitive commands (such as file deletions, hard resets, privilege escalation, or irreversible external changes) where human review and explicit consent are required.\n"
        "3. DENY (judgment='deny'): Malicious, completely unrelated, or prohibited actions that must not run under any circumstances.\n"
        "4. If an execution error occurred, you may suggest an alternative in 'suggested_command'; it is not authorized and must not run automatically.\n"
        "5. Return strictly one valid JSON object conforming to this schema:\n"
        "{\n"
        '  "judgment": "accept" | "approval" | "deny",\n'
        '  "reason": "Clear justification for the decision",\n'
        '  "suggested_command": "Alternative command to run if different from target, else empty",\n'
        '  "confidence": 0.0 - 1.0,\n'
        '  "risk_assessment": "Summary of safety and impact analysis"\n'
        "}\n"
        "Do not include markdown code fences or conversational prose."
    )

    user_payload: dict[str, Any] = {
        "command": clean_cmd,
        "reason": reason or "Execute command to advance or verify task",
        "task_goal": goal,
        "working_directory": str(working_directory or "."),
        "risk_level": risk_level,
        "action_id": action_id,
    }
    if error:
        user_payload["execution_error"] = str(error)

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
    ]

    try:
        raw_res = llm.invoke(messages) if hasattr(llm, "invoke") else llm(messages)
        text = extract_model_text(raw_res).strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        data = json.loads(text)
        raw_val = str(data.get("judgment") or "deny").strip().lower()
        if raw_val in {"accept", "approved", "approve", "allow", "yes"}:
            judgment_val: Literal["accept", "approval", "deny"] = "accept"
        elif raw_val in {"approval", "require_approval", "needs_approval", "ask_user", "human_approval"}:
            judgment_val = "approval"
        else:
            judgment_val = "deny"

        return ModelApprovalJudgment(
            judgment=judgment_val,
            reason=str(data.get("reason") or "Model evaluated command safety."),
            suggested_command=str(data.get("suggested_command") or "").strip(),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 1.0)))),
            risk_assessment=str(data.get("risk_assessment") or ""),
        )
    except Exception as exc:
        logger.warning("Model approval judgment evaluation failed: %s", exc)
        return ModelApprovalJudgment(
            judgment="approval",
            reason=f"Model approval judgment evaluation encountered an error ({exc}); human approval is required.",
            confidence=0.0,
            risk_assessment="evaluation_error",
        )


class ApprovalAgent(BaseAgent):
    """Specialist agent responsible for human approvals, decision gating, and verification."""

    def __init__(
        self,
        agent_id: str = "agent_approval_0001",
        *,
        role: AgentRole = AgentRole.APPROVAL,
        parent_agent_id: str | None = None,
        capabilities: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        mailbox: Any = None,
        taskboard: Any = None,
        message_bus: Any = None,
        inbox_service: Any = None,
        execution_supervisor: Any = None,
        llm: Any = None,
        **kwargs,
    ) -> None:
        from mana_agent.multi_agent.registry.capability_registry import DEFAULT_CAPABILITIES

        caps = (
            list(capabilities)
            if capabilities is not None
            else list(DEFAULT_CAPABILITIES.get(role, []))
        )
        tools_list = list(
            allowed_tools if allowed_tools is not None else APPROVAL_ALLOWED_TOOLS
        )
        super().__init__(
            agent_id=agent_id,
            role=role,
            parent_agent_id=parent_agent_id,
            capabilities=caps,
            allowed_tools=tools_list,
            mailbox=mailbox,
            taskboard=taskboard,
            message_bus=message_bus,
            **kwargs,
        )
        self.inbox_service = inbox_service
        self.execution_supervisor = execution_supervisor
        self.llm = llm

    def tools(self) -> list[str]:
        """Return the allowed tools for the approval agent."""
        return list(self.allowed_tools)

    def evaluate_approval(
        self,
        *,
        command: str = "",
        reason: str = "",
        goal: str = "",
        working_directory: str = "",
        risk_level: str = "medium",
        action_id: str = "",
        error: str = "",
    ) -> ModelApprovalJudgment:
        """Evaluate an approval request using this agent's model."""
        return judge_approval_request(
            command=command,
            reason=reason,
            goal=goal,
            working_directory=working_directory,
            risk_level=risk_level,
            llm=self.llm,
            action_id=action_id,
            error=error,
        )

    def may_continue(self, result: Any) -> bool:
        """Return True only when the approval verdict is explicitly approved."""
        if result == "approved":
            return True
        if isinstance(result, dict):
            return result.get("verdict") == "approved" and bool(result.get("may_continue", True))
        verdict = getattr(result, "verdict", None)
        if verdict is not None:
            may_cont = getattr(result, "may_continue", True)
            return verdict == "approved" and bool(may_cont)
        return False


__all__ = [
    "APPROVAL_ALLOWED_TOOLS",
    "ApprovalAgent",
    "ModelApprovalJudgment",
    "judge_approval_request",
]

