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
    """Structured decision produced by model judgment on approval requests."""

    judgment: Literal["accept", "deny"]
    reason: str
    suggested_command: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    risk_assessment: str = ""


_DANGEROUS_COMMAND_PATTERNS = (
    r"\brm\s+-[a-zA-Z]*r",
    r"\brm\s+-[a-zA-Z]*f",
    r"\brmdir\b",
    r"\bsudo\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[a-zA-Z]*f",
    r"\bgit\s+push\s+.*--force\b",
    r"\bformat\s+[a-zA-Z]:",
    r"\bmkfs\b",
    r"\b(shutdown|reboot)\b",
)


def judge_approval_request(
    *,
    command: str = "",
    reason: str = "",
    goal: str = "",
    working_directory: str = "",
    risk_level: str = "medium",
    llm: Any = None,
    action_id: str = "",
) -> ModelApprovalJudgment:
    """Evaluate an approval request using model judgment, enforcing deterministic safety invariants."""
    clean_cmd = str(command or "").strip()

    # Deterministic safety gate: hard-deny known destructive commands
    for pat in _DANGEROUS_COMMAND_PATTERNS:
        if re.search(pat, clean_cmd, re.IGNORECASE):
            return ModelApprovalJudgment(
                judgment="deny",
                reason=f"Deterministic security policy: destructive command pattern detected ({clean_cmd!r}).",
                suggested_command="",
                confidence=1.0,
                risk_assessment="high_risk_destructive_shell",
            )

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
        # Fail-closed when no model is available to make the decision
        return ModelApprovalJudgment(
            judgment="deny",
            reason="No model judgment engine is available to evaluate the approval request.",
            suggested_command="",
            confidence=0.0,
            risk_assessment="unverified_model_unavailable",
        )

    system_prompt = (
        "You are Mana-Agent's authoritative Model Approval Judge.\n"
        "Your role is to evaluate whether a requested shell command or tool execution should be ACCEPTED or DENIED.\n\n"
        "CRITICAL RULES:\n"
        "1. ACCEPT (judgment='accept') commands that are safe, non-destructive, and genuinely needed to inspect, build, test, verify, or advance the user's task goal.\n"
        "2. DENY (judgment='deny') any command that is destructive, dangerous, or unneeded.\n"
        "   - Commands like rm, rmdir, git reset --hard, git clean -fd, sudo, or accessing secrets must ALWAYS be DENIED.\n"
        "3. If a command is suboptimal or an alternative command should be run instead, provide the exact alternative in 'suggested_command'.\n"
        "4. Return strictly one valid JSON object conforming to this schema:\n"
        "{\n"
        '  "judgment": "accept" | "deny",\n'
        '  "reason": "Clear justification for accepting or denying",\n'
        '  "suggested_command": "Alternative command to run if different from target, else empty",\n'
        '  "confidence": 0.0 - 1.0,\n'
        '  "risk_assessment": "Summary of safety and impact analysis"\n'
        "}\n"
        "Do not include markdown code fences or conversational prose."
    )

    user_payload = {
        "command": clean_cmd,
        "reason": reason or "Execute command to advance or verify task",
        "task_goal": goal,
        "working_directory": str(working_directory or "."),
        "risk_level": risk_level,
        "action_id": action_id,
    }

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
        judgment_val = str(data.get("judgment") or "deny").strip().lower()
        if judgment_val not in {"accept", "deny"}:
            judgment_val = "accept" if judgment_val in {"approved", "approve", "allow", "yes"} else "deny"
        return ModelApprovalJudgment(
            judgment=judgment_val,  # type: ignore[arg-type]
            reason=str(data.get("reason") or "Model evaluated command safety."),
            suggested_command=str(data.get("suggested_command") or "").strip(),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 1.0)))),
            risk_assessment=str(data.get("risk_assessment") or ""),
        )
    except Exception as exc:
        logger.warning("Model approval judgment evaluation failed: %s", exc)
        return ModelApprovalJudgment(
            judgment="deny",
            reason=f"Model approval judgment evaluation encountered an error: {exc}",
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

