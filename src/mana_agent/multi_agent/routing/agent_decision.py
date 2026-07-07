from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from mana_agent.multi_agent.routing.policies import classify_request
from mana_agent.tools.contracts import coding_tool_contracts


AgentIntent = Literal[
    "answer",
    "repo_search",
    "web_research",
    "analyze",
    "plan",
    "edit",
    "verify",
    "review",
    "tool",
    "high_risk_tool",
]

KNOWN_AGENT_TOOLS = frozenset(
    {
        "web_search",
        "repo_search",
        "repo_batch_search",
        "read_file",
        "repo_batch_read",
        "apply_patch",
        "apply_patch_batch",
        "edit_file",
        "multi_edit_file",
        "write_file",
        "create_file",
        "delete_file",
        "run_command",
        "verify_project",
        "git_status",
        "git_diff",
        "semantic_search",
        "list_files",
        "ls",
        "find_symbols",
        "call_graph",
        "read_skill",
    }
)

SAFETY_COMMAND_RE = re.compile(r"\b(git reset|git clean|sudo|rm -rf|force push)\b", re.I)


@dataclass(slots=True)
class AgentDecision:
    intent: AgentIntent
    confidence: float
    selected_tools: list[str] = field(default_factory=list)
    tool_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    repo_context_needed: bool = False
    web_search_needed: bool = False
    code_editing_needed: bool = False
    reasoning_summary: str = ""
    source: str = "model"
    verifier_passed: bool = False
    verifier_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentDecisionVerification:
    passed: bool
    summary: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def agent_tool_descriptions() -> list[dict[str, Any]]:
    descriptions = [
        {
            "name": "web_search",
            "description": (
                "Search the public web for current facts, official docs, unknown terms, "
                "recent information, and topics not answerable from the local repository."
            ),
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        }
    ]
    for contract in coding_tool_contracts():
        descriptions.append(
            {
                "name": contract.name,
                "description": contract.description,
                "input_schema": contract.input_schema,
            }
        )
    return descriptions


AGENT_DECISION_PROMPT = """You are Mana-Agent's routing decision layer.
Choose tools from the provided tool descriptions based on the user's intent.
Do not route by keywords alone. Infer whether the user needs local repository context, public web research, code editing, verification, planning, review, or a plain answer.
Explicit commands such as /analyze, /plan, or "search repo" are hints, but ordinary words like "search", "find", "web", "repo", "read", "edit", and "analyze" do not bypass your intent decision.
Use web_search for public/current/unknown-topic research, official docs, and questions that the local repo is unlikely to answer.
Use repo_search/read_file for local repository inspection.
Use apply_patch or edit/write tools only when the user wants code or files changed.
Return JSON only with this schema:
{
  "intent": "answer|repo_search|web_research|analyze|plan|edit|verify|review|tool|high_risk_tool",
  "confidence": 0.0,
  "selected_tools": ["tool_name"],
  "tool_inputs": {"tool_name": {"query": "compact query"}},
  "repo_context_needed": true,
  "web_search_needed": false,
  "code_editing_needed": false,
  "reasoning_summary": "short reason"
}
"""


class AgentDecisionEngine:
    def __init__(self, *, llm: Any | None = None, tool_descriptions: list[dict[str, Any]] | None = None) -> None:
        self.llm = llm
        self.tool_descriptions = tool_descriptions or agent_tool_descriptions()

    def decide(
        self,
        *,
        user_request: str,
        repo_context: str = "",
        memory_context: str = "",
        command_hint: str = "",
    ) -> AgentDecision:
        request = str(user_request or "").strip()
        if SAFETY_COMMAND_RE.search(request):
            decision = AgentDecision(
                intent="high_risk_tool",
                confidence=1.0,
                selected_tools=["run_command"],
                tool_inputs={"run_command": {"command": request}},
                repo_context_needed=False,
                web_search_needed=False,
                code_editing_needed=False,
                reasoning_summary="High-risk shell or git operation requires safety routing.",
                source="safety",
            )
            return self._with_verification(decision, request)

        model_decision = self._model_decision(
            request,
            repo_context=repo_context,
            memory_context=memory_context,
            command_hint=command_hint,
        )
        if model_decision is not None:
            return self._with_verification(model_decision, request)
        return self._with_verification(self._fallback_decision(request), request)

    def _model_decision(
        self,
        request: str,
        *,
        repo_context: str,
        memory_context: str,
        command_hint: str,
    ) -> AgentDecision | None:
        if self.llm is None or not hasattr(self.llm, "invoke"):
            return None
        payload = {
            "user_request": request,
            "command_hint": command_hint,
            "repo_context": repo_context[:1200],
            "memory_context": memory_context[:1200],
            "tools": self.tool_descriptions,
        }
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=AGENT_DECISION_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                ]
            )
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
            data = _extract_json(str(content))
            return self._decision_from_payload(data)
        except Exception:
            return None

    def _decision_from_payload(self, data: dict[str, Any]) -> AgentDecision:
        intent = str(data.get("intent") or "answer").strip().lower()
        if intent not in {
            "answer",
            "repo_search",
            "web_research",
            "analyze",
            "plan",
            "edit",
            "verify",
            "review",
            "tool",
            "high_risk_tool",
        }:
            intent = "answer"
        selected_tools = _clean_tool_list(data.get("selected_tools"))
        tool_inputs = _clean_tool_inputs(data.get("tool_inputs"), selected_tools)
        return AgentDecision(
            intent=intent,  # type: ignore[arg-type]
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5) or 0.5))),
            selected_tools=selected_tools,
            tool_inputs=tool_inputs,
            repo_context_needed=bool(data.get("repo_context_needed", False)),
            web_search_needed=bool(data.get("web_search_needed", False)),
            code_editing_needed=bool(data.get("code_editing_needed", False)),
            reasoning_summary=str(data.get("reasoning_summary") or "Model-routed agent decision.")[:500],
            source="model",
        )

    def _fallback_decision(self, request: str) -> AgentDecision:
        kind = classify_request(request)
        if kind == "coding":
            return AgentDecision(
                intent="edit",
                confidence=0.45,
                selected_tools=["repo_search", "read_file", "apply_patch"],
                tool_inputs={"repo_search": {"query": request[:240]}},
                repo_context_needed=True,
                code_editing_needed=True,
                reasoning_summary="Fallback used because model routing was unavailable.",
                source="fallback",
            )
        if kind == "analyze":
            return AgentDecision(
                intent="analyze",
                confidence=0.45,
                selected_tools=["repo_search", "read_file"],
                tool_inputs={"repo_search": {"query": request[:240]}},
                repo_context_needed=True,
                reasoning_summary="Fallback used because model routing was unavailable.",
                source="fallback",
            )
        if kind == "planning":
            return AgentDecision(intent="plan", confidence=0.45, reasoning_summary="Fallback used because model routing was unavailable.", source="fallback")
        if kind == "tool":
            return AgentDecision(intent="tool", confidence=0.45, selected_tools=["run_command"], reasoning_summary="Fallback used because model routing was unavailable.", source="fallback")
        if kind == "high_risk_tool":
            return AgentDecision(intent="high_risk_tool", confidence=0.8, selected_tools=["run_command"], reasoning_summary="Safety fallback selected high-risk tool route.", source="fallback")
        return AgentDecision(intent="answer", confidence=0.4, reasoning_summary="Fallback used because model routing was unavailable.", source="fallback")

    def _with_verification(self, decision: AgentDecision, request: str) -> AgentDecision:
        verification = verify_agent_decision(decision, user_request=request, tool_descriptions=self.tool_descriptions)
        decision.verifier_passed = verification.passed
        decision.verifier_summary = verification.summary
        return decision


def verify_agent_decision(
    decision: AgentDecision,
    *,
    user_request: str,
    tool_descriptions: list[dict[str, Any]] | None = None,
) -> AgentDecisionVerification:
    available = {str(item.get("name") or "") for item in (tool_descriptions or agent_tool_descriptions())}
    warnings: list[str] = []
    unknown = [tool for tool in decision.selected_tools if tool not in available and tool not in KNOWN_AGENT_TOOLS]
    if unknown:
        warnings.append(f"unknown tools selected: {', '.join(unknown)}")
    if decision.web_search_needed and "web_search" not in decision.selected_tools:
        warnings.append("web_search_needed=true but web_search was not selected")
    if decision.repo_context_needed and not any(tool in decision.selected_tools for tool in ("repo_search", "repo_batch_search", "read_file", "repo_batch_read", "semantic_search", "list_files")):
        warnings.append("repo_context_needed=true but no repository read/search tool was selected")
    if decision.code_editing_needed and not any(tool in decision.selected_tools for tool in ("apply_patch", "apply_patch_batch", "edit_file", "multi_edit_file", "write_file", "create_file", "delete_file")):
        warnings.append("code_editing_needed=true but no mutation tool was selected")
    if decision.intent == "web_research" and not decision.web_search_needed:
        warnings.append("web_research intent must set web_search_needed=true")
    if decision.intent in {"repo_search", "analyze", "edit", "review"} and not decision.repo_context_needed:
        warnings.append(f"{decision.intent} intent should request repository context")
    if not str(user_request or "").strip():
        warnings.append("empty user request")
    return AgentDecisionVerification(
        passed=not warnings,
        summary="selected tools are consistent with the routed intent" if not warnings else "; ".join(warnings),
        warnings=warnings,
    )


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    return data if isinstance(data, dict) else {}


def _clean_tool_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [tool for tool in (str(item).strip() for item in value) if tool]


def _clean_tool_inputs(value: Any, selected_tools: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, dict[str, Any]] = {}
    for tool in selected_tools:
        raw = value.get(tool)
        if isinstance(raw, dict):
            cleaned[tool] = dict(raw)
    return cleaned
