from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

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
        "github_search",
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
        "browser_open",
        "browser_inspect",
        "browser_click",
        "browser_type",
        "browser_select",
        "browser_scroll",
        "browser_wait",
        "browser_screenshot",
        "browser_upload",
        "browser_download",
        "browser_check_links",
        "browser_back",
        "browser_tabs",
        "browser_switch_tab",
        "browser_close",
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
        },
        {
            "name": "github_search",
            "description": (
                "Search public GitHub repositories, code, issues, and project metadata for external open-source "
                "projects or examples. Use with web_search when the user asks for internet and GitHub research."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "github_kind": {"enum": ["repositories", "code", "issues"]},
                    "repo": {"type": "string"},
                },
                "required": ["query"],
            },
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
    from mana_agent.config.user_config import get_setting

    browser_tool_contracts = None
    if bool(get_setting("MANA_BROWSER_ENABLED", True)):
        try:
            from mana_agent.connectors.browser.contracts import browser_tool_contracts
        except ImportError:
            browser_tool_contracts = None
    if browser_tool_contracts is not None:
        for contract in browser_tool_contracts():
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
Browser/search boundary: web_search cannot inspect a rendered target page or interact with it. When the user supplies a target URL and asks to check the website, inspect visible controls, forms, buttons, page title, navigation, functionality, authentication, or broken links, select browser tools with intent="tool". Do not select web_search as a substitute, and do not combine web_search with browser tools unless the user separately requests broader public-web research.
Use github_search for public GitHub project/repository/code research.
Use both web_search and github_search when the user asks for internet/web plus GitHub.
Use repo_search/read_file for local repository inspection.
Use browser tools for interactive website tasks that require navigation, page inspection, forms, clicks, uploads, downloads, tabs, account creation, sign-up, login, or authenticated browser state. Website actions do not edit repository code: set intent="tool", repo_context_needed=false, and code_editing_needed=false. Words such as create, change, submit, delete, or edit refer to the website when their target is a page, account, form, or URL; they must not select repository mutation tools. Select browser_open and browser_inspect plus the browser interaction capabilities the browser operator may need. Choose each concrete action later from current page evidence; do not assume a website-specific workflow.
Never select browser actions intended to bypass CAPTCHA, MFA, access restrictions, or website security controls. Sensitive or irreversible final actions require explicit user approval before execution.
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

AGENT_DECISION_REVIEW_PROMPT = """You are Mana-Agent's routing decision reviewer.
Review the proposed structured decision against the user request and available
tool descriptions. Return one complete corrected decision using exactly the
same JSON schema as the routing decision. This review is required before any
tool executes.

Enforce these boundaries:
- Rendered website inspection, visible controls, page functionality, forms,
  navigation, authentication, and link checking require browser_* tools with
  intent=tool, repo_context_needed=false, web_search_needed=false, and
  code_editing_needed=false.
- web_search is only for public information retrieval and cannot substitute for
  inspecting or interacting with a target URL.
- Website account/form/content actions are not repository edits.
- Repository mutation tools are valid only when repository files must change.
- Select only tools present in available_tools. Preserve a valid proposal; fix
  an invalid one. Do not infer a static route from keywords alone.
Return JSON only.
"""


class AgentDecisionEngine:
    def __init__(
        self,
        *,
        llm: Any | None = None,
        tool_descriptions: list[dict[str, Any]] | None = None,
        enable_fallback: bool = True,
    ) -> None:
        self.llm = llm
        self.tool_descriptions = tool_descriptions or agent_tool_descriptions()
        self.enable_fallback = False

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
        return self._with_verification(
            AgentDecision(
                intent="answer",
                confidence=0.0,
                reasoning_summary="Model routing decision was unavailable. No alternate route was executed.",
                source="model_unavailable",
            ),
            request,
        )

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
            proposed = self._decision_from_payload(data)
            review_response = self.llm.invoke(
                [
                    SystemMessage(content=AGENT_DECISION_REVIEW_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "user_request": request,
                                "proposed_decision": proposed.to_dict(),
                                "available_tools": self.tool_descriptions,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                ]
            )
            review_content = getattr(review_response, "content", review_response)
            if isinstance(review_content, list):
                review_content = " ".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in review_content
                )
            return self._decision_from_payload(_extract_json(str(review_content)))
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
    if decision.web_search_needed and not any(tool in decision.selected_tools for tool in ("web_search", "github_search")):
        warnings.append("web_search_needed=true but no external search tool was selected")
    if decision.repo_context_needed and not any(tool in decision.selected_tools for tool in ("repo_search", "repo_batch_search", "read_file", "repo_batch_read", "semantic_search", "list_files")):
        warnings.append("repo_context_needed=true but no repository read/search tool was selected")
    if decision.code_editing_needed and not any(tool in decision.selected_tools for tool in ("apply_patch", "apply_patch_batch", "edit_file", "multi_edit_file", "write_file", "create_file", "delete_file")):
        warnings.append("code_editing_needed=true but no mutation tool was selected")
    browser_selected = any(tool.startswith("browser_") for tool in decision.selected_tools)
    if browser_selected and decision.intent != "tool":
        warnings.append("browser tools require intent=tool")
    if browser_selected and (decision.repo_context_needed or decision.code_editing_needed):
        warnings.append("browser tasks must not request repository context or code editing")
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
