from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.runtime.model_levels import resolve_model_for_role


RouteKind = Literal[
    "command",
    "tool_execution",
    "semantic_qa",
    "repo_search",
    "web_search",
    "github_search",
    "gitops",
    "coding_task",
    "analysis_task",
    "clarification",
    "unsupported",
]

ROUTE_KINDS: set[str] = {
    "command",
    "tool_execution",
    "semantic_qa",
    "repo_search",
    "web_search",
    "github_search",
    "gitops",
    "coding_task",
    "analysis_task",
    "clarification",
    "unsupported",
}


@dataclass(frozen=True)
class RouteDecision:
    kind: RouteKind
    confidence: float
    reason: str
    command_name: str | None = None
    command_args: list[str] = field(default_factory=list)
    tool_plan: list[dict[str, Any]] = field(default_factory=list)
    requires_index: bool = False
    requires_repo_context: bool = False
    requires_external_search: bool = False
    user_visible_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteRuntimeState:
    index_available: bool
    dir_mode: bool = False
    validation_error: str | None = None
    web_search_enabled: bool = True
    github_search_enabled: bool = True
    required_mcp_server: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouteDecisionError(RuntimeError):
    """Raised when the entry router cannot obtain a valid model decision."""


ENTRY_ROUTER_PROMPT = """You are Mana-Agent's ask/chat entry router.
Choose exactly one route for the user's request before any command, search, tool, or QnA chain runs.
Do not use fallback behavior. Do not choose actions by keyword alone.
Use runtime state to decide whether semantic indexes, repo context, external search, and commands are available.
If validation_error is present, choose a new valid route or explain why execution must stop.
If runtime_state.required_mcp_server is set, the user explicitly requires that
MCP provider. Select tool_execution with a tool from mcp__<required_mcp_server>__*
only. Never select web_search, github_search, or another provider as a substitute.

Route kinds:
- command: run a known Mana-Agent command only when command_name is in available_commands.
- tool_execution: execute a named tool/action from available_tools, such as command_inventory.
- semantic_qa: answer from the semantic index; requires_index must be true.
- repo_search: search/read local repository files directly.
- web_search: use public web search when enabled.
- github_search: use public GitHub search when enabled.
- gitops: execute explicit Git operations such as status, diff, add/stage, commit,
  amend, push, pull, fetch, branch, switch, checkout, merge, rebase, reset, revert,
  or tag through Git-capable tools. Choose this before repo_search for Git requests;
  branch names and words like commit/push are not repository search queries.
- coding_task: user wants repository files changed.
- analysis_task: user wants repository analysis.
- clarification: ask the user a concise clarifying question.
- unsupported: explain that no safe valid route is available.

For gitops, the route decision only selects the GitOps workflow. The GitOps
executor must inspect Git state first, then use a later model decision to choose
the exact commands, files to stage, commit message, branch action, and push action.
Do not encode a rigid Git command sequence in this route decision.

Return JSON only with this schema:
{
  "kind": "command|tool_execution|semantic_qa|repo_search|web_search|github_search|gitops|coding_task|analysis_task|clarification|unsupported",
  "confidence": 0.0,
  "reason": "short reason",
  "command_name": null,
  "command_args": [],
  "tool_plan": [{"tool": "command_inventory", "args": {}}],
  "requires_index": false,
  "requires_repo_context": false,
  "requires_external_search": false,
  "user_visible_message": null
}
"""


class EntryRouter:
    def __init__(self, *, llm: Any | None = None, router_model: str | None = None) -> None:
        self.llm = llm
        self.router_model = router_model or _router_model_name()

    def route(
        self,
        *,
        question: str,
        index_dir: str | Path | None,
        project_root: str | Path,
        available_commands: list[str] | None = None,
        available_tools: list[str] | None = None,
        runtime_state: RouteRuntimeState | dict[str, Any] | None = None,
    ) -> RouteDecision:
        if self.llm is None or not hasattr(self.llm, "invoke"):
            raise RouteDecisionError(
                "Model decision failed: entry_route. No action executed. Reason: routing model is unavailable."
            )
        state = runtime_state.to_dict() if isinstance(runtime_state, RouteRuntimeState) else dict(runtime_state or {})
        payload = {
            "question": str(question or ""),
            "index_dir": str(index_dir or ""),
            "project_root": str(Path(project_root).resolve()),
            "available_commands": sorted(set(available_commands or [])),
            "available_tools": sorted(set(available_tools or [])),
            "runtime_state": state,
        }
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=ENTRY_ROUTER_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                ]
            )
        except Exception as exc:  # noqa: BLE001 - converted to explicit routing failure
            raise RouteDecisionError(
                f"Model decision failed: entry_route. No action executed. Reason: {exc}"
            ) from exc
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        try:
            data = _extract_json(str(content))
            return validate_route_decision(data)
        except Exception as exc:  # noqa: BLE001 - validation failure blocks execution
            raise RouteDecisionError(
                f"Model decision failed: entry_route. No action executed. Reason: {exc}"
            ) from exc


def validate_route_decision(data: dict[str, Any]) -> RouteDecision:
    if not isinstance(data, dict):
        raise ValueError("router output must be a JSON object")
    kind = str(data.get("kind") or "").strip()
    if kind not in ROUTE_KINDS:
        raise ValueError(f"invalid route kind: {kind or '<missing>'}")
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    reason = str(data.get("reason") or "").strip()
    if not reason:
        raise ValueError("reason is required")
    command_name = data.get("command_name")
    if command_name is not None:
        command_name = str(command_name).strip() or None
    command_args = data.get("command_args") or []
    if not isinstance(command_args, list):
        raise ValueError("command_args must be a list")
    tool_plan = data.get("tool_plan") or []
    if not isinstance(tool_plan, list):
        raise ValueError("tool_plan must be a list")
    cleaned_plan: list[dict[str, Any]] = []
    for raw in tool_plan:
        if not isinstance(raw, dict):
            raise ValueError("tool_plan entries must be objects")
        tool = str(raw.get("tool") or "").strip()
        if not tool:
            raise ValueError("tool_plan entry missing tool")
        args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
        cleaned_plan.append({"tool": tool, "args": dict(args)})
    return RouteDecision(
        kind=kind,  # type: ignore[arg-type]
        confidence=confidence,
        reason=reason[:600],
        command_name=command_name,
        command_args=[str(item) for item in command_args],
        tool_plan=cleaned_plan,
        requires_index=bool(data.get("requires_index", False)),
        requires_repo_context=bool(data.get("requires_repo_context", False)),
        requires_external_search=bool(data.get("requires_external_search", False)),
        user_visible_message=(
            str(data.get("user_visible_message")).strip()
            if data.get("user_visible_message") is not None
            else None
        ),
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
    if not isinstance(data, dict):
        raise ValueError("router output must decode to an object")
    return data


def _router_model_name() -> str:
    main_model = os.getenv("MANA_MODEL_MAIN") or os.getenv("OPENAI_CHAT_MODEL") or ""
    return resolve_model_for_role(AgentRole.HEAD_DECISION, global_model=main_model).resolved_model
