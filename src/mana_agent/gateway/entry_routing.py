"""Model-driven entry routing and dynamic route availability for chat turns."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from mana_agent.evals.ids import stable_hash
from mana_agent.evals.recorder import record_current


EntryRouteName = Literal[
    "conversation",
    "coding",
    "gmail",
    "calendar",
    "computer",
    "browser",
    "search",
    "github",
    "repository",
    "memory",
    "automation",
    "artifact",
    "command",
    "unsupported",
    "capability_error",
]

RequiredSource = Literal[
    "repository", "browser", "search", "gmail", "calendar", "computer", "github",
    "memory", "artifact", "internal_knowledge", "none",
]

REQUIRED_SOURCES: set[str] = {
    "repository", "browser", "search", "gmail", "calendar", "computer", "github",
    "memory", "artifact", "internal_knowledge", "none",
}
TOOL_SOURCES = REQUIRED_SOURCES - {"internal_knowledge", "none"}


@dataclass(frozen=True, slots=True)
class RouteAvailability:
    available: bool
    configured: bool = True
    authorized: bool = True
    reason: str = ""
    setup_action: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RouteRegistration:
    name: EntryRouteName
    description: str
    availability: Callable[[], RouteAvailability]
    tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntryRouteContext:
    session_id: str
    conversation_id: str
    turn_id: str
    previous_route: str = ""
    conversation_summary: str = ""
    artifact_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EntryRoutingDecision:
    route: EntryRouteName
    confidence: float
    reason: str
    required_sources: tuple[RequiredSource, ...]
    target_urls: tuple[str, ...] = ()
    requires_live_data: bool = False
    reason_code: str = ""
    error_code: str = ""
    reuse_active_route: bool = False
    command_name: str = ""
    command_arguments: tuple[str, ...] = ()
    source: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntryRoutingError(RuntimeError):
    """The model did not return a valid entry-routing decision."""


class EntryRouteRegistry:
    """Registry of execution routes and their live runtime availability."""

    def __init__(self) -> None:
        self._routes: dict[str, RouteRegistration] = {}

    def register(self, registration: RouteRegistration) -> None:
        name = str(registration.name).strip()
        if not name:
            raise ValueError("entry route name is required")
        self._routes[name] = registration

    def get(self, name: str) -> RouteRegistration:
        try:
            return self._routes[str(name)]
        except KeyError as exc:
            raise EntryRoutingError(f"Unknown entry route: {name or '<missing>'}") from exc

    def snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in sorted(self._routes):
            registration = self._routes[name]
            availability = registration.availability()
            rows.append(
                {
                    "name": registration.name,
                    "description": registration.description,
                    "tools": list(registration.tools),
                    "availability": availability.to_dict(),
                }
            )
        return rows

    @property
    def names(self) -> set[str]:
        return set(self._routes)


ENTRY_ROUTER_PROMPT = """You are Mana-Agent's first-turn entry router.
Select exactly one registered execution route and its exact required information sources before any
conversational response is generated.
Routing is independent from response generation: return a decision only and never answer the user.

Use the supplied live route registry. A route may be selected when unavailable so its executor can
return the registry's truthful setup or authorization error; do not send a supported connector
request to conversation merely because its connector is unavailable.

Route semantics:
- command: a request equivalent to one command exposed by the supplied command route tools. Return
  its canonical command_name and structured string command_arguments; never execute it directly.
- conversation: ordinary discussion that needs no tool, connector, repository, or coding action.
- coding: repository code/file changes handled by the Codex coding workflow.
  Explicit, user-authorized terminal work—including SSH to a named host using a
  supplied local identity file or SSH-agent identity—selects coding so the
  model-selected tool workflow can execute and report it in chat. Preserve the
  exact host, user, identity-file path, and requested remote action. Never
  invent remote connection details, and do not select unsupported merely
  because a request is expressed as a command line.
- artifact: creation, editing, conversion, inspection, or export of a user-provided document, spreadsheet, presentation, PDF, or image. A user artifact is not repository code, even when it has a filename. Use the supplied artifact_evidence, including provenance and repository membership. Only select coding when the resolved target is a repository member and the requested change is a repository edit.
- gmail: inspect or act on the user's Gmail/email account through registered email tools.
- calendar: calendar account operations through a registered account/cloud calendar connector.
- computer: permission-aware control of the local desktop, installed applications, native calendar,
  media, notes, clipboard, screenshots, filesystem, notifications, browser application, or system.
- browser: direct public-page inspection using browser tools. A supplied public HTTP(S) URL is a
  strong signal for this route; page content, HTML, metadata, links, robots, and sitemap content
  require browser rather than search snippets.
- search: current public-web discovery, mentions, competitors, indexing, or other search-visible
  information. Search snippets never substitute for browser page inspection.
- github: connected/public GitHub information.
- repository: read-only local repository questions or inspection.
- memory: explicitly requested persisted memory retrieval.
- automation: create, inspect, or manage an automation.
- unsupported: no registered route can represent the request safely.

Repository context is only one possible evidence source. Current mailbox/account data is never ordinary conversation. Requests to check an inbox, latest
email, Gmail message, email thread, or mailbox must select gmail when that registered route
represents the request. The conversation route must never speculate about connector availability.
Use computer—not calendar—for an explicitly native/installed desktop calendar, and computer—not
browser—for the user's current installed browser page or tabs. Do not choose computer merely as an
availability fallback for a cloud calendar or isolated public-browser request.

Use previous_route and conversation_summary only for continuity. Reuse the active route for a true
follow-up; reroute when the user's intent changes. Do not route by isolated keywords alone.

The required_sources array is an execution contract: every listed tool source is mandatory and
must complete successfully before response generation. Never substitute a source or provider.
Use only identifiers from required_source_vocabulary in the request payload, and obey its
required_source_rules. Route registry `tools` are executor tool names, not source identifiers;
never copy a tool name or route name into required_sources unless it is explicitly present in the
source vocabulary. In particular, command is tool-free and requires exactly ["none"].
When no supported available capability can satisfy a required source, return route capability_error
with that source and an exact error_code. unsupported is a distinct model decision, never a
fallback. Direct URL signals are supplied separately; do not treat them as repository evidence.

Return JSON only:
{
  "route": "conversation|coding|artifact|command|gmail|calendar|computer|browser|search|github|repository|memory|automation|unsupported|capability_error",
  "confidence": 0.0,
  "reason": "short routing reason",
  "required_sources": ["browser"],
  "target_urls": ["https://example.com"],
  "requires_live_data": true,
  "reason_code": "DIRECT_PAGE_INSPECTION",
  "error_code": "",
  "reuse_active_route": false,
  "command_name": "sessions",
  "command_arguments": ["list"]
}

Examples:
- “Review https://example.com/about” -> browser, ["browser"], that URL.
- “Check https://example.com and prepare a complete SEO report” -> browser with search only when
  public indexing/discovery is independently required; both sources are mandatory if selected.
- “Find competitors for example.com” -> search, ["search"].
- “Improve metadata in this repository” -> coding, ["repository"].
- “Check my latest Gmail” -> gmail, ["gmail"], even when Gmail is unavailable; use
  capability_error with GMAIL_NOT_AVAILABLE rather than repository, memory, or conversation.
"""


class EntryRouter:
    """Obtain and validate the single entry decision for one gateway turn."""

    def __init__(self, *, llm: Any, registry: EntryRouteRegistry) -> None:
        self.llm = llm
        self.registry = registry

    def route(
        self,
        *,
        user_prompt: str,
        context: EntryRouteContext,
    ) -> EntryRoutingDecision:
        if self.llm is None or not callable(getattr(self.llm, "invoke", None)):
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: routing model is unavailable."
            )
        payload = {
            "user_prompt": str(user_prompt or "").strip(),
            "context": context.to_dict(),
            "routes": self.registry.snapshot(),
            "required_source_vocabulary": sorted(REQUIRED_SOURCES),
            "required_source_rules": {
                "conversation": [["none"]],
                "command": [["none"]],
                "unsupported": [["none"]],
                "coding": [["repository"]],
                "artifact": [["artifact"]],
                "gmail": [["gmail"]],
                "calendar": [["calendar"]],
                "computer": [["computer"]],
                "browser": [["browser"], ["browser", "search"]],
                "search": [["search"]],
                "github": [["github"]],
                "repository": [["repository"]],
                "memory": [["memory"]],
                "automation": [["repository"]],
                "capability_error": "one or more unavailable source identifiers",
            },
            "direct_url_signals": _public_urls(user_prompt),
        }
        try:
            started = time.perf_counter()
            response = self.llm.invoke(
                [
                    SystemMessage(content=ENTRY_ROUTER_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                ]
            )
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            decision = self._validate(_extract_json(str(content)))
            record_current(
                "model.decision",
                {
                    "boundary": "entry_router",
                    "prompt_template": "ENTRY_ROUTER_PROMPT",
                    "prompt_hash": stable_hash(ENTRY_ROUTER_PROMPT),
                    "request_hash": stable_hash(payload),
                    "response": decision.to_dict(),
                    "usage": getattr(response, "usage_metadata", None),
                    "latency_seconds": time.perf_counter() - started,
                },
            )
            return decision
        except EntryRoutingError:
            raise
        except Exception as exc:
            record_current("model.call.failed", {"boundary": "entry_router", "error_type": type(exc).__name__, "error": str(exc)})
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                f"Reason: {exc}"
            ) from exc

    def _validate(self, payload: dict[str, Any]) -> EntryRoutingDecision:
        route = str(payload.get("route") or "").strip()
        if route not in self.registry.names:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                f"Reason: unknown route {route or '<missing>'}."
            )
        try:
            confidence = float(payload.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: confidence must be numeric."
            ) from exc
        if not 0.0 <= confidence <= 1.0:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: confidence must be between 0 and 1."
            )
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: routing reason is required."
            )
        raw_sources = payload.get("required_sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: required_sources must be a non-empty list."
            )
        sources = tuple(str(item).strip() for item in raw_sources)
        unknown_sources = sorted({source for source in sources if source not in REQUIRED_SOURCES})
        if unknown_sources:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: required_sources contains unknown source identifier(s): "
                f"{', '.join(unknown_sources)}. Allowed values: {', '.join(sorted(REQUIRED_SOURCES))}."
            )
        if len(set(sources)) != len(sources) or ("none" in sources and len(sources) != 1):
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: invalid required_sources combination.")
        if route in {"conversation", "command", "unsupported"} and sources != ("none",):
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: tool-free routes require required_sources=[\"none\"].")
        if route == "capability_error" and not any(source in TOOL_SOURCES for source in sources):
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: capability_error requires an unavailable tool source.")
        if route not in {"conversation", "command", "unsupported", "capability_error"} and not any(source in TOOL_SOURCES for source in sources):
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: executable route requires a tool source.")
        target_urls = tuple(str(item).strip() for item in (payload.get("target_urls") or []) if str(item).strip())
        if not isinstance(payload.get("target_urls") or [], list) or any(not url.startswith(("http://", "https://")) for url in target_urls):
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: target_urls must contain valid HTTP(S) URLs.")
        if "browser" in sources and _public_urls(payload.get("user_prompt", "")):
            # User prompt is not returned by the model; direct-url validation is performed below
            # against target_urls when the model declares browser inspection.
            pass
        if "browser" in sources and not target_urls:
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: browser source requires target_urls.")
        error_code = str(payload.get("error_code") or "").strip()
        if route == "capability_error" and not error_code:
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: capability_error requires error_code.")
        availability = {row["name"]: bool(row["availability"]["available"]) for row in self.registry.snapshot()}
        source_routes = {"browser": "browser", "search": "search", "github": "github", "repository": "repository", "gmail": "gmail", "calendar": "calendar", "computer": "computer", "memory": "memory"}
        unavailable = [source for source in sources if source in source_routes and not availability.get(source_routes[source], False)]
        if unavailable and route != "capability_error":
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                f"Reason: selected unavailable source(s): {', '.join(unavailable)}."
            )
        if route == "capability_error":
            available_sources = [
                source
                for source in sources
                if source in source_routes and availability.get(source_routes[source], False)
            ]
            if available_sources:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: capability_error declared available source(s) unavailable: "
                    f"{', '.join(available_sources)}."
                )
        command_name = str(payload.get("command_name") or "").strip().lower().lstrip("/")
        raw_command_arguments = payload.get("command_arguments") or []
        command_registration = self.registry.get("command") if "command" in self.registry.names else None
        if route == "command":
            if command_registration is None or command_name not in command_registration.tools:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: command route selected an unknown command."
                )
            if not isinstance(raw_command_arguments, list) or any(not isinstance(item, str) for item in raw_command_arguments):
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: command_arguments must be a list of strings."
                )
        return EntryRoutingDecision(
            route=route,  # type: ignore[arg-type]
            confidence=confidence,
            reason=reason,
            required_sources=sources,  # type: ignore[arg-type]
            target_urls=target_urls,
            requires_live_data=bool(payload.get("requires_live_data", False)),
            reason_code=str(payload.get("reason_code") or "").strip(),
            error_code=error_code,
            reuse_active_route=bool(payload.get("reuse_active_route", False)),
            command_name=command_name,
            command_arguments=(
                tuple(raw_command_arguments)
                if isinstance(raw_command_arguments, list)
                and all(isinstance(item, str) for item in raw_command_arguments)
                else ()
            ),
        )


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("router output must be a JSON object")
    return payload


def _public_urls(text: object) -> list[str]:
    """Provide URL signals to the model; this never chooses a route."""
    import re

    return re.findall(r"https?://[^\s<>()\[\]{}\"']+", str(text or ""))


def gmail_route_availability() -> RouteAvailability:
    """Inspect local Gmail registration and credential presence without contacting Gmail."""
    from mana_agent.connectors.email.auth.credential_store import CredentialStore
    from mana_agent.connectors.email.config import load_accounts
    from mana_agent.connectors.email.exceptions import EmailConnectorError
    from mana_agent.connectors.email.models import EmailPermission

    accounts = [
        account
        for account in load_accounts()
        if account.enabled and account.provider == "gmail"
    ]
    if not accounts:
        return RouteAvailability(
            available=False,
            configured=False,
            authorized=False,
            reason="No enabled Gmail account is configured.",
            setup_action=(
                "Run `mana-agent connector email add --provider gmail "
                "--client-secret-file <google-client.json> --permissions email.read`."
            ),
        )
    readable = [account for account in accounts if EmailPermission.READ in account.granted_permissions]
    if not readable:
        return RouteAvailability(
            available=False,
            configured=True,
            authorized=False,
            reason="The configured Gmail account has not granted email.read permission.",
            setup_action=(
                f"Run `mana-agent connector email reconnect {accounts[0].id} "
                "--client-secret-file <google-client.json> --permissions email.read`."
            ),
            details={"account_id": accounts[0].id, "provider": "gmail"},
        )
    account = readable[0]
    if not account.secret_ref:
        return RouteAvailability(
            available=False,
            configured=True,
            authorized=False,
            reason="The Gmail credential reference is missing.",
            setup_action=f"Reconnect Gmail account `{account.id}`.",
            details={"account_id": account.id, "provider": "gmail"},
        )
    try:
        CredentialStore().get(account.secret_ref)
    except EmailConnectorError as exc:
        return RouteAvailability(
            available=False,
            configured=True,
            authorized=False,
            reason=str(exc),
            setup_action=f"Reconnect Gmail account `{account.id}`.",
            details={
                "account_id": account.id,
                "provider": "gmail",
                "provider_error": exc.to_payload(),
            },
        )
    return RouteAvailability(
        available=True,
        configured=True,
        authorized=True,
        details={"account_id": account.id, "provider": "gmail"},
    )
