"""Model-driven entry routing and dynamic route availability for chat turns."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, get_args

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from mana_agent.context_cost.accounting import ModelContextLimitError
from mana_agent.context_cost.models import ContextBudgetExceeded
from mana_agent.evals.ids import stable_hash
from mana_agent.evals.recorder import record_current
from mana_agent.media.models import MediaOperationDecision
from mana_agent.server.models import ServerActionKind


EntryRouteName = Literal[
    "multi_task",
    "conversation",
    "coding",
    "mcp",
    "gmail",
    "calendar",
    "computer",
    "browser",
    "search",
    "github",
    "repository",
    "memory",
    "automation",
    "api",
    "canvas",
    "remote_execution",
    "server",
    "artifact",
    "media",
    "command",
    "unsupported",
    "capability_error",
]

AutomationOperation = Literal[
    "create", "get", "list", "status", "update", "delete", "enable", "disable", "run_now",
]

RequiredSource = Literal[
    "repository", "browser", "search", "gmail", "calendar", "computer", "github",
    "memory", "artifact", "media", "remote_execution", "server", "canvas", "api", "mcp", "internal_knowledge", "none",
]

REQUIRED_SOURCES: set[str] = set(get_args(RequiredSource))
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
    memory_task_candidates: tuple[dict[str, str], ...] = ()
    memory_capsules_enabled: bool = False
    atomic_child: bool = False
    orchestration_parent_task_id: str = ""
    authenticated_user_id: str = ""
    envelope: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        base = {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "previous_route": self.previous_route,
            "conversation_summary": self.conversation_summary,
            "artifact_evidence": dict(self.artifact_evidence),
            "memory_task_candidates": [dict(c) for c in self.memory_task_candidates] if self.memory_capsules_enabled else [],
            "memory_capsules_enabled": self.memory_capsules_enabled,

            "atomic_child": self.atomic_child,
            "orchestration_parent_task_id": self.orchestration_parent_task_id,
            "authenticated_user_id": self.authenticated_user_id,
        }
        if self.envelope is not None and hasattr(self.envelope, "to_dict"):
            base["envelope"] = self.envelope.to_dict()
        return base


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
    remote_request: dict[str, Any] = field(default_factory=dict)
    server_request: dict[str, Any] = field(default_factory=dict)
    mcp_request: dict[str, Any] = field(default_factory=dict)
    memory_task_id: str = ""
    artifact_family: str = ""
    media_request: dict[str, Any] = field(default_factory=dict)
    automation_operation: AutomationOperation | str = ""
    source: str = "model"
    runtime_capability_change: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _StrictRoutingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntryRoutingSSHTarget(_StrictRoutingOutput):
    host: str = ""
    port: int = 22
    user: str = ""


class EntryRoutingSSHAuthentication(_StrictRoutingOutput):
    mode: Literal["agent", "key_path", "key"] = "agent"
    key_path: str = ""


class EntryRoutingRemoteCommand(_StrictRoutingOutput):
    argv: list[str] = Field(default_factory=list)


class EntryRoutingRemoteRequest(_StrictRoutingOutput):
    provider: Literal["remote-ssh", "reverse-worker"] = "remote-ssh"
    profile: str = ""
    worker_id: str = ""
    target: EntryRoutingSSHTarget | None = None
    authentication: EntryRoutingSSHAuthentication | None = None
    command: EntryRoutingRemoteCommand | None = None
    working_directory: str | None = None
    connect_timeout_seconds: int | None = None
    known_hosts_file: str | None = None
    timeout_seconds: int | None = None
    read_only: bool = True
    pty: bool = False


class EntryRoutingServerDecision(_StrictRoutingOutput):
    """Closed model-boundary representation of a server action decision."""

    decision_id: str = Field(min_length=1)
    server_id: str = Field(min_length=1)
    action: ServerActionKind
    tool_name: str = Field(min_length=1)
    arguments_json: str = "{}"
    required_capability: str = Field(min_length=1)
    read_only: bool
    consequential: bool
    destructive: bool = False
    affected_resources: list[str] = Field(default_factory=list)
    recovery_plan: str | None = None
    verification_commands: list[list[str]] = Field(default_factory=list)
    safe_to_continue: bool
    reason: str = Field(min_length=1)


class EntryRoutingServerRequest(_StrictRoutingOutput):
    decision: EntryRoutingServerDecision


class EntryRoutingMcpRequest(_StrictRoutingOutput):
    """Exact configured MCP provider selected by the entry-routing model."""

    provider_id: str = Field(min_length=1)


class EntryRoutingOutput(_StrictRoutingOutput):
    """Schema enforced at the model boundary before routing validation."""

    route: str
    confidence: float
    reason: str
    required_sources: list[str]
    target_urls: list[str] = Field(default_factory=list)
    requires_live_data: bool = False
    reason_code: str = ""
    error_code: str = ""
    reuse_active_route: bool = False
    command_name: str = ""
    command_arguments: list[str] = Field(default_factory=list)
    remote_request: EntryRoutingRemoteRequest | None = None
    server_request: EntryRoutingServerRequest | None = None
    mcp_request: EntryRoutingMcpRequest | None = None
    memory_task_id: str = ""
    artifact_family: Literal["", "spreadsheet", "document", "presentation", "pdf", "image"] = ""
    media_request: MediaOperationDecision | None = None
    automation_operation: Literal[
        "", "create", "get", "list", "status", "update", "delete", "enable", "disable", "run_now",
    ] = ""
    runtime_capability_change: bool


class EntryRoutingError(RuntimeError):
    """The model did not return a valid entry-routing decision."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "",
        phase: str = "entry_route",
        provider_call_executed: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase
        self.provider_call_executed = provider_call_executed
        self.details = details or {}



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
- multi_task: two or more actionable goals that need distinct execution lifecycles, different
  routes, independent verification, or an explicit dependency order. It is orchestration-only and
  requires exactly ["none"]. Do not select it for several implementation steps that naturally
  form one atomic coding workflow, such as changing one function and running its tests. The
  complete prompt, conversation context, registry, and capabilities decide this; never count
  keywords or conjunctions.
- command: a request equivalent to one command exposed by the supplied command route tools. Return
  its canonical command_name and structured string command_arguments; never execute it directly.
  Remote-worker lifecycle requests must select the `remote-worker` command with exactly
  `["register"|"start"|"stop", worker_id]`. This is a model decision, never keyword routing.
- conversation: ordinary discussion that needs no tool, connector, repository, or coding action.
  Do not select conversation for repository planning, plan continuation, verification, or review
  of repository work—even when the user says not to edit yet.
- coding: repository engineering workflows handled by the Codex coding path, including edits,
  implementation, plan-only design for repository changes, continuing an active plan without
  applying it, and verification/testing of the repository. Planning a CLI flag or similar
  repository change is coding, not conversation.
- mcp: execute a request through one configured Model Context Protocol provider. Return a complete
  mcp_request with one exact provider_id from the route availability details. MCP provider state and
  tool results are live external state, so requires_live_data must be true. The provider selection is
  a model decision; never invent a provider or substitute another provider.
- remote_execution: explicit user-authorized SSH work. Never select coding for
  SSH. Return an exact structured remote_request. Select `remote-ssh` for direct
  local OpenSSH execution or `reverse-worker` for an enrolled worker. Read the
  remote_execution route availability details: when
  `managed_worker_available` is false, provider MUST be `remote-ssh` and
  worker_id MUST be "". Use reverse-worker only when it is true; use its
  managed_worker_id rather than "auto". The remote_request must be valid JSON:
  use an argv array of plain strings and never place shell syntax in JSON keys.
  For an approved analysis request, choose a bounded command that emits the
  requested concise findings (counts, top entries, and relevant samples) rather
  than streaming an entire log; its stdout is shown back in chat after approval.
- server: management of an explicitly enrolled Linux server. Return a complete server_request
  containing a strict ServerActionDecision. Put the exact tool arguments in arguments_json as a
  JSON object encoded in a string; use "{}" when the tool takes no arguments. Use server rather than
  remote_execution for enrolled-server inspection, packages, services, files, users, networking,
  firewall, databases, containers, deployments, backups, provisioning, reboot, or shutdown.
  The decision must classify read_only, consequential, destructive, affected resources, recovery,
  verification commands, required capability, and whether it is safe to continue. Set decision_id
  to a non-empty unique opaque identifier for this exact model decision. Copy action,
  required_capability, read_only, consequential, and destructive exactly from the selected entry
  in the route availability tool_contracts, and follow that entry's non-empty
  arguments_json_example. For a package action whose manager is not established by server_catalog
  evidence, explicitly select manager "auto"; runtime discovery must observe one unambiguous
  supported manager. Select the server only from its non-secret
  server_catalog. If that enrolled server lacks the tool's required capability, still select the
  server route with the exact decision and safe_to_continue=false so server preflight can return
  the specific authorization guidance; capability_error is only for a route-wide unavailable
  source. Never invent an enrollment, credential, approval, server ID, tool, capability, or
  recovery point.
  The server catalog's login_user is the configured remote SSH user. For a path in that
  user's home directory, use a relative argv path with no cwd (for example
  ["mkdir", "-p", "mana-agent-test"]); do not copy a placeholder absolute home path.
  A directory listing is performed by server_directory_list, which establishes its own authenticated
  connection; wording such as "connect to SERVER and list DIRECTORY" does not require a separate
  server_connect decision. Its exact contract is action=file_read,
  required_capability=filesystem.read, read_only=true, consequential=false, destructive=false,
  and arguments_json must be {"path":"/absolute/directory"}. Never use the inspect action or
  inspect capability for a directory-list tool.
- artifact: creation, editing, conversion, inspection, or export of a user-provided document, spreadsheet, presentation, PDF, or image. A user artifact is not repository code, even when it has a filename. Use the supplied artifact_evidence, including provenance and repository membership. Only select coding when the resolved target is a repository member and the requested change is a repository edit. Return artifact_family for creation requests even when no existing filename or attachment supplies artifact evidence. Do not invent a filename.
- media: generate an image, spoken voice/audio, or video; inspect a media generation job; or cancel
  one. Return a complete typed media_request. Never route media generation to artifact, coding, or
  conversation. The configured media provider/model is authoritative; never select a fallback.
- gmail: inspect or act on the user's Gmail/email account immediately in the current turn through
  registered email tools. Do not select gmail when the requested mailbox action is deferred,
  scheduled for a specified time, or recurring.
- calendar: calendar account operations through a registered account/cloud calendar connector.
- computer: permission-aware control of the local desktop, installed applications, native calendar,
  media, notes, clipboard, screenshots, filesystem, notifications, browser application, or system.
  Bounded screen-recording requests select computer even when material recording parameters need a
  typed clarification; do not select unsupported merely because duration, display, or destination is absent.
  Do not select computer for reading or reviewing files that belong to the active code repository;
  those use repository (read-only) or coding (edit/plan/verify workflows).
- browser: direct public-page inspection using browser tools. A supplied public HTTP(S) URL is a
  strong signal for this route; page content, HTML, metadata, links, robots, and sitemap content
  require browser rather than search snippets.
- search: current public-web discovery, mentions, competitors, indexing, or other search-visible
  information. Search snippets never substitute for browser page inspection.
- github: connected/public GitHub information.
- repository: read-only local repository questions or inspection (find definitions, review architecture,
  read project files such as pyproject.toml, inspect source without editing). Prefer repository over
  computer when the target is the managed workspace/repository rather than an arbitrary desktop path.
- memory: explicitly requested persisted memory retrieval. When
  memory_capsules_enabled is true, select exactly one task ID from
  memory_task_candidates and return it as memory_task_id; private capsule reads
  never search across tasks. Leave memory_task_id empty for legacy memory.
- automation: create, inspect, or manage an automation, including a one-time or recurring future
  connector action. A request to perform another route's action later or at a specified time
  selects automation for the whole turn; the referenced connector becomes the persisted job and
  must not execute during automation creation. Return the exact automation_operation. Use create
  for a request to make or schedule an automation and list only when the user actually asks to see
  existing automations; listing is not a prerequisite for creation.
- api: import authorized external API documentation, inspect or configure saved API integrations,
  retrieve candidate operations, preview a validated request, or execute a saved operation. Use
  this dynamic route for arbitrary providers; never expose a raw unrestricted HTTP tool. A
  documentation URL belongs to api when the requested outcome is a reusable integration. Inspecting
  documentation, saving the resulting integration, and calling that API are one ordered API
  lifecycle, not separate browser and API children; select api for the complete request.
- canvas: create, update, inspect, wait on, or close an interactive Live Canvas/A2UI surface.
  Use this route only when the user requests a visual interactive workspace or when current
  conversation context is already operating on a canvas. Canvas tools require the supplied exact
  session, conversation, turn, and structured decision identifiers; never invent ownership.
- unsupported: no registered route can represent the request safely.

Repository context is only one possible evidence source. Current mailbox/account data is never ordinary conversation. Requests to check an inbox, latest
email, Gmail message, email thread, or mailbox must select gmail when that registered route
represents an immediate request. If the user asks to check that mailbox at a future or specified
time, select automation instead and do not select gmail as a preliminary action. The conversation
route must never speculate about connector availability.
Use computer—not calendar—for an explicitly native/installed desktop calendar, and computer—not
browser—for the user's current installed browser page or tabs. Do not choose computer merely as an
availability fallback for a cloud calendar or isolated public-browser request.

Use previous_route and conversation_summary only for continuity. Reuse the active route for a true
follow-up; reroute when the user's intent changes. Do not route by isolated keywords alone.
When routing_constraints.atomic_child is true, the request is a validated atomic child of an
existing compound plan. Select its exact executable, capability-error, or unsupported route and
never select multi_task. The parent conversation summary provides continuity only; it must not
cause the already-decomposed child to be orchestrated again.

The required_sources array is an execution contract: every listed tool source is mandatory and
must complete successfully before response generation. Never substitute a source or provider.
Use only identifiers from required_source_vocabulary in the request payload, and obey its
required_source_rules. Route registry `tools` are executor tool names, not source identifiers;
never copy a tool name or route name into required_sources unless it is explicitly present in the
source vocabulary. In particular, command is tool-free and requires exactly ["none"].
When no supported available capability can satisfy a required source, return route capability_error
with that source and an exact error_code. Only use capability_error when the live route registry
marks that route or source unavailable (available=false); if the registry reports the browser (or
other source) as available, select that route instead of inventing CONNECTOR_ERROR_* codes.
unsupported is a distinct model decision, never a fallback. Direct URL signals are supplied
separately; do not treat them as repository evidence.
Open-ended discovery requests (for example, finding remote jobs, companies, products, or current
opportunities) must select search with required_sources=["search"]. Browser is only for inspecting
one or more explicit direct URLs supplied by the user or selected in target_urls; never select a
browser source without at least one target URL.

required_sources is required for every decision and must never be omitted or empty. Use exactly
["none"] for conversation and unsupported. Use the route's corresponding source for ordinary
single-source decisions: coding/repository/automation -> ["repository"], api -> ["api"], server -> ["server"], gmail -> ["gmail"],
calendar -> ["calendar"], browser -> ["browser"], search -> ["search"], github -> ["github"],
canvas -> ["canvas"], media -> ["media"],
memory -> ["memory"], and mcp -> ["mcp"]. capability_error must name the unavailable tool source. Do not use an
empty array for a request that needs no external information.
memory_task_id must be empty string "" for all routes except memory. Only populate memory_task_id when route="memory" and memory_capsules_enabled is true.

Return JSON only:
{
  "route": "multi_task|conversation|coding|mcp|remote_execution|server|artifact|media|command|gmail|calendar|computer|browser|search|github|repository|memory|automation|api|canvas|unsupported|capability_error",
  "confidence": 0.0,
  "reason": "short routing reason",
  "required_sources": ["browser"],
  "target_urls": ["https://example.com"],
  "memory_task_id": "",
  "requires_live_data": true,
  "reason_code": "DIRECT_PAGE_INSPECTION",
  "error_code": "",
  "reuse_active_route": false,
  "command_name": "sessions",
  "command_arguments": ["list"],
  "remote_request": {"provider": "remote-ssh", "profile": "", "worker_id": "", "target": {"host": "example.com", "port": 22, "user": "root"}, "authentication": {"mode": "key_path", "key_path": "~/.ssh/id_ed25519"}, "command": {"argv": ["true"]}, "read_only": true},
  "server_request": null,
  "mcp_request": null,
  "artifact_family": "",
  "media_request": null,
  "automation_operation": "",
  "runtime_capability_change": false
}

Examples:
- “ping” -> conversation, ["none"].
- “What can you do?” -> conversation, ["none"].
- “Change this function and run its tests” -> coding, ["repository"] (one atomic workflow).
- “Plan how to add a harmless CLI flag, but do not edit anything” -> coding, ["repository"]
  (repository planning / plan-only still uses the coding workflow; not conversation).
- “Continue the active plan without applying it” -> coding, ["repository"] (plan continuation).
- “Verify the repository tests without modifying files” -> coding, ["repository"].
- “Check open GitHub issues and update the README” -> multi_task, ["none"] (independent routes).
- “Research the current API, then update the implementation from those findings” -> multi_task,
  ["none"] (the coding child depends on the research child).
- “Read my email, add its meeting to my calendar, create a summary document, and verify a remote
  service” -> multi_task, ["none"] (mixed connectors, artifact, and remote execution).
- “Review https://example.com/about” -> browser, ["browser"], that URL.
- “Check https://example.com and prepare a complete SEO report” -> browser with search only when
  public indexing/discovery is independently required; both sources are mandatory if selected.
- “Find competitors for example.com” -> search, ["search"].
- “Improve metadata in this repository” -> coding, ["repository"].
- “Find where AgentDecision is defined in this repository without changing files” -> repository,
  ["repository"].
- “Review the current repository architecture without changing files” -> repository, ["repository"].
- “Read pyproject.toml and do not use mutation tools” -> repository, ["repository"].
- “Inspect a local source file without changing it” -> repository, ["repository"] (workspace source;
  not computer desktop control).
- “Start a distinct repository review while another coding flow is active” -> repository,
  ["repository"] (new read-only repository task; do not select conversation clarification alone).
- “Create an image of a lunar greenhouse” -> media, ["media"], media_request.operation=image.generate with the exact prompt.
- “Read this response aloud” -> media, ["media"], media_request.operation=voice.generate with the exact text to speak.
- “Turn this prompt into a four-second video” -> media, ["media"], media_request.operation=video.generate with prompt and duration_seconds=4.
- “Check media generation media_abc” -> media, ["media"], media_request.operation=generation.status with generation_id=media_abc.
- “Check my latest Gmail” -> gmail, ["gmail"], even when Gmail is unavailable; use
  capability_error with GMAIL_NOT_AVAILABLE rather than repository, memory, or conversation.
- “Use the configured Kaggle MCP provider to upload this competition submission” -> mcp, ["mcp"],
  mcp_request.provider_id="kaggle", requires_live_data=true.
- “At 12:52, check my Gmail” -> automation, ["repository"], automation_operation=create; create
  only the scheduled one-time connector action in this turn and do not inspect Gmail now.
- “Fixture: routing model failure must stop with no fallback action.” -> unsupported, ["none"]
  (explicit no-fallback contract; never invent a route).
"""


def _routing_correction(validation_error: str) -> str:
    """Return model-only correction guidance for a bounded invalid-decision retry."""
    if "browser source requires target_urls" in validation_error:
        return (
            "Return a new complete routing decision. Open-ended discovery must use "
            'route="search" and required_sources=["search"]; browser requires target_urls.'
        )
    if "invalid server decision: Server decision does not match tool contract fields:" in validation_error:
        return (
            "Return a new complete server routing decision. Read the selected tool's exact "
            "action, required_capability, read_only, consequential, and destructive values "
            "from the live route availability tool_contracts and copy all five values exactly. "
            "Do not retain any mismatched values from the previous invalid decision."
        )
    if "memory_task_id is only valid for the memory route" in validation_error:
        return (
            'Return a new complete routing decision with memory_task_id set to empty string "". '
            'memory_task_id is only valid when route="memory" and memory_capsules_enabled is true.'
        )
    if "memory_task_id is only valid for private capsule retrieval" in validation_error:
        return (
            'Return a new complete routing decision with memory_task_id set to empty string "". '
            'Legacy memory does not select a memory_task_id.'
        )
    if "private memory retrieval requires a selected task ID" in validation_error:
        return (
            'Return a new complete routing decision. route="memory" with memory capsules '
            'requires selecting one valid task ID from context.memory_task_candidates into memory_task_id.'
        )
    if "memory route selected a task that was not offered" in validation_error:
        return (
            'Return a new complete routing decision. Select a task ID that is explicitly present '
            'in context.memory_task_candidates for memory_task_id.'
        )
    return ""



class EntryRouter:
    """Obtain and validate the single entry decision for one gateway turn."""

    def __init__(
        self,
        *,
        llm: Any,
        registry: EntryRouteRegistry,
        compactor: Any | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.compactor = compactor

    def route(
        self,
        *,
        user_prompt: str,
        context: EntryRouteContext,
        envelope: Any | None = None,
    ) -> EntryRoutingDecision:
        if self.llm is None or not callable(getattr(self.llm, "invoke", None)):
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: routing model is unavailable.",
                phase="entry_route",
                provider_call_executed=False,
            )
        routes = self.registry.snapshot()
        atomic_child = bool(getattr(context, "atomic_child", False))
        parent_task_id = str(getattr(context, "orchestration_parent_task_id", ""))
        disallowed_routes = ["multi_task"] if atomic_child else []
        if disallowed_routes:
            routes = [row for row in routes if row["name"] not in disallowed_routes]

        if self.compactor is None:
            try:
                from mana_agent.gateway.context_compactor import ContextCompactor

                self.compactor = ContextCompactor()
            except Exception:
                self.compactor = None

        if self.compactor is not None:
            compaction = self.compactor.compact_routing_context(
                user_prompt=user_prompt,
                system_prompt=ENTRY_ROUTER_PROMPT,
                context=context,
                envelope=envelope,
                routes=routes,
            )
            if not compaction.is_valid:
                raise EntryRoutingError(
                    f"Model decision failed: entry_route. No response was generated. "
                    f"Reason: Context budget blocked: context_limit_deficit:{compaction.deficit}. "
                    "No provider call was executed.",
                    code="context_budget_blocked",
                    phase="entry_route",
                    provider_call_executed=False,
                    details=compaction.diagnostic_details,
                )
            context = compaction.bounded_context
            effective_envelope = compaction.bounded_envelope
        else:
            effective_envelope = envelope or getattr(context, "envelope", None)

        payload = {
            "user_prompt": str(user_prompt or "").strip(),
            "context": context.to_dict(),
            "routes": routes,
            "routing_constraints": {
                "atomic_child": atomic_child,
                "disallowed_routes": disallowed_routes,
                "orchestration_parent_task_id": parent_task_id,
            },
            "required_source_vocabulary": sorted(REQUIRED_SOURCES),
            "required_source_rules": {
                "multi_task": [["none"]],
                "conversation": [["none"]],
                "command": [["none"]],
                "unsupported": [["none"]],
                "coding": [["repository"]],
                "mcp": [["mcp"]],
                "remote_execution": [["remote_execution"]],
                "server": [["server"]],
                "artifact": [["artifact"]],
                "media": [["media"]],
                "gmail": [["gmail"]],
                "calendar": [["calendar"]],
                "computer": [["computer"]],
                "browser": [["browser"], ["browser", "search"]],
                "search": [["search"]],
                "github": [["github"]],
                "repository": [["repository"]],
                "memory": [["memory"]],
                "automation": [["repository"]],
                "api": [["api"]],
                "canvas": [["canvas"]],
                "capability_error": "one or more unavailable source identifiers",
            },
            "direct_url_signals": _public_urls(user_prompt),
        }
        try:
            started = time.perf_counter()
            messages = [
                SystemMessage(content=ENTRY_ROUTER_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            ]
            structured_output = getattr(self.llm, "with_structured_output", None)
            response, decision_payload = _invoke_routing_model(
                self.llm,
                messages,
            )
            decision_payload = _coerce_routing_output(response)
            try:
                decision = self._validate(decision_payload, context=context)
            except EntryRoutingError as validation_error:
                # This is a bounded correction request, not a static reroute:
                # the model must supply a new, fully validated decision.
                correction = _routing_correction(str(validation_error))
                if not correction:
                    raise
                repair_payload = {
                    **payload,
                    "previous_invalid_decision": decision_payload,
                    "validation_error": str(validation_error),
                    "correction": correction,
                }
                repair_messages = [
                    SystemMessage(content=ENTRY_ROUTER_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            repair_payload, ensure_ascii=False, sort_keys=True
                        )
                    ),
                ]
                response, decision_payload = _invoke_routing_model(
                    self.llm,
                    repair_messages,
                )
                decision_payload = _coerce_routing_output(response)
                decision = self._validate(decision_payload, context=context)
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
        except (ContextBudgetExceeded, ModelContextLimitError) as exc:
            record_current(
                "model.call.failed",
                {
                    "boundary": "entry_router",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "phase": "entry_route",
                    "provider_call_executed": False,
                },
            )
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                f"Reason: {exc}",
                code="context_budget_blocked",
                phase="entry_route",
                provider_call_executed=False,
            ) from exc
        except Exception as exc:
            record_current("model.call.failed", {"boundary": "entry_router", "error_type": type(exc).__name__, "error": str(exc)})
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                f"Reason: {exc}"
            ) from exc


    def _validate(
        self,
        payload: dict[str, Any],
        *,
        context: EntryRouteContext | None = None,
    ) -> EntryRoutingDecision:
        route = str(payload.get("route") or "").strip()
        if route not in self.registry.names:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                f"Reason: unknown route {route or '<missing>'}."
            )
        if route == "coding" and "runtime_capability_change" not in payload:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: coding route must explicitly declare runtime_capability_change."
            )
        if context is not None and bool(getattr(context, "atomic_child", False)) and route == "multi_task":
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: an atomic compound child cannot select recursive multi_task routing."
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
        if route in {"multi_task", "conversation", "command", "unsupported"} and sources != ("none",):
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: tool-free routes require required_sources=[\"none\"].")
        if route == "capability_error" and not any(source in TOOL_SOURCES for source in sources):
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: capability_error requires an unavailable tool source.")
        if route not in {"multi_task", "conversation", "command", "unsupported", "capability_error"} and not any(source in TOOL_SOURCES for source in sources):
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: executable route requires a tool source.")
        if route == "media" and sources != ("media",):
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                'Reason: media route requires required_sources=["media"].'
            )
        if route == "server" and sources != ("server",):
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                'Reason: server route requires required_sources=["server"].'
            )
        if route == "mcp" and sources != ("mcp",):
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                'Reason: mcp route requires required_sources=["mcp"].'
            )
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
        source_routes = {"browser": "browser", "search": "search", "github": "github", "repository": "repository", "gmail": "gmail", "calendar": "calendar", "computer": "computer", "memory": "memory", "mcp": "mcp", "remote_execution": "remote_execution", "server": "server"}
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
        remote_request = payload.get("remote_request") or {}
        server_request = payload.get("server_request")
        mcp_request = payload.get("mcp_request")
        media_request = payload.get("media_request")
        requires_live_data = bool(payload.get("requires_live_data", False))
        if route == "mcp":
            try:
                mcp_request = EntryRoutingMcpRequest.model_validate(mcp_request).model_dump()
            except Exception as exc:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    f"Reason: mcp route requires a valid mcp_request: {exc}."
                ) from exc
            mcp_details = next(
                (
                    dict(row["availability"].get("details") or {})
                    for row in self.registry.snapshot()
                    if row["name"] == "mcp"
                ),
                {},
            )
            provider_ids = {
                str(item.get("id") or "").strip()
                for item in list(mcp_details.get("providers") or [])
                if isinstance(item, dict)
            }
            provider_id = str(mcp_request["provider_id"]).strip()
            if provider_id not in provider_ids:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: mcp route selected a provider that is not configured."
                )
            if not requires_live_data:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: mcp route must require live execution."
                )
        elif mcp_request is not None:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: mcp_request is only valid for the mcp route."
            )
        memory_task_id = str(payload.get("memory_task_id") or "").strip()
        if route == "memory":
            if context is not None and context.memory_capsules_enabled:
                offered_memory_tasks = {
                    str(item.get("task_id") or "").strip()
                    for item in context.memory_task_candidates
                }
                if not memory_task_id:
                    raise EntryRoutingError(
                        "Model decision failed: entry_route. No response was generated. "
                        "Reason: private memory retrieval requires a selected task ID."
                    )
                if memory_task_id not in offered_memory_tasks:
                    raise EntryRoutingError(
                        "Model decision failed: entry_route. No response was generated. "
                        "Reason: memory route selected a task that was not offered."
                    )
            elif memory_task_id:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: memory_task_id is only valid for private capsule retrieval."
                )
        elif memory_task_id:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: memory_task_id is only valid for the memory route."
            )
        if route == "media":
            try:
                media_request = MediaOperationDecision.model_validate(media_request).model_dump(
                    mode="json"
                )
            except Exception as exc:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    f"Reason: media route requires a valid media_request: {exc}."
                ) from exc
        elif media_request:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: media_request is only valid for the media route."
            )
        artifact_family = str(payload.get("artifact_family") or "").strip().lower()
        allowed_artifact_families = {
            "spreadsheet", "document", "presentation", "pdf", "image",
        }
        if artifact_family and artifact_family not in allowed_artifact_families:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: artifact_family is invalid."
            )
        if route == "artifact" and not artifact_family and context is not None:
            evidence_families = list(context.artifact_evidence.get("artifact_families") or [])
            if len(evidence_families) == 1:
                artifact_family = str(evidence_families[0])
        if route == "artifact" and not artifact_family:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: artifact route requires artifact_family when evidence does not resolve it."
            )
        automation_operation = str(payload.get("automation_operation") or "").strip().lower()
        allowed_automation_operations = {
            "create", "get", "list", "status", "update", "delete", "enable", "disable", "run_now",
        }
        if automation_operation and automation_operation not in allowed_automation_operations:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: automation_operation is invalid."
            )
        if route == "automation" and not automation_operation:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: automation route requires automation_operation."
            )
        if route != "automation" and automation_operation:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: automation_operation is only valid for the automation route."
            )
        if route == "remote_execution" and not isinstance(remote_request, dict):
            raise EntryRoutingError("Model decision failed: entry_route. No response was generated. Reason: remote_execution requires structured remote_request.")
        if route == "remote_execution":
            provider = str(remote_request.get("provider") or "").strip()
            worker_id = str(remote_request.get("worker_id") or "").strip()
            if provider not in {"remote-ssh", "reverse-worker"}:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: remote_execution provider must be remote-ssh or reverse-worker."
                )
            remote_details = next(
                (
                    dict(row["availability"].get("details") or {})
                    for row in self.registry.snapshot()
                    if row["name"] == "remote_execution"
                ),
                {},
            )
            worker_available = bool(remote_details.get("managed_worker_available"))
            if not worker_available and provider != "remote-ssh":
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: no managed worker is available; select provider remote-ssh."
                )
            if provider == "remote-ssh" and worker_id:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: direct SSH requests must not specify worker_id."
                )
            if provider == "reverse-worker" and (
                not worker_id or worker_id != str(remote_details.get("managed_worker_id") or "")
            ):
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: reverse-worker requests must use the available managed worker ID."
                )
        if route == "server":
            if not isinstance(server_request, dict) or not server_request:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    "Reason: server route requires a structured server_request."
                )
            try:
                from mana_agent.server.models import ServerActionDecision
                from mana_agent.server.runtime_tools import validate_tool_arguments
                from mana_agent.server.tools import validate_tool_decision

                raw_server_decision = server_request.get("decision")
                if not isinstance(raw_server_decision, dict):
                    raise ValueError("server_request.decision must be an object")
                decision_payload = dict(raw_server_decision)
                arguments_json = decision_payload.pop("arguments_json", None)
                if not isinstance(arguments_json, str):
                    raise ValueError("server decision arguments_json must be a JSON string")
                arguments = json.loads(arguments_json)
                if not isinstance(arguments, dict):
                    raise ValueError("server decision arguments_json must encode a JSON object")
                decision_payload["arguments"] = arguments
                validated_server_decision = ServerActionDecision.model_validate(
                    decision_payload
                )
                validate_tool_decision(validated_server_decision)
                validate_tool_arguments(validated_server_decision)
            except Exception as exc:
                raise EntryRoutingError(
                    "Model decision failed: entry_route. No response was generated. "
                    f"Reason: invalid server decision: {exc}."
                ) from exc
            server_request = {
                **server_request,
                "decision": validated_server_decision.model_dump(mode="json"),
            }
        elif server_request is not None:
            raise EntryRoutingError(
                "Model decision failed: entry_route. No response was generated. "
                "Reason: server_request is only valid for the server route."
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
            requires_live_data=requires_live_data,
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
            remote_request=dict(remote_request) if isinstance(remote_request, dict) else {},
            server_request=dict(server_request) if isinstance(server_request, dict) else {},
            mcp_request=dict(mcp_request) if isinstance(mcp_request, dict) else {},
            memory_task_id=memory_task_id,
            artifact_family=artifact_family,
            media_request=dict(media_request) if isinstance(media_request, dict) else {},
            automation_operation=automation_operation,
            runtime_capability_change=bool(payload.get("runtime_capability_change")),
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


def _invoke_routing_model(
    llm: Any,
    messages: list[Any],
) -> tuple[Any, dict[str, Any]]:
    """
    Invoke the entry-routing model exactly once.

    Prefer strict structured output, but retain the raw AIMessage so a provider
    that prepends/appends prose around an otherwise valid JSON object does not
    destroy the entire turn.

    Recovery does NOT bypass EntryRouter._validate().
    """
    structured_output = getattr(llm, "with_structured_output", None)

    if not callable(structured_output):
        response = llm.invoke(messages)
        return response, _coerce_routing_output(response)

    try:
        runner = structured_output(
            EntryRoutingOutput,
            method="json_schema",
            strict=True,
            include_raw=True,
        )
    except TypeError:
        try:
            runner = structured_output(
                EntryRoutingOutput,
                method="json_schema",
                strict=True,
            )
        except TypeError:
            runner = structured_output(EntryRoutingOutput)

    result = runner.invoke(messages)

    # LangChain include_raw=True contract:
    #
    # {
    #     "raw": AIMessage(...),
    #     "parsed": EntryRoutingOutput(...) | None,
    #     "parsing_error": Exception | None,
    # }
    if isinstance(result, dict) and (
        "raw" in result
        or "parsed" in result
        or "parsing_error" in result
    ):
        raw = result.get("raw")
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")

        if parsed is not None:
            return raw or parsed, _coerce_routing_output(parsed)

        # Important recovery path:
        # structured parsing failed, but the raw response may contain a
        # perfectly usable JSON object surrounded by provider/model prose.
        if raw is not None:
            try:
                return raw, _coerce_routing_output(raw)
            except Exception as raw_error:
                # Preserve the original structured-parser error when possible.
                if parsing_error is not None:
                    raise parsing_error from raw_error
                raise

        if parsing_error is not None:
            raise parsing_error

        raise ValueError(
            "structured entry-router response contained neither parsed nor raw output"
        )

    # Defensive compatibility with implementations that ignore include_raw.
    return result, _coerce_routing_output(result)

def _coerce_routing_output(response: Any) -> dict[str, Any]:
    if isinstance(response, EntryRoutingOutput):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    from mana_agent.utils.text import extract_model_text

    content = getattr(response, "content", response)
    extracted = _extract_json(extract_model_text(content))
    return extracted



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
