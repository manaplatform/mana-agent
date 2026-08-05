# Entry routing and chat sessions

Every gateway turn begins with one structured model decision before a conversational model, coding agent, search service, or connector can run. The entry decision selects one registered route, including `conversation`, `coding`, connectors, search, repository, automation, `multi_task`, and the fail-closed error routes.

The route registry is runtime data, not a keyword table. Each route publishes its description, tools, configuration state, authorization state, and any setup action. The model chooses the route from the request, prior route, conversation context, and this live registry snapshot. Invalid model output stops with an entry-decision error; it does not execute a default route or generate an integration refusal.

## Compound turns

The model selects `multi_task` only for two or more actionable goals that need separate execution lifecycles, different routes, independent verification, or a real dependency order. “Change this function and run its tests” remains one coding task; “check GitHub issues and update the README” is compound, while “research the current API and then update the implementation from those findings” is a dependency-ordered compound request. Mixed connector, artifact, browser, computer-control, and remote-execution requests use the same rule.

The parent route is orchestration-only and must declare `required_sources=["none"]`. A strict structured decomposition creates two to twelve children, rejects duplicate IDs, missing or self dependencies, and cycles, and persists the local-ID mapping. Invalid decomposition stops with `MultiTaskError`; conversation, coding, or another route is never used as a fallback.

One root TaskBoard item holds the complete goal and aggregate progress. Each child inherits its root, parent, session, workspace, repository, priority, risk, and budget lineage, then receives an independent entry-routing and model-routing decision. Children therefore own their exact source availability, permission request IDs, results, artifacts, changed files, and verification status. Child execution does not call public `process_turn()` and does not add duplicate chat messages.

The validated dependency DAG uses a bounded worker pool and the existing lane coordinator. Ready children may run concurrently, but lane capacity and repository/workspace/file locks remain authoritative; overlapping or unknown repository mutations serialize. A failed or blocked prerequisite blocks its dependents without preventing independent children from finishing. Cancellation propagates through the existing task tree, and persisted decomposition IDs prevent resume or event reprocessing from recreating children.

The root result reports every child as completed, blocked, failed, skipped, cancelled, or awaiting approval, plus aggregate progress and partial-completion warnings. `DONE` requires all required children to complete or be intentionally skipped; blocked or approval-waiting children make the root `BLOCKED`, an unrecoverable execution failure makes it `FAILED`, and cancellation makes it `CANCELLED`.

## Artifact creation

Attachment-free artifact creation runs document tools against the directory in
which Mana-Agent was launched, so a requested PDF is returned at that project
root rather than remaining under `~/.mana/artifacts/`. Attached files still use
an isolated staging workspace for safe inspection and updates.

PDF creation is governed by the model-selected `artifact_family="pdf"` decision.
The artifact executor must call `read_skill("pdf-create")` before
`document_create`; the tool rejects PDF creation if that skill was not loaded.
The PDF writer accepts structured titles, subtitles, sections, bullets, and
tables, paginates without truncating content, and emits a styled report with
consistent margins and page footers. Invalid paths, existing targets, missing
skills, or renderer failures stop safely without a fallback artifact.

## Specialist lane coordination

After entry routing and before agent execution, the same gateway assigns exactly one owning specialist lane: `artifact`, `media`, `canvas`, `coding`, `research`, `review`, `verify`, `release`, or `operations`. Validated route and agent-decision fields drive selection; an explicit valid model lane may resolve an ambiguous request. Invalid model lane output is ignored only when the validated entry route or intent already determines a lane. If no valid decision remains, execution stops.

The coordinator reserves lane/session token and cost capacity, checks duplicates, acquires canonical repository/workspace/file lock leases when the selected capability requires one, validates tool capabilities, and then invokes the existing turn engine. Artifact, media, and Canvas work are isolated from repository locks; coding owns mutations, research owns external investigation, review owns correctness/security/architecture review, verify owns tests and static checks, release owns version/package preparation, and operations owns deployment and monitoring work.

The default handoff graph is `research → coding`, `coding → review|verify|release`, `review → coding|verify`, `verify → coding`, `release → operations`, and `operations → coding`. A handoff retains the task, session, workspace, repository, budget, changed files, and verification state. It does not create another chat session or unrelated task lineage.

Read locks coexist. Repository writes are exclusive, file writes serialize overlapping paths, and review/verification repository snapshots block file mutation until released. Unknown mutation targets acquire a conservative repository lock. Lock leases and execution records persist under the workspace gateway state, expired leases are reclaimed at startup, and interrupted mutation work requires fresh validation before it can resume.

Gmail readiness is checked from enabled email-account metadata, granted `email.read` permission, and the referenced keyring credential. A configured Gmail request runs through an email-only AskAgent policy. Missing configuration or credentials returns the registry's actionable setup/reconnect error. Provider authorization errors from Gmail retain their provider code, HTTP status, and `reconnect_required` detail.

## Session lifecycle

A frontend opens exactly one workspace session for a chat. All turns, route decisions, connector calls, model calls, coding work, memory, and persisted messages reuse its `session_id`; its `conversation_id` and each `turn_id` are passed through the gateway result and connector execution context.

Session records use `active`, `closed`, or `abandoned` lifecycle states (`archived` remains readable for older records), with opening and closing timestamps. Normal frontend shutdown closes a session and preserves history. `/new` is deliberately different: it cancels an active turn when possible, closes session resources, clears session memory (or installs an authoritative content-free tombstone), physically deletes the validated session directory and `messages.jsonl`, removes gateway state, and binds a fresh session. Repository/workspace registrations and repository-scoped indexes, analysis, and memory remain intact.

On a newly opened chat, Mana-Agent creates a new session rather than reopening a closed chat. Active sessions owned by a process that no longer exists are finalized as `abandoned`; opening a new chat also abandons any previous active chat for the same repository before creating the new identity.

Frontends should construct or reuse `AgentChatGateway`, call `create_session()` once when the chat opens, use `process_turn()` for every message, and call `close_session()` on every shutdown path. They must not instantiate a gateway or workspace session per message.

`SessionService` is the frontend-independent authority for create, replace,
list, switch/reopen, rename, delete, active binding, titles, summaries, and exact
history loading. `/sessions` is canonical and `/session` is an alias. A switch
validates workspace ownership, rebinds gateway/memory/coding state, and emits a
timeline replacement; closed sessions reopen, while deleted or archived sessions
cannot be selected. The first meaningful user message supplies the default title.

## Shared commands and persistent processes

All frontends dispatch through `mana_agent.chat_commands`. Definitions carry
aliases, argument contracts, capabilities, frontend availability, confirmation,
secret acceptance, execution mode, and renderer metadata, and return a typed
`CommandResult`. Natural-language command intent must come from a structured
model resolver; absence or invalid output does not invoke keyword/default routing.

`mana_agent.background` persists registered worker records and bounded logs under
`~/.mana/runtime/processes/`. Launch uses argv arrays and a minimum environment;
secrets are resolved by name and never stored in argv or metadata. Stable process
identity is checked in addition to PID before group termination. `/tasks` remains
agent execution; `/processes` manages operating-system services.
