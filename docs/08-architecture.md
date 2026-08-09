# Architecture

Durable approval and clarification handling is provided by the bounded
`human_inbox` module. Its optimistic-versioned repository is authoritative;
identity resolution, signed responses, delivery adapters, audit, and
checkpoint/resume are separate typed boundaries. It reuses the execution
supervisor and transactional policy pipeline rather than routing around either.
See [Durable Human-in-the-Loop Inbox](32-durable-human-inbox.md).

Consequential tool execution is mediated by the durable policy-gated action layer described in [Policy-Gated Transactional Actions](31-policy-gated-transactional-actions.md). Tool-call generation and authorization are separate states; preview, deterministic policy, exact approval when required, observable verification, idempotency, and compensation evidence form the execution boundary.

## Teach Mode pipeline

Teach Mode follows `explicit start → semantic raw events → redaction →
normalization → parameterization → typed flow → review → safe replay →
observable verification`. It reuses the shared user data directory,
computer-control permission scopes, execution event hub, model tool selection,
API, and persistent automation service. It does not introduce a separate
executor or scheduler. Typed extension protocols and trust boundaries are
described in [Teach Mode](26-teach-mode.md#architecture-and-extension-points).

## Fleet verification boundary

`mana_agent.fleet` owns worker capability inventory, health, deterministic
selection, persisted runs, matrices, and ordered Fleet events. It does not own
a second command runner: every selected provider is validated and executed by
`ExecutionManager`. Authenticated reverse-worker transport supplies worker
identity and signed capabilities. See [Mana Fleet](24-fleet.md).

`mana-agent` is a Python CLI + service stack that performs repository analysis and
LLM-driven “agentic” workflows. The architecture is organized around three
axes:

1. **Evidence gathering** (search/read/indexing)
2. **Decision + orchestration** (flow + work queue + tool managers/workers)
3. **Artifacts + presentation** (generated files, reports, HTML/console UI)

The most recent architecture centers the *live* coding-work execution loop:
the planner produces a sequence of gated tool jobs, and a queue runner executes
them while a sniffer steers follow-up reads/edits/verification.

## Major Components

### Computer-control integration boundary

`mana_agent.integrations.computer_control` is the sole desktop automation
boundary for every frontend. Typed model actions pass through operation schema
validation, client policy, independent Mana/OS permissions, action-bound
confirmation, an application adapter or OS provider, cooperative
timeout/cancellation, live events, and sanitized audit logging. Platform
providers contain platform behavior; the core service contains no
platform-specific conditional tree. Unknown or invalid model decisions stop
without a default tool/provider/action. See
[`22-computer-control.md`](22-computer-control.md).

### Pluggable memory boundary

Scoped sharing is implemented by the capsule service inside this same boundary; it is not a second application memory system. Central ACL decisions, trusted namespaces, reauthorization after provider retrieval, staging/merge, lineage, retention, and audit rules are described in [Scoped shared-memory capsules](30-scoped-memory-capsules.md).

All application consumers import `mana_agent.memory.MemoryService`. The service
owns one validated **AI/semantic** backend selected centrally by
`memory/factory.py`; callers do not import internal or Mem0 implementations.
Canonical asynchronous add, search, get, update, delete, clear, health, and
close operations use provider-neutral models and a structured scope containing
user, agent, session, workspace, repository, conversation, and task identities.

`MemoryService.capabilities` separates domains:

* **AI memory** (conversation, semantic search, external multi-agent runtime
  adapter) — selected by mode/provider.
* **System state** (run evidence, coding-flow checkpoints/turn history) —
  always local durable stores so agent routes do not depend on hosted provider
  mapping for file-read evidence or plan continuation.

The `internal/mana` adapter remains the default for AI/semantic records and
wraps the existing local store without rewriting it. The optional
`external/mem0` and `external/supermemory` adapters are lazy, normalize provider
responses and failures, map scope fields in one mapper, apply timeouts, and
reuse their clients for the service lifecycle. Invalid configuration stops
before execution; external failure never rewrites **semantic AI memory** to an
internal provider backend. Local system stores remain available regardless.

`ChatStack` owns one canonical service instance and rebinds its identity scope
when the frontend opens a session; this does not construct a backend or create a
session. Successful chat turns are written with session/workspace/repository/
conversation scope only when capsules are disabled. With capsules enabled,
successful durable task results become compact private capsules and follow-up
turns recall only a model-selected related task after authorization. Session history remains authoritative and a
reported degraded-memory path may continue without recall, but never writes to
a different provider.

New providers implement `MemoryBackend`, add an isolated provider package, and
register one validated mode/provider branch in the factory. Provider-specific
types must not escape the adapter.

### CLI / command surface

- **`src/mana_agent/commands/`**: command entry points and “chat command” helpers.
  For example, `/analyze` format parsing and the numbered menu are centralized in
  `src/mana_agent/commands/analyze_formats.py` as the single source of truth for
  artifact filenames under `.mana/`.
  See: `src/mana_agent/commands/analyze_formats.py:1-174`.

Interactive commands are defined once in `mana_agent.chat_commands` and rendered
by CLI, Textual, dashboard/API, and connector adapters. `SessionService` owns the
canonical workspace-session identity and `BackgroundProcessManager` owns
persistent registered services. The dashboard `ConversationService` is now a
compatibility adapter over canonical sessions; its legacy storage is migrated
once and is not maintained in parallel.

### Chat gateway (runtime owner)

All chat frontends connect through **`src/mana_agent/gateway/`**:

- **`AgentChatGateway`** (`chat_gateway.py`): session management, stack ownership,
  `process_turn` / `send`, rich context for TUI.
- **`ChatGatewayConfig` + `build_chat_stack`**: construct AskService, ChatService,
  CodingAgent, ToolWorker, QueueManager (same stack the old chat CLI built).
- **`process_chat_turn`** (`turn_engine.py`): model decision routing, auto-chat
  modes, coding agent / auto-execute, web research, and classic ask path.
- **`GatewayRoutingAuthority`** (`routing.py`): the sole task-aware wrapper around
  the deployed evidence router; persists each request/decision and emits events.
- **`LaneCoordinator`** (`lane_coordinator.py`): the authoritative live task
  state machine, budget/concurrency owner, and repository/file lock manager.

Before a coding stack or multi-agent coding runtime starts, the shared
`WorkspaceService.prepare_repository` boundary resolves the selected working
directory and actual Git root, validates persisted workspace ownership, reuses
normal repositories and worktrees, or safely initializes an authorized non-Git
directory without staging or committing files. Repository persistence is
reconciled under the same preparation lock. Codex receives the repository root
and selected working directory separately and only performs defensive Git
validation; it does not initialize repositories itself.

Frontends (CLI flags/I/O, TUI, Telegram, dashboard) collect config and render
results; they should not rebuild CodingAgent independently.

### Adaptive model-routing boundary

All inference lanes resolve through the gateway-owned instance of `mana_agent.model_routing.ModelRouter`. Every invocation has a persisted request and decision. The gateway inventories repository language/framework/build and changed-scope metadata once per fingerprint, then the router validates profile capabilities, context, availability, latency, budget and verification reserve before applying its deterministic evidence score. Legacy logical levels seed profiles but never bypass the router. Invalid or missing decisions stop execution.

Simple single-model execution is the default. The main model may request decomposition or candidate competition, but the router can reject or reduce that request. Parallel candidates require positive evidence, two materially qualified models, isolation, an independent verifier, ownership safety, concurrency, latency, and reserved budget. Candidate executors use separate managed worktrees or patch roots; normalized diffs and executed check evidence go to the independently routed verifier. The winner alone may be promoted. See [Evidence-based model routing](model-routing.md).

After routing selects the final provider/model, the shared
`mana_agent.context_cost` service resolves its capability and Decimal pricing
profile, estimates the serialized payload and expected calls, validates context
and output capacity, and creates a durable reservation. Provider usage then
reconciles that reservation; missing usage is explicitly estimated. Gateway
lanes and the execution supervisor persist policy limits and estimate metadata
without treating a lane cap as model capacity. See
[Model-aware token accounting](model-token-accounting.md).

### Prompting and flow context

- **`src/mana_agent/agent/flow.py`** builds an `AgentFlow` (goal + phase + verification
  plan) that drives which parts of the workflow run.
  See: `src/mana_agent/agent/flow.py:1-109`.

- **`src/mana_agent/prompting/builder.py`** assembles the stable prompt (rules,
  tool policy, skill index hints, verification rules) and the ephemeral prompt
  (per-call task context, retrieved snippets, recent summaries).
  See: `src/mana_agent/prompting/builder.py:1-353`.

### Managed agent worktrees (isolated coding checkouts)

Parallel coding work must not silently mutate the user's primary checkout.
Mana-Agent therefore allocates a **managed Git worktree** for eligible multi-agent
coding/tool routes:

```text
Taskboard → QueueManager → WorkspaceManager → isolated Git worktree
  → CodingAgent → Verifier → Reviewer → merge_candidate
```

- **`src/mana_agent/multi_agent/worktrees/manager.py`** (`WorkspaceManager`) owns the full lifecycle:
  create/reuse/resume, status transitions, branch naming (`mana/<task-slug>`), dirty detection,
  reconciliation against `git worktree list --porcelain`, safe remove, and explicit merge.
- Worktree checkouts live outside the source tree under
  `~/.mana/repositories/<repository-id>/worktrees/`.
- Metadata (task, agent, branch, base revision, path, status) is persisted under
  `~/.mana/repositories/<repository-id>/managed_worktrees/`.
- Execution roots are passed explicitly via `TaskBoardItem.execution_repo_root`,
  `QueueJob.execution_repo_root`, and `ExecutionContext.execution_repo_root`. Tools never rely on
  mutating process `cwd`.
- Successful work becomes a **merge candidate** only. Merge requires validated user intent
  (`mana-agent worktree merge <task-id> --yes`) and the existing Git safety policy.
- Dirty, failed, interrupted, or review-rejected workspaces are retained for inspection;
  destructive cleanup requires explicit force intent.

CLI: `mana-agent worktree list|create|status|resume|diff|merge|remove`.

Disable with `MANA_MANAGED_WORKTREES_ENABLED=false` when the existing non-worktree coding path is required.

### Coding orchestration (work queue + decision lifecycle)

The adaptive scope, evidence, delegation, communication, and stop contracts are
described in [`adaptive-coding-runtime.md`](adaptive-coding-runtime.md).

- **`src/mana_agent/multi_agent/runtime/agent_work_queue.py`** is the orchestration spine.
  It defines:
  - `WorkItem`: a unit of work (tool call + gate + priority + dependencies +
    fingerprint)
  - `AgentWorkQueue`: a dependency-aware, fingerprint-deduplicated queue
  - `EventBus` + `TaskBoard`: status transitions and a live renderable progress view
  - `WorkQueueRunner`: `claim -> execute -> classify -> broadcast -> sniff`
  - `JobSniffer`: a hook for the coding-agent to emit follow-up jobs
  See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py:1-1969`.

Key lifecycle details from the implementation:

1. **Queue submission**: jobs are enqueued via `AgentWorkQueue.submit()`.
   Fingerprints prevent redundant work for idempotent kinds (discover/search/read).
2. **Claiming**: the runner claims the highest-priority job whose dependencies are
   satisfied.
3. **Execution + classification**: execution is done through an injected
   `execute(item) -> WorkResult`. Runner classifies results into `done/failed/retry`.
4. **Sniffed follow-ups**: when a job finishes successfully, the sniffer may enqueue
   additional jobs.
5. **Mutation phase + verification gate**: edit jobs compile/execute registered
   mutation tools, then verification is summarized from the execution trace.
   If required deliverables are missing or verification fails, the run is blocked
   without fabricating content.

### Tool worker and tool manager processes

This architecture assumes tool execution is delegated to a “worker” client.
The queue runner itself stays deterministic by injecting the worker executor:

- `QueueManager` owns a `worker_client` and builds a worker executor using
  `make_worker_executor()` (imported in `QueueManager.run`).
  See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py` around `QueueManager.run`.

### Mutation commands (the contract for edits)

Mutation tool execution uses a typed command contract:

- The queue executor calls `execute_registered_mutation_command(repo_root, command)`.
  This validates the `MutationCommand` and routes to safe repository mutation tools
  (`write_file`, `create_file`, `delete_file`, `apply_patch`, `apply_patch_batch`).
  See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py` `execute_registered_mutation_command`.

### Repository access and mutation tools

- **Patch application**: `src/mana_agent/tools/apply_patch.py` implements a Codex-style
  patch format with strict path validation and optional read-before-patch safety.
  It also writes patch history under the configured logs directory.
  See: `src/mana_agent/tools/apply_patch.py:1-447`.

### Analysis / ask service (indexed Q&A)

- **`src/mana_agent/services/ask_service.py`** is the central orchestration layer for
  answering questions over indexed code context.
  It supports:
  - “classic ask” using FAISS semantic search when the semantic index exists
  - a fallback to direct project search when the FAISS index is missing/empty
  - an agent/tool path (`ask_with_tools` / `ask_with_tools_dir_mode`)
  See: `src/mana_agent/services/ask_service.py:1-448`.

Important behaviors:

- Semantic index missing triggers a direct project search fallback with explicit
  warnings. See: `src/mana_agent/services/ask_service.py:1-448` (constants and fallback paths).

### Indexing and embeddings (FAISS vector store)

- **`src/mana_agent/vector_store/embeddings.py`** constructs an embeddings client
  compatible with the configured `base_url` / provider.
  In particular, it supports NVIDIA Build / NIM endpoints by:
  - resolving `NVIDIA_API_KEY` + `NVIDIA_BASE_URL` through
    `resolve_inference_connection` (never via `OPENAI_API_KEY`)
  - disabling client-side tokenization (`check_embedding_ctx_length=False`)
  - setting `extra_body["input_type"]` to `"query"` vs `"passage"`
  See: `src/mana_agent/vector_store/embeddings.py` and
  `src/mana_agent/config/inference_provider.py`.

- **Inference providers** are registered in
  `src/mana_agent/config/provider_registry.py` (`openai`, `openrouter`,
  `nvidia`, `custom`). Runtime transport is OpenAI-compatible Chat Completions
  for all of them; credentials and base URLs stay provider-isolated.

- The ask service uses the FAISS store (`FaissStore`) and falls back when the on-disk
  index under `.mana/` or the requested `index_dir` is missing.
  See: `src/mana_agent/services/ask_service.py:1-448`.

### Skills and progressive skill loading

- **`src/mana_agent/skills/manager.py`** loads skills from:
  - project-local `skills/`
  - global user skills under `~/.mana/skills/`
  - built-in skills bundled in the package

It also supports building a skill index and reading individual skills.
See: `src/mana_agent/skills/manager.py:1-441`.

The trusted Experience-to-Skill Workshop is isolated under
`src/mana_agent/builtin_skills/skill_creator/`. It runs only after the normal
act, verify, summarize, and outcome-recording lifecycle:

```text
Task outcome -> deterministic eligibility -> typed model draft -> validation
  -> ~/.mana/skill-proposals/ -> explicit review -> ~/.mana/skills/
```

The evaluator and confidence calculation are deterministic. The model-dependent
generator returns a typed `SkillDraft`; recorded task events remain the evidence
authority. `ProposalStorage` performs locked atomic writes and keeps proposals
and quarantine outside active skill loading. Installation revalidates, preserves
provenance, updates the active index, and refuses silent overwrite. Workshop
errors emit shared execution events without changing the original task status.

### UI and rendering layer

- **`src/mana_agent/ui/banner.py`** renders the CLI banner and compact mode headers.
  See: `src/mana_agent/ui/banner.py:1-56`.

- **`src/mana_agent/renderers/html_report.py`** renders HTML for analyze/describe/report
  flows using helper primitives (sections, badges, tables, details blocks, copy buttons).
  See: `src/mana_agent/renderers/html_report.py:1-613`.

## Data Flow (end-to-end)

### A. Analyze / describe / report artifacts
1. A CLI/command handler selects an analyze/describe/report mode.
2. Services run analysis and generate artifact payloads.
3. Renderers convert payloads into Markdown/HTML artifacts.
4. Output format selection is shared via `commands/analyze_formats.py`.

### B. Ask (indexed Q&A)
1. `AskService.ask()` chooses semantic index search when present.
2. If FAISS semantic index is missing, it falls back to direct `project_search()`.
3. `QnAChain` produces the final answer from retrieved snippets.
See: `src/mana_agent/services/ask_service.py:1-448`.

### C. Coding / mutation workflows (agentic loop)
1. **Flow building**: `build_agent_flow()` computes goal/phase/verification plan.
   See: `src/mana_agent/agent/flow.py:1-109`.
2. **Prompt assembly**: `prompting/builder.py` composes stable + ephemeral prompt
   context for the coding agent.
   See: `src/mana_agent/prompting/builder.py:1-353`.
3. **Work queue planning/execution**:
   - queue runner executes gated tool jobs
   - sniffer emits additional read/edit/verify jobs on successful job completion
   See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py:1-1969`.
4. **Mutation execution**:
   - mutation commands compile and execute against safe repository mutation tools
   - verification is derived from the tool execution trace
   - missing deliverables or failed checks block the final result
   See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py` and `src/mana_agent/tools/apply_patch.py`.

## Repository Layout

```text
src/mana_agent/
  agent/                 # flow + phase selection + verification planning
  analysis/              # static analysis + chunk helpers
  commands/              # CLI/chat command surface and output format contracts
  config/                # settings, runtime config
  dependencies/          # dependency graph support
  describe/              # repository description flow
  multi_agent/runtime/                   # prompt chains, agents, tool managers/workers, queue
  parsers/               # source parsing entry points
  prompting/             # stable/ephemeral prompt assembly and memory snapshots
  renderers/            # HTML rendering and export helpers
  services/             # ask/analyze/report orchestration services
  skills/               # skill loading and skill index matching
  builtin_skills/       # trusted non-user-loadable capabilities (skill-creator)
  tools/                # repository access + safe mutation tools
  ui/                   # console UI helpers
  utils/                # guards, IO, discovery, helper glue
  vector_store/         # FAISS store and embedding construction
```

## Related Docs

- [Overview](./01-overview.md)
- [Live Canvas and A2UI](./live-canvas.md)
- [Project Diagram](./07-diagram.md)
- [README](../README.md)
- [Resilient Execution Supervisor](./29-resilient-execution.md)
