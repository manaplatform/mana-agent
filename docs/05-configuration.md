# Configuration

## Durable human inbox

Inbox state lives under `~/.mana/inbox` (or the existing test/managed
`MANA_HOME`). Reviewer role and group membership is explicit in
`~/.mana/inbox/identities.json`; missing role/group membership leaves requests
pending with a configuration error. Per-request typed policies define expiry,
reminders, escalation targets, and behavior. No repository `.env` or ambient
administrator fallback changes reviewer routing. The signing key and protected
context use owner-only local storage. See
[Durable Human-in-the-Loop Inbox](32-durable-human-inbox.md#reviewer-directory).

## Scoped memory capsules

Scoped capsules are enabled with conservative review and retention defaults. The persisted `~/.mana/config.toml` keys correspond to these typed settings:

```toml
MANA_MEMORY_CAPSULES_ENABLED = true
MANA_MEMORY_CAPSULES_DEFAULT_MAX_CAPSULES = 12
MANA_MEMORY_CAPSULES_DEFAULT_MAX_TOKENS = 4000
MANA_MEMORY_CAPSULES_SHARED_WRITES_REQUIRE_REVIEW = true
MANA_MEMORY_CAPSULES_ORGANISATION_SCOPE_ENABLED = false
MANA_MEMORY_CAPSULES_USER_SCOPE_ENABLED = true
MANA_MEMORY_CAPSULES_RECORD_ACCESS_EVENTS = true
MANA_MEMORY_CAPSULES_QUARANTINE_PROMPT_INJECTION = true
MANA_MEMORY_CAPSULES_RETENTION_PRIVATE_DAYS = 7
MANA_MEMORY_CAPSULES_RETENTION_PARENT_CHILD_DAYS = 30
MANA_MEMORY_CAPSULES_RETENTION_TEAM_DAYS = 90
MANA_MEMORY_CAPSULES_RETENTION_PROJECT_DAYS = 180
MANA_MEMORY_CAPSULES_RETENTION_ORGANISATION_DAYS = 365
```

Disabling capsules preserves capsule storage and selects the existing supported memory path. It does not promote, migrate, or copy capsule data. Shared writes without review are rejected, and organisation scope remains disabled in the initial rollout. See [Scoped shared-memory capsules](30-scoped-memory-capsules.md).

## Execution supervisor

The supervisor is enabled by default and stores process-independent state below
`~/.mana/execution`. The typed settings are:

```toml
MANA_EXECUTION_SUPERVISOR_ENABLED = true
MANA_EXECUTION_SUPERVISOR_LEASE_SECONDS = 60
MANA_EXECUTION_SUPERVISOR_HEARTBEAT_SECONDS = 15
MANA_EXECUTION_SUPERVISOR_CHECKPOINT_SECONDS = 60
MANA_EXECUTION_SUPERVISOR_RETRY_BUDGET = 3
MANA_EXECUTION_SUPERVISOR_MAX_REPLANS = 2
MANA_EXECUTION_SUPERVISOR_MAX_CHILD_DEPTH = 5
MANA_EXECUTION_SUPERVISOR_MAX_CHILDREN = 20
MANA_EXECUTION_SUPERVISOR_MAX_TOTAL_SUBTASKS = 100
MANA_EXECUTION_SUPERVISOR_MAX_CONCURRENT_CHILDREN = 4
MANA_EXECUTION_SUPERVISOR_STARTUP_RECOVERY = true
MANA_EXECUTION_SUPERVISOR_VERIFY_ARTIFACTS = true
MANA_EXECUTION_SUPERVISOR_ALLOW_UNKNOWN_RETRY = false
```

Heartbeat duration must be shorter than lease duration. Unknown-side-effect
retry is false by design; enabling it is an explicit operator risk decision,
not a routing fallback. Normal Mana precedence still applies: persisted
`~/.mana/config.toml`, protected secrets where applicable, then safe defaults.
Repository `.env` files do not override these persisted settings.

Disabling the supervisor is fail-closed: execution entry points refuse to start
unsupervised work. Disabling artifact verification likewise leaves submitted
results in `completed_pending_verification`; it never converts verification-off
into an implicit success path.

## Context and cost governor

Every gateway session owns one context/cost governor shared by chat, routing,
the internal coding runtime, tool workers, and Codex. It reuses the existing
`MANA_ROUTING_*` limits; these settings control allocation and visibility:

`AskAgent` construction requires an explicit `ContextCostGovernor`. Gateway,
worker, Telegram, and TUI factories must pass the governor with the session
identity; a missing governor is a construction error and does not select an
ungoverned fallback path.

```toml
MANA_CONTEXT_GOVERNOR_ENABLED = true
MANA_CONTEXT_GOVERNOR_MODE = "observe" # observe, soft, enforce
MANA_CONTEXT_WARNING_RATIO = 0.70
MANA_CONTEXT_COMPACT_RATIO = 0.80
MANA_CONTEXT_MAX_UTILIZATION = 0.85
MANA_CONTEXT_HARD_LIMIT_RATIO = 0.95
MANA_CONTEXT_RESPONSE_RESERVE_RATIO = 0.12
MANA_CONTEXT_RESPONSE_RESERVE_TOKENS = 0
MANA_CONTEXT_TOOL_RESULT_MAX_TOKENS = 2000
MANA_CONTEXT_HISTORY_MAX_TOKENS = 8000
MANA_CONTEXT_RETRIEVAL_MAX_TOKENS = 12000
MANA_CONTEXT_LAZY_CAPABILITIES = true
MANA_CONTEXT_CAPABILITY_IDLE_STEPS = 3
MANA_CONTEXT_ARTIFACT_RETENTION_DAYS = 30
MANA_CONTEXT_COST_LOG_ENABLED = true
MANA_CONTEXT_COST_LOG_RETENTION_DAYS = 30
```

Ratios must increase strictly from warning through hard limit. Start in
`observe`, inspect `mana-agent context report --since 7d`, move to `soft` for
reversible compaction, then use `enforce` for hard rejection. Configured model
profile prices are exact; fallback prices and missing usage are labeled
estimated. A `MANA_MODEL_PROFILES` entry whose configuration contains
`pricing_fallback=true` supplies the fallback input/output rates without
creating a second task-cost budget. Redacted analytics rotate under
`~/.mana/logs/context-cost/`, while
lossless permitted tool results live under
`~/.mana/context-cache/tool-results/`; both honor `MANA_HOME`.

## Teach Mode

The optional `[teach]` table controls semantic event sources, user-data storage,
retention, browser/native capture, exclusions, redaction, verification, replay,
correction checkpoints, private Flow Cards, and experimental sharing. Screenshot
and voice persistence default off, redaction defaults on, and imports always
require dry run. The complete typed keys are documented in
[Teach Mode](26-teach-mode.md#configuration).

`desktop_capture` defaults to false. `allowed_applications`,
`excluded_applications`, and event-source settings bound native monitoring; OS
privacy approval and the separate Teach grant store are still required when the
configuration enables it.

## Mana Fleet

Fleet is disabled by default with `MANA_FLEET_ENABLED=false`. Worker, timeout,
concurrency, capability TTL, log/artifact limit, retention, repair, and trust
settings are listed in [the Fleet guide](24-fleet.md#configuration). Enabling
Fleet does not register or trust a worker; enrollment and a fresh authenticated
capability inventory are still required.

## Computer control

Computer control is disabled by default and configured through the Textual
Settings → Computer control tab, Dashboard → Computer Control, or the
`[computer_control]` table. Permission scopes, allowed filesystem roots, remote
restrictions, exact-action confirmations, audit retention, defaults, and the
complete example are documented in
[`22-computer-control.md`](22-computer-control.md). `MANA_COMPUTER_CONTROL_ENABLED`
is a compatibility flat switch; the validated table is authoritative.

## Experience-to-Skill Workshop

The workshop reads the `[experience_to_skill]` table from the existing user
configuration at `~/.mana/config.toml`:

```toml
[experience_to_skill]
enabled = true
auto_propose = true
minimum_confidence = 0.80
needs_attention_confidence = 0.60
minimum_successful_runs = 1
require_verification = true
require_user_acceptance = false
semantic_duplicate_threshold = 0.88
retain_rejected_days = 90
quarantine_on_validation_failure = true
```

`MANA_SKILLS_ROOT`, `MANA_SKILL_PROPOSALS_ROOT`, and
`MANA_SKILL_QUARANTINE_ROOT` override the three storage roots. Every scalar also
has a `MANA_EXPERIENCE_TO_SKILL_*` environment override documented in
[`19-experience-to-skill-workshop.md`](19-experience-to-skill-workshop.md).
Tests should set these paths to temporary directories and must not use the
developer's actual `~/.mana` state.

`mana-agent` stores and reads its managed settings from the user-level `~/.mana`
directory. Repository `.env` files and shell environment variables are not used
for Mana-managed configuration.

## First-Run Wizard

Run:

```bash
mana-agent
```

When no saved user config exists, the CLI prints the Mana banner first, then starts a keyboard-selectable setup wizard. The wizard can:

- Configure OpenAI, OpenAI-compatible, NVIDIA OpenAI-compatible, or manual provider settings.
- Enter API keys without echoing them back to the terminal.
- Fetch models from `GET {OPENAI_BASE_URL}/models`.
- Select chat, tool-worker, coding-planner, and embedding models.
- Assign model levels for Mana roles such as main, planner, coding, verifier, reviewer, tool, and summarizer.
- Configure web and GitHub search providers.
- Save a masked config summary for review.

Saved files:

- `~/.mana/config.toml` for non-secret settings.
- `~/.mana/secrets.toml` for API keys and tokens. The file is written atomically
  with owner-only permissions where supported.
- `~/.mana/model_cache.json` for fetched model IDs keyed by provider/base URL.

The Memory tab stores its API key in a recommended operating-system keyring
when one is available. On headless systems without a recommended keyring
backend, it stores the key in Mana's protected `secrets.toml`; normal
`config.toml` data contains only an explicit `mana-secrets:`
`MANA_MEMORY_SECRET_REF`. Managed deployments may instead inject `MEM0_API_KEY`
or `SUPERMEMORY_API_KEY` through their secret manager/environment.

The config directory is created with private permissions where the OS allows it. Secret values are masked in display output.

## Settings Menu

The root menu includes:

- Chat with repo
- Analyze repo
- Create implementation plan
- Settings
- Exit

Settings includes:

- Change model provider/API key
- Refresh model list
- Change selected models
- Change model role levels
- Configure search providers
- Show current config summary

## Precedence

Effective settings are resolved from `~/.mana/config.toml` and
`~/.mana/secrets.toml`, then safe defaults. This repository-independent policy
prevents a project's `.env` or a shell variable from replacing credentials or
model settings selected through the Mana-Agent Settings menu.

`OPENAI_CHAT_MODEL` is the canonical chat model value. `LLM_MODEL` remains a
backward-compatible alias when `OPENAI_CHAT_MODEL` is not saved.

## Non-Interactive Use

Use `--no-interactive` in CI or scripts:

```bash
mana-agent --no-interactive chat --root-dir .
```

In non-interactive mode, Mana-Agent does not open menus or prompts. Commands that require model configuration fail clearly if required values such as `OPENAI_API_KEY` are missing.

## Memory providers

Exactly two modes are supported:

- `internal` with provider `mana` keeps memory locally managed and remains the
  compatibility-preserving default.
- `external` with provider `mem0` or `supermemory` uses the optional hosted
  provider adapter selected in configuration.

Install external support with `pip install "mana-agent[mem0]"` or
`pip install "mana-agent[supermemory]"`. Configure it in the Memory tab or set:

```bash
MANA_MEMORY_MODE=external
MANA_MEMORY_PROVIDER=mem0
MEM0_API_KEY="m0-..."
MEM0_ORG_ID=
MEM0_PROJECT_ID=
MEM0_BASE_URL=
MANA_MEMORY_TIMEOUT_SECONDS=15
MANA_MEMORY_FALLBACK_TO_INTERNAL=false
```

Or:

```bash
MANA_MEMORY_MODE=external
MANA_MEMORY_PROVIDER=supermemory
SUPERMEMORY_API_KEY="sm_..."
MANA_MEMORY_TIMEOUT_SECONDS=15
MANA_MEMORY_FALLBACK_TO_INTERNAL=false
```

Invalid mode/provider pairs, missing credentials, missing optional dependencies,
authentication failures, connectivity failures, and provider failures stop with
typed errors. There is no silent fallback that rewrites **semantic / conversation
AI memory** to the local provider store, and no automatic upload of existing
local memory. If a runtime explicitly permits degraded memory, it may continue
the turn without semantic memory, but it must report that state. Switch back with
`MANA_MEMORY_MODE=internal` and `MANA_MEMORY_PROVIDER=mana`.

External mode selects the hosted provider for **AI memory only**
(conversation, semantic search, and multi-agent records adapted through the
external runtime). Local **system-state stores** remain available regardless of
provider so agent routes do not crash:

| Domain | External mode backend |
| --- | --- |
| Conversation / semantic search | Hosted provider (`mem0` / `supermemory`) |
| Multi-agent task / decision records | External runtime adapter (hosted writes) |
| Run evidence (file-read cache) | Local durable store under Mana runs |
| Coding-flow checkpoints / turn history | Local SQLite system store |
| Scoped capsules metadata | Local capsule service |

`MemoryService.capabilities` declares these domains. Routes that need evidence
or coding-flow continuity call the local system stores; they do not require the
external provider to implement run evidence. Configuration errors are raised
only when no safe backend exists for a requested domain.

External memory has different privacy and retention implications because
selected content, identity scopes, and metadata leave the local machine. Review
the selected provider policy before enabling it. Local system stores never send
run evidence or coding-flow checkpoints to the hosted provider.

Chat follow-ups use the gateway-owned shared memory service in addition to the
durable session transcript. The service records successful user/assistant turn
pairs and recalls relevant records only within the active conversation scope.
A new conversation receives a new scope. The gateway explicitly permits
degraded follow-up memory: provider failures are included in turn warnings while
the transcript remains usable, and no internal semantic AI-memory fallback write
occurs.

## Core configuration keys

Set these through the Settings menu; Mana-Agent writes them to `~/.mana`.

```bash
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_CHAT_MODEL="gpt-4.1"
LLM_MODEL="gpt-4.1"
OPENAI_TOOL_WORKER_MODEL="gpt-4.1"
OPENAI_CODING_PLANNER_MODEL="gpt-4.1"
OPENAI_EMBED_MODEL="text-embedding-3-small"
MODEL_LEVEL_3_HIGH_REASONING="gpt-4.1"
MODEL_LEVEL_2_CODING="gpt-4.1"
MODEL_LEVEL_1_FAST_TOOL="gpt-4.1-mini"
DEFAULT_TOP_K=8
MANA_LLM_LOG_FILE=
MANA_LLM_API_MODE=auto
MANA_LLM_REASONING_EFFORT=
MANA_LLM_SUPPORTS_RESPONSES_API=
MANA_LLM_SUPPORTS_CHAT_COMPLETIONS=
MANA_LLM_SUPPORTS_TOOLS=
MANA_LLM_SUPPORTS_REASONING=
MANA_LLM_SUPPORTS_TOOLS_WITH_CHAT_REASONING=
MANA_MANAGED_WORKTREES_ENABLED=true
MANA_CODEX_ENABLED=true
MANA_CODEX_BIN=codex
MANA_CODEX_MAX_WORKERS=2
MANA_CODEX_STREAM_EVENTS=true
MANA_CODEX_WORKTREE_ISOLATION=false
MANA_CODEX_TASK_TIMEOUT_SECONDS=1800
MANA_CODEX_ALLOW_NETWORK=false
MANA_CODEX_MODEL=
MANA_LANE_GLOBAL_WORKER_LIMIT=8
MANA_LANE_SESSION_TOKEN_BUDGET=0
MANA_LANE_GLOBAL_TOKEN_BUDGET=0
```

### Specialist lane coordinator

All frontends use the gateway's six specialist lanes. Defaults are conservative and require no configuration. `0` means unlimited for the optional session/global token caps. Provider/model concurrency can be supplied as a JSON object through `MANA_LANE_PROVIDER_LIMITS`.

Lane overrides use the existing user configuration as a table/object. Only `enabled`, `max_concurrent_jobs`, `max_subagents`, `token_budget`, `cost_budget`, `priority`, `timeout_seconds`, and `allowed_models` are configurable; invalid lane names, fields, priorities, or non-positive limits stop gateway construction with an actionable validation error.

For execution routing, Mana uses the more restrictive of the selected model's
capacity, `MANA_ROUTING_TASK_TOKEN_BUDGET` /
`MANA_ROUTING_TASK_COST_BUDGET`, remaining session/task budget, and an optional
validated lane `token_budget` / `cost_budget`. Lane caps are product policy, not
model context metadata. See [Model-aware token accounting](model-token-accounting.md).

```toml
MANA_LANE_GLOBAL_WORKER_LIMIT = 8
MANA_LANE_SESSION_TOKEN_BUDGET = 120000

[MANA_LANE_PROVIDER_LIMITS]
"openai/gpt-4.1" = 3

[MANA_LANE_CONTRACTS.coding]
max_concurrent_jobs = 2
max_subagents = 2
token_budget = 80000
cost_budget = 25.0
priority = "interactive"
timeout_seconds = 1800
allowed_models = ["openai/gpt-4.1"]
```

Coordinator events are written to the existing taskboard/event history. Normal clients show concise lane/lock progress; `lane.*`, `lock.*`, and `resource.*` metadata is intended for verbose diagnostics and the dashboard.

### Codex coding runtime

Codex is the authoritative coding runtime and is enabled by default. It requires
the official `codex` CLI; Mana-Agent communicates with `codex app-server` and
does not depend on an unofficial Python SDK. Disabling Codex makes coding turns
fail explicitly; it does not activate the legacy planner or executor. Writing
tasks require an isolated clean worktree. See
[`20-codex-integration.md`](20-codex-integration.md).

### Managed agent worktrees

`MANA_MANAGED_WORKTREES_ENABLED` (default `true`) controls whether multi-agent
coding/tool routes allocate an isolated Git worktree under
`~/.mana/repositories/<repository-id>/worktrees/` instead of mutating the primary
checkout. Explicit merge intent is still required after review
(`mana-agent worktree merge <task-id> --yes`).

Set `MANA_MANAGED_WORKTREES_ENABLED=false` to preserve the legacy in-checkout coding path.

All LLM credentials, base URLs, chat/planner/tool-worker models, role mappings,
reasoning options, and provider capability flags are resolved from
`~/.mana/config.toml` and `~/.mana/secrets.toml`. Shell variables and repository
`.env` files do not override them. Tool-worker subprocesses receive those values
through their validated initialization payload and remove conflicting LLM
configuration variables from the inherited process environment.

## OpenAI-Compatible LLM Capabilities

Mana-Agent automatically uses the Responses API for tool calls when the active
endpoint is OpenAI. This also supports reasoning models that enable reasoning
by default and reject function tools through Chat Completions. Custom
`OPENAI_BASE_URL` gateways are treated as Chat Completions-only by default, so
tool calls stay enabled and incompatible reasoning is sent as `none`.

For a verified nonstandard gateway, configure these optional values in
`~/.mana/config.toml`:

```bash
MANA_LLM_API_MODE=auto # auto, responses, or chat_completions
MANA_LLM_REASONING_EFFORT=high
MANA_LLM_SUPPORTS_RESPONSES_API=true
MANA_LLM_SUPPORTS_TOOLS_WITH_CHAT_REASONING=false
```

Only enable Responses API support when that gateway implements `/v1/responses`.

## Adaptive model routing

Adaptive routing is enabled by default. Explicit profiles can be supplied as a JSON/TOML list in `MANA_MODEL_PROFILES`; see [Evidence-based model routing](model-routing.md) for its schema, scoring, history, budget, verifier, and competition behavior.

```bash
MANA_ADAPTIVE_ROUTING_ENABLED=true
MANA_ROUTING_COMPLEXITY_THRESHOLD=high
MANA_ROUTING_RISK_THRESHOLD=high
MANA_ROUTING_MAX_CANDIDATES=2
MANA_ROUTING_MIN_CONFIDENCE=0.55
MANA_ROUTING_TASK_TOKEN_BUDGET=32000
MANA_ROUTING_TASK_COST_BUDGET=
MANA_ROUTING_SESSION_COST_BUDGET=
MANA_ROUTING_COMPETITION_COST_BUDGET=
MANA_ROUTING_VERIFICATION_COST_BUDGET=
MANA_ROUTING_RETRY_COST_BUDGET=
MANA_ROUTING_VERIFICATION_RESERVE_RATIO=0.15
MANA_ROUTING_BENCHMARK_WEIGHTS={}
MANA_ROUTING_LANGUAGE_PREFERENCES={}
MANA_ROUTING_EVIDENCE_RETENTION_DAYS=90
MANA_ROUTING_CIRCUIT_BREAKER_FAILURES=3
MANA_ROUTING_CIRCUIT_BREAKER_WINDOW_SECONDS=900
MANA_ROUTING_RELIABILITY_DECAY_SECONDS=3600
MANA_ROUTING_MODEL_FAILURE_PENALTY_WEIGHT=0.08
MANA_ROUTING_PROVIDER_FAILURE_PENALTY_WEIGHT=0.04
MANA_CONTEXT_ESTIMATION_SAFETY_MARGIN_RATIO=0.05
MANA_CONTEXT_DEFAULT_OUTPUT_RATIO=0.20
MANA_CONTEXT_HISTORICAL_PREDICTION_ENABLED=true
MANA_CONTEXT_UNKNOWN_MODEL_POLICY=conservative
MANA_CONTEXT_UNKNOWN_MODEL_CONTEXT_WINDOW=16384
MANA_CONTEXT_UNKNOWN_MODEL_MAX_OUTPUT_TOKENS=4096
```

Invalid profile, capability, latency, or budget configuration stops selection. Disabling adaptive routing also stops model execution; it does not restore static routing.

An explicit profile includes `provider`, `model_id`, `supported_roles`, `supported_tools`, `reasoning_settings`, required `context_window` and `max_output_tokens`, optional `tokenizer`, input/cached-input/output/reasoning pricing or `logical_cost_per_1k_tokens`, `reliability_score`, `supported_languages`, `benchmark_scores`, and the `can_patch`, `can_structured_output`, `can_tool_call`, `can_verify`, and `available` flags. Duplicate provider/model IDs and invalid ranges fail validation. Unknown pricing remains unknown rather than becoming zero.

### Legacy model-level migration

`MODEL_LEVEL_*` variables contain actual model IDs. `MANA_MODEL_*` variables remain accepted and seed adaptive candidate profiles. Levels contribute initial cost, latency, reliability, and benchmark hints but do not lock a role to a model.

```bash
MODEL_LEVEL_3_HIGH_REASONING=gpt-4.1
MODEL_LEVEL_2_CODING=gpt-4.1
MODEL_LEVEL_1_FAST_TOOL=gpt-4.1-mini

MANA_MODEL_MAIN=MODEL_LEVEL_3_HIGH_REASONING
MANA_MODEL_HEAD_DECISION=MODEL_LEVEL_3_HIGH_REASONING
MANA_MODEL_PLANNER=MODEL_LEVEL_3_HIGH_REASONING
MANA_MODEL_CODING=MODEL_LEVEL_2_CODING
MANA_MODEL_VERIFIER=MODEL_LEVEL_2_CODING
MANA_MODEL_REVIEWER=MODEL_LEVEL_3_HIGH_REASONING
MANA_MODEL_TOOL=MODEL_LEVEL_1_FAST_TOOL
MANA_MODEL_SUMMARIZER=MODEL_LEVEL_1_FAST_TOOL
```

Allowed model levels:

- `MODEL_LEVEL_1_FAST_TOOL`
- `MODEL_LEVEL_2_CODING`
- `MODEL_LEVEL_3_HIGH_REASONING`

## Search Providers

The wizard can configure:

- Disabled
- Tavily
- Brave Search API
- Exa
- SerpAPI
- Google Programmable Search / Custom Search JSON API
- Custom HTTP provider

Relevant variables:

```bash
MANA_GITHUB_TOKEN=
MANA_SEARCH_ENABLE_WEB=true
MANA_SEARCH_ENABLE_GITHUB=true
MANA_SEARCH_MAX_RESULTS=8
MANA_SEARCH_TIMEOUT_SECONDS=15
MANA_SEARCH_MEMORY_TTL_DAYS=14
MANA_WEB_SEARCH_PROVIDER=tavily
MANA_WEB_SEARCH_API_KEY=
MANA_WEB_SEARCH_MAX_RESULTS=8
MANA_WEB_SEARCH_ENGINE_ID=
MANA_WEB_SEARCH_BASE_URL=
MANA_WEB_SEARCH_ENDPOINT=
```

GitHub tokens are optional. Without a token, GitHub search may still work with unauthenticated rate limits.

## Live Canvas / A2UI

Live Canvas is configured in `~/.mana/config.toml`. Defaults are intentionally bounded; inline catalogs and non-HTTPS remote images are disabled. Loopback HTTP is allowed by default for local development only.

```toml
MANA_CANVAS_ENABLED = true
MANA_CANVAS_PROTOCOL_VERSIONS = "v0.9"
MANA_CANVAS_DEFAULT_PROTOCOL_VERSION = "v0.9"
MANA_CANVAS_ALLOWED_CATALOGS = "https://mana-agent.dev/a2ui/catalogs/core/v1/catalog.json"
MANA_CANVAS_ACCEPT_INLINE_CATALOGS = false
MANA_CANVAS_ALLOW_LOCALHOST = true
MANA_CANVAS_MAX_ACTIVE_SURFACES = 16
MANA_CANVAS_MAX_COMPONENTS = 250
MANA_CANVAS_MAX_EVENT_BYTES = 262144
MANA_CANVAS_MAX_DEPTH = 24
MANA_CANVAS_SNAPSHOT_INTERVAL = 20
MANA_CANVAS_GENERATION_TIMEOUT_SECONDS = 30
MANA_CANVAS_SURFACE_EXPIRY_SECONDS = 86400
MANA_CANVAS_ACTION_TIMEOUT_SECONDS = 900
MANA_CANVAS_VALIDATION_RETRY_LIMIT = 1
MANA_CANVAS_MAX_UPDATES_PER_SECOND = 20
MANA_CANVAS_WEBSOCKET_QUEUE_SIZE = 256
MANA_CANVAS_ALLOWED_IMAGE_SCHEMES = "https"
MANA_CANVAS_ALLOWED_ARTIFACT_SCHEMES = "https,artifact"
MANA_CANVAS_DEVELOPER_DIAGNOSTICS = false
```

The built-in catalog is also served at
`http://localhost:<api-port>/api/v1/canvas/catalogs/core/v1/catalog.json`.
Only `localhost` and `127.0.0.1` may use HTTP. Setting
`MANA_CANVAS_ALLOW_LOCALHOST = false` disables these aliases. Unsupported
versions, unsafe URL schemes, missing catalogs, excessive retry limits, and
non-positive resource bounds fail configuration validation. See
[Live Canvas and A2UI](./live-canvas.md).
