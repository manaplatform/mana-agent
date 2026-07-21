# Configuration

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
- `~/.mana/secrets.toml` for API keys and tokens.
- `~/.mana/model_cache.json` for fetched model IDs keyed by provider/base URL.

Mem0 credentials are an exception to the legacy secrets file: the Memory tab
stores the API key in the operating-system keyring and writes only a
`MANA_MEMORY_SECRET_REF` reference to `config.toml`. Headless deployments should
inject `MEM0_API_KEY` directly through their secret manager/environment.

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
- `external` with provider `mem0` uses the optional hosted provider adapter.

Install external support with `pip install "mana-agent[mem0]"`. Configure it in
the Memory tab or set:

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

Invalid mode/provider pairs, missing credentials, missing optional dependencies,
authentication failures, connectivity failures, and provider failures stop with
typed errors. There is no silent fallback or automatic upload of existing local
memory. If a runtime explicitly permits degraded memory, it may continue the
turn without memory, but it must report that state. Switch back with
`MANA_MEMORY_MODE=internal` and `MANA_MEMORY_PROVIDER=mana`.

External memory has different privacy and retention implications because
selected content, identity scopes, and metadata leave the local machine. Review
the provider policy before enabling it.

Chat follow-ups use the gateway-owned shared memory service in addition to the
durable session transcript. The service records successful user/assistant turn
pairs and recalls relevant records only within the active conversation scope.
A new conversation receives a new scope. The gateway explicitly permits
degraded follow-up memory: provider failures are included in turn warnings while
the transcript remains usable, and no internal fallback write occurs.

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

## Model Role Levels

`MODEL_LEVEL_*` variables contain actual model IDs. `MANA_MODEL_*` variables map each Mana role to one of those model levels.

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
