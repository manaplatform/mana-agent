# Configuration

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
```

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
