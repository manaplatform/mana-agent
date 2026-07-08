# Configuration

`mana-agent` supports an interactive first-run setup wizard and still supports existing environment-variable and `.env` workflows.

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

Effective settings are resolved in this order:

1. CLI flags, when a command exposes a flag.
2. Environment variables and project `.env`.
3. `~/.mana/config.toml` and `~/.mana/secrets.toml`.
4. Safe defaults.

`OPENAI_CHAT_MODEL` is the canonical chat model value. `LLM_MODEL` remains a backward-compatible alias and is honored when `OPENAI_CHAT_MODEL` is not set in the environment.

## Non-Interactive Use

Use `--no-interactive` in CI or scripts:

```bash
mana-agent --no-interactive chat --root-dir .
```

In non-interactive mode, Mana-Agent does not open menus or prompts. Commands that require model configuration fail clearly if required values such as `OPENAI_API_KEY` are missing.

## Core Variables

```bash
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_CHAT_MODEL="gpt-4.1"
LLM_MODEL="gpt-4.1"
OPENAI_TOOL_WORKER_MODEL="gpt-4.1"
OPENAI_CODING_PLANNER_MODEL="gpt-4.1"
OPENAI_EMBED_MODEL="text-embedding-3-small"
DEFAULT_TOP_K=8
MANA_LLM_LOG_FILE=
```

## Model Role Levels

```bash
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
