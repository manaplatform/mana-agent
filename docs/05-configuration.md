# Configuration

`mana-agent` reads its runtime configuration from environment variables and optional `.env` files. The repository’s quick-start documentation shows the expected settings and the command modules expose the knobs that consume them. [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119) [src/mana_agent/commands/analyze_cli.py:1-444](src/mana_agent/commands/analyze_cli.py:1-444) [src/mana_agent/commands/ask_cli.py:1-262](src/mana_agent/commands/ask_cli.py:1-262) [src/mana_agent/commands/chat_cli.py:1-2332](src/mana_agent/commands/chat_cli.py:1-2332)

## Environment variables

The quick-start guide documents these core variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_CHAT_MODEL`
- `OPENAI_TOOL_WORKER_MODEL`
- `OPENAI_CODING_PLANNER_MODEL`
- `OPENAI_EMBED_MODEL`
- `DEFAULT_TOP_K` [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119)

A typical local setup uses a `.env` file with those values. The docs point to `.env.example` as the template. [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119)

## Model selection

The CLI exposes model selection on multiple commands, so configuration can be overridden per invocation:

- `analyze` accepts `--model` and related analysis controls. [docs/04-commands.md:1-171](docs/04-commands.md:1-171)
- `ask` accepts `--model` for question-answering. [docs/04-commands.md:1-171](docs/04-commands.md:1-171)
- `chat` accepts `--model` and additional workflow-related options. [docs/04-commands.md:1-171](docs/04-commands.md:1-171)

This makes the environment the default source of truth, with CLI flags used for one-off overrides.

## Search and retrieval defaults

The documentation shows `DEFAULT_TOP_K=8`, which matches the search-oriented command defaults documented elsewhere in the repo. [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119)

Relevant command options include:

- `--k` for retrieval depth
- `--index-dir` for selecting an existing index
- `--ephemeral-index` for temporary indexing
- `--dir-mode` for directory-aware indexing and retrieval [docs/04-commands.md:1-171](docs/04-commands.md:1-171)

External search is routed separately from repository-local retrieval. The agent first decides whether outside sources are useful, then reuses fresh search memory before calling configured providers. Useful results are compacted and stored under `.mana/search_memory.jsonl`.

External search variables:

- `MANA_SEARCH_ENABLE_WEB`
- `MANA_SEARCH_ENABLE_GITHUB`
- `MANA_SEARCH_MAX_RESULTS`
- `MANA_SEARCH_TIMEOUT_SECONDS`
- `MANA_SEARCH_MEMORY_TTL_DAYS`
- `MANA_GITHUB_TOKEN`
- `MANA_WEB_SEARCH_PROVIDER`
- `MANA_WEB_SEARCH_API_KEY`
- `MANA_WEB_SEARCH_ENDPOINT`
- `MANA_WEB_SEARCH_MAX_RESULTS`

## Working directories and outputs

Several commands write generated artifacts under `.mana/` inside the target project:

- `analyze` writes report artifacts such as `analyze.json`, `analyze.md`, `analyze.html`, `analyze.dot`, and `analyze.graphml`. [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119)
- `chat` can persist coding memory at `<project>/.mana/index/chat_memory.sqlite3`. [docs/04-commands.md:1-171](docs/04-commands.md:1-171)
- `continue` resumes runs from `<root>/.mana/runs/<run_id>`. [docs/04-commands.md:1-171](docs/04-commands.md:1-171)

That means configuration should be kept alongside the project being analyzed so runs remain reproducible.

## Recommended local setup

1. Create a virtual environment.
2. Put API keys and model settings in `.env` or export them in your shell.
3. Verify the configuration by running `mana-agent --help` and a small command such as `mana-agent ask ...` or `mana-agent analyze ...`. [docs/02-installation.md:1-26](docs/02-installation.md:1-26) [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119)

If you need to tune behavior for a specific run, prefer CLI flags over editing global environment settings.
