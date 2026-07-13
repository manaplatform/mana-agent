# Quick Start

`mana-agent` is a Python CLI for analyzing local repositories, asking questions with repository context, and opening an interactive coding-agent chat. The project is packaged as an installable tool named `mana-agent` in `pyproject.toml`, and the README shows that the main entry point is the `mana-agent` command. [pyproject.toml:1-51](pyproject.toml:1-51) [README.md:1-242](README.md:1-242)

## Requirements

- Python 3.10 through 3.14. [pyproject.toml:1-51](pyproject.toml:1-51)
- An OpenAI-compatible chat and embedding endpoint. [README.md:1-242](README.md:1-242)
- An OpenAI API key plus model settings saved through Mana-Agent's first-run
  Settings wizard under `~/.mana`. [README.md:1-242](README.md:1-242)

The default dependency set includes Typer, Rich, Pydantic, python-dotenv, LangChain, OpenAI, FAISS, Safety, Redis, RQ, and Tenacity. [pyproject.toml:1-51](pyproject.toml:1-51)

## Install

Create and activate a virtual environment, then install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For local development and checks, the README also installs pytest, ruff, and mypy alongside the package. [README.md:1-242](README.md:1-242)

```bash
python -m pip install -e .
python -m pip install pytest ruff mypy
```

## Configure

Run `mana-agent` in an interactive terminal and enter your provider settings in
the first-run wizard. Mana-Agent saves the values under `~/.mana`; it does not
read credentials from the current repository's `.env` file.

```bash
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_CHAT_MODEL="gpt-4.1"
OPENAI_TOOL_WORKER_MODEL="gpt-4.1"
OPENAI_CODING_PLANNER_MODEL="gpt-4.1"
OPENAI_EMBED_MODEL="text-embedding-3-small"
DEFAULT_TOP_K=8
```

Use the Settings menu to update the saved provider configuration later.

## First run

From a repository you want to inspect, run unified analysis:

```bash
mana-agent analyze /path/to/project
```

The `analyze` command resolves the target path, indexes the project, runs dependency and structure analysis, performs static findings, optionally runs LLM-assisted findings, and writes artifacts under the analyzed project’s `.mana/` directory. [src/mana_agent/commands/analyze_cli.py:1-444](src/mana_agent/commands/analyze_cli.py:1-444)

If you want semantic search results included in the report, add a query:

```bash
mana-agent analyze /path/to/project --query "authentication flow"
```

To emit machine-readable output, use JSON mode:

```bash
mana-agent analyze /path/to/project --json
```

### What `analyze` writes

When the command succeeds, it writes these artifacts under `<project>/.mana/`:

- `analyze.json`
- `analyze.md`
- `analyze.html`
- `analyze.dot`
- `analyze.graphml`

Those paths are built directly in the command implementation. [src/mana_agent/commands/analyze_cli.py:1-444](src/mana_agent/commands/analyze_cli.py:1-444)

## Ask a question

Use `ask` for repository-grounded Q&A:

```bash
mana-agent ask "How is configuration loaded?" --root-dir /path/to/project
```

The `ask` command supports an optional index directory, directory-aware mode, ephemeral indexes, agent tool use, and JSON output. It returns sources when available. [src/mana_agent/commands/ask_cli.py:1-262](src/mana_agent/commands/ask_cli.py:1-262)

## Open chat

Start the interactive chat session with a project root:

```bash
mana-agent chat --root-dir /path/to/project
```

The chat command supports classic and directory-aware index modes, coding-agent workflows, tool-worker execution, coding memory, planning mode, auto-execute flows, and multiline input. [src/mana_agent/commands/chat_cli.py:1-2332](src/mana_agent/commands/chat_cli.py:1-2332)

## Typical workflow

1. Install the package in a virtual environment.
2. Configure `OPENAI_API_KEY` and model settings.
3. Run `mana-agent analyze <project>` to build the initial index and reports.
4. Use `mana-agent ask ...` for targeted questions.
5. Use `mana-agent chat ...` for interactive repository analysis or coding tasks.

## Optional browser tasks

Install Mana-Agent and its managed Chromium runtime:

```bash
python -m pip install -e .
python -m playwright install chromium
```

Browser tools are available from the normal chat experience. For example:

```text
Check https://example.test and summarize the page's forms and interactive controls.
```

The model selects each structured browser action. It pauses before payments,
publishing, deletion, legal acceptance, or final form submission, and it never
attempts to bypass CAPTCHA, MFA, or access controls. See
[Browser Automation](./17-browser-automation.md).

## Useful flags

The README documents several global flags that work across commands:

```bash
mana-agent --verbose analyze .
mana-agent --log-dir .mana/logs ask "summarize the parser"
mana-agent --output-dir .mana/output chat
```

It also documents command-specific options such as `--query`, `--k`, `--json`, `--root-dir`, `--index-dir`, `--dir-mode`, `--auto-index-missing`, `--agent-tools`, `--agent-max-steps`, `--planning-mode`, `--auto-execute-plan`, `--full-auto`, `--coding-memory`, `--tool-worker-process`, `--multiline-input`, and diagram rendering flags. [README.md:1-242](README.md:1-242) [src/mana_agent/commands/analyze_cli.py:1-444](src/mana_agent/commands/analyze_cli.py:1-444) [src/mana_agent/commands/ask_cli.py:1-262](src/mana_agent/commands/ask_cli.py:1-262) [src/mana_agent/commands/chat_cli.py:1-2332](src/mana_agent/commands/chat_cli.py:1-2332)
