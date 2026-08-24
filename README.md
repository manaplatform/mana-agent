<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

<h1 align="center">Mana-Agent</h1>

<p align="center">
  <strong>Repository intelligence and safe multi-agent coding automation.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.14-blue" /></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green" /></a>
  <img alt="Version" src="https://img.shields.io/badge/version-v0.1.8-purple" />
</p>

Mana-Agent is a Python CLI and optional dashboard for understanding, operating, and safely changing software repositories.

## Features

- Repository analysis and evidence-backed chat
- Multi-agent planning, coding, review, and verification
- Safe Git, document, browser, and computer-control tools
- Reusable external API integrations with validated requests, approvals, and SSRF protection
- Adaptive model routing and repository-scoped memory
- Visible context/cost budgets, lazy tool schemas, and lossless-backed result compression
- Automations, Teach Mode, Live Canvas, and media generation
- Gmail, Telegram, ACP, A2A, MCP, SSH, and reverse-worker integrations
- CLI, dashboard, and reusable analysis artifacts
- Durable execution supervision with leases, checkpoints, safe recovery, cancellation, result escrow, and verified completion

## Install

### pipx

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

### From source

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install all optional features:

```bash
python -m pip install -e ".[full]"
```

## Quick start

```bash
mana-agent --configure
cd /path/to/project
mana-agent
```

Useful commands:

```bash
mana-agent doctor
mana-agent chat --root-dir .
mana-agent dashboard --root-dir .
mana-agent git -- status
mana-agent context report --since 7d
mana-agent tasks list --incomplete
mana-agent tasks status <task-id>
# `mana-agent runs ...` is the visible root-help alias for the same commands.
```

Inside chat:

```text
/analyze
/tasks
/models
/route explain
```

## How it works

Each user message is stored as an independent durable chat turn. A completed
task therefore does not close its conversation: later messages are classified
as independent work, a linked follow-up, a safe retry/resume, a status request,
or ordinary conversation. Re-delivery is idempotent only for the same client
message ID; verified task artifacts remain immutable inputs to follow-up work.

```text
Request → Router → Planner → Taskboard → Execution Supervisor → Tools → Verifier → Verified Result
```

Repository changes run through constrained tools, isolated worktrees, permission gates, and verification before being returned as merge candidates.

## Optional capabilities

```bash
python -m pip install -e ".[dashboard]"
python -m pip install -e ".[automations]"
python -m pip install -e ".[observability]"
python -m pip install -e ".[protocols]"
```

Browser support:

```bash
python -m playwright install chromium
```

## Configuration

Run:

```bash
mana-agent --configure
```

Mana-Agent stores configuration under:

```text
~/.mana/config.toml
~/.mana/secrets.toml
```

Credentials remain separate from normal settings.

## Documentation

See the [`docs/`](docs/) directory for installation, architecture, commands,
connectors, [connector health](docs/33-connector-health.md),
[resilient execution](docs/29-resilient-execution.md),
[API Manager](docs/api-manager.md), automation, Teach Mode, media generation,
workers, protocols, and development guides.

## Development

```bash
python -m pip install -e ".[full]"
python -m pip install pytest ruff mypy

pytest -q
ruff check src tests
mypy src tests
```

## License

Released under the [MIT License](LICENSE).
