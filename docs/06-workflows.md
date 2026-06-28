# Workflows

`mana-agent` supports a few common repository-assistant workflows: initial analysis, targeted questions, interactive chat, and resuming a saved automation run. The command docs describe the available entry points and the outputs they produce. [docs/04-commands.md:1-171](docs/04-commands.md:1-171) [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119)

## 1. Open an interactive chat session

Use `chat` when you want a conversational workspace for repository exploration or coding tasks:

```bash
mana-agent chat --root-dir /path/to/project
```

The chat command supports coding-agent workflows, tool-worker execution, coding memory, planning mode, and auto-execute flows. It is the most flexible option for iterative problem solving. [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119) [docs/04-commands.md:1-171](docs/04-commands.md:1-171)

Typical chat use cases include:

- exploring unfamiliar modules
- tracing a bug through multiple files
- planning a change before editing code
- iterating on a task with memory of prior context

The documentation notes that coding memory may be stored at `<project>/.mana/index/chat_memory.sqlite3`. [docs/04-commands.md:1-171](docs/04-commands.md:1-171)

While inside a chat session you can generate analysis artifacts without leaving the REPL using the `/analyze` slash command:

```text
/analyze            # opens the format menu (1-7)
/analyze all        # generate every artifact
/analyze json markdown html
/analyze --format json,markdown,html
```

This writes the selected reports under the project `.mana/` directory (for example `.mana/analyze.json`, `.mana/analyze.md`, `.mana/diagram.mmd`) and runs before the message reaches the model, so it is fast and read-only apart from those artifacts. [docs/04-commands.md:1-171](docs/04-commands.md:1-171)

## 2. Resume an automated run

Use `continue` to pick up a persisted auto-execute session:

```bash
mana-agent continue --root-dir /path/to/project --run-id <run_id>
```

This command resumes a run from `<root>/.mana/runs/<run_id>` and accepts limits for passes, tool calls, runtime, cost, and resume cycles. [docs/04-commands.md:1-171](docs/04-commands.md:1-171)

This workflow is useful when an earlier automated run stopped before finishing and you want to continue it without restarting from scratch.

## Suggested sequence

1. Configure the environment and verify the CLI.
2. Run `mana-agent analyze` to build repository context.
3. Use `mana-agent ask` for precise questions.
4. Switch to `mana-agent chat` for iterative analysis or editing.
5. Use `mana-agent continue` when resuming a previous auto-execute run.

## Operational tips

- Use `--json` when you want machine-readable output. [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119) [docs/04-commands.md:1-171](docs/04-commands.md:1-171)
- Keep project-generated artifacts under `.mana/` so they stay tied to the analyzed repository. [docs/03-quick-start.md:1-119](docs/03-quick-start.md:1-119)
- Prefer `analyze` before `ask` when you want better context for a complex question.
