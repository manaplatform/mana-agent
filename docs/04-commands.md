# Commands

This page documents the commands exposed by `mana-agent` and the `/analyze` slash command used inside chat. The console script is defined in `pyproject.toml` and points to `mana_agent.commands.cli:app`. [pyproject.toml:1-52](../pyproject.toml#L1-L52)

The Typer app is created in `src/mana_agent/commands/cli_internal.py`, where the top-level CLI registers `continue`; the interactive chat command is registered in `src/mana_agent/commands/chat_cli.py`. [src/mana_agent/commands/cli_internal.py:68-69](../src/mana_agent/commands/cli_internal.py#L68-L69) [src/mana_agent/commands/cli_internal.py:191-262](../src/mana_agent/commands/cli_internal.py#L191-L262) [src/mana_agent/commands/chat_cli.py:1-1](../src/mana_agent/commands/chat_cli.py#L1-L1) [src/mana_agent/commands/chat_cli.py:196-196](../src/mana_agent/commands/chat_cli.py#L196-L196)

## Commands found in the project

From the CLI implementation, the commands available to users are:

- `mana-agent chat`
- `mana-agent continue`
- `mana-agent worktree` (managed agent Git worktrees for isolated coding)

The README’s CLI section only highlights `chat`, but the code shows `continue` is also a first-class command. [README.md:1-337](../README.md#L1-L337) [src/mana_agent/commands/cli_internal.py:191-262](../src/mana_agent/commands/cli_internal.py#L191-L262) [src/mana_agent/commands/chat_cli.py:196-196](../src/mana_agent/commands/chat_cli.py#L196-L196)

## Command reference

### `mana-agent chat`

`chat` starts the interactive assistant. It can work with an index, build an ephemeral index, use directory-aware mode, enable agent tools, enable the coding agent, persist coding memory, and run in full-auto or planning modes. The command also supports diagram rendering and JSON output. [src/mana_agent/commands/chat_cli.py:196-357](../src/mana_agent/commands/chat_cli.py#L196-L357)

Example:

```bash
mana-agent chat --root-dir /path/to/project
mana-agent chat --root-dir . --planning-mode --coding-memory
mana-agent chat --dir-mode --auto-index-missing
```

Notable options implemented in the command signature include:

- `--model`
- `--index-dir`
- `--k`
- `--ephemeral-index`
- `--dir-mode`
- `--root-dir`
- `--max-indexes`
- `--auto-index-missing/--no-auto-index-missing`
- `--agent-tools/--no-agent-tools`
- `--coding-agent/--no-coding-agent`
- `--tool-worker-process/--no-tool-worker-process`
- `--tool-worker-strict/--no-tool-worker-strict`
- `--tool-exec-backend`
- `--redis-url`
- `--toolsmanager-parallel-requests`
- `--redis-queue-name`
- `--redis-ttl-seconds`
- `--coding-memory/--no-coding-memory`
- `--flow-id`
- `--coding-plan-max-steps`
- `--coding-search-budget`
- `--coding-read-budget`
- `--coding-require-read-files`
- `--planning-mode`
- `--planning-max-questions`
- `--auto-execute-plan/--no-auto-execute-plan`
- `--auto-execute-max-passes`
- `--auto-continue/--no-auto-continue`
- `--execution-profile`
- `--full-auto`
- `--full-auto-status-every`
- `--agent-max-steps`
- `--agent-unlimited/--no-agent-unlimited`
- `--agent-timeout-seconds`
- `--multiline-input/--no-multiline-input`
- `--multiline-terminator`
- `--diagram-render-images/--no-diagram-render-images`
- `--diagram-output-dir`
- `--diagram-format`
- `--diagram-open/--no-diagram-open`
- `--diagram-timeout-seconds`
- `--json` [src/mana_agent/commands/chat_cli.py:196-357](../src/mana_agent/commands/chat_cli.py#L196-L357)

The chat implementation uses a read-only answer path when coding-agent features are not enabled, and a coding-agent path when edits are allowed. It also supports direct command fast paths, exact search, planning questions, and the `/analyze` slash command. [src/mana_agent/commands/chat_cli.py:1-2579](../src/mana_agent/commands/chat_cli.py#L1-L2579)

### `mana-agent continue`

`continue` resumes a saved auto-execute run from `.mana/runs/<run_id>`. It requires `--run-id` and can be constrained with pass, tool-call, runtime, cost, and progress caps. [src/mana_agent/commands/cli_internal.py:191-262](../src/mana_agent/commands/cli_internal.py#L191-L262)

### `mana-agent worktree`

Manage isolated Git worktrees used by coding agents. Worktrees are stored under
`~/.mana/repositories/<repository-id>/worktrees/` with metadata in
`~/.mana/repositories/<repository-id>/managed_worktrees/`.

```bash
mana-agent worktree list --root-dir .
mana-agent worktree create <task-id> --root-dir . --title "Fix auth"
mana-agent worktree status <task-id> --root-dir .
mana-agent worktree resume <task-id> --root-dir .
mana-agent worktree diff <task-id> --root-dir .
mana-agent worktree merge <task-id> --root-dir . --yes
mana-agent worktree remove <task-id> --root-dir .
mana-agent worktree remove <task-id> --root-dir . --force --yes
mana-agent worktree reconcile --root-dir .
```

| Command | Behavior |
| --- | --- |
| `list` | Task ID, branch, status, worktree path, assigned agent, dirty state |
| `create` | Deterministic worktree + `mana/<task-slug>` branch for a task |
| `status` | Repository identity, base revision, HEAD, Git state, recovery notes |
| `resume` | Reconnect interrupted task workspaces when safe |
| `diff` | Diff against the recorded task base revision |
| `merge` | Merge into the source checkout only with `--yes`; never force-push |
| `remove` | Refuses dirty/unmerged cleanup unless `--force --yes` |
| `reconcile` | Match metadata to `git worktree list --porcelain` (never auto-deletes user worktrees) |

Implementation: [src/mana_agent/commands/worktree_cli.py](../src/mana_agent/commands/worktree_cli.py), [src/mana_agent/multi_agent/worktrees/](../src/mana_agent/multi_agent/worktrees/).

Example:

```bash
mana-agent continue --run-id my-run --root-dir /path/to/project
```

Options implemented by the command:

- `--run-id`
- `--root-dir`
- `--pass-cap`
- `--auto-continue/--no-auto-continue`
- `--max-passes`
- `--max-tool-calls/--max-total-tool-calls`
- `--max-runtime-minutes`
- `--max-cost`
- `--max-no-progress-passes`
- `--timeout`
- `--k`
- `--max-steps`
- `--max-resume-cycles` [src/mana_agent/commands/cli_internal.py:191-262](../src/mana_agent/commands/cli_internal.py#L191-L262)

## In-chat `/analyze`

Inside `mana-agent chat`, the `/analyze` slash command analyzes the current project and writes report artifacts under `.mana/`. The slash command is detected before normal chat routing and is implemented in `src/mana_agent/commands/chat_analyze_command.py`. [src/mana_agent/commands/chat_analyze_command.py:1-84](../src/mana_agent/commands/chat_analyze_command.py#L1-L84) [src/mana_agent/commands/chat_cli.py:1430-1470](../src/mana_agent/commands/chat_cli.py#L1430-L1470)

The supported analyze artifact formats are defined in `src/mana_agent/commands/analyze_formats.py`:

- `json` → `.mana/analyze.json`
- `markdown` / `md` → `.mana/analyze.md`
- `html` → `.mana/analyze.html`
- `dot` → `.mana/analyze.dot`
- `graphml` → `.mana/analyze.graphml`
- `mermaid` → `.mana/diagram.mmd`
- `all` → every artifact above [src/mana_agent/commands/analyze_formats.py:1-174](../src/mana_agent/commands/analyze_formats.py#L1-L174)

Direct forms accepted by the parser include:

```text
/analyze all
/analyze json
/analyze markdown
/analyze md
/analyze html
/analyze dot
/analyze graphml
/analyze mermaid
/analyze json markdown html
/analyze --format json,markdown,html
```

If no format is supplied, the slash command opens the numbered menu. The menu offers JSON, Markdown, HTML, DOT graph, GraphML, Mermaid diagram, and an all-formats option. [src/mana_agent/commands/chat_analyze_command.py:32-42](../src/mana_agent/commands/chat_analyze_command.py#L32-L42) [src/mana_agent/commands/analyze_formats.py:49-63](../src/mana_agent/commands/analyze_formats.py#L49-L63)

## Quick diff against the markdown commands list

The previous markdown already mentioned the following commands or slash command:

- `chat`
- `/analyze`

The code-based command inventory adds the missing top-level CLI commands:

- `continue`

So the command list in this file is now aligned with the implementation. [src/mana_agent/commands/cli_internal.py:148-191](../src/mana_agent/commands/cli_internal.py#L148-L191) [src/mana_agent/commands/chat_cli.py:196-196](../src/mana_agent/commands/chat_cli.py#L196-L196)
