# Commands

## Durable tasks

`mana-agent tasks` reads and controls the durable execution state shared by the
gateway, TUI, API, and dashboard:

The same command group is shown as `mana-agent runs` in root help. `tasks`
remains a supported hidden alias so existing operator instructions and scripts
continue to work without reintroducing the retired `ask` branding substring in
the root command table.

```bash
mana-agent tasks list [--incomplete]
mana-agent tasks status <task-id>
mana-agent tasks tree <task-id>
mana-agent tasks logs <task-id> [--limit 200]
mana-agent tasks artefacts <task-id>
mana-agent tasks cancel <task-id> --reason "operator request" [--attempt-id <attempt-id>]
mana-agent tasks retry <task-id> --decision-json recovery.json
mana-agent tasks resume <task-id> [--decision-json recovery.json]
mana-agent tasks recover
```

Retry and first-time resume require a schema-valid `RecoveryDecision`; the CLI
does not pick a default worker, model, retry category, or workflow. A scheduled
retry can be released after its recorded backoff, and a validated replan can be
released with `resume`, without another decision. Unknown and
non-idempotent ambiguous actions are refused even when retry is requested. See
[Resilient Execution](29-resilient-execution.md).

## Teach Mode

`mana-agent teach` provides `start`, `pause`, `resume`, `status`, `explain`,
`stop`, `cancel`, `review`, `replay`, `edit`, `repair`, `export`, `import`,
`card`, `schedule`, and `doctor`. Drafts require review, dry runs never commit
side effects, and real success requires observable verification. See
[Teach Mode](26-teach-mode.md) for examples and safety behavior.

`teach grant` manages separate, explicit local grants and may open OS-owned
privacy settings. `teach start --desktop` attaches the optional persistent
native recorder only after every required grant and dependency validates.

## Fleet

`mana-agent fleet` lists and manages trusted workers, inspects jobs/logs/
artifacts, compares immutable matrices, and starts explicit cross-platform
verification. Run `mana-agent fleet --help` and
`mana-agent fleet verify --help` for the complete options. Verification requires
at least one explicit `--platform` and `--command`; no platform, provider, or
local fallback is inferred. See [Mana Fleet](24-fleet.md).

## Computer permission and confirmation

When a computer permission is configured as `ask`, the active Textual chat opens
a decision modal and the Dashboard chat timeline displays an actionable card
with deny, once, session, and always choices. A local terminal can make the same
decision explicitly:

```text
/computer-permission list
/computer-permission <permission-request-id> deny|once|session|always
```

An allow choice executes the already stored exact action immediately.

When an enabled computer-control tool proposes a high-risk or critical action,
it stops with a preview and short-lived request ID. A trusted local CLI or
Textual client can inspect and approve it explicitly:

```text
/computer-confirm list
/computer-confirm <confirmation-request-id>
```

Approval remains bound to the exact action and expires quickly. The model and
remote connectors cannot run this command. See
[`22-computer-control.md`](22-computer-control.md).

## Skill proposal workshop

```bash
mana-agent skill proposals [--status STATUS] [--min-confidence SCORE] [--risk LEVEL]
mana-agent skill proposal show <proposal-id>
mana-agent skill proposal review <proposal-id>
mana-agent skill proposal install <proposal-id> [--version VERSION]
mana-agent skill proposal edit <proposal-id> (--draft-file FILE | --skill-file FILE)
mana-agent skill proposal reject <proposal-id> [--reason TEXT]
mana-agent skill proposal quarantine <proposal-id> --reason TEXT
mana-agent skill create-from-session <session-id> [--draft-file FILE]
```

`install` is the explicit approval action. It revalidates and refuses to
overwrite an active skill. `edit` resets the proposal to `needs_attention` and
reruns validation. `reject` retains metadata to prevent immediate regeneration;
`quarantine` moves the proposal outside every active or pending loader.

This page documents the commands exposed by `mana-agent` and the `/analyze` slash command used inside chat. The console script is defined in `pyproject.toml` and points to `mana_agent.commands.cli:app`. [pyproject.toml:1-52](../pyproject.toml#L1-L52)

The Typer app is created in `src/mana_agent/commands/cli_internal.py`, where the top-level CLI registers `continue`; the interactive chat command is registered in `src/mana_agent/commands/chat_cli.py`. [src/mana_agent/commands/cli_internal.py:68-69](../src/mana_agent/commands/cli_internal.py#L68-L69) [src/mana_agent/commands/cli_internal.py:191-262](../src/mana_agent/commands/cli_internal.py#L191-L262) [src/mana_agent/commands/chat_cli.py:1-1](../src/mana_agent/commands/chat_cli.py#L1-L1) [src/mana_agent/commands/chat_cli.py:196-196](../src/mana_agent/commands/chat_cli.py#L196-L196)

## Commands found in the project

From the CLI implementation, the commands available to users are:

- `mana-agent chat`
- `mana-agent continue`
- `mana-agent worktree` (managed agent Git worktrees for isolated coding)
- `mana-agent codex` (optional Codex backend status and authentication)

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

### `mana-agent codex`

```bash
mana-agent codex status --repo .
mana-agent codex doctor --repo .
```

`status` and `doctor` perform read-only executable, version, enablement, and
repository checks. Mana-Agent supplies its selected provider credential only to
the isolated Codex child process; this command group does not mutate the user's
normal Codex authentication. See
[`20-codex-integration.md`](20-codex-integration.md).

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
