# Tool System

Fleet verification is an orchestration service, not a broad remote-command
tool. CLI and chat inputs compile into typed selection and verification
contracts. Provider execution, argv validation, artifact confinement, and
cleanup remain within the existing execution tool boundary.

## Computer tools

When `[computer_control].enabled` is true, the normal model-selected tool loop
loads narrow `computer_*`, `calendar_*`, `media_*`, `notes_*`, desktop
`browser_*`, and `clipboard_*` contracts. No raw computer-command tool exists.
Every call carries the validated source decision ID; the runtime supplies the
authenticated gateway client identity outside model arguments. High/critical
calls require an exact-action token. See
[`22-computer-control.md`](22-computer-control.md) for the full tool, permission,
risk, and provider contract.

`mana-agent` includes a repository-aware tool system for search, inspection, and
controlled file changes during agent workflows.

## Purpose

The tool system lets the agent gather evidence from the repository before it
acts. It supports codebase search, targeted reads, batch reads/searches, symbol
inspection, file patching, file writing, and verification steps.

## Evidence Flow

1. Search for relevant code or text, using `repo_batch_search` for independent queries.
2. Read the files that contain the evidence, using `repo_batch_read` for multiple files.
3. Use symbols or call graphs when structural detail is needed.
4. Apply constrained edits with edit, patch, or `apply_patch_batch` tools.
   `apply_patch` recovers stale context deterministically: re-read → match unique
   anchors → rebuild → bounded retry. Ambiguous locations fail without writing.
5. Run checks to confirm the change, using `run_script_once` for grouped checks.

## Dashboard Chat and Events

The optional Streamlit dashboard uses multipage navigation and canonical
workspace session history under `~/.mana/sessions/<session_id>/`.
Runtime activity is the same normalized `ChatEvent` model used by the CLI/TUI,
published through `ExecutionEventHub` and persisted for durable timeline
recovery. `mana-agent dashboard` starts a loopback API beside Streamlit and
serves the live chat reducer from that same origin. The reducer subscribes
before submission, renders optimistic user messages, updates correlated tool
cards in place, and resumes the WebSocket
(`/api/v1/ws/conversations/{id}`) from its last persisted sequence after
reconnect. Direct `streamlit run` development requires a separately configured
API through `MANA_DASHBOARD_API_BASE`.
Analyze from the dashboard starts `ProjectAnalyzeService` jobs (not a separate
pipeline) and surfaces repository analysis artifacts.

## Available Tool Categories

- Search tools: semantic search, text search, and grouped text search.
- Inspection tools: file listing, file reads, batch file reads, chunked reads,
  symbol lookup, and call graph inspection.
- Change tools: exact string edits, multi-edit batches, Codex-style patch
  application, batch patch application, guarded whole-file writes, file creation,
  and file deletion.
- Validation tools: project verification, single command execution, grouped
  script execution, and git status or diff review.
- Browser tools: model-selected page navigation, DOM and accessibility
  inspection, interaction, screenshots, tabs, uploads, downloads, and isolated
  session cleanup. Sensitive final actions require exact-action confirmation.
- Reporting tools: the in-chat `/analyze` slash command runs the existing
  analysis services (dependency graph, project structure, static checks) and
  writes report artifacts under `.mana/` (`analyze.json`, `analyze.md`,
  `analyze.html`, `analyze.dot`, `analyze.graphml`, `diagram.mmd`). It is
  read-only apart from those artifacts and never calls the model. See
  [src/mana_agent/commands/chat_analyze_command.py](../src/mana_agent/commands/chat_analyze_command.py)
  and [src/mana_agent/commands/analyze_formats.py](../src/mana_agent/commands/analyze_formats.py).

## Tool Use Rules

- Prefer `repo_batch_search` when searching more than one pattern.
- Prefer `repo_batch_read` when reading more than one file.
- Prefer `run_script_once` when several safe commands/checks are needed.
- Prefer `apply_patch_batch` for multiple related patches.
- Prefer search before reading broad files.
- Read files before editing them.
- Keep edits focused and traceable.
- Verify changes when the repository supports it.
- Use repository-local tools only for repository work.

## Related Docs

- [Architecture](./08-architecture.md)
- [Agent Behavior](./09-agent-behavior.md)
- [README](../README.md)
- [Browser Automation](./17-browser-automation.md)
