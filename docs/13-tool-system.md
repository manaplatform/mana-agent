# Tool System

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
5. Run checks to confirm the change, using `run_script_once` for grouped checks.

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
