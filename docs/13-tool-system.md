# Tool System

`mana-agent` includes a repository-aware tool system for search, inspection, and
controlled file changes during agent workflows.

## Purpose

The tool system lets the agent gather evidence from the repository before it
acts. It supports codebase search, targeted reads, symbol inspection, file
patching, file writing, and verification steps.

## Evidence Flow

1. Search for relevant code or text.
2. Read the files that contain the evidence.
3. Use symbols or call graphs when structural detail is needed.
4. Apply constrained edits with patch or write tools.
5. Run checks to confirm the change.

## Available Tool Categories

- Search tools: semantic search and text search.
- Inspection tools: file listing, file reads, chunked reads, symbol lookup, and
  call graph inspection.
- Change tools: exact string edits, multi-edit batches, Codex-style patch
  application, guarded whole-file writes, file creation, and file deletion.
- Validation tools: project verification, command execution, and git status or
  diff review.
- Reporting tools: the in-chat `/analyze` slash command runs the existing
  analysis services (dependency graph, project structure, static checks) and
  writes report artifacts under `.mana/` (`analyze.json`, `analyze.md`,
  `analyze.html`, `analyze.dot`, `analyze.graphml`, `diagram.mmd`). It is
  read-only apart from those artifacts and never calls the model. See
  [src/mana_agent/commands/chat_analyze_command.py](../src/mana_agent/commands/chat_analyze_command.py)
  and [src/mana_agent/commands/analyze_formats.py](../src/mana_agent/commands/analyze_formats.py).

## Tool Use Rules

- Prefer search before reading broad files.
- Read files before editing them.
- Keep edits focused and traceable.
- Verify changes when the repository supports it.
- Use repository-local tools only for repository work.

## Related Docs

- [Architecture](./08-architecture.md)
- [Agent Behavior](./09-agent-behavior.md)
- [README](../README.md)
