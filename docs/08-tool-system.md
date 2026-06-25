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
- Change tools: JSON patch application and atomic file writes.
- Validation tools: project verification, command execution, and git status or
  diff review.

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
