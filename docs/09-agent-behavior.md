# Agent Behavior

This document describes the expected behavior of the coding agent used by
`mana-agent`.

## Behavior Principles

The agent should:

- understand the user request and current context,
- gather repository evidence before concluding,
- prefer direct citations from repository files,
- make concrete changes only when evidence supports them,
- run checks after edits when possible,
- report what changed and what was not verified.

## Typical Workflow

1. Clarify the task.
2. Search the repository for relevant code or docs.
3. Read the source files that support the answer or change.
4. Edit only the necessary files.
5. Verify the change with tests or smoke checks.
6. Summarize the result with file citations.

## Reporting Expectations

When finishing a task, the agent should report:

- changed files,
- key checks run,
- any skipped checks,
- remaining risks or unknowns.

## Related Docs

- [Architecture](./08-architecture.md)
- [Tool System](./08-tool-system.md)
- [README](../README.md)
