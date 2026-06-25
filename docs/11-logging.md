# Logging

`mana-agent` should use logging to make repository activity, decisions, and
failures observable without overwhelming the user.

## Goals

- explain what the tool or command is doing,
- record useful context for debugging,
- keep output concise in normal workflows,
- avoid exposing unnecessary internal noise.

## What to Log

- command start and completion,
- repository discovery and analysis steps,
- file reads, edits, and verification actions,
- recoverable warnings and failures,
- key paths, commands, and identifiers involved in a task.

## Logging Expectations

- Use consistent message wording for similar operations.
- Include enough context to correlate an error with the action that caused it.
- Prefer structured or semistructured messages where practical.
- Keep user-facing output distinct from diagnostic logging when possible.
- Pair logs with verification results after edits.

## Related Docs

- [Error Handling](./10-error-handling.md)
- [Tool System](./08-tool-system.md)
- [Agent Behavior](./09-agent-behavior.md)
