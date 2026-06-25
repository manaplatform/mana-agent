# Error Handling

`mana-agent` should handle failures in a way that is clear, recoverable, and
traceable for users and developers.

## Principles

- Fail fast when a requested operation cannot continue safely.
- Surface actionable error messages instead of low-level tracebacks when
  possible.
- Preserve enough context for debugging and follow-up investigation.
- Prefer graceful degradation over silent failure.

## Common Error Sources

- missing files or directories,
- invalid CLI arguments or configuration,
- parser and analysis failures,
- repository access or write failures,
- external tool or runtime exceptions.

## Handling Expectations

- Validate inputs before starting expensive work.
- Catch expected exceptions at service boundaries.
- Log or report the operation that failed and the relevant path or command.
- Avoid partial writes unless the workflow explicitly supports them.
- Leave the repository in a consistent state after failure.

## Related Docs

- [Tool System](./08-tool-system.md)
- [Agent Behavior](./09-agent-behavior.md)
- [Logging](./11-logging.md)
