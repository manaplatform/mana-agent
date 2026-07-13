# Testing

`mana-agent` should be tested at the unit, integration, and smoke-test levels
so repository changes can be validated with confidence.

## Test Layers

- Unit tests for small functions and helpers.
- Integration tests for command flows, services, and repository interactions.
- Smoke checks for CLI commands and end-to-end verification.

## Testing Goals

- catch regressions early,
- verify behavior after edits,
- protect repository-specific workflows,
- document assumptions with repeatable checks.

## Expectations

- Add or update tests when behavior changes.
- Keep tests focused on observable outcomes.
- Prefer deterministic fixtures for repository workflows.
- Run the narrowest useful verification first, then expand if needed.
- Report any skipped checks and why they were skipped.

## Browser Integration

Browser integration coverage runs against a deterministic local HTTP server,
not a public website. Install Chromium to enable it:

```bash
python -m playwright install chromium
python -m pytest tests/integration/test_browser_playwright.py -q
```

The integration module skips cleanly when Playwright or Chromium is absent.

## Related Docs

- [Error Handling](./10-error-handling.md)
- [Logging](./11-logging.md)
- [Tool System](./13-tool-system.md)
