# Testing

The deterministic Fleet smoke suite is `python -m pytest tests/fleet -q`. It
uses fake/local providers and temporary Git repositories; it requires no public
network, cloud account, or remote host. Normal CI runs the project on Ubuntu
with Python 3.10, 3.12, and 3.14, plus Windows and macOS with Python 3.12.

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

## Coding workspace fixtures

Coding backend tests should import `WorkspaceContext` from
`mana_agent.coding.models` and provide both `repository_path` and
`worktree_path`. Tests that intentionally use the same temporary directory for
both paths must opt into `sandbox="readOnly"` (or explicitly authorize the
in-place write workflow); do not rely on a nonexistent `mana_agent.coding.workspace`
module or omit the repository path. This preserves the workspace isolation
contract while keeping fixtures aligned with the production model.

## Async tests and dispatcher path handling

The approval/TUI regression tests keep async coverage independent of an optional
pytest async plugin: async application methods are invoked from ordinary test
functions with `anyio.run` or `asyncio.run`. Do not mark a test with
`pytest.mark.asyncio` unless `pytest-asyncio` is declared and installed for the
test environment. The `ToolExecutionDispatcher` also uses the explicit
`FilePath` alias for path construction so a function-local `Path` import cannot
shadow the filesystem path type and trigger `UnboundLocalError` during initialization.

**Do not reintroduce:** keep dispatcher path references unshadowable and keep
async tests compatible with the dependencies declared by the project.

The live-steering TUI integration tests previously used `pytest.mark.asyncio`
without declaring the plugin in the `dev` extra installed by CI. A clean CI
environment therefore reported an unknown marker and could not execute the two
coroutines. The tests now use synchronous pytest entry points that run their
unchanged async scenarios with `asyncio.run`; no runtime or dependency change
is required. Do not reintroduce async test markers without installing their
plugin in the test environment. User-owned verification (not run by the agent):
`python -m pytest tests/execution/test_live_steering_integration.py -q`.

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
