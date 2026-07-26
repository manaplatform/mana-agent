# Contributing

Thanks for your interest in contributing to `mana-agent`.

## How to contribute

1. **Check existing issues**: look for open bugs and feature requests.
2. **Create a plan**: describe what you want to change and why.
3. **Follow the code style**: keep changes focused and readable.
4. **Add tests**: include tests for both safe and unsafe cases when relevant.

## Security-related changes

If your contribution involves security-sensitive areas (authentication/authorization, secrets handling, input validation, or safe logging):

- review the project `SECURITY.md` policy
- include tests that cover safe and unsafe cases
- avoid leaking secrets in logs, error messages, or prompts

## Development workflow (high level)

- Run the test suite and ensure it passes.
- Prefer small, well-scoped pull requests.
- New pull requests load `.github/pull_request_template.md`. The `pr-autofill` workflow then replaces empty/template bodies with commits, changed files, inferred change types, and related issue refs. Edit the result and complete testing/checklist items yourself.
- Update `CHANGELOG.md` for user-visible or repository behavior changes.
- For versioned releases, see [docs/14-release.md](docs/14-release.md). The stable GitHub Release tag is derived from the application version by `.github/workflows/release.yml`; its version-tag trigger must match.

## Pull request tips

- Keep the PR focused on one concern when practical.
- Include the commands you ran under **Testing and verification**.
- Call out breaking changes explicitly; write `None` when there are none.
- Link related issues with `Fixes #…` / `Closes #…` when appropriate.
- Do not introduce keyword-routing or fallback behavior that bypasses model decisions.

## License

By contributing, you agree that your contributions will be licensed under the project’s existing license.
