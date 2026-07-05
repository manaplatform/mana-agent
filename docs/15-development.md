# Development

This repository is structured around a small Python package with a doc-driven
workflow. Development should follow the repository's evidence-first tool and
verification flow so changes stay grounded in the codebase.

## Development Workflow

1. Inspect the repository before editing.
2. Read the relevant source files and docs for the area you are changing.
3. Make focused mutations with `create_file`, `write_file`, `apply_patch`, or `delete_file`.
4. Verify the result with the narrowest useful checks.
5. Report the files that changed and any checks that were skipped.

## Source Areas to Check

When working in this project, start with the code that defines the feature:

- `src/mana_agent/multi_agent/runtime/` for agent orchestration and tool handling.
- `src/mana_agent/tools/` for mutation and contract behavior.
- `src/mana_agent/services/` for service-layer logic.
- `src/mana_agent/parsers/` and `src/mana_agent/renderers/` for input/output processing.
- `tests/` for behavior coverage and fixtures.

## Practical Rules

- Prefer repository-local evidence over assumptions.
- Use search tools before broad reads.
- Keep changes traceable to the source code and existing docs.
- If a change affects documented behavior, update the doc set alongside code.
- Confirm any file-creation task uses the requested path exactly; this project
  includes logic to detect misplaced deliverables and treat them as failures.

## Verification Notes

- Run project checks after edits when possible.
- For doc-only changes, a light verification pass is usually sufficient.
- For code changes, prefer targeted tests that exercise the modified area.

## Related Docs

- [Testing](./12-testing.md)
- [Tool System](./13-tool-system.md)
- [Release](./14-release.md)
