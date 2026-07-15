# Release

Release work in this repository should be handled with the same evidence-first
approach used during development: inspect the code, verify the affected paths,
and confirm the deliverables before publishing or tagging.

## Versioning

- Package version lives in `pyproject.toml` under `[project].version`.
- Runtime version resolution uses `mana_agent._version.get_version()` (pyproject first, then installed metadata).
- Stable Git tags use the form `vMAJOR.MINOR.PATCH` (for example `v0.0.15`).
- Keep the README version badge in sync when cutting a documented release.

## GitHub automation

Configuration lives under `.github/`:

| Path | Purpose |
| --- | --- |
| `.github/pull_request_template.md` | Default pull request form for new PRs |
| `.github/release.yml` | Categories for GitHub-generated release notes |
| `.github/scripts/build_release_notes.py` | Builds polished release Markdown from tags, PRs, and CHANGELOG highlights |
| `.github/workflows/ci.yml` | Continuous integration tests |
| `.github/workflows/release.yml` | Build package + binaries; publish GitHub Releases |

### Stable releases (`v*.*.*` tags)

1. Update `pyproject.toml` version and user-facing docs as needed.
2. Record notable changes in `CHANGELOG.md`.
3. Merge to `main` after review and CI.
4. Create and push an annotated tag: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.
5. The `release` workflow builds wheels, sdists, platform binaries, checksums, then publishes a GitHub Release using the standardized body from `build_release_notes.py`.
6. Re-running the workflow for the same tag **updates** that release (assets and notes) instead of creating a duplicate.

Release notes are **not** static placeholders. They combine:

- GitHub “generate release notes” content (commits / PRs between tags)
- Categories from `.github/release.yml` when PR labels match
- Recent top-level bullets from `CHANGELOG.md` as highlights when available
- Install / upgrade instructions, documentation links, contributor list, and compare links

### Development channel (`latest-dev`)

Pushes to `main` (and manual `workflow_dispatch`) move the `latest-dev` tag and overwrite the `latest-dev` prerelease assets. Prefer tagged `v*.*.*` releases for production.

### Permissions

- Build jobs use read-only `contents` access.
- Only the publish job requests `contents: write` (tags, releases, asset upload).

## Release Checklist

- Review the code and docs that changed.
- Confirm package metadata, commands, and workflows still align.
- Run the appropriate test or smoke-check subset.
- Update `CHANGELOG.md` and version metadata.
- Verify documentation links and numbering remain consistent.
- Check `git status` or `git diff` to ensure only intended files changed.
- Tag and push only after checks pass.

## What to Check Before Releasing

- `src/mana_agent/` for functional changes.
- `tests/` for updated coverage or fixtures.
- `docs/` for any user-facing behavior changes.
- Packaging and metadata files when versioning or distribution behavior changes.
- `.github/workflows/release.yml` if release packaging or notes behavior must change.

## Recommended Release Flow

1. Inspect the feature or fix in source control.
2. Run focused verification for the affected area.
3. Update changelog, version, and docs if needed.
4. Merge via pull request (PR template under `.github/pull_request_template.md`).
5. Tag `vX.Y.Z` and let GitHub Actions publish the release.
6. Confirm the GitHub Release body, assets, and checksums look correct.

## Related Docs

- [Development](./15-development.md)
- [Testing](./12-testing.md)
- [Installation](./02-installation.md)
- [Contributing](../CONTRIBUTING.md)
