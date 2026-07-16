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
| `.github/pull_request_template.md` | Fallback PR form (static; GitHub cannot auto-fill it alone) |
| `.github/workflows/pr-autofill.yml` | On PR open, fills description from commits and changed files |
| `.github/scripts/fill_pr_body.py` | Builds the auto-filled PR body |
| `.github/release.yml` | Categories for GitHub-generated release notes |
| `.github/scripts/build_release_notes.py` | Builds polished release Markdown from tags, PRs, and CHANGELOG highlights |
| `.github/workflows/ci.yml` | Continuous integration tests |
| `.github/workflows/release.yml` | Build package + binaries; publish GitHub Releases |
| `.github/workflows/publish-pypi.yml` | Validate, build, and publish releases to PyPI with OIDC |

### Stable releases (`v*.*.*` tags)

1. Update `pyproject.toml` version and user-facing docs as needed.
2. Record notable changes in `CHANGELOG.md`.
3. Merge to `main` after review and CI.
4. Create and push an annotated tag: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.
5. The `release` workflow builds wheels, sdists, platform binaries, checksums, then publishes a GitHub Release using the standardized body from `build_release_notes.py`.
6. Publishing that GitHub Release triggers `publish-pypi.yml`. It independently runs the complete test suite, verifies that the tag and package version match, checks that the version is not already on PyPI, builds and validates one wheel and one source distribution, and uploads those exact files to PyPI.
7. Re-running the GitHub release workflow for the same tag **updates** that release (assets and notes) instead of creating a duplicate. PyPI is different: published versions are immutable and cannot be overwritten.

Release notes are **not** static placeholders. They combine:

- GitHub “generate release notes” content (commits / PRs between tags)
- Categories from `.github/release.yml` when PR labels match
- Recent top-level bullets from `CHANGELOG.md` as highlights when available
- Install / upgrade instructions, documentation links, contributor list, and compare links

### Development channel (`latest-dev`)

Pushes to `main` (and manual `workflow_dispatch`) move the `latest-dev` tag and overwrite the `latest-dev` prerelease assets. Prefer tagged `v*.*.*` releases for production.

### One-time PyPI Trusted Publisher setup

Mana-Agent publishes without an API token or repository secret. A PyPI project owner must configure a Trusted Publisher for the `mana-agent` project using these exact values:

| PyPI field | Value |
| --- | --- |
| GitHub owner | `manadevelopment23` |
| Repository | `mana-agent` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` |

In PyPI, open the `mana-agent` project publishing settings, add a GitHub Actions Trusted Publisher, and enter the values above. In GitHub, create the `pypi` environment; environment protection rules and required reviewers are recommended for an additional deployment approval gate. Do not add `PYPI_API_TOKEN` or any PyPI password to GitHub.

### Publishing a version

1. Set `[project].version` in `pyproject.toml` to the intended version, for example `0.0.11`, and update release notes and documentation.
2. Merge the release-ready commit to `main` after CI passes.
3. Create and push a matching tag, for example: `git tag -a v0.0.11 -m "v0.0.11" && git push origin v0.0.11`.
4. The existing release workflow creates the GitHub Release. Once that release is published, the PyPI workflow validates and publishes the package.
5. Confirm that both `mana-agent-0.0.11.tar.gz` and the matching wheel are present on PyPI.

After removing one leading `v`, the GitHub Release tag must exactly match `[project].version`; otherwise publication stops. PyPI versions are immutable and cannot be overwritten or replaced, so increment the package version before every new publication.

Manual runs of `publish-pypi.yml` require an existing tag and may validate or rebuild artifacts only. The production publish job is restricted to the `release.published` event, so `workflow_dispatch`, pushes, and pull requests cannot publish to PyPI.

### Permissions

- Build jobs use read-only `contents` access.
- Only the publish job requests `contents: write` (tags, releases, asset upload).
- The PyPI deployment job has only `contents: read` and `id-token: write`; the latter is the short-lived OIDC credential used by Trusted Publishing.

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
