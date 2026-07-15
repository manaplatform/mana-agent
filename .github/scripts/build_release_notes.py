#!/usr/bin/env python3
"""Build a polished GitHub Release body for mana-agent version tags.

Reads repository context (tag, previous tag, generated notes, optional CHANGELOG
excerpt) and writes Markdown suitable for GitHub Releases. Intended for use by
``.github/workflows/release.yml``; does not publish anything itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


SECTION_ORDER = (
    ("Breaking Changes", "⚠️ Breaking Changes"),
    ("New Features", "✨ New Features"),
    ("Improvements", "🚀 Improvements"),
    ("Bug Fixes", "🐛 Bug Fixes"),
    ("Documentation", "📚 Documentation"),
    ("CI / Tooling / Release", "🧰 CI / Tooling / Release"),
    ("Dependencies", "📦 Dependencies"),
    ("Other Changes", "🧾 Other Changes"),
)

HEADING_ALIASES = {
    "breaking change": "Breaking Changes",
    "breaking changes": "Breaking Changes",
    "what's new": "New Features",
    "new feature": "New Features",
    "new features": "New Features",
    "features": "New Features",
    "enhancements": "Improvements",
    "improvements": "Improvements",
    "performance": "Improvements",
    "bug fix": "Bug Fixes",
    "bug fixes": "Bug Fixes",
    "fixes": "Bug Fixes",
    "documentation": "Documentation",
    "docs": "Documentation",
    "ci": "CI / Tooling / Release",
    "tooling": "CI / Tooling / Release",
    "maintenance": "CI / Tooling / Release",
    "dependencies": "Dependencies",
    "other changes": "Other Changes",
    "other": "Other Changes",
}


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def normalize_version(tag: str) -> str:
    tag = tag.strip()
    if tag.startswith("v") and re.match(r"^v\d", tag):
        return tag[1:]
    return tag


def previous_tag(current_tag: str) -> str | None:
    """Return the nearest previous version tag, excluding the current tag."""
    tags = [
        line.strip()
        for line in run_git("tag", "--list", "v*", "--sort=-v:refname").splitlines()
        if line.strip()
    ]
    if current_tag in tags:
        idx = tags.index(current_tag)
        if idx + 1 < len(tags):
            return tags[idx + 1]
    # Fallback: describe parent
    desc = run_git("describe", "--tags", "--abbrev=0", "--match", "v*", f"{current_tag}^")
    return desc or None


def github_api_json(url: str, token: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "mana-agent-release-notes",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {exc.code} for {url}: {body}") from exc


def generate_notes(
    repo: str,
    tag: str,
    previous: str | None,
    token: str,
    target_commitish: str | None = None,
) -> str:
    payload: dict[str, str] = {"tag_name": tag}
    if previous:
        payload["previous_tag_name"] = previous
    if target_commitish:
        payload["target_commitish"] = target_commitish
    result = github_api_json(
        f"https://api.github.com/repos/{repo}/releases/generate-notes",
        token=token,
        payload=payload,
    )
    return str(result.get("body") or "").strip()


def parse_generated_sections(notes: str) -> dict[str, list[str]]:
    """Map GitHub-generated notes into canonical sections."""
    sections: dict[str, list[str]] = {name: [] for name, _ in SECTION_ORDER}
    current = "Other Changes"
    for raw_line in notes.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        heading_match = re.match(r"^#{1,3}\s+(.+?)\s*$", stripped)
        if heading_match:
            title = heading_match.group(1).strip()
            # Drop leading emoji noise for matching
            title_key = re.sub(r"^[^\w]+", "", title).strip().lower()
            current = HEADING_ALIASES.get(title_key, title if title in sections else "Other Changes")
            if current not in sections:
                sections[current] = []
            continue
        if stripped.startswith(("- ", "* ")):
            item = stripped[2:].strip()
            if item:
                sections.setdefault(current, []).append(item)
            continue
        # Full changelog footer lines from GitHub (e.g. **Full Changelog**: ...)
        if stripped.lower().startswith("**full changelog**"):
            continue
        if stripped.startswith("**") and ":" in stripped:
            continue
    return sections


def extract_changelog_highlights(changelog_path: Path, limit: int = 8) -> list[str]:
    """Pull recent dated bullets from CHANGELOG.md when present."""
    if not changelog_path.is_file():
        return []
    text = changelog_path.read_text(encoding="utf-8")
    bullets: list[str] = []
    in_entry = False
    for line in text.splitlines():
        if re.match(r"^##\s+\d{4}-\d{2}-\d{2}\s*$", line.strip()):
            if bullets:
                break
            in_entry = True
            continue
        if in_entry and line.startswith("## "):
            break
        if in_entry and line.startswith("- "):
            item = line[2:].strip()
            # Prefer top-level bullets only
            if item and not item.lower().startswith("verification:"):
                bullets.append(item)
            if len(bullets) >= limit:
                break
    return bullets


def short_summary(sections: dict[str, list[str]], changelog_highlights: list[str]) -> str:
    for key in ("New Features", "Improvements", "Bug Fixes", "Breaking Changes", "Other Changes"):
        items = sections.get(key) or []
        if items:
            first = items[0]
            # Strip trailing " by @user in #n" style tails for a cleaner blurb
            first = re.sub(r"\s+by\s+@[\w-]+\s+in\s+https?://\S+$", "", first)
            first = re.sub(r"\s+in\s+https?://\S+$", "", first)
            return first.rstrip(".")
    if changelog_highlights:
        return changelog_highlights[0].rstrip(".")
    return "Maintenance and quality updates for mana-agent."


def collect_contributors(notes: str, sections: dict[str, list[str]]) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"@([A-Za-z0-9-]+)", notes):
        user = match.group(1)
        if user.lower().endswith("[bot]") or user.lower() in {"github-actions", "dependabot"}:
            continue
        if user not in found:
            found.append(user)
    # Also scan section items
    for items in sections.values():
        for item in items:
            for match in re.finditer(r"@([A-Za-z0-9-]+)", item):
                user = match.group(1)
                if user.lower().endswith("[bot]"):
                    continue
                if user not in found:
                    found.append(user)
    return found


def build_body(
    *,
    tag: str,
    version: str,
    previous: str | None,
    repo: str,
    sections: dict[str, list[str]],
    changelog_highlights: list[str],
    contributors: list[str],
    generated_raw: str,
) -> str:
    repo_url = f"https://github.com/{repo}"
    compare_url = (
        f"{repo_url}/compare/{previous}...{tag}" if previous else f"{repo_url}/commits/{tag}"
    )
    summary = short_summary(sections, changelog_highlights)

    lines: list[str] = [
        f"# 🚀 mana-agent `{tag}`",
        "",
        f"**{summary}**",
        "",
        "---",
        "",
        "## ✨ Highlights",
        "",
    ]

    highlight_items = changelog_highlights[:5]
    if not highlight_items:
        for key in ("New Features", "Improvements", "Bug Fixes", "Breaking Changes"):
            for item in sections.get(key, [])[:2]:
                cleaned = re.sub(r"\s+by\s+@[\w-]+\s+in\s+\S+$", "", item)
                highlight_items.append(cleaned)
            if len(highlight_items) >= 5:
                break
    if highlight_items:
        for item in highlight_items[:5]:
            lines.append(f"- {item}")
    else:
        lines.append("- See **What's Changed** below for commits and pull requests in this release.")
    lines.append("")

    # Canonical categorized sections
    for key, heading in SECTION_ORDER:
        items = sections.get(key) or []
        if not items:
            continue
        lines.extend([f"## {heading}", ""])
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    # If GitHub returned unstructured content only, include a compact fallback
    if not any(sections.get(k) for k, _ in SECTION_ORDER) and generated_raw:
        lines.extend(
            [
                "## 🧾 What's Changed",
                "",
                generated_raw,
                "",
            ]
        )

    lines.extend(
        [
            "## 📦 Installation / Upgrade",
            "",
            "### From source (recommended)",
            "",
            "```bash",
            f"pipx install --force git+{repo_url}.git@{tag}",
            "# or",
            f'python -m pip install --upgrade "git+{repo_url}.git@{tag}"',
            "```",
            "",
            "### Editable local install",
            "",
            "```bash",
            f"git clone --branch {tag} {repo_url}.git",
            "cd mana-agent",
            "python -m pip install -e \".[dev]\"",
            "```",
            "",
            "### Standalone binaries",
            "",
            "Download platform binaries and checksums attached to this release:",
            "",
            "| Platform | Asset |",
            "| --- | --- |",
            "| Linux x64 | `mana-agent-linux-x64` |",
            "| macOS Intel | `mana-agent-macos-x64` |",
            "| macOS Apple Silicon | `mana-agent-macos-arm64` |",
            "| Windows x64 | `mana-agent-windows-x64.exe` |",
            "",
            "Verify checksums with the matching `*.sha256` files before running binaries.",
            "",
            "Python package artifacts (`*.whl`, `*.tar.gz`) are also attached when the release workflow completes successfully.",
            "",
            "## 📚 Documentation",
            "",
            f"- [README]({repo_url}/blob/{tag}/README.md)",
            f"- [CHANGELOG]({repo_url}/blob/{tag}/CHANGELOG.md)",
            f"- [Installation guide]({repo_url}/blob/{tag}/docs/02-installation.md)",
            f"- [Release process]({repo_url}/blob/{tag}/docs/14-release.md)",
            f"- [Contributing]({repo_url}/blob/{tag}/CONTRIBUTING.md)",
            "",
        ]
    )

    if contributors:
        lines.extend(["## 🙌 Contributors", ""])
        for user in contributors:
            lines.append(f"- [@{user}](https://github.com/{user})")
        lines.append("")

    lines.extend(
        [
            "## 🔗 Full changelog",
            "",
            f"- Compare: [`{previous or 'initial'}...{tag}`]({compare_url})",
            f"- Tag: [`{tag}`]({repo_url}/releases/tag/{tag})",
            f"- Package version: `{version}`",
            "",
            "---",
            "",
            f"*Automated release for **mana-agent** `{tag}`. Built by the repository release workflow from tags, commits, and pull requests.*",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def release_exists(repo: str, tag: str, token: str) -> bool:
    try:
        github_api_json(
            f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
            token=token,
        )
        return True
    except RuntimeError as exc:
        if " 404 " in str(exc) or ": 404" in str(exc) or "Not Found" in str(exc):
            return False
        # Treat other errors as unknown; let publish step decide
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Git tag name, e.g. v0.0.15")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "",
    )
    parser.add_argument(
        "--output",
        default="release-body.md",
        help="Path to write the release body Markdown",
    )
    parser.add_argument(
        "--meta-output",
        default="release-meta.env",
        help="Path to write KEY=value metadata for GitHub Actions",
    )
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to CHANGELOG.md for highlight extraction",
    )
    parser.add_argument(
        "--target-commitish",
        default=os.environ.get("GITHUB_SHA", ""),
        help="Commit SHA associated with the tag (optional)",
    )
    args = parser.parse_args(argv)

    if not args.repo:
        print("error: --repo or GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2
    if not args.token:
        print("error: --token, GITHUB_TOKEN, or GH_TOKEN is required", file=sys.stderr)
        return 2

    tag = args.tag.strip()
    version = normalize_version(tag)
    previous = previous_tag(tag)
    target = args.target_commitish.strip() or None

    generated = generate_notes(
        repo=args.repo,
        tag=tag,
        previous=previous,
        token=args.token,
        target_commitish=target,
    )
    sections = parse_generated_sections(generated)
    changelog_highlights = extract_changelog_highlights(Path(args.changelog))
    contributors = collect_contributors(generated, sections)
    body = build_body(
        tag=tag,
        version=version,
        previous=previous,
        repo=args.repo,
        sections=sections,
        changelog_highlights=changelog_highlights,
        contributors=contributors,
        generated_raw=generated,
    )

    output_path = Path(args.output)
    output_path.write_text(body, encoding="utf-8")

    exists = release_exists(args.repo, tag, args.token)
    meta_path = Path(args.meta_output)
    meta_lines = [
        f"RELEASE_TAG={tag}",
        f"RELEASE_VERSION={version}",
        f"PREVIOUS_TAG={previous or ''}",
        f"RELEASE_EXISTS={'true' if exists else 'false'}",
        f"RELEASE_BODY_PATH={output_path.as_posix()}",
        f"RELEASE_NAME=mana-agent {tag}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Wrote release body to {output_path} ({len(body)} bytes)")
    print(f"Previous tag: {previous or '(none)'}")
    print(f"Existing release: {exists}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
