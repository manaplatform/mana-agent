#!/usr/bin/env python3
"""Validate that a release tag is safe to publish to PyPI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib

from packaging.version import InvalidVersion, Version


def read_project_metadata(pyproject_path: Path) -> tuple[str, str]:
    """Return the non-empty project name and canonical release version."""
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Could not read package metadata from {pyproject_path}: {exc}") from exc

    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml is missing the [project] table")

    name = project.get("name")
    version_text = project.get("version")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("pyproject.toml [project].name is missing or empty")
    if not isinstance(version_text, str) or not version_text.strip():
        raise ValueError("pyproject.toml [project].version is missing or empty")

    version_text = version_text.strip()
    try:
        version = Version(version_text)
    except InvalidVersion as exc:
        raise ValueError(f"Package version {version_text!r} is not PEP 440 compliant") from exc

    if str(version) != version_text:
        raise ValueError(
            f"Package version {version_text!r} is not canonical; use {str(version)!r}"
        )
    if version.is_devrelease or version.local is not None:
        raise ValueError(
            f"Package version {version_text!r} is unsuitable for a production PyPI release "
            "because development and local versions are not allowed"
        )

    return name.strip(), version_text


def validate_tag_name(tag: str) -> None:
    """Require a non-empty GitHub release tag with the standard v prefix."""
    tag = tag.strip()
    if not tag:
        raise ValueError("Release tag is missing or empty")
    if not tag.startswith("v"):
        raise ValueError(f"Release tag {tag!r} must start with 'v'")


def validate_tag(tag: str, package_version: str) -> None:
    """Require the tag, with at most one leading v removed, to match exactly."""
    validate_tag_name(tag)
    tag = tag.strip()
    tag_version = tag[1:] if tag.startswith("v") else tag
    if not tag_version:
        raise ValueError(f"Release tag {tag!r} does not contain a version")
    if tag_version != package_version:
        raise ValueError(
            f"Release tag {tag!r} resolves to version {tag_version!r}, but "
            f"pyproject.toml defines {package_version!r}"
        )


def ensure_version_is_unpublished(package_name: str, package_version: str) -> None:
    """Fail closed unless PyPI confirms that this immutable version is absent."""
    url = (
        "https://pypi.org/pypi/"
        f"{quote(package_name, safe='')}/{quote(package_version, safe='')}/json"
    )
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "mana-agent-release-validation"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return
        raise ValueError(f"PyPI availability check failed with HTTP {exc.code}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"PyPI availability check failed: {exc}") from exc

    published_version = payload.get("info", {}).get("version") if isinstance(payload, dict) else None
    raise ValueError(
        f"PyPI already contains immutable version {published_version or package_version!r} "
        f"for {package_name!r}; increment the project version before releasing"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="GitHub Release tag to validate")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument(
        "--allow-tag-version-mismatch",
        action="store_true",
        help="allow a GitHub release tag that differs from the package version",
    )
    parser.add_argument(
        "--check-pypi",
        action="store_true",
        help="fail if the package version already exists on production PyPI",
    )
    args = parser.parse_args()

    try:
        name, version = read_project_metadata(args.pyproject)
        if args.allow_tag_version_mismatch:
            validate_tag_name(args.tag)
        else:
            validate_tag(args.tag, version)
        if args.check_pypi:
            ensure_version_is_unpublished(name, version)
    except ValueError as exc:
        print(f"Release version validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Validated release tag {args.tag!r} for {name} {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
