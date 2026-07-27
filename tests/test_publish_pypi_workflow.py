"""Safety and structure checks for the production PyPI workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_PATH = _ROOT / ".github" / "workflows" / "publish-pypi.yml"
_RELEASE_WORKFLOW_PATH = _ROOT / ".github" / "workflows" / "release.yml"
_VALIDATOR_PATH = _ROOT / ".github" / "scripts" / "validate_release_version.py"


def _workflow_text() -> str:
    return _WORKFLOW_PATH.read_text(encoding="utf-8")


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_release_version", _VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_job_is_unreachable_from_push_pull_request_and_dispatch() -> None:
    workflow = _workflow_text()
    assert "types: [published]" in workflow
    assert "push:" not in workflow
    assert "pull_request:" not in workflow
    assert "if: github.event_name == 'release' && github.event.action == 'published'" in workflow
    assert "PYPI_API_TOKEN" not in workflow
    assert "password:" not in workflow
    assert "skip-existing" not in workflow


def test_publish_job_uses_verified_artifact_oidc_and_production_guardrails() -> None:
    workflow = _workflow_text()
    assert "needs: validate-build" in workflow
    assert workflow.count("python -m build") == 1
    assert "name: mana-agent-pypi-distributions" in workflow
    assert "environment:\n      name: pypi" in workflow
    assert "id-token: write" in workflow
    assert "contents: read" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "pypa/gh-action-pypi-publish@" in workflow
    assert "packages-dir: dist/" in workflow


def test_stable_github_release_keeps_its_trigger_tag_and_uses_the_application_version_as_title() -> None:
    workflow = _RELEASE_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "name: Resolve stable release metadata" in workflow
    assert 'release_tag = os.environ["GITHUB_REF_NAME"]' not in workflow
    assert 'trigger_tag = os.environ["GITHUB_REF_NAME"]' in workflow
    assert 'env_file.write(f"RELEASE_TAG={trigger_tag}\\n")' in workflow
    assert 'env_file.write(f"RELEASE_TITLE=v{version}\\n")' in workflow
    assert "tag_name: ${{ env.RELEASE_TAG }}" in workflow
    assert "name: ${{ env.RELEASE_TITLE }}" in workflow
    assert '--tag "${RELEASE_TAG}"' in workflow
    assert '--display-tag "${RELEASE_TITLE}"' in workflow


def test_pypi_release_validation_allows_the_explicit_release_tag_mismatch() -> None:
    workflow = _workflow_text()

    assert "--allow-tag-version-mismatch --check-pypi" in workflow


def test_release_version_validator_requires_exact_canonical_version(tmp_path: Path) -> None:
    validator = _load_validator()
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "mana-agent"\nversion = "0.0.15"\n', encoding="utf-8")

    name, version = validator.read_project_metadata(pyproject)
    assert (name, version) == ("mana-agent", "0.0.15")
    validator.validate_tag("v0.0.15", version)

    with pytest.raises(ValueError, match="defines '0.0.15'"):
        validator.validate_tag("v0.0.14", version)

    validator.validate_tag_name("v2026.07.26")
    with pytest.raises(ValueError, match="must start with 'v'"):
        validator.validate_tag_name("2026.07.26")


@pytest.mark.parametrize("version", ["", "not-a-version", "0.0.15.dev1", "0.0.15+local"])
def test_release_version_validator_rejects_unsuitable_versions(tmp_path: Path, version: str) -> None:
    validator = _load_validator()
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'[project]\nname = "mana-agent"\nversion = "{version}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validator.read_project_metadata(pyproject)
