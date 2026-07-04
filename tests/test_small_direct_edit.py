from __future__ import annotations

from pathlib import Path

from mana_agent.llm.small_direct_edit import (
    classify_edit_intent,
    handle_small_direct_edit,
    resolve_explicit_path,
)


def test_direct_readme_version_update_uses_single_patch(monkeypatch, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Mana Agent\n\nCurrent documented version: **v0.0.7**.\n\nMore docs.\n",
        encoding="utf-8",
    )
    patch_calls: list[str] = []

    from mana_agent.llm import small_direct_edit

    real_apply = small_direct_edit.apply_patch_tool.safe_apply_patch

    def _spy_apply_patch(*, repo_root: Path, patch: str):
        patch_calls.append(patch)
        return real_apply(repo_root=repo_root, patch=patch)

    monkeypatch.setattr(small_direct_edit.apply_patch_tool, "safe_apply_patch", _spy_apply_patch)

    result = handle_small_direct_edit(tmp_path, "update version in readme.md to 0.0.8")

    assert result.handled is True
    assert result.ok is True
    assert patch_calls == [
        "*** Begin Patch\n"
        "*** Update File: README.md\n"
        "@@\n"
        "-Current documented version: **v0.0.7**.\n"
        "+Current documented version: **v0.0.8**.\n"
        "*** End Patch\n"
    ]
    assert [row["tool_name"] for row in result.trace] == ["read_file", "apply_patch", "read_file"]
    assert result.trace[0]["path"] == "README.md"
    assert all(row["tool_name"] not in {"repo_search", "list_files", "semantic_search", "verify_project", "tool_worker"} for row in result.trace)
    assert readme.read_text(encoding="utf-8").splitlines()[2] == "Current documented version: **v0.0.8**."
    assert "Verification skipped: docs-only one-line edit" in result.answer


def test_readme_case_duplicate_guard_reads_one_canonical_path(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Current documented version: **v0.0.7**.\n", encoding="utf-8")

    assert resolve_explicit_path(tmp_path, "readme.md") == "README.md"
    intent = classify_edit_intent(tmp_path, "update version in readme.md to 0.0.8")

    assert intent.kind == "small_direct_edit"
    assert intent.explicit_path == "README.md"
    assert intent.docs_only is True
    assert intent.requires_verification is False


def test_docs_only_verification_policy_is_reported(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Current documented version: **0.0.7**.\n", encoding="utf-8")

    result = handle_small_direct_edit(tmp_path, "set README.md version to 0.0.8")

    assert result.ok is True
    assert "Verification skipped: docs-only one-line edit" in result.answer
    assert "Verification: passed" not in result.answer
    assert all(row["tool_name"] != "verify_project" for row in result.trace)


def test_non_doc_code_edit_uses_normal_routing(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("__version__ = '0.0.7'\n", encoding="utf-8")

    intent = classify_edit_intent(tmp_path, "update version in src/app.py to 0.0.8")
    result = handle_small_direct_edit(tmp_path, "update version in src/app.py to 0.0.8")

    assert intent.kind == "small_direct_edit"
    assert intent.docs_only is False
    assert intent.requires_verification is True
    assert result.handled is False
