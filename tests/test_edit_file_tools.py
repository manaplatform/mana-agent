from __future__ import annotations

from pathlib import Path

from mana_agent.tools.edit_file import safe_edit_file, safe_multi_edit_file


def test_edit_file_succeeds_with_one_exact_match(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = safe_edit_file(
        repo_root=tmp_path,
        path="src/app.py",
        old_string="alpha\n",
        new_string="omega\n",
    )

    assert result["ok"] is True
    assert result["files_changed"] == ["src/app.py"]
    assert result["before_sha256"] != result["after_sha256"]
    assert result["changed_ranges"] == [{"start": 1, "end": 1}]
    assert target.read_text(encoding="utf-8") == "omega\nbeta\n"


def test_edit_file_fails_when_old_string_missing(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = safe_edit_file(
        repo_root=tmp_path,
        path="src/app.py",
        old_string="gamma\n",
        new_string="omega\n",
    )

    assert result["ok"] is False
    assert result["error_code"] == "old_string_not_found"
    assert result["nearest_snippets"]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_edit_file_fails_when_old_string_is_ambiguous(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")

    result = safe_edit_file(
        repo_root=tmp_path,
        path="src/app.py",
        old_string="alpha\n",
        new_string="omega\n",
    )

    assert result["ok"] is False
    assert result["error_code"] == "ambiguous_old_string"
    assert result["match_lines"] == [1, 3]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\nalpha\n"


def test_multi_edit_file_applies_sequential_edits_atomically(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = safe_multi_edit_file(
        repo_root=tmp_path,
        path="src/app.py",
        edits=[
            {"old_string": "alpha\n", "new_string": "omega\n"},
            {"old_string": "omega\nbeta\n", "new_string": "omega\ndelta\n"},
        ],
    )

    assert result["ok"] is True
    assert result["files_changed"] == ["src/app.py"]
    assert target.read_text(encoding="utf-8") == "omega\ndelta\n"


def test_multi_edit_file_does_not_write_partial_changes_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = safe_multi_edit_file(
        repo_root=tmp_path,
        path="src/app.py",
        edits=[
            {"old_string": "alpha\n", "new_string": "omega\n"},
            {"old_string": "missing\n", "new_string": "delta\n"},
        ],
    )

    assert result["ok"] is False
    assert result["error_code"] == "old_string_not_found"
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_multi_edit_file_updates_skill_registry_without_line_numbers(tmp_path: Path) -> None:
    target = tmp_path / "src" / "mana_agent" / "skills" / "manager.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'DEFAULT_SKILL_NAMES = ("fastapi", "reactjs")\n'
        '_KEYWORDS: dict[str, tuple[str, ...]] = {\n'
        '    "fastapi": ("fastapi",),\n'
        '    "reactjs": ("react",),\n'
        '}\n',
        encoding="utf-8",
    )

    result = safe_multi_edit_file(
        repo_root=tmp_path,
        path="src/mana_agent/skills/manager.py",
        edits=[
            {
                "old_string": 'DEFAULT_SKILL_NAMES = ("fastapi", "reactjs")',
                "new_string": 'DEFAULT_SKILL_NAMES = ("fastapi", "reactjs", "laravel")',
            },
            {
                "old_string": '    "reactjs": ("react",),\n}',
                "new_string": '    "reactjs": ("react",),\n    "laravel": ("laravel", "php", "artisan"),\n}',
            },
        ],
    )

    assert result["ok"] is True
    content = target.read_text(encoding="utf-8")
    assert '"laravel"' in content
    assert result["files_changed"] == ["src/mana_agent/skills/manager.py"]
