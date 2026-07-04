from __future__ import annotations

from pathlib import Path

from mana_agent.tools.repository import apply_patch_batch, list_files, repo_batch_read, repo_batch_search, run_script_once


def test_list_files_flat_and_recursive_globs_include_existing_files(tmp_path: Path) -> None:
    skill_dir = tmp_path / "src" / "mana_agent" / "default_skills"
    nested = skill_dir / "nested"
    nested.mkdir(parents=True)
    (skill_dir / "cli.md").write_text("# CLI\n", encoding="utf-8")
    (skill_dir / "reactjs.md").write_text("# ReactJS\n", encoding="utf-8")
    (nested / "extra.md").write_text("# Extra\n", encoding="utf-8")

    flat = list_files(tmp_path, glob="src/mana_agent/default_skills/*.md")["files"]
    recursive_star = list_files(tmp_path, glob="src/mana_agent/default_skills/**/*")["files"]
    recursive_dir = list_files(tmp_path, glob="src/mana_agent/default_skills/**")["files"]

    assert flat == [
        "src/mana_agent/default_skills/cli.md",
        "src/mana_agent/default_skills/reactjs.md",
    ]
    assert "src/mana_agent/default_skills/cli.md" in recursive_star
    assert "src/mana_agent/default_skills/nested/extra.md" in recursive_star
    assert recursive_star == recursive_dir


def test_repo_batch_read_returns_multiple_files_and_errors(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("beta\n", encoding="utf-8")

    result = repo_batch_read(tmp_path, files=["src/a.py", "src/b.py", "../escape.py", "missing.py"])

    assert result["ok"] is False
    ok_files = [item for item in result["files"] if item["ok"]]
    assert [item["path"] for item in ok_files] == ["src/a.py", "src/b.py"]
    assert ok_files[0]["content"] == "alpha\n"
    assert {item["error"] for item in result["errors"]} == {"path_outside_repo", "file_not_found"}


def test_repo_batch_search_groups_results_per_query(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("class Package:\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("SkillIndexItem = object\n", encoding="utf-8")

    result = repo_batch_search(
        tmp_path,
        patterns=[
            {"query": "class Package", "glob": "**/*.py", "regex": False, "limit": 10},
            {"query": "SkillIndexItem", "glob": "**/*.py", "regex": False, "limit": 10},
        ],
    )

    assert result["ok"] is True
    assert len(result["results"]) == 2
    assert result["results"][0]["matches"][0]["file"] == "src/a.py"
    assert result["results"][1]["matches"][0]["file"] == "src/b.py"


def test_run_script_once_returns_exit_code_output_and_duration(tmp_path: Path) -> None:
    result = run_script_once(tmp_path, script="printf 'hello'")

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["stdout"] == "hello"
    assert result["duration_ms"] >= 0


def test_apply_patch_batch_validates_and_applies_multiple_patches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("old a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("old b\n", encoding="utf-8")

    result = apply_patch_batch(
        tmp_path,
        patches=[
            {"path": "a.txt", "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-old a\n+new a\n*** End Patch"},
            {"path": "b.txt", "patch": "*** Begin Patch\n*** Update File: b.txt\n@@\n-old b\n+new b\n*** End Patch"},
        ],
    )

    assert result["ok"] is True
    assert result["changed_files"] == ["a.txt", "b.txt"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "new a\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "new b\n"
