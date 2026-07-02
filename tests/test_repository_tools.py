from __future__ import annotations

from pathlib import Path

from mana_agent.tools.repository import list_files


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
