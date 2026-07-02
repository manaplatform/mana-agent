from pathlib import Path

from typer.testing import CliRunner

from mana_agent.commands.cli import app
from mana_agent.skills.manager import SkillManager, build_default_skill_registry_text

runner = CliRunner()


def test_plan_mode_loads_matching_skills_and_saves(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--repo",
            str(tmp_path),
            "--no-code",
            "Add CLI smoke test",
        ],
    )

    assert result.exit_code == 0
    assert "Loaded skills: cli, testing" in result.output
    assert "# Implementation Plan" in result.output
    assert (tmp_path / ".mana" / "plans" / "add-cli-smoke-test.md").exists()


def test_skills_init_list_show_uses_root_directory(tmp_path: Path) -> None:
    result_init = runner.invoke(app, ["skills", "init", "--repo", str(tmp_path)])
    assert result_init.exit_code == 0
    assert (tmp_path / "skills" / "cli.md").exists()

    custom = "# CLI Skill\n\ncustom root skill\n"
    (tmp_path / "skills" / "cli.md").write_text(custom, encoding="utf-8")
    result_init_again = runner.invoke(app, ["skills", "init", "--repo", str(tmp_path)])
    assert result_init_again.exit_code == 0
    assert (tmp_path / "skills" / "cli.md").read_text(encoding="utf-8") == custom

    result_list = runner.invoke(app, ["skills", "list", "--repo", str(tmp_path)])
    assert result_list.exit_code == 0
    assert "Project Root Skills" in result_list.output
    assert "- cli" in result_list.output

    result_show = runner.invoke(app, ["skills", "show", "cli", "--repo", str(tmp_path)])
    assert result_show.exit_code == 0
    assert "custom root skill" in result_show.output


def test_root_flag_dispatches_plan_mode(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--plan", "--repo", str(tmp_path)],
        input="Add CLI banner\n",
    )

    assert result.exit_code == 0
    assert "Plan Mode" in result.output
    assert "Loaded skills: cli" in result.output


def test_new_framework_built_in_skills_are_registered(tmp_path: Path) -> None:
    manager = SkillManager(project_root=tmp_path)

    listed = manager.list_by_source()["Built-in Skills"]
    for name in ("nestjs", "nextjs", "reactjs", "fastapi", "laravel"):
        assert name in listed
        skill = manager.get(name)
        assert skill is not None
        assert skill.source == "built-in"
        assert "## When to use" in skill.content
        assert "## Rules" in skill.content
        assert "## Verification" in skill.content


def test_default_skill_registry_builder_inserts_by_markers() -> None:
    source = '''from __future__ import annotations

DEFAULT_SKILL_NAMES = (
    "vue",
    "telegram-bot",
)

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "vue": ("vue",),
    "telegram-bot": ("telegram",),
}
'''

    updated = build_default_skill_registry_text(source, ["reactjs", "nextjs"])

    assert '    "nextjs",\n    "reactjs",\n    "telegram-bot",' in updated
    assert '    "nextjs": ("next", "nextjs", "next.js", "ssr", "app router", "routing", "server components"),' in updated
    assert '    "reactjs": ("react", "reactjs", "jsx", "tsx", "component", "hooks"),' in updated
