from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from mana_agent.commands.cli import app
from mana_agent.skills.manager import SkillManager, build_default_skill_registry_text, read_skill
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path

runner = CliRunner()


def test_plan_command_passes_explicit_skill_and_export_to_workflow(tmp_path: Path, monkeypatch) -> None:
    from mana_agent.chat_commands.models import CommandResult

    calls = []

    def start(*args, **kwargs):
        calls.append((args, kwargs))
        return CommandResult(status="success", message="# Implementation Plan")

    class Gateway:
        def __init__(self, root, **_kwargs):
            assert root == tmp_path.resolve()
            self.mode_workflow = SimpleNamespace(start=start)

        def create_new_session(self, **_kwargs):
            return "review-session"

    monkeypatch.setattr("mana_agent.gateway.chat_gateway.AgentChatGateway", Gateway)
    result = runner.invoke(app, [
        "plan", "--repo", str(tmp_path), "--no-code", "--skill", "cli", "Add CLI smoke test",
    ])
    assert result.exit_code == 0, result.output
    assert "# Implementation Plan" in result.output
    assert calls[0][0] == ("review-session", "Add CLI smoke test")
    assert calls[0][1]["no_code"]
    assert calls[0][1]["save"]
    assert calls[0][1]["skill_names"] == ("cli",)


def test_skills_init_list_show_uses_root_directory(tmp_path: Path) -> None:
    result_init = runner.invoke(app, ["skills", "init", "--repo", str(tmp_path)])
    assert result_init.exit_code == 0
    assert (tmp_path / "skills" / "cli" / "SKILL.md").exists()
    pdf_skill = tmp_path / "skills" / "pdf-create" / "SKILL.md"
    assert pdf_skill.exists()
    assert "Mana-Agent launch directory" in pdf_skill.read_text(encoding="utf-8")

    custom = "# CLI Skill\n\ncustom root skill\n"
    (tmp_path / "skills" / "cli" / "SKILL.md").write_text(custom, encoding="utf-8")
    result_init_again = runner.invoke(app, ["skills", "init", "--repo", str(tmp_path)])
    assert result_init_again.exit_code == 0
    assert (tmp_path / "skills" / "cli" / "SKILL.md").read_text(encoding="utf-8") == custom

    result_list = runner.invoke(app, ["skills", "list", "--repo", str(tmp_path)])
    assert result_list.exit_code == 0
    assert "Project Root Skills" in result_list.output
    assert "- cli" in result_list.output

    result_show = runner.invoke(app, ["skills", "show", "cli", "--repo", str(tmp_path)])
    assert result_show.exit_code == 0
    assert "custom root skill" in result_show.output


def test_skill_index_reads_metadata_and_read_skill_loads_body_on_demand(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "django"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: django\n"
        "description: Django development skill.\n"
        "trigger: Use when migrations are mentioned.\n"
        "---\n\n"
        "# Django\n\n"
        "FULL BODY CONTENT\n",
        encoding="utf-8",
    )
    manager = SkillManager(project_root=tmp_path)

    item = next(item for item in manager.build_index(include_builtins=False) if item.name == "django")

    assert item.description == "Django development skill."
    assert item.trigger == "Use when migrations are mentioned."
    assert "FULL BODY CONTENT" not in item.description
    assert manager._content_cache == {}  # noqa: SLF001
    assert manager.read_skill("django").endswith("FULL BODY CONTENT\n")
    assert "django" in manager._content_cache  # noqa: SLF001


def test_read_skill_returns_safe_error_for_missing_or_invalid_skill(tmp_path: Path) -> None:
    assert read_skill("../secret", project_root=tmp_path) == "Error: invalid skill name."
    assert read_skill("missing", project_root=tmp_path) == "Error: skill 'missing' was not found."


def test_root_flag_dispatches_plan_mode(tmp_path: Path, monkeypatch) -> None:
    import sys
    from mana_agent.chat_commands.models import CommandResult

    calls = []
    gateway = SimpleNamespace(
        create_new_session=lambda **_kw: "review-session",
        mode_workflow=SimpleNamespace(start=lambda *args, **kw: (
            calls.append(args) or CommandResult(status="success", message="# Implementation Plan")
        )),
    )
    monkeypatch.setattr("mana_agent.gateway.chat_gateway.AgentChatGateway", lambda *a, **kw: gateway)
    cli_sys = SimpleNamespace(**vars(sys))
    cli_sys.stdin = SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr("mana_agent.commands.cli_internal.sys", cli_sys)
    result = runner.invoke(app, ["--plan", "--repo", str(tmp_path)], input="Add CLI banner\n")
    assert result.exit_code == 0, result.output
    assert "Plan Mode" in result.output
    assert calls == [("review-session", "Add CLI banner")]
    assert not (tmp_path / ".git").exists()


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
