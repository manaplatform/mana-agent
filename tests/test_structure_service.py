from __future__ import annotations

import json
from pathlib import Path

from mana_agent.services.structure_service import StructureService


def _write_sample_project(root: Path) -> None:
    (root / "src" / "sample").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "frontend").mkdir(parents=True)
    (root / "app").mkdir(parents=True)
    (root / "cmd" / "spiderly").mkdir(parents=True)

    (root / "pyproject.toml").write_text(
        """
[project]
name = "sample"
version = "0.1.0"
dependencies = ["typer>=0.12", "httpx>=0.27", "numpy>=1.24"]
[project.optional-dependencies]
dev = ["pytest>=8.3"]
[project.scripts]
sample = "sample.cli:app"
""".strip(),
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "ci.yml").write_text("name: ci\n", encoding="utf-8")
    (root / "src" / "sample" / "__init__.py").write_text('__all__ = ["Foo"]\n', encoding="utf-8")
    (root / "src" / "sample" / "cli.py").write_text(
        """
import typer
from dataclasses import dataclass

app = typer.Typer()
CONST_VALUE = 1

@dataclass
class Foo:
    x: int

    def method(self) -> int:
        return self.x

@app.command()
def run() -> None:
    pass
""".strip(),
        encoding="utf-8",
    )
    (root / "frontend" / "main.ts").write_text("export function boot() {}\n", encoding="utf-8")
    (root / "app" / "pubspec.yaml").write_text("name: demo\ndependencies:\n  flutter:\n    sdk: flutter\n", encoding="utf-8")
    (root / "cmd" / "spiderly" / "main.go").write_text(
        """
package main

import "github.com/spf13/cobra"

type Chunk struct {
    URL string
}

func BuildRoot() *cobra.Command {
    return &cobra.Command{Use: "spiderly run <url>"}
}

func (c *Chunk) Count() int {
    return 1
}
""".strip(),
        encoding="utf-8",
    )
    (root / "tests" / "test_one.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")


def test_structure_service_detects_dependencies_commands_and_exports(tmp_path: Path) -> None:
    _write_sample_project(tmp_path)
    service = StructureService(include_tests=False)
    report = service.analyze_project(tmp_path)
    payload = report.to_dict()

    assert "Typer" in payload["frameworks"]
    assert any("pytest" in item for item in payload["dependencies_dev"])
    assert "run" in payload["commands"]
    assert "spiderly run <url>" in payload["commands"]
    assert any(item["mechanism"] == "__all__" and item["symbol"] == "Foo" for item in payload["exports"])
    assert any(item["symbol"] == "BuildRoot" and item["mechanism"] == "public-function" for item in payload["exports"])
    assert any(item["name"] == "Foo" and "x" in item["fields"] for item in payload["data_structures"])
    assert any(item["name"] == "Chunk" and "URL" in item["fields"] for item in payload["data_structures"])
    assert "python" in payload["language_counts"]
    assert "typescript" in payload["language_counts"]
    assert any(item.endswith("src") for item in payload["directories"])
    assert any(item["root_path"] == "app" for item in payload["subprojects"])


def test_structure_service_output_formats(tmp_path: Path) -> None:
    _write_sample_project(tmp_path)
    service = StructureService(include_tests=True)
    report = service.analyze_project(tmp_path)

    as_json = json.dumps(report.to_dict())
    as_markdown = service.render_markdown(report)

    assert '"modules"' in as_json
    assert '"directories"' in as_json
    assert "## Modules" in as_markdown
    assert "## Directory Tree" in as_markdown
    assert "## APIs and Exports" in as_markdown
    assert "Summary: detected" in as_markdown
    assert "test_one.py" in as_markdown
