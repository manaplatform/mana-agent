from __future__ import annotations

import json
from pathlib import Path

from mana_agent.services.dependency_service import DependencyService


def _write_repo(root: Path) -> None:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "package.json").write_text(
        json.dumps(
            {
                "dependencies": {"react": "^18.0.0", "axios": "^1.0.0"},
                "devDependencies": {"typescript": "^5.0.0"},
            }
        ),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        """
[project]
dependencies = ["typer>=0.12", "httpx>=0.27"]
[project.optional-dependencies]
dev = ["pytest>=8.0"]
""".strip(),
        encoding="utf-8",
    )
    (root / "src" / "pkg" / "a.py").write_text(
        """
import typer
import pkg.b
from mana_agent.llm.engine import NativeLLMEngine
""".strip(),
        encoding="utf-8",
    )
    (root / "src" / "pkg" / "b.py").write_text("import json\n", encoding="utf-8")


def _write_polyrepo(root: Path) -> None:
    (root / "app").mkdir(parents=True)
    (root / "backend" / "src").mkdir(parents=True)
    (root / "website").mkdir(parents=True)

    (root / "app" / "pubspec.yaml").write_text(
        """
name: sample_app
dependencies:
  flutter:
    sdk: flutter
  cupertino_icons: ^1.0.0
dev_dependencies:
  flutter_test:
    sdk: flutter
""".strip(),
        encoding="utf-8",
    )

    (root / "backend" / "nest-cli.json").write_text("{}", encoding="utf-8")
    (root / "backend" / "package.json").write_text(
        json.dumps(
            {
                "dependencies": {
                    "@nestjs/common": "^10.0.0",
                    "@nestjs/core": "^10.0.0",
                    "reflect-metadata": "^0.2.0",
                },
                "devDependencies": {"typescript": "^5.0.0"},
            }
        ),
        encoding="utf-8",
    )
    (root / "backend" / "src" / "main.ts").write_text("import { NestFactory } from '@nestjs/core';\n", encoding="utf-8")

    (root / "website" / "package.json").write_text(
        json.dumps(
            {
                "dependencies": {"react": "^18.0.0", "vite": "^5.0.0"},
                "devDependencies": {"typescript": "^5.0.0"},
            }
        ),
        encoding="utf-8",
    )
    (root / "website" / "vite.config.ts").write_text("export default {};\n", encoding="utf-8")


def test_dependency_service_builds_graph_and_detects_stack(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    report = DependencyService().analyze(tmp_path)

    assert "Typer" in report.frameworks
    assert "React" in report.frameworks
    assert "httpx" in report.runtime_dependencies
    assert "pytest" in report.dev_dependencies
    assert any(edge.target.endswith("pkg.b") for edge in report.module_edges)
    assert any(edge.target == "typer" for edge in report.dependency_edges)


def test_dependency_service_detects_nested_polyrepo_frameworks(tmp_path: Path) -> None:
    _write_polyrepo(tmp_path)
    report = DependencyService().analyze(tmp_path)

    assert "Flutter" in report.frameworks
    assert "NestJS" in report.frameworks
    assert "React" in report.frameworks
    assert "Vite" in report.frameworks
    assert "pub" in report.package_managers
    assert "npm" in report.package_managers
    assert any(item.endswith("app/pubspec.yaml") for item in report.manifests)
    assert any(item.endswith("backend/nest-cli.json") for item in report.manifests)


def test_dependency_export_formats(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    report = DependencyService().analyze(tmp_path)

    dot = report.to_dot()
    graphml = report.to_graphml()

    assert "digraph mana_agent" in dot
    assert "<graphml" in graphml
