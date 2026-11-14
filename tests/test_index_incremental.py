from __future__ import annotations

import json
from pathlib import Path

from mana_agent.analysis.chunker import CodeChunker
from mana_agent.analysis.models import CodeChunk
from mana_agent.parsers.multi_parser import MultiLanguageParser
from mana_agent.services.index_service import IndexService


def test_incremental_index_updates_only_changed_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    file_a = project / "a.py"
    file_b = project / "b.py"
    file_a.write_text('"""A"""\n\ndef one():\n    """one"""\n    return 1\n', encoding="utf-8")
    file_b.write_text('"""B"""\n\ndef two():\n    """two"""\n    return 2\n', encoding="utf-8")

    index_dir = tmp_path / "index"
    service = IndexService(MultiLanguageParser(), CodeChunker())

    first = service.index(project, index_dir, rebuild=False)
    assert first["indexed_files"] == 2

    file_a.write_text('"""A"""\n\ndef one():\n    """changed"""\n    return 10\n', encoding="utf-8")
    second = service.index(project, index_dir, rebuild=False)

    assert second["indexed_files"] == 1
    assert second["removed_chunks"] >= 1


def test_incremental_index_includes_non_python_source_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    py_file = project / "a.py"
    ts_file = project / "b.ts"
    dart_file = project / "c.dart"
    py_file.write_text("def a():\n    return 1\n", encoding="utf-8")
    ts_file.write_text("export function b() { return 2; }\n", encoding="utf-8")
    dart_file.write_text("String c() { return '3'; }\n", encoding="utf-8")

    index_dir = tmp_path / "index"
    service = IndexService(MultiLanguageParser(), CodeChunker())

    result = service.index(project, index_dir, rebuild=False)
    assert result["indexed_files"] == 3


def test_incremental_index_rebuilds_chunks_when_chunk_schema_changes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    service = IndexService(MultiLanguageParser(), CodeChunker())
    service.index(project, index_dir, rebuild=False)

    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunk_schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = service.index(project, index_dir, rebuild=False)

    assert result["indexed_files"] == 1
    assert result["removed_chunks"] >= 1
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["chunk_schema_version"] == 2
