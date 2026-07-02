from __future__ import annotations

import hashlib
from pathlib import Path

from mana_agent.tools.write_file import (
    build_create_file_tool,
    build_write_file_tool,
    safe_create_file,
    safe_finalize_file_parts,
    safe_write_file,
    safe_write_file_part,
)


def test_safe_write_file_part_then_finalize(tmp_path: Path) -> None:
    part1 = safe_write_file_part(repo_root=tmp_path, path="src/big.txt", content="hello ", part_index=1)
    part2 = safe_write_file_part(repo_root=tmp_path, path="src/big.txt", content="world", part_index=2)

    assert part1["ok"] is True
    assert part2["ok"] is True

    finalize = safe_finalize_file_parts(repo_root=tmp_path, path="src/big.txt")
    assert finalize["ok"] is True
    assert finalize["files_changed"] == ["src/big.txt"]
    assert (tmp_path / "src" / "big.txt").read_text(encoding="utf-8") == "hello world"
    assert not (tmp_path / "src" / ".big.txt.parts").exists()


def test_safe_finalize_file_parts_requires_parts(tmp_path: Path) -> None:
    result = safe_finalize_file_parts(repo_root=tmp_path, path="src/missing.txt")
    assert result["ok"] is False
    assert "no parts directory found" in result["error"]


def test_write_file_tool_chunk_then_finalize(tmp_path: Path) -> None:
    tool = build_write_file_tool(repo_root=tmp_path, allowed_prefixes=None)

    r1 = tool.invoke({"path": "docs/out.md", "content": "A", "part_index": 1})
    r2 = tool.invoke({"path": "docs/out.md", "content": "B", "part_index": 2})
    r3 = tool.invoke({"path": "docs/out.md", "finalize": True})

    assert r1["ok"] is True
    assert r2["ok"] is True
    assert r3["ok"] is True
    assert (tmp_path / "docs" / "out.md").read_text(encoding="utf-8") == "AB"


def test_safe_create_file_refuses_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "note.md"
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")

    result = safe_create_file(repo_root=tmp_path, path="docs/note.md", content="new\n")

    assert result["ok"] is False
    assert "already exists" in result["error"]
    assert target.read_text(encoding="utf-8") == "old\n"


def test_create_file_tool_creates_missing_parent_dirs(tmp_path: Path) -> None:
    tool = build_create_file_tool(repo_root=tmp_path, allowed_prefixes=None)

    result = tool.invoke({"path": "docs/new/note.md", "content": "# Note\n"})

    assert result["ok"] is True
    assert result["files_changed"] == ["docs/new/note.md"]
    assert (tmp_path / "docs" / "new" / "note.md").read_text(encoding="utf-8") == "# Note\n"


def test_safe_write_file_requires_hash_or_force_for_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "note.md"
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")

    blocked = safe_write_file(repo_root=tmp_path, path="docs/note.md", content="new\n")
    wrong_hash = safe_write_file(
        repo_root=tmp_path,
        path="docs/note.md",
        content="new\n",
        expected_sha256="bad",
    )
    current_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    accepted = safe_write_file(
        repo_root=tmp_path,
        path="docs/note.md",
        content="new\n",
        expected_sha256=current_hash,
    )

    assert blocked["ok"] is False
    assert wrong_hash["ok"] is False
    assert accepted["ok"] is True
    assert target.read_text(encoding="utf-8") == "new\n"


def test_safe_finalize_file_parts_requires_hash_or_force_for_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "out.md"
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")
    safe_write_file_part(repo_root=tmp_path, path="docs/out.md", content="new\n", part_index=1)

    blocked = safe_finalize_file_parts(repo_root=tmp_path, path="docs/out.md", cleanup_parts=False)
    wrong_hash = safe_finalize_file_parts(
        repo_root=tmp_path,
        path="docs/out.md",
        cleanup_parts=False,
        expected_sha256="bad",
    )
    current_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    accepted = safe_finalize_file_parts(
        repo_root=tmp_path,
        path="docs/out.md",
        expected_sha256=current_hash,
    )

    assert blocked["ok"] is False
    assert wrong_hash["ok"] is False
    assert accepted["ok"] is True
    assert target.read_text(encoding="utf-8") == "new\n"
