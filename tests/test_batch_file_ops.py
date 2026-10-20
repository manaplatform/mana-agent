from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mana_agent.tools.batch_file_ops import (
    BatchFileSystemManager,
    batch_apply_patches,
    batch_delete_files,
    batch_list_files,
    batch_move_files,
    batch_read_files,
    batch_search_files,
    batch_write_files,
)


def test_batch_read_files_success_and_slicing(tmp_path: Path) -> None:
    f1 = tmp_path / "file1.txt"
    f1.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
    f2 = tmp_path / "file2.txt"
    f2.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    res = batch_read_files(
        tmp_path,
        files=[
            {"path": "file1.txt", "start_line": 2, "end_line": 4},
            "file2.txt",
        ],
    )
    assert res["ok"] is True
    assert res["total_files"] == 2
    assert res["successful_count"] == 2

    by_path = {f["path"]: f for f in res["files"]}
    assert "file1.txt" in by_path
    assert by_path["file1.txt"]["content"] == "line2\nline3\nline4\n"
    assert by_path["file1.txt"]["start_line"] == 2
    assert by_path["file1.txt"]["end_line"] == 4
    assert len(by_path["file1.txt"]["sha256"]) == 64

    assert "file2.txt" in by_path
    assert by_path["file2.txt"]["content"] == "alpha\nbeta\ngamma\n"
    assert by_path["file2.txt"]["total_lines"] == 3


def test_batch_read_files_bounds_and_binary_exclusion(tmp_path: Path) -> None:
    text_file = tmp_path / "code.py"
    text_file.write_text("x = 100\n", encoding="utf-8")

    binary_file = tmp_path / "data.bin"
    binary_file.write_bytes(b"some\x00binary\x00data")

    res = batch_read_files(
        tmp_path,
        files=["code.py", "data.bin", "nonexistent.py"],
    )
    assert res["ok"] is False  # Has errors for binary and nonexistent
    by_path = {f["path"]: f for f in res["files"]}

    assert by_path["code.py"]["ok"] is True
    assert by_path["data.bin"]["ok"] is False
    assert by_path["data.bin"]["error_code"] == "BINARY_FILE"
    assert by_path["nonexistent.py"]["ok"] is False
    assert by_path["nonexistent.py"]["error_code"] == "FILE_NOT_FOUND"


def test_batch_write_files_staged_atomic_success(tmp_path: Path) -> None:
    writes = [
        {"path": "src/pkg/module1.py", "content": "def func1(): return 1\n"},
        {"path": "src/pkg/module2.py", "content": "def func2(): return 2\n"},
    ]
    res = batch_write_files(tmp_path, writes=writes)
    assert res["ok"] is True
    assert sorted(res["changed_files"]) == ["src/pkg/module1.py", "src/pkg/module2.py"]
    assert (tmp_path / "src/pkg/module1.py").read_text(encoding="utf-8") == "def func1(): return 1\n"
    assert (tmp_path / "src/pkg/module2.py").read_text(encoding="utf-8") == "def func2(): return 2\n"


def test_batch_write_files_optimistic_conflict_detection(tmp_path: Path) -> None:
    f = tmp_path / "config.json"
    f.write_text('{"version": 1}\n', encoding="utf-8")

    # Attempt write with wrong expected_sha256
    writes = [
        {"path": "config.json", "content": '{"version": 2}\n', "expected_sha256": "0000000000000000000000000000000000000000000000000000000000000000"},
    ]
    res = batch_write_files(tmp_path, writes=writes)
    assert res["ok"] is False
    assert res["error_code"] == "WRITE_CONFLICT"
    assert len(res["conflicts"]) == 1
    # File content must remain unchanged
    assert f.read_text(encoding="utf-8") == '{"version": 1}\n'


def test_batch_write_files_transactional_rollback_on_failure(tmp_path: Path) -> None:
    f1 = tmp_path / "existing.txt"
    f1.write_text("original content\n", encoding="utf-8")

    # Batch with 1 valid write and 1 invalid write (directory traversal escape)
    writes = [
        {"path": "existing.txt", "content": "mutated content\n"},
        {"path": "../escape.txt", "content": "malicious write"},
    ]
    res = batch_write_files(tmp_path, writes=writes, rollback_on_failure=True)
    assert res["ok"] is False
    assert res["error_code"] == "PATH_TRAVERSAL"
    # Pre-validation aborted batch before mutation; existing.txt is untouched
    assert f1.read_text(encoding="utf-8") == "original content\n"


def test_batch_search_files_concurrency(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("class Authenticator:\n    def login(self): pass\n", encoding="utf-8")
    (tmp_path / "src" / "db.py").write_text("class Database:\n    def connect(self): pass\n", encoding="utf-8")

    patterns = [
        {"query": "Authenticator", "glob": "**/*.py"},
        {"query": "Database", "glob": "**/*.py"},
        {"query": "def login", "glob": "src/auth.py"},
    ]
    res = batch_search_files(tmp_path, patterns=patterns, concurrency=3)
    assert res["ok"] is True
    assert res["total_queries"] == 3
    assert len(res["results"]) == 3

    auth_res = res["results"][0]
    assert auth_res["query"] == "Authenticator"
    assert len(auth_res["matches"]) >= 1
    assert auth_res["matches"][0]["file"] == "src/auth.py"


def test_batch_list_files_single_traversal(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("print(2)", encoding="utf-8")
    (tmp_path / "tests" / "test_main.py").write_text("pass", encoding="utf-8")

    res = batch_list_files(tmp_path, queries=["src/*.py", "tests/*.py"])
    assert res["ok"] is True
    assert res["total_patterns"] == 2

    src_spec = next(item for item in res["results"] if item["pattern"] == "src/*.py")
    assert "src/main.py" in src_spec["files"]
    assert "src/util.py" in src_spec["files"]
    assert "tests/test_main.py" not in src_spec["files"]


def test_batch_move_files_and_rollback(tmp_path: Path) -> None:
    (tmp_path / "old").mkdir()
    src_file = tmp_path / "old" / "item.txt"
    src_file.write_text("content to move\n", encoding="utf-8")

    moves = [
        {"source": "old/item.txt", "destination": "new/item.txt"},
    ]
    res = batch_move_files(tmp_path, moves=moves)
    assert res["ok"] is True
    assert not (tmp_path / "old" / "item.txt").exists()
    assert (tmp_path / "new" / "item.txt").read_text(encoding="utf-8") == "content to move\n"


def test_batch_delete_files_and_rollback(tmp_path: Path) -> None:
    f1 = tmp_path / "del1.txt"
    f1.write_text("delete me 1", encoding="utf-8")
    f2 = tmp_path / "del2.txt"
    f2.write_text("delete me 2", encoding="utf-8")

    res = batch_delete_files(tmp_path, paths=["del1.txt", "del2.txt"])
    assert res["ok"] is True
    assert not f1.exists()
    assert not f2.exists()
    assert sorted(res["deleted_files"]) == ["del1.txt", "del2.txt"]


def test_path_safety_and_traversal_prevention(tmp_path: Path) -> None:
    res_r = batch_read_files(tmp_path, files=["../../etc/passwd"])
    assert res_r["ok"] is False
    assert res_r["files"][0]["error_code"] == "PATH_TRAVERSAL"

    res_w = batch_write_files(tmp_path, writes=[{"path": "/absolute/path/file.txt", "content": "abc"}])
    assert res_w["ok"] is False
    assert res_w["error_code"] in {"INVALID_PATH", "PATH_OUTSIDE_WORKSPACE", "ABSOLUTE_DRIVE_PATH"}
