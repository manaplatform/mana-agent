from __future__ import annotations

from pathlib import Path

from mana_agent.tools.apply_patch import build_apply_patch_tool
from mana_agent.tools.write_file import build_create_file_tool, build_delete_file_tool, build_write_file_tool, safe_delete_file


def test_apply_patch_tool_accepts_patch_alias(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_safe_apply_patch(
        *,
        repo_root: Path,
        patch: str,
        allowed_prefixes,
        check_only: bool,
        **_kwargs: object,
    ) -> dict:
        captured["repo_root"] = repo_root
        captured["patch"] = patch
        captured["allowed_prefixes"] = allowed_prefixes
        captured["check_only"] = check_only
        return {"ok": True, "touched_files": ["sample.py"], "check_only": check_only}

    monkeypatch.setattr("mana_agent.tools.apply_patch.safe_apply_patch", _fake_safe_apply_patch)

    tool = build_apply_patch_tool(repo_root=tmp_path, allowed_prefixes=None)
    result = tool.invoke({"patch": "*** Begin Patch\n*** End Patch", "check_only": True})

    assert result["ok"] is True
    assert captured["patch"] == "*** Begin Patch\n*** End Patch"
    assert captured["check_only"] is True


def test_apply_patch_tool_accepts_nested_patch_payload(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    patch_text = "*** Begin Patch\n*** Update File: sample.py\n@@\n-old\n+new\n*** End Patch\n"

    def _fake_safe_apply_patch(
        *,
        repo_root: Path,
        patch: str,
        allowed_prefixes,
        check_only: bool,
        **_kwargs: object,
    ) -> dict:
        captured["patch"] = patch
        return {"ok": True, "touched_files": ["sample.py"], "check_only": check_only}

    monkeypatch.setattr("mana_agent.tools.apply_patch.safe_apply_patch", _fake_safe_apply_patch)

    tool = build_apply_patch_tool(repo_root=tmp_path, allowed_prefixes=None)
    result = tool.invoke({"patch": {"patch": patch_text}})

    assert result["ok"] is True
    assert captured["patch"] == patch_text


def test_apply_patch_tool_rejects_structured_patch_list(tmp_path: Path) -> None:
    tool = build_apply_patch_tool(repo_root=tmp_path, allowed_prefixes=None)
    result = tool.invoke({"patch": [{"path": "sample.py", "hunks": []}]})

    assert result["ok"] is False
    assert result["error_code"] == "invalid_patch_format"


def test_apply_patch_tool_accepts_input_alias(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    patch_text = "*** Begin Patch\n*** Update File: sample.py\n@@\n-old\n+new\n*** End Patch\n"

    def _fake_safe_apply_patch(
        *,
        repo_root: Path,
        patch: str,
        allowed_prefixes,
        check_only: bool,
        **_kwargs: object,
    ) -> dict:
        captured["patch"] = patch
        return {"ok": True, "touched_files": ["sample.py"], "check_only": check_only}

    monkeypatch.setattr("mana_agent.tools.apply_patch.safe_apply_patch", _fake_safe_apply_patch)

    tool = build_apply_patch_tool(repo_root=tmp_path, allowed_prefixes=None)
    result = tool.invoke({"input": patch_text})

    assert result["ok"] is True
    assert captured["patch"] == patch_text


def test_write_file_tool_accepts_text_alias(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_safe_write_file(
        *,
        repo_root: Path,
        path: str,
        content: str,
        allowed_prefixes,
        expected_sha256=None,
        force: bool = False,
    ) -> dict:
        _ = (expected_sha256, force)
        captured["repo_root"] = repo_root
        captured["path"] = path
        captured["content"] = content
        captured["allowed_prefixes"] = allowed_prefixes
        return {"ok": True, "path": path, "bytes_written": len(content.encode("utf-8")), "sha256": "", "error": ""}

    monkeypatch.setattr("mana_agent.tools.write_file.safe_write_file", _fake_safe_write_file)

    tool = build_write_file_tool(repo_root=tmp_path, allowed_prefixes=None)
    result = tool.invoke({"path": "src/new_file.py", "text": "print('ok')\n"})

    assert result["ok"] is True
    assert captured["path"] == "src/new_file.py"
    assert captured["content"] == "print('ok')\n"


def test_create_file_tool_accepts_text_alias(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_safe_create_file(
        *,
        repo_root: Path,
        path: str,
        content: str,
        allowed_prefixes,
    ) -> dict:
        captured["repo_root"] = repo_root
        captured["path"] = path
        captured["content"] = content
        captured["allowed_prefixes"] = allowed_prefixes
        return {"ok": True, "path": path, "bytes_written": len(content.encode("utf-8")), "sha256": "", "error": ""}

    monkeypatch.setattr("mana_agent.tools.write_file.safe_create_file", _fake_safe_create_file)

    tool = build_create_file_tool(repo_root=tmp_path, allowed_prefixes=None)
    result = tool.invoke({"path": "src/new_file.py", "text": "print('ok')\n"})

    assert result["ok"] is True
    assert captured["path"] == "src/new_file.py"
    assert captured["content"] == "print('ok')\n"


def test_safe_delete_file_deletes_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "src" / "old.py"
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")

    result = safe_delete_file(repo_root=tmp_path, path="src/old.py")

    assert result["ok"] is True
    assert result["path"] == "src/old.py"
    assert result["deleted"] is True
    assert result["files_changed"] == ["src/old.py"]
    assert not target.exists()


def test_safe_delete_file_rejects_missing_directory_and_traversal(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()

    missing = safe_delete_file(repo_root=tmp_path, path="src/missing.py")
    directory = safe_delete_file(repo_root=tmp_path, path="src")
    traversal = safe_delete_file(repo_root=tmp_path, path="../outside.py")

    assert missing["ok"] is False
    assert "does not exist" in missing["error"]
    assert directory["ok"] is False
    assert "not a file" in directory["error"]
    assert traversal["ok"] is False
    assert "traversal" in traversal["error"]


def test_delete_file_tool_calls_safe_delete_file(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_safe_delete_file(*, repo_root: Path, path: str, allowed_prefixes) -> dict:
        captured["repo_root"] = repo_root
        captured["path"] = path
        captured["allowed_prefixes"] = allowed_prefixes
        return {"ok": True, "path": path, "deleted": True, "files_changed": [path], "error": ""}

    monkeypatch.setattr("mana_agent.tools.write_file.safe_delete_file", _fake_safe_delete_file)

    tool = build_delete_file_tool(repo_root=tmp_path, allowed_prefixes=None)
    result = tool.invoke({"path": "src/old.py"})

    assert result["ok"] is True
    assert captured["path"] == "src/old.py"
