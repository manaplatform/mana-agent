from __future__ import annotations

from pathlib import Path

from mana_agent.tools.apply_patch import extract_patch_touched_files, safe_apply_patch


def test_apply_patch_accepts_codex_patch_without_line_numbers(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\n", encoding="utf-8")
    payload = """*** Begin Patch
*** Update File: src/example.py
@@
-old
+new
*** End Patch
"""
    result = safe_apply_patch(repo_root=tmp_path, patch=payload)

    assert result["ok"] is True
    assert result["touched_files"] == ["src/example.py"]
    assert target.read_text(encoding="utf-8") == "new\n"


def test_apply_patch_rejects_non_codex_patch_text(tmp_path: Path) -> None:
    result = safe_apply_patch(repo_root=tmp_path, patch="[]")

    assert result["ok"] is False
    assert result["error_code"] == "invalid_patch_format"
    assert "Codex patch format" in str(result.get("error", ""))
    assert "perl" not in str(result.get("error", ""))


def test_apply_patch_rejects_json_hunks_with_old_start(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\n", encoding="utf-8")

    patch_payload = """[
  {
    "path": "src/example.py",
    "create": false,
    "hunks": [
      {
        "old_start": 1,
        "old_lines": ["old"],
        "new_lines": ["new"]
      }
    ]
  }
]"""

    result = safe_apply_patch(repo_root=tmp_path, patch=patch_payload)

    assert result["ok"] is False
    assert result["error_code"] == "invalid_patch_format"
    assert target.read_text(encoding="utf-8") == "old\n"


def test_extract_patch_touched_files_accepts_codex_payload() -> None:
    patch_payload = """*** Begin Patch
*** Update File: src/example.py
@@
-old
+new
*** End Patch"""

    from_text = extract_patch_touched_files(patch_payload)
    from_nested = extract_patch_touched_files({"patch": patch_payload})

    assert from_text == {"ok": True, "touched_files": ["src/example.py"]}
    assert from_nested == {"ok": True, "touched_files": ["src/example.py"]}
