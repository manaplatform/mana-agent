from __future__ import annotations

from pathlib import Path

from mana_agent.tools.apply_patch import extract_patch_touched_files, safe_apply_patch


def test_apply_patch_accepts_git_unified_diff_payload(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\n", encoding="utf-8")
    payload = """diff --git a/src/example.py b/src/example.py
--- a/src/example.py
+++ b/src/example.py
@@ -1,1 +1,1 @@
-old
+new
"""
    result = safe_apply_patch(repo_root=tmp_path, patch=payload)

    assert result["ok"] is True
    assert result["touched_files"] == ["src/example.py"]
    assert target.read_text(encoding="utf-8") == "new\n"


def test_apply_patch_rejects_command_strategy_hint(tmp_path: Path) -> None:
    result = safe_apply_patch(repo_root=tmp_path, patch="[]", strategy_hint="command")

    assert result["ok"] is False
    assert "invalid strategy_hint 'command'" in str(result.get("error", ""))
    assert "auto" in str(result.get("error", ""))
    assert "py" in str(result.get("error", ""))
    assert "perl" not in str(result.get("error", ""))


def test_apply_patch_auto_applies_json_patch_with_py_strategy(tmp_path: Path) -> None:
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

    result = safe_apply_patch(repo_root=tmp_path, patch=patch_payload, strategy_hint="auto")

    assert result["ok"] is True
    assert result["strategy"] == "py"
    assert target.read_text(encoding="utf-8") == "new\n"


def test_extract_patch_touched_files_accepts_structured_payloads() -> None:
    patch_payload = [
        {
            "path": "src/example.py",
            "hunks": [
                {
                    "old_start": 1,
                    "old_lines": ["old"],
                    "new_lines": ["new"],
                }
            ],
        }
    ]

    from_list = extract_patch_touched_files(patch_payload)
    from_nested = extract_patch_touched_files({"patch": patch_payload})

    assert from_list == {"ok": True, "touched_files": ["src/example.py"]}
    assert from_nested == {"ok": True, "touched_files": ["src/example.py"]}
