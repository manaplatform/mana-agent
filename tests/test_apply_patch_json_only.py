from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.tools.apply_patch import (
    PatchFormatError,
    extract_patch_operations,
    extract_patch_touched_files,
    safe_apply_patch,
)


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


@pytest.mark.parametrize(
    ("bad_line", "diagnostic"),
    [
        ("*** End Patch", "premature *** End Patch at patch line 6"),
        ("missing a hunk prefix", "invalid patch hunk line at patch line 6"),
    ],
)
def test_apply_patch_rejects_bad_syntax_before_matching_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_line: str, diagnostic: str,
) -> None:
    target = tmp_path / "example.txt"
    target.write_text("different\n", encoding="utf-8")
    patch = (
        "*** Begin Patch\n*** Update File: example.txt\n@@\n-old\n+new\n"
        f"{bad_line}\n@@\n-different\n+replacement\n*** End Patch\n"
    )
    read_bytes = Path.read_bytes

    def forbid_target_read(path: Path) -> bytes:
        if path == target:
            pytest.fail("Malformed patches must fail before reading the target")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_target_read)
    with pytest.raises(PatchFormatError) as error:
        extract_patch_operations(patch)
    assert diagnostic in str(error.value)
    assert not extract_patch_touched_files(patch)["ok"]

    for check_only in (True, False):
        result = safe_apply_patch(repo_root=tmp_path, patch=patch, check_only=check_only)
        assert not result["ok"]
        assert result["error_code"] == "invalid_patch_format"
        assert diagnostic in result["error"]
        assert not result["attempts"]
        assert not result.get("action_id")
        assert target.read_text(encoding="utf-8") == "different\n"


def test_apply_patch_preserves_signed_terminators_inside_a_hunk(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("*** End Patch\nbefore\n", encoding="utf-8")
    body = (
        "*** Begin Patch\n*** Update File: example.txt\n@@\n"
        " *** End Patch\n-before\n+*** End Patch\n"
    )
    patch = body + "*** End Patch\n"
    result = safe_apply_patch(repo_root=tmp_path, patch=patch)
    assert result["ok"], result["error"]
    assert target.read_text(encoding="utf-8") == "*** End Patch\n*** End Patch\n"

    missing_terminator = body + " *** End Patch\n"
    result = safe_apply_patch(repo_root=tmp_path, patch=missing_terminator, check_only=True)
    assert result["error_code"] == "invalid_patch_format"
    assert "patch must end with *** End Patch" in result["error"]


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
