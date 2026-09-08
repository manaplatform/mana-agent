from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.multi_agent.runtime.ask_agent import AskAgent
from mana_agent.tools.contracts import coding_tool_contracts
from mana_agent.config.settings import default_logs_dir
from mana_agent.tools.apply_patch import safe_apply_patch
from mana_agent.tools.repository import _run_check, call_graph, verify_project


def test_tool_contracts_are_machine_readable() -> None:
    contracts = coding_tool_contracts()

    names = {item.name for item in contracts}
    assert {
        "read_file",
        "edit_file",
        "multi_edit_file",
        "apply_patch",
        "create_file",
        "delete_file",
        "verify_project",
        "repo_search",
        "find_symbols",
        "call_graph",
    } <= names
    for contract in contracts:
        payload = contract.model_dump()
        assert payload["name"]
        assert payload["description"]
        assert payload["input_schema"]["type"] == "object"
        assert payload["output_schema"]["type"] == "object"
        assert "error" in payload["error_format"]
        assert payload["safety_rules"]
        assert payload["examples"]


def test_safe_file_read_rejects_outside_root_and_binary(tmp_path: Path) -> None:
    agent = object.__new__(AskAgent)
    agent.project_root = tmp_path.resolve()
    binary = tmp_path / "asset.bin"
    binary.write_bytes(b"abc\x00def")

    with pytest.raises(ValueError):
        agent._resolve_read_path(str(tmp_path.parent / "outside.txt"))
    assert agent._is_binary_path(binary) is True


def test_patch_rejects_outside_root_path(tmp_path: Path) -> None:
    result = safe_apply_patch(
        repo_root=tmp_path,
        patch="""*** Begin Patch
*** Add File: ../outside.py
+outside
*** End Patch""",
    )

    assert result["ok"] is False
    assert "traversal" in result["error"]


def test_patch_rejects_unread_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")

    result = safe_apply_patch(
        repo_root=tmp_path,
        patch="""*** Begin Patch
*** Update File: src/example.py
@@
-old
+new
*** End Patch""",
        require_read=True,
        read_files=[],
    )

    assert result["ok"] is False
    assert "unread files" in result["error"]
    assert target.read_text(encoding="utf-8") == "old\n"


def test_successful_patch_flow_records_history(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")

    result = safe_apply_patch(
        repo_root=tmp_path,
        patch="""*** Begin Patch
*** Update File: src/example.py
@@
-old
+new
*** End Patch""",
        require_read=True,
        read_files=["src/example.py"],
    )

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == "new\n"
    history = list(default_logs_dir(tmp_path).glob("apply_patch_*.json"))
    assert history


def test_verification_command_reports_missing_tool(tmp_path: Path) -> None:
    result = _run_check(tmp_path, "missing", ["definitely-not-a-mana-tool"])

    assert result.status == "skipped"
    assert "not found" in result.reason
    assert result.failure_code == "verification_tool_missing"


def test_verify_project_docs_only_returns_structured_skips(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")

    result = verify_project(tmp_path, quick=True, changed_files=["README.md"])

    assert result["ok"] is True
    assert result["verification_class"] == "documentation"
    assert result["selected_commands"] == ["verify_changed_artifacts"]
    assert any("pytest" in reason for reason in result["skipped_checks"])
    assert result["failure_code"] == ""


def test_quick_project_verification_does_not_select_full_pytest(monkeypatch, tmp_path: Path) -> None:
    selected: list[list[str]] = []

    def _fake_run_check(repo_root, name, command, timeout=120, **kwargs):  # noqa: ANN001
        selected.append(command)
        from mana_agent.tools.repository import VerificationCheck

        return VerificationCheck(name=name, command=command, status="passed", **kwargs)

    monkeypatch.setattr("mana_agent.tools.repository._run_check", _fake_run_check)

    result = verify_project(tmp_path, quick=True)

    assert result["ok"] is True
    assert selected
    assert all(command[:2] != ["pytest", "-q"] for command in selected)


def test_call_graph_reports_python_ast_call_edges(tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "demo.py"
    source.parent.mkdir()
    source.write_text(
        """
def helper():
    return 1

class Runner:
    def run(self):
        helper()
        self.finish()
""".lstrip(),
        encoding="utf-8",
    )

    result = call_graph(tmp_path, query="Runner.run", limit=20)

    assert result["ok"] is True
    edges = result["edges"]
    assert {"file": "pkg/demo.py", "line": 6, "caller": "Runner.run", "callee": "helper"} in edges
    assert {"file": "pkg/demo.py", "line": 7, "caller": "Runner.run", "callee": "self.finish"} in edges
