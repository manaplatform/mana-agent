from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.multi_agent.runtime.ask_agent import AskAgent
from mana_agent.multi_agent.runtime.coding_agent_models import CodingAgentStateMachine
from mana_agent.tools.contracts import coding_tool_contracts
from mana_agent.config.settings import default_logs_dir
from mana_agent.tools.apply_patch import safe_apply_patch
from mana_agent.tools.repository import _run_check, call_graph


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


def test_coding_agent_phase_machine_blocks_patch_until_read() -> None:
    machine = CodingAgentStateMachine()
    machine.transition("plan", reason="request understood")
    machine.transition("search", reason="need files")
    machine.transition("read", reason="inspect target")

    with pytest.raises(ValueError):
        machine.transition("patch", targets=["src/a.py"])

    machine.mark_read("src/a.py")
    machine.transition("patch", targets=["src/a.py"])
    machine.transition("verify")
    machine.transition("finalize")
    assert machine.phase == "finalize"
