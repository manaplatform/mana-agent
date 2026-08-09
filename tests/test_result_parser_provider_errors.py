"""Result parser surfaces provider diagnostics instead of reconnect banners."""

from __future__ import annotations

from pathlib import Path

from mana_agent.coding.models import CodingTask, WorkspaceContext
from mana_agent.integrations.codex.result_parser import parse_codex_result


def test_turn_failed_preserves_http_400_diagnostic(tmp_path: Path) -> None:
    task = CodingTask(task_id="t1", goal="fix bug")
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="readOnly",
    )
    result = parse_codex_result(
        task=task,
        workspace=workspace,
        worker_id="w1",
        thread_id="th1",
        turn_id="tu1",
        notifications=[
            {
                "method": "turn/failed",
                "params": {
                    "message": {
                        "message": "Reconnecting... 1/5",
                        "codexErrorInfo": {
                            "responseStreamDisconnected": {"httpStatusCode": None}
                        },
                        "additionalDetails": (
                            "stream disconnected before completion: "
                            "NVIDIA rejected the request (HTTP 400). "
                            "model=deepseek-ai/deepseek-v4-flash"
                        ),
                    }
                },
            }
        ],
        changed_files=[],
    )
    assert result.status == "failed"
    joined = " ".join(result.errors)
    assert "HTTP 400" in joined or "400" in joined
    assert "rejected" in joined.lower() or "stream" in joined.lower()
