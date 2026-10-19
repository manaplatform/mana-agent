"""Unit tests for InternalCodingEngine OpenAI Batch API integration and revision fencing."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mana_agent.batch.manager import BatchLifecycleManager
from mana_agent.batch.models import (
    BatchConfig,
    BatchItemResult,
    BatchJobRecord,
    BatchLifecycleStatus,
)
from mana_agent.batch.store import BatchJobStore
from mana_agent.coding.internal_engine import InternalCodingEngine
from mana_agent.coding.models import CodingTask, WorkspaceContext


def test_submit_code_analysis_batch(tmp_path: Path) -> None:
    # Prepare dummy files in workspace
    file1 = tmp_path / "service.py"
    file1.write_text("class Service:\n    pass\n", encoding="utf-8")
    file2 = tmp_path / "models.py"
    file2.write_text("class Model:\n    pass\n", encoding="utf-8")

    file1_sha = hashlib.sha256(file1.read_text("utf-8").encode("utf-8")).hexdigest()

    mock_client = MagicMock()
    mock_client.config = BatchConfig()
    mock_client.upload_batch_file.return_value = "input-file-001"
    mock_client.create_batch.return_value = BatchJobRecord(
        batch_id="batch_code_1",
        status=BatchLifecycleStatus.VALIDATING.value,
        endpoint="/v1/responses",
    )

    store = BatchJobStore(":memory:")
    batch_mgr = BatchLifecycleManager(client=mock_client, store=store)

    # Mock TaskBoard
    mock_tb_item = SimpleNamespace(task_id="task-code-1", queue_job_ids=[])
    mock_taskboard = MagicMock()
    mock_taskboard.get_task.return_value = mock_tb_item

    engine = InternalCodingEngine(
        model="gpt-4o",
        batch_manager=batch_mgr,
        taskboard=mock_taskboard,
    )

    task = CodingTask(
        task_id="task-code-1",
        goal="Audit codebase for performance bugs",
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        allow_in_place_write=True,
    )

    job = engine.submit_code_analysis_batch(
        task=task,
        workspace=workspace,
        files=["service.py", "models.py"],
        prompt="Identify any missing index or quadratic loop",
    )

    assert job.batch_id == "batch_code_1"
    # Verify items stored in sqlite store
    items = store.get_items("batch_code_1")
    assert len(items) == 2
    items_map = {it.file_path: it for it in items}
    assert "service.py" in items_map
    assert items_map["service.py"].expected_sha256 == file1_sha

    # Verify queue_job_ids recorded on taskboard
    assert "batch_code_1" in mock_tb_item.queue_job_ids
    mock_taskboard.save.assert_called()


def test_retrieve_and_apply_batch_results_revision_fencing(tmp_path: Path) -> None:
    file1 = tmp_path / "math.py"
    file1.write_text("def multiply(a, b): return a * b\n", encoding="utf-8")
    initial_sha = hashlib.sha256(file1.read_text("utf-8").encode("utf-8")).hexdigest()

    file2 = tmp_path / "util.py"
    file2.write_text("def ping(): return True\n", encoding="utf-8")
    initial_sha2 = hashlib.sha256(file2.read_text("utf-8").encode("utf-8")).hexdigest()

    mock_client = MagicMock()
    mock_client.config = BatchConfig()
    store = BatchJobStore(":memory:")

    # Prepare job and items in store
    job = BatchJobRecord(
        batch_id="batch_fencing_test",
        status=BatchLifecycleStatus.COMPLETED.value,
        output_file_id="out-file-1",
    )
    store.save_job(job)

    from mana_agent.batch.models import BatchItemRequest

    items = [
        BatchItemRequest(
            custom_id="req-math",
            body={"input": "test"},
            file_path="math.py",
            expected_sha256=initial_sha,
        ),
        BatchItemRequest(
            custom_id="req-util",
            body={"input": "test"},
            file_path="util.py",
            expected_sha256=initial_sha2,
        ),
    ]
    store.save_items("batch_fencing_test", items)

    # Download returns results
    mock_client.download_results.return_value = [
        BatchItemResult(
            custom_id="req-math",
            status_code=200,
            response_body={"output_text": "Suggestion for math"},
        ),
        BatchItemResult(
            custom_id="req-util",
            status_code=200,
            response_body={"output_text": "Suggestion for util"},
        ),
    ]

    # Concurrent modification: mutate math.py before retrieving results!
    file1.write_text("def multiply(a, b): return a * b * 2  # Mutated concurrently!\n", encoding="utf-8")

    batch_mgr = BatchLifecycleManager(client=mock_client, store=store)
    engine = InternalCodingEngine(batch_manager=batch_mgr)

    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        allow_in_place_write=True,
    )
    applied = engine.retrieve_and_apply_batch_results("batch_fencing_test", workspace)

    assert len(applied) == 2
    applied_map = {a["custom_id"]: a for a in applied}

    # math.py should be REJECTED due to revision drift
    assert applied_map["req-math"]["status"] == "stale_revision_rejected"
    assert "revision changed" in applied_map["req-math"]["error"]

    # util.py should be VERIFIED and READY
    assert applied_map["req-util"]["status"] == "verified_and_ready"
    assert applied_map["req-util"]["output_text"] == "Suggestion for util"


def test_internal_engine_execute_batch_mode(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.config = BatchConfig()
    mock_client.upload_batch_file.return_value = "file-batch-input-99"
    mock_client.create_batch.return_value = BatchJobRecord(
        batch_id="batch_exec_99",
        status=BatchLifecycleStatus.VALIDATING.value,
        endpoint="/v1/responses",
        total_requests=1,
    )

    store = BatchJobStore(":memory:")
    batch_mgr = BatchLifecycleManager(client=mock_client, store=store)

    engine = InternalCodingEngine(
        model="gpt-6-luna",
        batch_manager=batch_mgr,
    )

    task = CodingTask(
        task_id="task-batch-exec-1",
        goal="Audit security headers and dependencies",
        context={"execution_mode": "batch"},
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        allow_in_place_write=True,
    )

    result = asyncio.run(engine.execute(task, workspace))
    assert result.status == "completed"
    assert "OpenAI Batch Job Submitted" in result.summary
    assert "batch_exec_99" in result.summary
    assert "50% token cost discount" in result.summary

    # Ensure batch job is saved in SQLite and not cancelled
    job = store.get_job("batch_exec_99")
    assert job is not None
    assert job.batch_id == "batch_exec_99"
    assert job.status != BatchLifecycleStatus.CANCELLED.value
