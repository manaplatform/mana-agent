from __future__ import annotations

from pathlib import Path

from mana_agent.gateway.artifact_routing import artifact_handler_availability, artifact_routing_evidence
from mana_agent.gateway.lane_coordinator import LaneCoordinator
from mana_agent.gateway.lanes import LaneId, select_lane


def test_uploaded_xls_is_spreadsheet_evidence_and_never_needs_repository_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    upload = tmp_path / "uploads" / "test.XLS"
    upload.parent.mkdir()
    upload.write_bytes(b"placeholder")

    evidence = artifact_routing_evidence(
        root=repo,
        user_prompt="test.xls\nin cell under age add average of age.",
        attachments=[{"path": str(upload), "mime_type": "application/vnd.ms-excel"}],
    )

    assert evidence["artifact_families"] == ["spreadsheet"]
    assert evidence["has_user_artifact"] is True
    assert evidence["references"][0]["repository_member"] is False
    assert artifact_handler_availability(evidence) == (True, "")
    assert select_lane(entry_route="artifact") is LaneId.ARTIFACT

    coordinator = LaneCoordinator(repo)
    reservation = coordinator.reserve(
        normalized_intent="spreadsheet edit",
        lane_id=LaneId.ARTIFACT,
        session_id="session",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id="",
        target_files=[str(upload)],
        capabilities=("artifact_read", "artifact_write"),
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    coordinator.start(reservation)
    held = [lease.mode for lease in coordinator._locks.values() if lease.task_id == reservation.execution.task_id]
    assert held == []
    coordinator.finish(reservation.execution.task_id)


def test_repository_source_is_not_a_user_artifact(tmp_path: Path) -> None:
    source = tmp_path / "src" / "service.py"
    source.parent.mkdir()
    source.write_text("pass\n")
    evidence = artifact_routing_evidence(root=tmp_path, user_prompt="Edit src/service.py", target_files=["src/service.py"])

    assert evidence["artifact_families"] == []
    assert evidence["has_user_artifact"] is False


def test_image_attachment_is_multimodal_not_document_artifact(tmp_path: Path) -> None:
    upload = tmp_path / "uploads" / "logo.png"
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_bytes(b"\x89PNG\r\n\x1a\n")

    evidence = artifact_routing_evidence(
        root=tmp_path,
        user_prompt="check what is this?",
        attachments=[{"path": str(upload), "mime_type": "image/png"}],
    )

    # Image attachments belong to multimodal conversation, not document tools
    assert evidence["artifact_families"] == []
    assert evidence["has_user_artifact"] is False
    assert any(ref["filename"] == "logo.png" and ref["family"] is None for ref in evidence["references"])

    # If an unconfigured family is directly evaluated, guidance points to conversation or media
    available, reason = artifact_handler_availability({"artifact_families": ["image"]})
    assert available is False
    assert "No configured artifact handler can execute: image." in reason
    assert "conversation" in reason

