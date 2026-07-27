"""Safe, immutable local workspace snapshots."""

from __future__ import annotations

import json
import tarfile
import uuid
from pathlib import Path

from mana_agent.execution.artifacts import confined_path, sha256_file
from mana_agent.execution.errors import SnapshotError
from mana_agent.execution.models import SnapshotRef, SnapshotRequest


def create_archive_snapshot(workspace: Path, request: SnapshotRequest, root: Path, provider: str) -> SnapshotRef:
    snapshot_id = f"snap_{uuid.uuid4().hex}"
    directory = root / snapshot_id
    directory.mkdir(parents=True, exist_ok=False)
    archive = directory / "workspace.tar.gz"
    manifest = directory / "manifest.json"
    try:
        with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as bundle:
            for relative in request.include_paths:
                source = confined_path(workspace, relative)
                bundle.add(source, arcname=str(source.relative_to(workspace.resolve())), recursive=True)
        checksum = sha256_file(archive)
        payload = {"schema_version": 1, "snapshot_id": snapshot_id, "provider": provider, "checksum": checksum}
        manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return SnapshotRef(snapshot_id=snapshot_id, provider=provider, location=str(archive), checksum=checksum)
    except Exception as exc:
        raise SnapshotError(f"snapshot creation failed: {exc}", provider=provider) from exc


def restore_archive_snapshot(snapshot: SnapshotRef, destination: Path) -> None:
    archive = Path(snapshot.location).resolve()
    if not archive.is_file() or sha256_file(archive) != snapshot.checksum:
        raise SnapshotError("snapshot checksum validation failed", provider=snapshot.provider)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        root = destination.resolve()
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise SnapshotError("snapshot contains a path traversal entry") from exc
            if member.issym() or member.islnk():
                raise SnapshotError("snapshot links are not allowed")
        bundle.extractall(root, filter="data")
