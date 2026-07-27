"""Provider-independent safe local artifact collection."""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
from pathlib import Path

from mana_agent.execution.errors import ArtifactError
from mana_agent.execution.models import ArtifactRequest, ArtifactResult


def confined_path(root: Path, relative: str, *, require_exists: bool = True) -> Path:
    root = root.resolve()
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ArtifactError("artifact paths must be relative to the sandbox workspace")
    resolved = (root / candidate).resolve(strict=require_exists)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactError(f"artifact path escapes workspace: {relative}") from exc
    if resolved.is_symlink():
        raise ArtifactError(f"artifact symlinks are not allowed: {relative}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_local_artifacts(root: Path, request: ArtifactRequest) -> list[ArtifactResult]:
    destination = request.destination.resolve() if request.destination else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    results: list[ArtifactResult] = []
    total = 0
    for relative in request.paths:
        try:
            source = confined_path(root, relative)
        except (FileNotFoundError, ArtifactError):
            if request.missing_ok:
                continue
            raise
        files = [source] if source.is_file() else [item for item in source.rglob("*") if item.is_file() and not item.is_symlink()]
        for item in files:
            size = item.stat().st_size
            if size > request.max_file_bytes:
                raise ArtifactError(f"artifact exceeds per-file limit: {item.name}")
            total += size
            if total > request.max_total_bytes:
                raise ArtifactError("artifact collection exceeds total size limit")
            rel = item.relative_to(root.resolve())
            local_path = None
            if destination:
                local_path = destination / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, local_path, follow_symlinks=False)
            digest = sha256_file(item)
            results.append(ArtifactResult(
                reference=f"sha256:{digest}", source_path=str(rel), local_path=local_path,
                size_bytes=size, sha256=digest,
                mime_type=mimetypes.guess_type(item.name)[0] or "application/octet-stream",
            ))
    return results
