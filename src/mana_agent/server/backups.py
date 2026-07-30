"""Backup evidence and checksum verification contracts."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from pydantic import Field

from .models import StrictModel, utc_now


class BackupResult(StrictModel):
    backup_id: str
    server_id: str
    destination: str
    checksum_algorithm: str = "sha256"
    checksum: str
    verified: bool
    restore_tested: bool = False
    created_at: datetime = Field(default_factory=utc_now)


def verify_local_backup(path: Path, expected_checksum: str) -> bool:
    if not path.is_file() or not expected_checksum:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_checksum


def postgres_backup_argv(database: str, destination: str) -> list[str]:
    if not database or database.startswith("-"):
        raise ValueError("An exact PostgreSQL database name is required.")
    if not destination.startswith("/"):
        raise ValueError("Backup destination must be an absolute remote path.")
    return ["pg_dump", "--format=custom", "--file", destination, "--", database]
