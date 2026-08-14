"""Scoped, content-addressed storage for lossless permitted tool results."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mana_agent.context_cost.models import ArtifactReference
from mana_agent.workspaces.paths import mana_home


class ArtifactAccessError(PermissionError):
    pass


class ContextArtifactStore:
    def __init__(self, root: Path | None = None, *, retention_days: int = 30) -> None:
        self.root = (root or mana_home() / "context-cache" / "tool-results").resolve()
        self.retention_days = max(1, int(retention_days))

    def put(
        self,
        content: Any,
        *,
        session_id: str,
        repository_id: str,
        workspace_id: str,
        content_type: str,
    ) -> ArtifactReference:
        body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        record = {
            "version": 1,
            "session_id": str(session_id),
            "repository_id": str(repository_id),
            "workspace_id": str(workspace_id),
            "content_type": str(content_type),
            "content": body,
            "content_hash": content_hash,
        }
        canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        target = self.root / f"{digest}.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        return ArtifactReference(
            artifact_id=f"sha256:{digest}", content_hash=content_hash,
            session_id=str(session_id), repository_id=str(repository_id), workspace_id=str(workspace_id),
            content_type=str(content_type), byte_length=len(body.encode("utf-8")),
        )

    def read(
        self,
        reference: ArtifactReference | str,
        *,
        session_id: str,
        repository_id: str,
        workspace_id: str,
        offset: int = 0,
        limit: int = 16_000,
        line_start: int | None = None,
        line_end: int | None = None,
        json_path: str | None = None,
        section: str | None = None,
        record_start: int | None = None,
        record_count: int | None = None,
        search: str | None = None,
        query: str | None = None,
    ) -> Any:
        digest = reference.artifact_id.removeprefix("sha256:") if isinstance(reference, ArtifactReference) else str(reference).removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ArtifactAccessError("invalid context artifact reference")
        path = self.root / f"{digest}.json"
        if path.parent != self.root or not path.is_file():
            raise FileNotFoundError(f"context artifact not found: sha256:{digest}")
        record = json.loads(path.read_text(encoding="utf-8"))
        canonical_record = {key: value for key, value in record.items() if key != "created_at"}
        canonical = json.dumps(canonical_record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != digest:
            raise ArtifactAccessError("context artifact failed its content hash check")
        expected = (str(session_id), str(repository_id), str(workspace_id))
        actual = (str(record.get("session_id", "")), str(record.get("repository_id", "")), str(record.get("workspace_id", "")))
        if actual != expected:
            raise ArtifactAccessError("context artifact scope does not match this session, repository, and workspace")
        content = str(record.get("content", ""))
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != str(record.get("content_hash") or ""):
            raise ArtifactAccessError("context artifact content does not match its recorded hash")
        if json_path:
            value: Any = json.loads(content)
            parts = [name or index for name, index in re.findall(r"([^.\[\]]+)|\[(\d+)\]", json_path.removeprefix("$"))]
            for part in parts:
                value = value[int(part)] if isinstance(value, list) else value[part]
            return value
        if section:
            lines = content.splitlines()
            target_heading = section.strip().lstrip("#").strip().casefold()
            capturing = False
            captured_lines: list[str] = []
            capture_level = 0
            for line in lines:
                match = re.match(r"^(#{1,6})\s+(.*)$", line)
                if match:
                    level = len(match.group(1))
                    heading_text = match.group(2).strip().casefold()
                    if capturing:
                        if level <= capture_level:
                            break
                        captured_lines.append(line)
                    elif target_heading in heading_text:
                        capturing = True
                        capture_level = level
                        captured_lines.append(line)
                elif capturing:
                    captured_lines.append(line)
            if captured_lines:
                return "\n".join(captured_lines)[: max(1, min(int(limit), 64_000))]
        if record_start is not None or record_count is not None:
            start_rec = max(0, int(record_start or 0))
            count_rec = max(1, min(int(record_count or 50), 500))
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data[start_rec : start_rec + count_rec]
                if isinstance(data, dict):
                    keys = list(data.keys())[start_rec : start_rec + count_rec]
                    return {k: data[k] for k in keys}
            except (json.JSONDecodeError, TypeError):
                lines = content.splitlines()
                return "\n".join(lines[start_rec : start_rec + count_rec])
        if line_start is not None:
            lines = content.splitlines()
            start = max(1, int(line_start)) - 1
            end = min(len(lines), int(line_end or line_start))
            return "\n".join(lines[start:end])
        target_search = search or query
        if target_search:
            lines = content.splitlines()
            matches = [line for line in lines if target_search.casefold() in line.casefold()]
            if not matches:
                terms = [t.casefold() for t in target_search.split() if len(t) > 2]
                if terms:
                    scored = []
                    for line in lines:
                        score = sum(t in line.casefold() for t in terms)
                        if score > 0:
                            scored.append((score, line))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    matches = [line for _, line in scored]
            return "\n".join(matches[:100])[: max(1, min(int(limit), 64_000))]
        bounded_limit = max(1, min(int(limit), 64_000))
        bounded_offset = max(0, int(offset))
        return content[bounded_offset : bounded_offset + bounded_limit]

    def cleanup(self, *, now: datetime | None = None) -> int:
        if not self.root.exists():
            return 0
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=self.retention_days)
        removed = 0
        for path in self.root.glob("*.json"):
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        return removed


__all__ = ["ArtifactAccessError", "ContextArtifactStore"]
