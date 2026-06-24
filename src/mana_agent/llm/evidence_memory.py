from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


ReadMode = Literal["line", "full"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceMemory:
    """Append-only, run-scoped evidence cache under ``.mana/runs/<run_id>``."""

    def __init__(self, *, repo_root: Path, run_id: str | None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.run_id = str(run_id or "").strip()
        self.run_dir = self.repo_root / ".mana" / "runs" / self.run_id if self.run_id else None
        self.path = self.run_dir / "read_evidence.jsonl" if self.run_dir else None
        self._index: dict[str, list[dict[str, Any]]] = {}
        self._loaded = False

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def enabled(self) -> bool:
        return self.path is not None

    def normalize_path(self, path: str | Path) -> Path:
        requested = Path(path)
        resolved = requested if requested.is_absolute() else (self.repo_root / requested)
        resolved = resolved.resolve()
        resolved.relative_to(self.repo_root)
        return resolved

    def to_repo_rel(self, path: str | Path) -> str:
        return self.normalize_path(path).relative_to(self.repo_root).as_posix()

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path is None or not self.path.exists():
            return
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            normalized = str(row.get("normalized_path") or "").strip()
            event = str(row.get("event") or "").strip()
            if not normalized:
                continue
            if event == "invalidate":
                self._index.pop(normalized, None)
                continue
            if event == "read":
                self._index.setdefault(normalized, []).insert(0, row)

    def _append(self, row: dict[str, Any]) -> None:
        if self.path is None or self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def read_files(self) -> set[str]:
        self._load()
        out: set[str] = set()
        for normalized in self._index:
            try:
                out.add(Path(normalized).relative_to(self.repo_root).as_posix())
            except ValueError:
                out.add(normalized)
        return out

    def invalidate(self, path: str | Path) -> None:
        if not self.enabled():
            return
        try:
            normalized = str(self.normalize_path(path))
        except Exception:
            return
        self._load()
        self._index.pop(normalized, None)
        self._append({"event": "invalidate", "timestamp": _utc_now(), "normalized_path": normalized})

    def invalidate_many(self, paths: list[str] | set[str] | tuple[str, ...]) -> None:
        for path in paths:
            self.invalidate(path)

    def _fresh_rows(self, resolved: Path) -> tuple[list[dict[str, Any]], bool]:
        self._load()
        normalized = str(resolved)
        rows = list(self._index.get(normalized, []))
        if not rows:
            return [], False
        stat = resolved.stat()
        fresh: list[dict[str, Any]] = []
        stale = False
        for row in rows:
            if int(row.get("stat_size", -1) or -1) != int(stat.st_size):
                stale = True
                continue
            if int(row.get("stat_mtime_ns", -1) or -1) != int(stat.st_mtime_ns):
                stale = True
                continue
            content = str(row.get("content") or "")
            if self.content_hash(content) != str(row.get("content_hash") or ""):
                stale = True
                continue
            fresh.append(row)
        if stale:
            self.invalidate(resolved)
            for row in reversed(fresh):
                self._index.setdefault(normalized, []).insert(0, row)
        return fresh, stale

    def lookup(
        self,
        *,
        resolved: Path,
        mode: ReadMode,
        start_line: int,
        end_line: int,
    ) -> tuple[dict[str, Any] | None, bool]:
        if not self.enabled():
            return None, False
        rows, invalidated = self._fresh_rows(resolved)
        if not rows:
            return None, invalidated
        full = next((row for row in rows if str(row.get("mode")) == "full"), None)
        if full is not None:
            return self._payload_from_row(full, mode=mode, start_line=start_line, end_line=end_line), invalidated
        if mode == "line":
            for row in rows:
                row_start = int(row.get("start_line", 1) or 1)
                row_end = int(row.get("end_line", row_start) or row_start)
                if row_start <= start_line and row_end >= end_line:
                    return self._payload_from_row(row, mode="line", start_line=start_line, end_line=end_line), invalidated
        return None, invalidated

    def _payload_from_row(
        self,
        row: dict[str, Any],
        *,
        mode: ReadMode,
        start_line: int,
        end_line: int,
    ) -> dict[str, Any]:
        row_mode = "full" if str(row.get("mode")) == "full" else "line"
        line_count = int(row.get("line_count", 0) or 0)
        if mode == "full":
            covered = [1, line_count]
            content = str(row.get("content") or "")
            actual_end = line_count
        elif row_mode == "full":
            actual_end = min(max(end_line, start_line), line_count)
            content = "\n".join(str(row.get("content") or "").splitlines()[start_line - 1 : actual_end])
            covered = [start_line, actual_end]
        else:
            row_start = int(row.get("start_line", 1) or 1)
            row_end = int(row.get("end_line", row_start) or row_start)
            actual_end = min(max(end_line, start_line), row_end)
            lines = str(row.get("content") or "").splitlines()
            slice_start = max(start_line, row_start) - row_start
            slice_end = min(actual_end, row_end) - row_start + 1
            content = "\n".join(lines[slice_start:max(slice_start, slice_end)])
            covered = [start_line, actual_end]
        return {
            "file_path": str(row.get("normalized_path") or ""),
            "normalized_path": str(row.get("normalized_path") or ""),
            "original_path": str(row.get("original_path") or ""),
            "mode": mode,
            "start_line": 1 if mode == "full" else start_line,
            "end_line": actual_end,
            "line_count": line_count,
            "content": content,
            "cache_hit": True,
            "source": "memory",
            "cache_source": "run_evidence_full" if row_mode == "full" else "run_evidence_range",
            "cache_invalidated": False,
            "full_file_cached": row_mode == "full",
            "covered_range": covered,
        }

    def store(
        self,
        *,
        original_path: str,
        resolved: Path,
        mode: ReadMode,
        start_line: int,
        end_line: int,
        line_count: int,
        content: str,
        summary: str,
    ) -> None:
        if not self.enabled():
            return
        stat = resolved.stat()
        row = {
            "event": "read",
            "timestamp": _utc_now(),
            "normalized_path": str(resolved),
            "original_path": str(original_path),
            "start_line": int(start_line),
            "end_line": int(end_line),
            "mode": mode,
            "content_hash": self.content_hash(content),
            "stat_size": int(stat.st_size),
            "stat_mtime": float(stat.st_mtime),
            "stat_mtime_ns": int(stat.st_mtime_ns),
            "line_count": int(line_count),
            "summary": summary,
            "content": content,
        }
        self._load()
        rows = self._index.setdefault(str(resolved), [])
        row_key = (mode, int(start_line), int(end_line))
        rows[:] = [
            item
            for item in rows
            if (str(item.get("mode")), int(item.get("start_line", 0) or 0), int(item.get("end_line", 0) or 0)) != row_key
        ]
        rows.insert(0, row)
        self._append(row)
