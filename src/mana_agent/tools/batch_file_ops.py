"""Native filesystem batch execution layer for Mana-Agent coding engine.

Provides high-performance, concurrency-safe batched file operations:
- batch_read_files: Parallel/batched reading with per-file truncation and hashing.
- batch_write_files: Staged atomic writes with optimistic concurrency and rollback.
- batch_apply_patches: Multi-file Codex patch application with rollback.
- batch_search_files: Concurrent multi-pattern text and regex searches.
- batch_list_files: Multi-pattern file listing without redundant filesystem scans.
- batch_move_files: Path-safe file relocation with transactional recovery.
- batch_delete_files: Bounded, safe file deletion with rollback capability.
"""

from __future__ import annotations

import concurrent.futures
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from mana_agent.tools.apply_patch import safe_apply_patch

logger = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git",
    ".mana",
    ".mana_logs",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


def _safe_resolve_path(repo_root: Path, user_path: str) -> tuple[Path | None, str | None, str | None]:
    """Validate user path against directory traversal, symlink escapes, and null bytes.

    Returns:
        (resolved_target_path, normalized_relative_posix_path, error_code_if_any)
    """
    raw = str(user_path or "").strip().replace("\\", "/")
    if not raw:
        return None, None, "EMPTY_PATH"
    if "\x00" in raw:
        return None, None, "NULL_BYTE_IN_PATH"

    root_resolved = repo_root.resolve()

    # Check for drive paths on Windows
    if re.match(r"^[a-zA-Z]:", raw):
        target = Path(raw).resolve()
        try:
            if not target.is_relative_to(root_resolved):
                return None, None, "ABSOLUTE_DRIVE_PATH"
            rel = target.relative_to(root_resolved).as_posix()
            return target, rel, None
        except (ValueError, RuntimeError):
            return None, None, "PATH_OUTSIDE_WORKSPACE"

    # Check for absolute posix paths
    if raw.startswith("/"):
        target = Path(raw).resolve()
        try:
            if not target.is_relative_to(root_resolved):
                return None, None, "PATH_OUTSIDE_WORKSPACE"
            rel = target.relative_to(root_resolved).as_posix()
            return target, rel, None
        except (ValueError, RuntimeError):
            return None, None, "PATH_OUTSIDE_WORKSPACE"

    # Relative path checks
    parts = [p for p in raw.split("/") if p not in {"", "."}]
    if any(p == ".." for p in parts):
        return None, None, "PATH_TRAVERSAL"

    target = (root_resolved / Path(*parts)).resolve()

    try:
        if not target.is_relative_to(root_resolved):
            return None, None, "PATH_OUTSIDE_WORKSPACE"
        rel = target.relative_to(root_resolved).as_posix()
        return target, rel, None
    except (ValueError, RuntimeError):
        return None, None, "PATH_RESOLUTION_FAILED"


def _is_binary_file(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:4096]
        return b"\x00" in sample
    except Exception:
        return False


def _compute_sha256(content_bytes: bytes) -> str:
    return hashlib.sha256(content_bytes).hexdigest()


def _matches_pattern(rel_path: str, pattern: str) -> bool:
    rel_path = rel_path.replace("\\", "/").lstrip("./")
    pattern = str(pattern or "**/*").replace("\\", "/").lstrip("./")
    if pattern in {"**", "**/*", "*"}:
        return True
    for suffix in ("/**/*", "/**"):
        if pattern.endswith(suffix):
            prefix = pattern[:-len(suffix)].rstrip("/")
            return rel_path.startswith(f"{prefix}/") if prefix else True
    if "/" not in pattern and fnmatch.fnmatch(Path(rel_path).name, pattern):
        return True
    return PurePosixPath(rel_path).match(pattern)


@dataclass
class FileRollbackSnapshot:
    """Captures original file bytes or nonexistent state for transactional rollback."""
    exists: bool
    content: bytes | None
    sha256: str | None


class BatchFileSystemManager:
    """Manages transactional batch filesystem operations with rollback."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def _capture_snapshot(self, paths: Sequence[Path]) -> dict[Path, FileRollbackSnapshot]:
        snapshots: dict[Path, FileRollbackSnapshot] = {}
        for p in paths:
            if p.is_file():
                try:
                    data = p.read_bytes()
                    snapshots[p] = FileRollbackSnapshot(exists=True, content=data, sha256=_compute_sha256(data))
                except Exception as exc:
                    logger.warning("Could not snapshot existing file %s: %s", p, exc)
                    snapshots[p] = FileRollbackSnapshot(exists=True, content=None, sha256=None)
            else:
                snapshots[p] = FileRollbackSnapshot(exists=False, content=None, sha256=None)
        return snapshots

    def _execute_rollback(self, snapshots: dict[Path, FileRollbackSnapshot]) -> list[str]:
        restored: list[str] = []
        for p, snap in snapshots.items():
            try:
                if snap.exists and snap.content is not None:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    # Atomic rewrite
                    tmp = tempfile.NamedTemporaryFile(dir=p.parent, delete=False)
                    try:
                        tmp.write(snap.content)
                        tmp.flush()
                        os.fsync(tmp.fileno())
                        tmp.close()
                        os.replace(tmp.name, p)
                        restored.append(p.as_posix())
                    finally:
                        if os.path.exists(tmp.name):
                            try:
                                os.unlink(tmp.name)
                            except OSError:
                                pass
                elif not snap.exists and p.exists():
                    p.unlink(missing_ok=True)
                    restored.append(p.as_posix())
            except Exception as exc:
                logger.error("Failed to restore %s during rollback: %s", p, exc)
        return restored


# =============================================================================
# 1. BATCH READ FILES
# =============================================================================

def batch_read_files(
    repo_root: Path,
    *,
    files: list[str | dict[str, Any]],
    max_bytes: int = 2_000_000,
    max_bytes_per_file: int = 150_000,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Read multiple repository text files with per-file limits, hashing, and concurrency."""
    root = repo_root.resolve()
    requested = list(files or [])[:100]
    total_budget = max(1_000, min(int(max_bytes or 2_000_000), 10_000_000))
    per_file_cap = max(500, min(int(max_bytes_per_file or 150_000), total_budget))

    file_specs: list[dict[str, Any]] = []
    for item in requested:
        if isinstance(item, str):
            file_specs.append({"path": item})
        elif isinstance(item, dict):
            file_specs.append(dict(item))

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _read_single(spec: dict[str, Any]) -> dict[str, Any]:
        raw_path = spec.get("path") or spec.get("file_path") or ""
        target, rel, err_code = _safe_resolve_path(root, str(raw_path))
        if err_code or target is None or rel is None:
            return {
                "path": str(raw_path),
                "ok": False,
                "error": f"Invalid path: {err_code or 'UNKNOWN'}",
                "error_code": err_code or "INVALID_PATH",
            }

        if not target.exists() or not target.is_file():
            return {
                "path": rel,
                "ok": False,
                "error": "File not found",
                "error_code": "FILE_NOT_FOUND",
            }

        if _is_binary_file(target):
            return {
                "path": rel,
                "ok": False,
                "error": "Binary file skipped",
                "error_code": "BINARY_FILE",
            }

        try:
            raw_bytes = target.read_bytes()
            file_size = len(raw_bytes)
            file_sha256 = _compute_sha256(raw_bytes)

            start_line = spec.get("start_line")
            end_line = spec.get("end_line")

            text = raw_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
            lines = text.splitlines(keepends=True)
            total_lines = len(lines)

            if start_line is not None or end_line is not None:
                s = max(1, int(start_line or 1))
                e = min(total_lines, int(end_line or total_lines))
                sliced = "".join(lines[s - 1 : e])
                sliced_bytes = sliced.encode("utf-8")
                truncated = len(sliced_bytes) > per_file_cap
                chunk = sliced_bytes[:per_file_cap].decode("utf-8", errors="replace")
                return {
                    "path": rel,
                    "ok": True,
                    "content": chunk,
                    "sha256": file_sha256,
                    "start_line": s,
                    "end_line": e,
                    "total_lines": total_lines,
                    "bytes_read": len(chunk.encode("utf-8")),
                    "file_size_bytes": file_size,
                    "truncated": truncated,
                }

            truncated = file_size > per_file_cap
            chunk_bytes = raw_bytes[:per_file_cap]
            chunk_str = chunk_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")

            return {
                "path": rel,
                "ok": True,
                "content": chunk_str,
                "sha256": file_sha256,
                "total_lines": total_lines,
                "bytes_read": len(chunk_bytes),
                "file_size_bytes": file_size,
                "truncated": truncated,
            }
        except Exception as exc:
            return {
                "path": rel,
                "ok": False,
                "error": str(exc),
                "error_code": "READ_ERROR",
            }

    # Parallelize file reads using ThreadPoolExecutor
    workers = max(1, min(concurrency, len(file_specs) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_read_single, spec): spec for spec in file_specs}
        for future in concurrent.futures.as_completed(future_map):
            res = future.result()
            path_key = res["path"]
            if path_key in seen:
                continue
            seen.add(path_key)
            results.append(res)
            if not res.get("ok"):
                errors.append(res)

    results.sort(key=lambda r: r.get("path", ""))
    return {
        "ok": len(errors) == 0,
        "files": results,
        "errors": errors,
        "total_files": len(results),
        "successful_count": len(results) - len(errors),
    }


# =============================================================================
# 2. BATCH WRITE FILES
# =============================================================================

def batch_write_files(
    repo_root: Path,
    *,
    writes: list[dict[str, Any]],
    rollback_on_failure: bool = True,
    check_only: bool = False,
) -> dict[str, Any]:
    """Execute multiple file writes with optimistic concurrency, staged atomic writes, and rollback."""
    root = repo_root.resolve()
    items = list(writes or [])
    if not items:
        return {"ok": True, "results": [], "changed_files": []}

    manager = BatchFileSystemManager(root)
    resolved_ops: list[dict[str, Any]] = []
    targets_to_snapshot: list[Path] = []
    conflicts: list[dict[str, Any]] = []

    # 1. Validation and optimistic concurrency pre-check
    for idx, item in enumerate(items):
        raw_path = item.get("path") or ""
        content = str(item.get("content") if item.get("content") is not None else item.get("text", ""))
        expected_sha = item.get("expected_sha256")
        force = bool(item.get("force", False))

        target, rel, err_code = _safe_resolve_path(root, raw_path)
        if err_code or target is None or rel is None:
            return {
                "ok": False,
                "error": f"Invalid path '{raw_path}': {err_code}",
                "error_code": err_code or "INVALID_PATH",
                "results": [],
                "changed_files": [],
            }

        targets_to_snapshot.append(target)
        # Check optimistic concurrency
        current_sha: str | None = None
        if target.exists() and target.is_file():
            try:
                current_sha = _compute_sha256(target.read_bytes())
            except Exception:
                current_sha = None

        if expected_sha and current_sha and expected_sha != current_sha and not force:
            conflicts.append({
                "path": rel,
                "error": f"Optimistic lock conflict: expected {expected_sha[:8]}, found {current_sha[:8]}",
                "expected_sha256": expected_sha,
                "current_sha256": current_sha,
                "status": "conflict",
            })

        resolved_ops.append({
            "index": idx,
            "target": target,
            "rel": rel,
            "content": content,
            "expected_sha256": expected_sha,
            "force": force,
        })

    if conflicts:
        return {
            "ok": False,
            "error": "Optimistic concurrency conflicts detected before write execution.",
            "error_code": "WRITE_CONFLICT",
            "conflicts": conflicts,
            "results": conflicts,
            "changed_files": [],
        }

    if check_only:
        return {
            "ok": True,
            "stage": "check_only",
            "results": [{"path": op["rel"], "status": "validated"} for op in resolved_ops],
            "changed_files": [],
        }

    # 2. Capture snapshots for potential rollback
    snapshots = manager._capture_snapshot(targets_to_snapshot) if rollback_on_failure else {}

    results: list[dict[str, Any]] = []
    changed_files: list[str] = []
    temp_files: list[Path] = []
    failed_index: int | None = None

    try:
        # 3. Stage writes into temporary files in target parent directories
        for op in resolved_ops:
            target = op["target"]
            rel = op["rel"]
            content = op["content"]

            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(dir=target.parent, prefix=".mana_batch_", delete=False)
            tmp_path = Path(tmp.name)
            temp_files.append(tmp_path)

            raw_bytes = content.encode("utf-8")
            tmp.write(raw_bytes)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()

            # Atomic rename
            os.replace(tmp_path, target)
            changed_files.append(rel)

            results.append({
                "path": rel,
                "ok": True,
                "status": "success",
                "bytes_written": len(raw_bytes),
                "sha256": _compute_sha256(raw_bytes),
            })

    except Exception as exc:
        logger.exception("Batch write failed during execution")
        failed_index = len(results)
        err_msg = str(exc)

        rollback_performed = False
        restored_files: list[str] = []
        if rollback_on_failure and snapshots:
            restored_files = manager._execute_rollback(snapshots)
            rollback_performed = True

        return {
            "ok": False,
            "error": f"Batch write failed: {err_msg}",
            "error_code": "BATCH_WRITE_FAILED",
            "failed_index": failed_index,
            "rollback_performed": rollback_performed,
            "restored_files": restored_files,
            "results": results,
            "changed_files": [] if rollback_performed else changed_files,
        }
    finally:
        # Clean up any residual temp files
        for tmp_p in temp_files:
            if tmp_p.exists():
                try:
                    tmp_p.unlink(missing_ok=True)
                except OSError:
                    pass

    return {
        "ok": True,
        "results": results,
        "changed_files": sorted(set(changed_files)),
        "rollback_performed": False,
    }


# =============================================================================
# 3. BATCH APPLY PATCHES
# =============================================================================

def batch_apply_patches(
    repo_root: Path,
    *,
    patches: list[dict[str, Any]],
    rollback_on_failure: bool = True,
    check_only: bool = False,
) -> dict[str, Any]:
    """Validate and apply multiple Codex patches with transactional rollback."""
    root = repo_root.resolve()
    patch_rows = list(patches or [])[:50]
    if not patch_rows:
        return {"ok": True, "results": [], "changed_files": []}

    # 1. Dry run validation pass
    validation_results: list[dict[str, Any]] = []
    affected_paths: set[Path] = set()

    for idx, item in enumerate(patch_rows):
        patch_text = str(item.get("patch", "") if isinstance(item, dict) else "")
        v_res = safe_apply_patch(repo_root=root, patch=patch_text, check_only=True)
        validation_results.append({"index": idx, **v_res})
        if not v_res.get("ok"):
            return {
                "ok": False,
                "stage": "validation",
                "error": f"Patch at index {idx} failed validation: {v_res.get('error')}",
                "error_index": idx,
                "results": validation_results,
                "changed_files": [],
            }
        for p in v_res.get("files_changed") or []:
            target, _, _ = _safe_resolve_path(root, str(p))
            if target:
                affected_paths.add(target)

    if check_only:
        return {"ok": True, "stage": "check_only", "results": validation_results, "changed_files": []}

    # 2. Snapshot affected files
    manager = BatchFileSystemManager(root)
    snapshots = manager._capture_snapshot(list(affected_paths)) if rollback_on_failure else {}

    results: list[dict[str, Any]] = []
    changed_files: list[str] = []

    for idx, item in enumerate(patch_rows):
        patch_text = str(item.get("patch", "") if isinstance(item, dict) else "")
        app_id = str(item.get("action_approval_id", "") if isinstance(item, dict) else "")
        res = safe_apply_patch(repo_root=root, patch=patch_text, check_only=False, action_approval_id=app_id)
        results.append({"index": idx, **res})

        for p in res.get("files_changed") or []:
            changed_files.append(str(p))

        if not res.get("ok"):
            rollback_done = False
            restored = []
            if rollback_on_failure and snapshots:
                restored = manager._execute_rollback(snapshots)
                rollback_done = True
            return {
                "ok": False,
                "stage": "apply",
                "error": f"Patch application failed at index {idx}: {res.get('error')}",
                "error_index": idx,
                "rollback_performed": rollback_done,
                "restored_files": restored,
                "results": results,
                "changed_files": [] if rollback_done else sorted(set(changed_files)),
            }

    return {
        "ok": True,
        "results": results,
        "changed_files": sorted(set(changed_files)),
        "rollback_performed": False,
    }


# =============================================================================
# 4. BATCH SEARCH FILES
# =============================================================================

def batch_search_files(
    repo_root: Path,
    *,
    patterns: list[dict[str, Any]],
    concurrency: int = 4,
) -> dict[str, Any]:
    """Execute multiple regex or literal searches concurrently across the repository."""
    root = repo_root.resolve()
    queries = list(patterns or [])[:50]
    if not queries:
        return {"ok": True, "results": [], "errors": []}

    from mana_agent.tools.repository import repo_search

    def _execute_search(item: dict[str, Any], idx: int) -> dict[str, Any]:
        query = str(item.get("query", ""))
        glob_p = str(item.get("glob", "**/*"))
        regex = bool(item.get("regex", False))
        limit = int(item.get("limit", 50))
        res = repo_search(root, query=query, glob=glob_p, regex=regex, limit=limit)
        return {"index": idx, "query": query, "glob": glob_p, "regex": regex, **res}

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    workers = max(1, min(concurrency, len(queries)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_execute_search, item, i): i for i, item in enumerate(queries)}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results.append(row)
            if not row.get("ok"):
                errors.append(row)

    results.sort(key=lambda r: r.get("index", 0))
    return {
        "ok": len(errors) == 0,
        "results": results,
        "errors": errors,
        "total_queries": len(results),
    }


# =============================================================================
# 5. BATCH LIST FILES
# =============================================================================

def batch_list_files(
    repo_root: Path,
    *,
    queries: list[str | dict[str, Any]],
    limit: int = 200,
) -> dict[str, Any]:
    """List repository files matching multiple patterns in a single filesystem traversal."""
    root = repo_root.resolve()
    query_specs: list[dict[str, Any]] = []
    for item in queries or ["**/*"]:
        if isinstance(item, str):
            query_specs.append({"pattern": item, "limit": limit})
        elif isinstance(item, dict):
            query_specs.append({"pattern": item.get("pattern") or item.get("glob") or "**/*", "limit": item.get("limit", limit)})

    # Gather repository files once
    repo_files: list[str] = []
    for path in root.rglob("*"):
        rel_parts = path.relative_to(root).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        if path.is_file():
            repo_files.append(path.relative_to(root).as_posix())
    repo_files.sort()

    grouped_results: list[dict[str, Any]] = []
    for spec in query_specs:
        pat = spec["pattern"]
        lim = max(1, min(int(spec["limit"] or limit), 2000))
        matched = [f for f in repo_files if _matches_pattern(f, pat)]
        grouped_results.append({
            "pattern": pat,
            "files": matched[:lim],
            "total_matches": len(matched),
            "truncated": len(matched) > lim,
        })

    return {
        "ok": True,
        "results": grouped_results,
        "total_patterns": len(grouped_results),
    }


# =============================================================================
# 6. BATCH MOVE FILES
# =============================================================================

def batch_move_files(
    repo_root: Path,
    *,
    moves: list[dict[str, Any]],
    rollback_on_failure: bool = True,
) -> dict[str, Any]:
    """Move/rename multiple files safely with boundary checks and rollback."""
    root = repo_root.resolve()
    move_items = list(moves or [])
    if not move_items:
        return {"ok": True, "results": [], "changed_files": []}

    manager = BatchFileSystemManager(root)
    resolved_moves: list[dict[str, Any]] = []
    affected_targets: list[Path] = []

    for idx, item in enumerate(move_items):
        src_raw = item.get("src") or item.get("source") or ""
        dst_raw = item.get("dst") or item.get("destination") or ""
        overwrite = bool(item.get("overwrite", False))

        src_target, src_rel, src_err = _safe_resolve_path(root, src_raw)
        if src_err or not src_target or not src_rel:
            return {"ok": False, "error": f"Invalid source '{src_raw}': {src_err}", "error_code": src_err or "INVALID_SRC"}
        if not src_target.is_file():
            return {"ok": False, "error": f"Source file does not exist: {src_rel}", "error_code": "FILE_NOT_FOUND"}

        dst_target, dst_rel, dst_err = _safe_resolve_path(root, dst_raw)
        if dst_err or not dst_target or not dst_rel:
            return {"ok": False, "error": f"Invalid destination '{dst_raw}': {dst_err}", "error_code": dst_err or "INVALID_DST"}
        if dst_target.exists() and not overwrite:
            return {"ok": False, "error": f"Destination file already exists: {dst_rel}", "error_code": "DESTINATION_EXISTS"}

        affected_targets.extend([src_target, dst_target])
        resolved_moves.append({
            "index": idx,
            "src": src_target,
            "src_rel": src_rel,
            "dst": dst_target,
            "dst_rel": dst_rel,
            "overwrite": overwrite,
        })

    snapshots = manager._capture_snapshot(affected_targets) if rollback_on_failure else {}

    results: list[dict[str, Any]] = []
    changed_files: list[str] = []

    try:
        for op in resolved_moves:
            src = op["src"]
            dst = op["dst"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            changed_files.extend([op["src_rel"], op["dst_rel"]])
            results.append({
                "source": op["src_rel"],
                "destination": op["dst_rel"],
                "ok": True,
                "status": "moved",
            })
    except Exception as exc:
        rollback_done = False
        restored = []
        if rollback_on_failure and snapshots:
            restored = manager._execute_rollback(snapshots)
            rollback_done = True
        return {
            "ok": False,
            "error": f"Move failed: {exc}",
            "rollback_performed": rollback_done,
            "restored_files": restored,
            "results": results,
            "changed_files": [] if rollback_done else changed_files,
        }

    return {
        "ok": True,
        "results": results,
        "changed_files": sorted(set(changed_files)),
        "rollback_performed": False,
    }


# =============================================================================
# 7. BATCH DELETE FILES
# =============================================================================

def batch_delete_files(
    repo_root: Path,
    *,
    paths: list[str | dict[str, Any]],
    rollback_on_failure: bool = True,
) -> dict[str, Any]:
    """Delete multiple existing repository files safely with rollback support."""
    root = repo_root.resolve()
    items = list(paths or [])
    if not items:
        return {"ok": True, "results": [], "deleted_files": []}

    manager = BatchFileSystemManager(root)
    resolved_deletes: list[tuple[Path, str]] = []
    targets: list[Path] = []

    for item in items:
        raw_p = item.get("path", "") if isinstance(item, dict) else str(item)
        target, rel, err_code = _safe_resolve_path(root, raw_p)
        if err_code or not target or not rel:
            return {"ok": False, "error": f"Invalid path '{raw_p}': {err_code}", "error_code": err_code or "INVALID_PATH"}
        if target.is_dir():
            return {"ok": False, "error": f"Path '{rel}' is a directory; batch_delete_files only removes files", "error_code": "DIRECTORY_NOT_ALLOWED"}
        if target.exists():
            targets.append(target)
            resolved_deletes.append((target, rel))

    snapshots = manager._capture_snapshot(targets) if rollback_on_failure else {}

    results: list[dict[str, Any]] = []
    deleted_files: list[str] = []

    try:
        for target, rel in resolved_deletes:
            target.unlink(missing_ok=True)
            deleted_files.append(rel)
            results.append({"path": rel, "ok": True, "status": "deleted"})
    except Exception as exc:
        rollback_done = False
        restored = []
        if rollback_on_failure and snapshots:
            restored = manager._execute_rollback(snapshots)
            rollback_done = True
        return {
            "ok": False,
            "error": f"Delete failed: {exc}",
            "rollback_performed": rollback_done,
            "restored_files": restored,
            "results": results,
            "deleted_files": [] if rollback_done else deleted_files,
        }

    return {
        "ok": True,
        "results": results,
        "deleted_files": sorted(deleted_files),
        "rollback_performed": False,
    }


__all__ = [
    "BatchFileSystemManager",
    "batch_read_files",
    "batch_write_files",
    "batch_apply_patches",
    "batch_search_files",
    "batch_list_files",
    "batch_move_files",
    "batch_delete_files",
]
