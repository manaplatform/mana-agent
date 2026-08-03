"""
mana_agent.tools.write_file

A safe file-write tool for coding agents.

Key properties:
- Refuses path traversal / absolute paths.
- Refuses writing outside repo_root.
- Optionally restricts writes to allowed path prefixes (e.g. "src/", "tests/").
- Atomic write (temp file + replace).
- Returns a JSON-serialisable result for tool-calling agents.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# None => allow any path under repo_root
DEFAULT_ALLOWED_PREFIXES: Optional[tuple[str, ...]] = None

# Guidance text injected into tool descriptions to reduce wrong-format failures.
PATCH_FORMAT_GUIDANCE = (
    "NOTE: This tool writes FULL FILE CONTENT, not patches. "
    "For partial edits, use edit_file/multi_edit_file with exact old_string text, "
    "or apply_patch with Codex patch text."
)


@dataclass(frozen=True)
class WriteFileResult:
    ok: bool
    path: str
    bytes_written: int = 0
    sha256: str = ""
    files_changed: list[str] | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["files_changed"] is None:
            data["files_changed"] = []
        return data


@dataclass(frozen=True)
class DeleteFileResult:
    ok: bool
    path: str
    deleted: bool = False
    files_changed: list[str] | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["files_changed"] is None:
            data["files_changed"] = []
        return data


_DRIVE_LETTER_RE = re.compile(r"^[a-zA-Z]:[\\/]")


def _normalise_user_path(path: str) -> str:
    """
    Convert user-supplied paths to a safe-ish posix-like relative path string:
    - Replace backslashes with forward slashes
    - Strip leading "./"
    - Strip leading slashes
    - Collapse redundant separators
    """
    p = path.replace("\\", "/").strip()

    # remove leading "./" repeatedly
    while p.startswith("./"):
        p = p[2:]

    # remove leading "/" repeatedly (treat as attempt at absolute)
    while p.startswith("/"):
        p = p[1:]

    # collapse double slashes
    while "//" in p:
        p = p.replace("//", "/")

    return p


def _normalise_prefixes(prefixes: Optional[Sequence[str]]) -> Optional[tuple[str, ...]]:
    """
    Normalise allowed prefixes to posix style and ensure they behave like directory prefixes.
    - None or empty => no restriction
    - Each prefix becomes:
        - posix slashes
        - stripped leading "./" and leading "/"
        - ensured to end with "/" (unless it's "" which means repo root)
    """
    if not prefixes:
        return None

    out: list[str] = []
    for raw in prefixes:
        p = _normalise_user_path(raw)
        if p and not p.endswith("/"):
            p = p + "/"
        out.append(p)
    return tuple(out)


def _is_allowed_prefix(rel_posix: str, allowed_prefixes: Optional[Sequence[str]]) -> bool:
    """
    Check that rel_posix is within one of allowed prefixes.
    Prefixes are treated as directory prefixes: 'src/' matches 'src/x.py' but not 'src_old/x.py'.
    """
    if not allowed_prefixes:
        return True

    # Ensure rel_posix is normalised with forward slashes and no leading slash.
    rel_posix = _normalise_user_path(rel_posix)

    norm = _normalise_prefixes(allowed_prefixes)
    if not norm:
        return True

    for prefix in norm:
        if prefix == "":
            return True
        if rel_posix == prefix[:-1] or rel_posix.startswith(prefix):
            return True
    return False


def _reject_obvious_bad_paths(original: str) -> Optional[str]:
    """
    Fast checks before any filesystem resolution.
    Returns an error string if blocked, else None.
    """
    if "\x00" in original:
        return "Blocked: NUL byte in path"

    # Block Windows drive letter absolute-ish paths like "C:\foo" or "C:/foo"
    if _DRIVE_LETTER_RE.match(original.strip()):
        return "Blocked: drive-letter paths are not allowed"

    p = original.strip().replace("\\", "/")

    # Absolute path (posix)
    if p.startswith("/"):
        return "Blocked: absolute paths are not allowed"

    # Prevent explicit traversal segments even before resolve()
    # (resolve() + relative_to() also protects, but this gives clearer errors)
    parts = [seg for seg in p.split("/") if seg not in ("", ".")]
    if any(seg == ".." for seg in parts):
        return "Blocked: path traversal ('..') is not allowed"

    return None


def _resolve_target_path(
    *,
    repo_root: Path,
    path: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
) -> tuple[Path | None, str | None, str | None]:
    """
    Resolve and validate user path.
    Returns: (target_absolute_path, rel_posix_path, error_message)
    """
    pre_err = _reject_obvious_bad_paths(path)
    if pre_err:
        return None, None, pre_err

    user_path = Path(path)
    if user_path.is_absolute():
        return None, None, "Blocked: absolute paths are not allowed"

    repo_root = repo_root.resolve()
    normalised = _normalise_user_path(path)
    rel_pp = PurePosixPath(normalised)
    if str(rel_pp) in ("", "."):
        return None, None, "Blocked: empty path"

    target = (repo_root / Path(str(rel_pp))).resolve()
    try:
        rel = target.relative_to(repo_root)
    except ValueError:
        return None, None, "Blocked: path escapes repository root"

    rel_posix = rel.as_posix()
    if not _is_allowed_prefix(rel_posix, allowed_prefixes):
        return None, None, f"Blocked: writes restricted to prefixes {list(_normalise_prefixes(allowed_prefixes) or [])}"

    return target, rel_posix, None


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    finally:
        # If something went wrong before replace, cleanup temp file
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:  # noqa: BLE001
            pass


def _atomic_create_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            target.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        raise


def _parts_dir_for_target(target: Path) -> Path:
    return target.parent / f".{target.name}.parts"


def safe_write_file_part(
    *,
    repo_root: Path,
    path: str,
    content: str,
    part_index: int,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
) -> dict[str, Any]:
    try:
        if part_index < 1:
            return WriteFileResult(ok=False, path=path, error="Blocked: part_index must be >= 1").to_dict()

        target, rel_posix, err = _resolve_target_path(repo_root=repo_root, path=path, allowed_prefixes=allowed_prefixes)
        if err or target is None or rel_posix is None:
            return WriteFileResult(ok=False, path=path, error=err or "Error: invalid path").to_dict()

        parts_dir = _parts_dir_for_target(target)
        part_file = parts_dir / f"{part_index:06d}.part"

        data = content.encode("utf-8")
        from mana_agent.transactional_actions.runtime import execute_file_action
        part_relative = part_file.relative_to(repo_root.resolve()).as_posix()
        gated = execute_file_action(
            workspace_root=repo_root,
            operation="edit" if part_file.exists() else "create",
            path=part_relative,
            content=data,
            originating_agent="write_file",
            parent_task_id="write-file-part",
        )
        if not gated.get("ok"):
            return gated

        logger.info("Wrote file part: %s (part=%06d, %d bytes)", rel_posix, part_index, len(data))
        return WriteFileResult(
            ok=True,
            path=rel_posix,
            bytes_written=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        ).to_dict() | {key: gated[key] for key in ("action_id", "action_state", "verification") if key in gated}
    except Exception as exc:  # noqa: BLE001
        logger.exception("write_file part failed for %s", path)
        return WriteFileResult(ok=False, path=path, error=f"Error: {exc}").to_dict()


def safe_finalize_file_parts(
    *,
    repo_root: Path,
    path: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    cleanup_parts: bool = True,
    expected_sha256: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    try:
        target, rel_posix, err = _resolve_target_path(repo_root=repo_root, path=path, allowed_prefixes=allowed_prefixes)
        if err or target is None or rel_posix is None:
            return WriteFileResult(ok=False, path=path, error=err or "Error: invalid path").to_dict()
        if target.exists() and not force:
            if not expected_sha256:
                return WriteFileResult(
                    ok=False,
                    path=rel_posix,
                    error="Blocked: existing file overwrite requires expected_sha256 or force=true",
                ).to_dict()
            current_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if current_hash != expected_sha256:
                return WriteFileResult(
                    ok=False,
                    path=rel_posix,
                    error="Blocked: existing file hash does not match expected_sha256; re-read before writing",
                ).to_dict()

        parts_dir = _parts_dir_for_target(target)
        if not parts_dir.exists() or not parts_dir.is_dir():
            return WriteFileResult(ok=False, path=path, error="Blocked: no parts directory found to finalize").to_dict()

        part_files = sorted(p for p in parts_dir.iterdir() if p.is_file() and p.name.endswith(".part"))
        if not part_files:
            return WriteFileResult(ok=False, path=path, error="Blocked: no part files found to finalize").to_dict()

        data_chunks: list[bytes] = []
        total = 0
        for part in part_files:
            b = part.read_bytes()
            data_chunks.append(b)
            total += len(b)
        final_data = b"".join(data_chunks)

        from mana_agent.transactional_actions.runtime import execute_file_action
        gated = execute_file_action(
            workspace_root=repo_root,
            operation="edit" if target.exists() else "create",
            path=rel_posix,
            content=final_data,
            originating_agent="write_file",
            parent_task_id="write-file-finalize",
        )
        if not gated.get("ok"):
            return gated

        cleanup_result: dict[str, Any] | None = None
        if cleanup_parts:
            cleanup_result = _cleanup_generated_parts(
                repo_root=repo_root,
                target=target,
                parts_dir=parts_dir,
                part_files=part_files,
                final_sha256=hashlib.sha256(final_data).hexdigest(),
            )
            if not cleanup_result.get("ok"):
                return {
                    **cleanup_result,
                    "final_file_action_id": gated.get("action_id", ""),
                    "final_file_committed": True,
                    "path": rel_posix,
                }

        logger.info("Finalized file from parts: %s (parts=%d, %d bytes)", rel_posix, len(part_files), total)
        payload = WriteFileResult(
            ok=True,
            path=rel_posix,
            bytes_written=total,
            sha256=hashlib.sha256(final_data).hexdigest(),
            files_changed=[rel_posix],
        ).to_dict()
        payload.update({key: gated[key] for key in ("action_id", "action_state", "verification") if key in gated})
        if cleanup_result is not None:
            payload["cleanup_action_id"] = cleanup_result.get("action_id", "")
            payload["cleanup_verification"] = cleanup_result.get("verification", {})
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.exception("write_file finalize failed for %s", path)
        return WriteFileResult(ok=False, path=path, error=f"Error: {exc}").to_dict()


def _cleanup_generated_parts(
    *,
    repo_root: Path,
    target: Path,
    parts_dir: Path,
    part_files: list[Path],
    final_sha256: str,
) -> dict[str, Any]:
    """Remove redundant chunk artifacts through a verified transactional action."""
    from mana_agent.transactional_actions.adapters import ActionAdapter, ActionInvalidatedError
    from mana_agent.transactional_actions.gateway import ApprovalRequired
    from mana_agent.transactional_actions.models import (
        ActionIntent,
        ActionPreview,
        BlastRadius,
        DataDisclosure,
        Reversibility,
        VerificationEvidence,
    )
    from mana_agent.transactional_actions.runtime import default_action_gateway

    root = repo_root.resolve()
    directory = parts_dir.resolve()
    files = [path.resolve() for path in part_files]
    part_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    material = "\x1f".join(
        [str(root), str(target.resolve()), final_sha256]
        + [f"{path}:{digest}" for path, digest in sorted(part_hashes.items())]
    )

    class GeneratedPartsCleanupAdapter(ActionAdapter):
        def build_intent(self) -> ActionIntent:
            return ActionIntent(
                parent_task_id="write-file-finalize",
                actor="model_tool",
                originating_agent="write_file",
                tool_name="file",
                operation_name="cleanup_generated_parts",
                target_resources=[str(path) for path in files] + [str(directory)],
                normalized_arguments={
                    "parts_directory": str(directory),
                    "part_hashes": part_hashes,
                    "final_target": str(target.resolve()),
                    "final_sha256": final_sha256,
                    "aggregate_matches_target": True,
                    "ephemeral_generated": True,
                },
                requested_capabilities=["file.cleanup_generated"],
                expected_side_effects=["remove verified redundant write-file chunks"],
                data_disclosure=DataDisclosure.NONE,
                blast_radius=BlastRadius.MULTIPLE_RESOURCES,
                reversibility=Reversibility.PARTIALLY_REVERSIBLE,
                idempotency_key="chunk-cleanup:" + hashlib.sha256(material.encode("utf-8")).hexdigest(),
                verification_plan=["verify every generated part and its directory are absent"],
                compensation_strategy="Chunks are redundant with the independently verified final file; generic compensation is unavailable.",
            )

        def preview(self, action: ActionIntent) -> ActionPreview:
            return ActionPreview(
                summary="remove verified redundant write-file chunks",
                resources=[
                    {"path": str(path), "change": "delete_generated_part", "before_sha256": part_hashes[str(path)]}
                    for path in files
                ] + [{"path": str(directory), "change": "remove_empty_generated_directory"}],
                exact_invocation=action.normalized_arguments,
                expected_side_effects=action.expected_side_effects,
                risks=["chunk artifacts cannot be reconstructed individually after cleanup"],
                externally_visible=False,
                potentially_billable=False,
            )

        def execute(self, action: ActionIntent) -> dict[str, Any]:
            actual_entries = sorted(path.resolve() for path in directory.iterdir()) if directory.is_dir() else []
            if actual_entries != sorted(files):
                raise ActionInvalidatedError("generated parts directory changed after preview")
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != final_sha256:
                raise ActionInvalidatedError("finalized file no longer matches the verified chunk aggregate")
            for path in files:
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != part_hashes[str(path)]:
                    raise ActionInvalidatedError("a generated part changed after preview")
            for path in files:
                path.unlink()
            directory.rmdir()
            return {"removed_parts": len(files), "parts_directory": str(directory)}

        def verify(self, action: ActionIntent, result: dict[str, Any]) -> VerificationEvidence:
            directory_absent = not directory.exists()
            parts_absent = all(not path.exists() for path in files)
            complete = directory_absent and parts_absent
            return VerificationEvidence(
                complete=complete,
                summary="Generated chunk artifacts were removed." if complete else "Chunk cleanup verification failed.",
                checks=[
                    {"check": "parts_absent", "expected": True, "observed": parts_absent},
                    {"check": "directory_absent", "expected": True, "observed": directory_absent},
                ],
            )

    try:
        outcome = default_action_gateway(root).execute(GeneratedPartsCleanupAdapter())
    except ApprovalRequired as exc:
        return {
            "ok": False,
            "error_code": "approval_required",
            "error": str(exc),
            "permission_required": True,
            "permission_request_id": exc.action.action_id,
            "action_id": exc.action.action_id,
            "preview": exc.action.preview.redacted() if exc.action.preview else {},
            "preview_digest": exc.action.preview_digest,
            "policy_decision": exc.action.policy_decision.model_dump(mode="json") if exc.action.policy_decision else {},
        }
    return {
        "ok": outcome.action.state.value == "committed",
        **outcome.result,
        "action_id": outcome.action.action_id,
        "action_state": outcome.action.state.value,
        "verification": outcome.action.verification.model_dump(mode="json") if outcome.action.verification else {},
    }


def safe_write_file(
    *,
    repo_root: Path,
    path: str,
    content: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    expected_sha256: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    try:
        target, rel_posix, err = _resolve_target_path(repo_root=repo_root, path=path, allowed_prefixes=allowed_prefixes)
        if err or target is None or rel_posix is None:
            return WriteFileResult(ok=False, path=path, error=err or "Error: invalid path").to_dict()
        if target.exists() and not force:
            if not expected_sha256:
                return WriteFileResult(
                    ok=False,
                    path=rel_posix,
                    error="Blocked: existing file overwrite requires expected_sha256 or force=true",
                ).to_dict()
            current_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if current_hash != expected_sha256:
                return WriteFileResult(
                    ok=False,
                    path=rel_posix,
                    error="Blocked: existing file hash does not match expected_sha256; re-read before writing",
                ).to_dict()

        # Friendly footgun detection: users sometimes pass patch payloads to write_file.
        # Don't block (might be intentional), but provide a clear hint if it looks like a patch.
        stripped = content.lstrip()
        if stripped.startswith("diff --git ") or (
            "\n@@ " in stripped and "\n--- " in stripped and "\n+++ " in stripped
        ):
            logger.warning("write_file received content that looks like a diff for %s", rel_posix)

        data = content.encode("utf-8")
        from mana_agent.transactional_actions.runtime import execute_file_action
        gated = execute_file_action(workspace_root=repo_root, operation="edit" if target.exists() else "create", path=rel_posix, content=data)
        if not gated.get("ok"):
            return gated

        result = WriteFileResult(
            ok=True,
            path=rel_posix,
            bytes_written=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            files_changed=[rel_posix],
        )
        logger.info("Wrote file: %s (%d bytes)", rel_posix, result.bytes_written)
        payload = result.to_dict()
        payload.update({key: gated[key] for key in ("action_id", "action_state", "preview_digest", "verification", "duplicate_suppressed") if key in gated})
        return payload

    except Exception as exc:  # noqa: BLE001
        logger.exception("write_file failed for %s", path)
        return WriteFileResult(ok=False, path=path, error=f"Error: {exc}").to_dict()


def safe_create_file(
    *,
    repo_root: Path,
    path: str,
    content: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
) -> dict[str, Any]:
    try:
        target, rel_posix, err = _resolve_target_path(repo_root=repo_root, path=path, allowed_prefixes=allowed_prefixes)
        if err or target is None or rel_posix is None:
            return WriteFileResult(ok=False, path=path, error=err or "Error: invalid path").to_dict()
        if target.exists():
            return WriteFileResult(ok=False, path=rel_posix, error="Blocked: target file already exists").to_dict()

        data = content.encode("utf-8")
        from mana_agent.transactional_actions.runtime import execute_file_action
        gated = execute_file_action(workspace_root=repo_root, operation="create", path=rel_posix, content=data)
        if not gated.get("ok"):
            return gated

        result = WriteFileResult(
            ok=True,
            path=rel_posix,
            bytes_written=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            files_changed=[rel_posix],
        )
        logger.info("Created file: %s (%d bytes)", rel_posix, result.bytes_written)
        payload = result.to_dict()
        payload.update({key: gated[key] for key in ("action_id", "action_state", "preview_digest", "verification", "duplicate_suppressed") if key in gated})
        return payload

    except Exception as exc:  # noqa: BLE001
        logger.exception("create_file failed for %s", path)
        return WriteFileResult(ok=False, path=path, error=f"Error: {exc}").to_dict()


def safe_delete_file(
    *,
    repo_root: Path,
    path: str,
    allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES,
    action_approval_id: str = "",
) -> dict[str, Any]:
    try:
        target, rel_posix, err = _resolve_target_path(repo_root=repo_root, path=path, allowed_prefixes=allowed_prefixes)
        if err or target is None or rel_posix is None:
            return DeleteFileResult(ok=False, path=path, error=err or "Error: invalid path").to_dict()
        if not target.exists():
            return DeleteFileResult(ok=False, path=rel_posix, error="Blocked: target file does not exist").to_dict()
        if not target.is_file():
            return DeleteFileResult(ok=False, path=rel_posix, error="Blocked: target is not a file").to_dict()

        from mana_agent.transactional_actions.runtime import execute_file_action
        gated = execute_file_action(workspace_root=repo_root, operation="delete", path=rel_posix, approval_id=action_approval_id)
        if not gated.get("ok"):
            return gated

        result = DeleteFileResult(ok=True, path=rel_posix, deleted=True, files_changed=[rel_posix])
        logger.info("Deleted file: %s", rel_posix)
        payload = result.to_dict()
        payload.update({key: gated[key] for key in ("action_id", "action_state", "preview_digest", "verification", "duplicate_suppressed") if key in gated})
        return payload

    except Exception as exc:  # noqa: BLE001
        logger.exception("delete_file failed for %s", path)
        return DeleteFileResult(ok=False, path=path, error=f"Error: {exc}").to_dict()


def build_write_file_tool(*, repo_root: Path, allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES):
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore

    def _tool(
        path: str,
        content: str | None = None,
        text: str | None = None,
        body: str | None = None,
        part_index: int | None = None,
        finalize: bool = False,
        cleanup_parts: bool = True,
        expected_sha256: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        # Compatibility: some tool-calling models send `text`/`body` instead of `content`.
        effective_content = content if content is not None else (text if text is not None else body)

        if finalize and part_index is not None:
            return WriteFileResult(
                ok=False,
                path=path,
                error="Error: `finalize` and `part_index` are mutually exclusive",
            ).to_dict()

        if finalize:
            return safe_finalize_file_parts(
                repo_root=repo_root,
                path=path,
                allowed_prefixes=allowed_prefixes,
                cleanup_parts=cleanup_parts,
                expected_sha256=expected_sha256,
                force=force,
            )

        if part_index is not None:
            if effective_content is None:
                return WriteFileResult(
                    ok=False,
                    path=path,
                    error="Error: missing file content (expected `content`, `text`, or `body`). "
                    + PATCH_FORMAT_GUIDANCE,
                ).to_dict()
            return safe_write_file_part(
                repo_root=repo_root,
                path=path,
                content=effective_content,
                part_index=part_index,
                allowed_prefixes=allowed_prefixes,
            )

        if effective_content is None:
            return WriteFileResult(
                ok=False,
                path=path,
                error="Error: missing file content (expected `content`, `text`, or `body`). "
                + PATCH_FORMAT_GUIDANCE,
            ).to_dict()

        return safe_write_file(
            repo_root=repo_root,
            path=path,
            content=effective_content,
            allowed_prefixes=allowed_prefixes,
            expected_sha256=expected_sha256,
            force=force,
        )

    return StructuredTool.from_function(
        func=_tool,
        name="write_file",
        description=(
            "Safely write text content to a file under the repository root. "
            "Refuses absolute paths, path traversal, and paths escaping the repository root. "
            "Refuses to overwrite an existing file unless expected_sha256 matches current content or force=true. "
            "Performs atomic write (temp + replace). "
            "For large files: send chunks with `part_index` (writes to .<filename>.parts/NNNNNN.part), "
            "then call again with `finalize=true` to atomically assemble the final file. "
            + PATCH_FORMAT_GUIDANCE
        ),
    )


def build_create_file_tool(*, repo_root: Path, allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES):
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore

    def _tool(
        path: str,
        content: str | None = None,
        text: str | None = None,
        body: str | None = None,
    ) -> dict[str, Any]:
        # Compatibility: some tool-calling models send `text`/`body` instead of `content`.
        effective_content = content if content is not None else (text if text is not None else body)
        if effective_content is None:
            return WriteFileResult(
                ok=False,
                path=path,
                error="Error: missing file content (expected `content`, `text`, or `body`). "
                + PATCH_FORMAT_GUIDANCE,
            ).to_dict()
        return safe_create_file(
            repo_root=repo_root,
            path=path,
            content=effective_content,
            allowed_prefixes=allowed_prefixes,
        )

    return StructuredTool.from_function(
        func=_tool,
        name="create_file",
        description=(
            "Safely create a new text file under the repository root. "
            "Refuses absolute paths, path traversal, paths escaping the repository root, and existing targets. "
            "Creates parent directories as needed and writes atomically. "
            + PATCH_FORMAT_GUIDANCE
        ),
    )


def build_delete_file_tool(*, repo_root: Path, allowed_prefixes: Optional[Sequence[str]] = DEFAULT_ALLOWED_PREFIXES):
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import StructuredTool  # type: ignore

    def _tool(path: str, action_approval_id: str = "") -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "repo_root": repo_root,
            "path": path,
            "allowed_prefixes": allowed_prefixes,
        }
        if action_approval_id:
            kwargs["action_approval_id"] = action_approval_id
        return safe_delete_file(**kwargs)

    return StructuredTool.from_function(
        func=_tool,
        name="delete_file",
        description=(
            "Safely delete one existing file under the repository root. "
            "Refuses absolute paths, path traversal, paths escaping the repository root, directories, and missing targets."
        ),
    )
