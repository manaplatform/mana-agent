from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import ast
import re
from time import perf_counter
from typing import Any, Literal, Sequence, Optional
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langchain_core.callbacks.base import BaseCallbackHandler
from mana_analyzer.analysis.models import AskResponseWithTrace, SearchHit, ToolInvocationTrace
from mana_analyzer.llm.prompts import ASK_AGENT_SYSTEM_PROMPT
from mana_analyzer.analysis.chunker import CodeChunker
from mana_analyzer.services.structure_service import StructureService
from mana_analyzer.llm.run_logger import LlmRunLogger
from mana_analyzer.config.settings import default_index_dir
from mana_analyzer.services.coding_memory_service import CodingMemoryService
from mana_analyzer.services.search_service import SearchService

logger = logging.getLogger(__name__)


class _SemanticSearchInput(BaseModel):
    query: str = Field(description="Query used for semantic code search")
    k: int = Field(default=8, ge=1, le=50, description="Top results to return")


class _ReadFileInput(BaseModel):
    path: str = Field(description="Absolute or project-relative file path")
    mode: Literal["line", "full"] = Field(default="line")
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=200, ge=1)


class _RunCommandInput(BaseModel):
    cmd: str = Field(description="Shell command to execute in project root")

class _ChunkFileInput(BaseModel):
    path: str = Field(description="Absolute or project-relative file path")

class _ListToolsInput(BaseModel):
    pass

class _LsInput(BaseModel):
    pass


class AskAgent:
    READ_FULL_FILE_MAX_LINES = 5000
    READ_FULL_FILE_MAX_CHARS = 250000

    _BLOCKED_PATTERNS = [
        "rm ",
        "mv ",
        "git reset --hard",
        "git checkout --",
        "sudo ",
        "dd ",
        "mkfs",
        "shutdown",
        "reboot",
        "chmod ",
        "chown ",
        ">",
        ">>",
    ]

    def __init__(
        self,
        api_key: str,
        model: str,
        search_service: SearchService,
        project_root: str | Path,
        base_url: str | None = None,
        coding_memory_service: CodingMemoryService | None = None,
    ) -> None:
        kwargs = {"api_key": api_key, "model": model}
        if base_url:
            kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**kwargs)
        self.model = model
        self.search_service = search_service
        self.project_root = Path(project_root).resolve()
        self.coding_memory_service = coding_memory_service
        self._resolved_index = default_index_dir(self.project_root)
        self._resolved_indexes = [self._resolved_index]
        self.run_logger = LlmRunLogger()

        # ✅ NEW: allow external code to register extra tools (e.g. write_file/apply_patch)
        self.tools: list[BaseTool] = []

    def _is_blocked_command(self, cmd: str) -> bool:
        lowered = f"{cmd.lower()} "
        return any(pattern in lowered for pattern in self._BLOCKED_PATTERNS)

    def _rewrite_python_command(self, cmd: str) -> tuple[str, bool]:
        raw = str(cmd or "").strip()
        if not raw:
            return raw, False
        try:
            parts = shlex.split(raw)
        except Exception:
            return raw, False
        if not parts:
            return raw, False
        head = str(parts[0]).strip()
        if not head:
            return raw, False
        head_name = Path(head).name.lower()
        if head_name != "python":
            return raw, False
        venv_python = (self.project_root / ".venv" / "bin" / "python3").resolve()
        if venv_python.exists() and os.access(venv_python, os.X_OK):
            parts[0] = str(venv_python)
        else:
            parts[0] = "python3"
        rewritten = shlex.join(parts)
        return rewritten, rewritten != raw

    @staticmethod
    def _coerce_tool_payload(content: Any) -> dict[str, Any] | None:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return None
        text = content.strip()
        if not text:
            return None
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
        # Some models/tools may return Python dict repr with single quotes.
        try:
            loaded = ast.literal_eval(text)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            return None
        return None

    @classmethod
    def _is_apply_patch_failure(cls, content: Any) -> bool:
        payload = cls._coerce_tool_payload(content)
        if payload is not None:
            ok = payload.get("ok")
            if ok is False:
                return True
            error = str(payload.get("error", "")).strip()
            if error:
                return True
            touched = payload.get("touched_files")
            if isinstance(touched, list) and not touched and not payload.get("check_only", False):
                return True
            return False
        text = str(content or "")
        lowered = text.lower()
        # Tolerate stringified success payloads that may still include per-attempt
        # "error: ..." details from failed sub-strategies.
        if re.search(r"""['"]ok['"]\s*:\s*true""", lowered):
            return False
        return "error" in lowered or "ok': false" in lowered or '"ok": false' in lowered

    @staticmethod
    def _normalize_search_key(args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip().lower()
        query = re.sub(r"\s+", " ", query)
        path = str(args.get("path", "")).strip().lower()
        k = int(args.get("k", 0) or 0)
        return f"q={query}|p={path}|k={k}"

    @classmethod
    def _search_error_detail(cls, content: Any) -> str:
        payload = cls._coerce_tool_payload(content)
        if payload is not None:
            detail = str(payload.get("error", "")).strip()
            lowered = detail.lower()
            # Suppress noisy env/config-only web-search warnings.
            if "duckduckgo fallback failed" in lowered and "tavily_api_key not set" in lowered:
                return ""
            return detail
        return ""

    @classmethod
    def _extract_model_text(cls, content: Any) -> str:
        """Normalize model message content to user-facing text.

        Newer model responses may return list-based content blocks that include
        non-display items like ``{"type": "reasoning"}``. This extractor keeps
        only text-like blocks.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            value = content.get("text")
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, dict):
                nested_value = value.get("value")
                if isinstance(nested_value, str):
                    return nested_value.strip()
            for key in ("content", "value"):
                nested = content.get(key)
                if nested is not None:
                    extracted = cls._extract_model_text(nested).strip()
                    if extracted:
                        return extracted
            return ""
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    block_type = str(item.get("type", "")).strip().lower()
                    if block_type and block_type not in {"text", "output_text"}:
                        continue
                extracted = cls._extract_model_text(item).strip()
                if extracted:
                    parts.append(extracted)
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _normalize_read_mode(mode: str | None) -> Literal["line", "full"]:
        return "full" if str(mode or "").strip().lower() == "full" else "line"

    def _resolve_read_path(self, path: str) -> Path:
        requested = Path(path)
        resolved = requested if requested.is_absolute() else (self.project_root / requested)
        resolved = resolved.resolve()
        if self.project_root not in resolved.parents and resolved != self.project_root:
            raise ValueError("path is outside project root")
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        return resolved

    @staticmethod
    def _normalize_line_request(start_line: int, end_line: int, line_window: int) -> tuple[int, int]:
        start = max(int(start_line or 1), 1)
        end = max(int(end_line or start), start)
        end = min(end, start + max(200, min(int(line_window or 400), 2000)))
        return start, end

    @staticmethod
    def _read_cache_row_key(row: dict[str, Any]) -> tuple[str, int, int]:
        return (
            str(row.get("mode", "")).strip().lower(),
            int(row.get("start_line", 0) or 0),
            int(row.get("end_line", 0) or 0),
        )

    def _get_persistent_read_cache_rows(self, flow_id: str | None, file_path: str) -> list[dict[str, Any]]:
        resolved_flow = str(flow_id or "").strip()
        service = getattr(self, "coding_memory_service", None)
        if not resolved_flow or service is None:
            return []
        try:
            return service.get_read_cache_rows(resolved_flow, file_path)
        except Exception:
            logger.debug("Failed to load persistent read cache rows", exc_info=True)
            return []

    def _invalidate_read_cache_for_file(
        self,
        *,
        flow_id: str | None,
        file_path: str,
        ephemeral_read_cache: dict[str, list[dict[str, Any]]] | None,
    ) -> None:
        if ephemeral_read_cache is not None:
            ephemeral_read_cache.pop(file_path, None)
        resolved_flow = str(flow_id or "").strip()
        service = getattr(self, "coding_memory_service", None)
        if resolved_flow and service is not None:
            try:
                service.delete_read_cache_for_file(resolved_flow, file_path)
            except Exception:
                logger.debug("Failed to invalidate persistent read cache", exc_info=True)

    def _store_read_cache_row(
        self,
        *,
        flow_id: str | None,
        row: dict[str, Any],
        ephemeral_read_cache: dict[str, list[dict[str, Any]]] | None,
    ) -> None:
        file_path = str(row.get("file_path", "")).strip()
        if not file_path:
            return
        if ephemeral_read_cache is not None:
            existing = list(ephemeral_read_cache.get(file_path, []))
            row_key = self._read_cache_row_key(row)
            existing = [item for item in existing if self._read_cache_row_key(item) != row_key]
            existing.insert(0, dict(row))
            ephemeral_read_cache[file_path] = existing[:20]
        resolved_flow = str(flow_id or "").strip()
        service = getattr(self, "coding_memory_service", None)
        if resolved_flow and service is not None:
            try:
                service.upsert_read_cache_row(
                    flow_id=resolved_flow,
                    file_path=file_path,
                    mode=str(row.get("mode", "line")),
                    start_line=int(row.get("start_line", 1) or 1),
                    end_line=int(row.get("end_line", 0) or 0),
                    line_count=int(row.get("line_count", 0) or 0),
                    content_text=str(row.get("content_text", "")),
                    file_size_bytes=int(row.get("file_size_bytes", 0) or 0),
                    file_mtime_ns=int(row.get("file_mtime_ns", 0) or 0),
                )
                service.prune_read_cache(resolved_flow, keep_per_file=20)
            except Exception:
                logger.debug("Failed to persist read cache row", exc_info=True)

    def _build_cached_read_payload(
        self,
        *,
        row: dict[str, Any],
        requested_mode: Literal["line", "full"],
        start_line: int,
        end_line: int,
        cache_source: Literal["flow_full", "flow_range"],
    ) -> dict[str, Any]:
        file_path = str(row.get("file_path", "")).strip()
        file_line_count = max(0, int(row.get("line_count", 0) or 0))
        row_mode = self._normalize_read_mode(str(row.get("mode", "line")))
        if requested_mode == "full":
            return {
                "file_path": file_path,
                "mode": "full",
                "start_line": 1,
                "end_line": file_line_count,
                "line_count": file_line_count,
                "content": str(row.get("content_text", "")),
                "cache_hit": True,
                "cache_source": cache_source,
                "cache_invalidated": False,
                "full_file_cached": row_mode == "full",
            }

        actual_end = min(max(int(end_line), int(start_line)), file_line_count)
        if row_mode == "full":
            lines = str(row.get("content_text", "")).splitlines()
            segment = lines[int(start_line) - 1 : actual_end]
        else:
            row_start = max(1, int(row.get("start_line", 1) or 1))
            row_end = max(row_start, int(row.get("end_line", row_start) or row_start))
            row_lines = str(row.get("content_text", "")).splitlines()
            slice_start = max(int(start_line), row_start) - row_start
            slice_end = min(actual_end, row_end) - row_start + 1
            segment = row_lines[slice_start:max(slice_start, slice_end)]
        return {
            "file_path": file_path,
            "mode": "line",
            "start_line": int(start_line),
            "end_line": actual_end,
            "line_count": file_line_count,
            "content": "\n".join(segment),
            "cache_hit": True,
            "cache_source": cache_source,
            "cache_invalidated": False,
            "full_file_cached": row_mode == "full",
        }

    def _lookup_read_cache(
        self,
        *,
        resolved: Path,
        requested_mode: Literal["line", "full"],
        start_line: int,
        end_line: int,
        flow_id: str | None,
        ephemeral_read_cache: dict[str, list[dict[str, Any]]] | None,
        invalidate_stale: bool,
    ) -> tuple[dict[str, Any] | None, bool]:
        file_path = str(resolved)
        rows = list((ephemeral_read_cache or {}).get(file_path, []))
        rows.extend(self._get_persistent_read_cache_rows(flow_id, file_path))
        if not rows:
            return None, False
        stat = resolved.stat()
        stale = any(
            int(row.get("file_size_bytes", -1) or -1) != int(stat.st_size)
            or int(row.get("file_mtime_ns", -1) or -1) != int(stat.st_mtime_ns)
            for row in rows
        )
        if stale:
            if invalidate_stale:
                self._invalidate_read_cache_for_file(
                    flow_id=flow_id,
                    file_path=file_path,
                    ephemeral_read_cache=ephemeral_read_cache,
                )
            return None, True

        full_row = next(
            (
                row
                for row in rows
                if self._normalize_read_mode(str(row.get("mode", ""))) == "full"
            ),
            None,
        )
        if full_row is not None:
            return (
                self._build_cached_read_payload(
                    row=full_row,
                    requested_mode=requested_mode,
                    start_line=start_line,
                    end_line=end_line,
                    cache_source="flow_full",
                ),
                False,
            )
        if requested_mode == "line":
            for row in rows:
                if self._normalize_read_mode(str(row.get("mode", ""))) != "line":
                    continue
                row_start = max(1, int(row.get("start_line", 1) or 1))
                row_end = max(row_start, int(row.get("end_line", row_start) or row_start))
                if row_start <= int(start_line) and row_end >= int(end_line):
                    return (
                        self._build_cached_read_payload(
                            row=row,
                            requested_mode="line",
                            start_line=start_line,
                            end_line=end_line,
                            cache_source="flow_range",
                        ),
                        False,
                    )
        return None, False

    def _can_serve_read_from_cache(
        self,
        *,
        path: str,
        mode: str | None,
        start_line: int,
        end_line: int,
        flow_id: str | None,
        ephemeral_read_cache: dict[str, list[dict[str, Any]]] | None,
        line_window: int,
    ) -> bool:
        try:
            resolved = self._resolve_read_path(path)
        except Exception:
            return False
        requested_mode = self._normalize_read_mode(mode)
        normalized_start, normalized_end = self._normalize_line_request(start_line, end_line, line_window)
        payload, _ = self._lookup_read_cache(
            resolved=resolved,
            requested_mode=requested_mode,
            start_line=normalized_start,
            end_line=normalized_end,
            flow_id=flow_id,
            ephemeral_read_cache=ephemeral_read_cache,
            invalidate_stale=False,
        )
        return payload is not None

    def _build_tools(
        self,
        k_default: int,
        timeout_seconds: int,
        read_line_window: int = 400,
        flow_id: str | None = None,
        ephemeral_read_cache: dict[str, list[dict[str, Any]]] | None = None,
        read_telemetry: dict[str, int] | None = None,
    ) -> tuple[list[BaseTool], list[ToolInvocationTrace], list[SearchHit], list[str]]:
        traces: list[ToolInvocationTrace] = []
        sources: list[SearchHit] = []
        warnings: list[str] = []
        safe_read_line_window = max(200, min(int(read_line_window or 400), 2000))
        resolved_indexes = list(getattr(self, "_resolved_indexes", []) or [])
        if not resolved_indexes:
            fallback_index = Path(getattr(self, "_resolved_index", default_index_dir(self.project_root))).resolve()
            resolved_indexes = [fallback_index]
            self._resolved_indexes = resolved_indexes

        def semantic_search(query: str, k: int = k_default) -> str:
            started = perf_counter()
            status = "ok"
            output_preview = ""
            args_summary = f"query={query!r} k={k}"
            try:
                payload: list[dict] = []
                all_hits: list[SearchHit] = []
                for index_dir in resolved_indexes:
                    try:
                        hits = self.search_service.search(index_dir=index_dir, query=query, k=k)
                    except Exception as exc:
                        warning = f"Skipped unusable index {index_dir}: {exc}"
                        warnings.append(warning)
                        payload.append({"index_dir": str(index_dir), "error": str(exc)})
                        continue
                    all_hits.extend(hits)
                    payload.extend([{"index_dir": str(index_dir), **item.to_dict()} for item in hits])
                sources.extend(all_hits)
                encoded = json.dumps(payload)
                output_preview = encoded
                return encoded
            except Exception as exc:
                status = "error"
                output_preview = str(exc)
                return json.dumps({"error": str(exc)})
            finally:
                traces.append(
                    ToolInvocationTrace(
                        tool_name="semantic_search",
                        args_summary=args_summary,
                        duration_ms=(perf_counter() - started) * 1000,
                        status=status,
                        output_preview=output_preview,
                    )
                )

        def read_file(path: str, mode: str = "line", start_line: int = 1, end_line: int = 200) -> str:
            started = perf_counter()
            status = "ok"
            output_preview = ""
            args_summary = f"path={path!r} mode={mode!r} start={start_line} end={end_line}"
            try:
                resolved = self._resolve_read_path(path)
                requested_mode = self._normalize_read_mode(mode)
                start, end = self._normalize_line_request(start_line, end_line, safe_read_line_window)
                cached_payload, invalidated = self._lookup_read_cache(
                    resolved=resolved,
                    requested_mode=requested_mode,
                    start_line=start,
                    end_line=end,
                    flow_id=flow_id,
                    ephemeral_read_cache=ephemeral_read_cache,
                    invalidate_stale=True,
                )
                if invalidated and read_telemetry is not None:
                    read_telemetry["read_cache_invalidations"] = int(read_telemetry.get("read_cache_invalidations", 0)) + 1
                if cached_payload is not None:
                    if read_telemetry is not None:
                        read_telemetry["read_cache_hits"] = int(read_telemetry.get("read_cache_hits", 0)) + 1
                        if requested_mode == "full":
                            read_telemetry["read_full_mode_used"] = int(read_telemetry.get("read_full_mode_used", 0)) + 1
                    encoded = json.dumps(cached_payload)
                    output_preview = encoded
                    return encoded

                content_text = resolved.read_text(encoding="utf-8")
                stat = resolved.stat()
                lines = content_text.splitlines()
                line_count = len(lines)
                char_count = len(content_text)
                if read_telemetry is not None:
                    read_telemetry["read_cache_misses"] = int(read_telemetry.get("read_cache_misses", 0)) + 1

                if requested_mode == "full":
                    if line_count > self.READ_FULL_FILE_MAX_LINES or char_count > self.READ_FULL_FILE_MAX_CHARS:
                        if read_telemetry is not None:
                            read_telemetry["read_full_mode_blocked"] = int(read_telemetry.get("read_full_mode_blocked", 0)) + 1
                        result = {
                            "error": "full mode exceeds safe read caps; use mode='line'",
                            "file_path": str(resolved),
                            "mode": "full",
                            "line_count": line_count,
                            "char_count": char_count,
                            "max_lines": self.READ_FULL_FILE_MAX_LINES,
                            "max_chars": self.READ_FULL_FILE_MAX_CHARS,
                            "cache_invalidated": bool(invalidated),
                        }
                        encoded = json.dumps(result)
                        output_preview = encoded
                        return encoded
                    cache_row = {
                        "file_path": str(resolved),
                        "mode": "full",
                        "start_line": 1,
                        "end_line": line_count,
                        "line_count": line_count,
                        "content_text": content_text,
                        "file_size_bytes": int(stat.st_size),
                        "file_mtime_ns": int(stat.st_mtime_ns),
                    }
                    self._store_read_cache_row(
                        flow_id=flow_id,
                        row=cache_row,
                        ephemeral_read_cache=ephemeral_read_cache,
                    )
                    if read_telemetry is not None:
                        read_telemetry["read_full_mode_used"] = int(read_telemetry.get("read_full_mode_used", 0)) + 1
                    result = {
                        "file_path": str(resolved),
                        "mode": "full",
                        "start_line": 1,
                        "end_line": line_count,
                        "line_count": line_count,
                        "content": content_text,
                        "cache_hit": False,
                        "cache_source": "disk",
                        "cache_invalidated": bool(invalidated),
                        "full_file_cached": True,
                    }
                    encoded = json.dumps(result)
                    output_preview = encoded
                    return encoded

                actual_end = min(end, len(lines))
                segment = lines[start - 1 : end]
                cache_row = {
                    "file_path": str(resolved),
                    "mode": "line",
                    "start_line": start,
                    "end_line": actual_end,
                    "line_count": line_count,
                    "content_text": "\n".join(segment),
                    "file_size_bytes": int(stat.st_size),
                    "file_mtime_ns": int(stat.st_mtime_ns),
                }
                self._store_read_cache_row(
                    flow_id=flow_id,
                    row=cache_row,
                    ephemeral_read_cache=ephemeral_read_cache,
                )
                result = {
                    "file_path": str(resolved),
                    "mode": "line",
                    "start_line": start,
                    "end_line": actual_end,
                    "line_count": line_count,
                    "content": "\n".join(segment),
                    "cache_hit": False,
                    "cache_source": "disk",
                    "cache_invalidated": bool(invalidated),
                    "full_file_cached": False,
                }
                encoded = json.dumps(result)
                output_preview = encoded
                return encoded
            except Exception as exc:
                status = "error"
                output_preview = str(exc)
                return json.dumps({"error": str(exc)})
            finally:
                traces.append(
                    ToolInvocationTrace(
                        tool_name="read_file",
                        args_summary=args_summary,
                        duration_ms=(perf_counter() - started) * 1000,
                        status=status,
                        output_preview=output_preview,
                    )
                )

        def run_command(cmd: str) -> str:
            started = perf_counter()
            status = "ok"
            output_preview = ""
            args_summary = f"cmd={cmd!r}"
            executed_cmd = str(cmd or "")
            rewritten = False
            try:
                if self._is_blocked_command(cmd):
                    raise PermissionError("command blocked by safety policy")
                executed_cmd, rewritten = self._rewrite_python_command(cmd)
                if self._is_blocked_command(executed_cmd):
                    raise PermissionError("command blocked by safety policy")
                shlex.split(executed_cmd)
                completed = subprocess.run(
                    executed_cmd,
                    cwd=self.project_root,
                    shell=True,
                    check=False,
                    timeout=timeout_seconds,
                    capture_output=True,
                    text=True,
                )
                payload = {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[:4000],
                    "stderr": completed.stderr[:4000],
                    "original_cmd": str(cmd or ""),
                    "executed_cmd": executed_cmd,
                    "interpreter_rewritten": bool(rewritten),
                }
                encoded = json.dumps(payload)
                output_preview = json.dumps(
                    {
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "executed_cmd": executed_cmd,
                        "interpreter_rewritten": bool(rewritten),
                    }
                )
                return encoded
            except subprocess.TimeoutExpired:
                status = "timeout"
                output_preview = "command timed out"
                return json.dumps({"error": f"command timed out after {timeout_seconds}s"})
            except Exception as exc:
                status = "error"
                output_preview = str(exc)[:400]
                return json.dumps({"error": str(exc)})
            finally:
                traces.append(
                    ToolInvocationTrace(
                        tool_name="run_command",
                        args_summary=args_summary,
                        duration_ms=(perf_counter() - started) * 1000,
                        status=status,
                        output_preview=output_preview,
                    )
                )

        # chunking and listing helpers
        def chunk_file(path: str) -> str:
            fp = Path(self.project_root / path).resolve()
            text = fp.read_text(encoding="utf-8")
            chunks = CodeChunker()._chunk_text(text)
            return json.dumps({"chunks": chunks})

        def list_tools() -> str:
            names = [tool.name for tool in base_tools + list(self.tools or [])]
            return json.dumps({"tools": names})

        def ls() -> str:
            dirs = StructureService._list_directories(self.project_root)
            return json.dumps({"directories": dirs})

        base_tools: list[BaseTool] = [
            StructuredTool.from_function(
                func=semantic_search,
                name="semantic_search",
                description="Search indexed code semantically and return JSON list of hits.",
                args_schema=_SemanticSearchInput,
            ),
            StructuredTool.from_function(
                func=read_file,
                name="read_file",
                description=(
                    "Read a repository file: call read_file(path, mode='full') first; "
                    "if full mode exceeds caps, call chunk_file(path). "
                    "Use mode='line' for targeted slices."
                ),
                args_schema=_ReadFileInput,
            ),
            StructuredTool.from_function(
                func=run_command,
                name="run_command",
                description="Run a non-destructive shell command in project root and return JSON stdout/stderr.",
                args_schema=_RunCommandInput,
            ),
            StructuredTool.from_function(
                func=chunk_file,
                name="chunk_file",
                description="Chunk a file into text parts when full-mode read is blocked.",
                args_schema=_ChunkFileInput,
            ),
            StructuredTool.from_function(
                func=list_tools,
                name="list_tools",
                description="List available tool names.",
                args_schema=_ListToolsInput,
            ),
            StructuredTool.from_function(
                func=ls,
                name="ls",
                description="List project directories relative to the root.",
                args_schema=_LsInput,
            ),
        ]

        # include externally-registered tools (write_file/apply_patch/etc)
        all_tools = [*base_tools, *list(getattr(self, "tools", []) or [])]
        return all_tools, traces, sources, warnings

    # ✅ NEW: public "ask" API (what your CodingAgent expects)
    def ask(
        self,
        question: str,
        *,
        index_dir: str | Path,
        k: int,
        max_steps: int = 9999999999999,
        timeout_seconds: int = 9999999999999,
        index_dirs: list[str | Path] | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        system_prompt: str | None = None,
        tool_policy: dict[str, Any] | None = None,
        tool_use: bool = True,
        flow_id: str | None = None,
    ) -> AskResponseWithTrace:
        # tool_use kept for compatibility; this AskAgent is tool-based by design.
        return self.run(
            question=question,
            index_dir=index_dir,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            index_dirs=index_dirs,
            callbacks=callbacks,
            system_prompt=system_prompt,
            tool_policy=tool_policy,
            flow_id=flow_id,
        )

    def run(
        self,
        question: str,
        index_dir: str | Path,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        index_dirs: list[str | Path] | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        system_prompt: str | None = None,  # ✅ NEW
        tool_policy: dict[str, Any] | None = None,
        flow_id: str | None = None,
    ) -> AskResponseWithTrace:
        started = perf_counter()

        self._resolved_index = Path(index_dir).resolve()
        if index_dirs:
            self._resolved_indexes = sorted({Path(item).resolve() for item in index_dirs}, key=lambda item: str(item))
        else:
            self._resolved_indexes = [self._resolved_index]

        policy = dict(tool_policy or {})
        allowed_tools = {str(x) for x in (policy.get("allowed_tools") or []) if str(x).strip()}
        search_budget = int(policy.get("search_budget", 0) or 0)
        read_budget = int(policy.get("read_budget", 0) or 0)
        read_line_window = max(200, min(int(policy.get("read_line_window", 400) or 400), 2000))
        require_read_files = int(policy.get("require_read_files", 0) or 0)
        block_internet = bool(policy.get("block_internet", False))
        search_repeat_limit = int(policy.get("search_repeat_limit", 1) or 1)
        search_seen: dict[str, int] = defaultdict(int)
        max_semantic_k = int(policy.get("max_semantic_k", 50) or 50)
        read_telemetry: dict[str, int] = {
            "read_cache_hits": 0,
            "read_cache_misses": 0,
            "read_full_mode_used": 0,
            "read_full_mode_blocked": 0,
            "read_cache_invalidations": 0,
        }
        ephemeral_read_cache: dict[str, list[dict[str, Any]]] = {}

        tools, traces, sources, warnings = self._build_tools(
            k_default=k,
            timeout_seconds=timeout_seconds,
            read_line_window=read_line_window,
            flow_id=flow_id,
            ephemeral_read_cache=ephemeral_read_cache,
            read_telemetry=read_telemetry,
        )
        tool_map = {tool.name: tool for tool in tools}

        bound = self.llm.bind_tools(tools)

        messages = [
            SystemMessage(content=system_prompt or ASK_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        cfg: dict[str, Any] = {"callbacks": list(callbacks) if callbacks else []}
        apply_patch_failures = 0
        forced_patch_fallback = False
        seen_tool_args: dict[tuple[str, str], int] = defaultdict(int)
        tool_counts: dict[str, int] = defaultdict(int)
        unique_read_files: set[str] = set()
        disk_read_count = 0

        final_answer = ""
        for _ in range(max_steps):
            try:
                ai_msg = bound.invoke(messages, config=cfg)
            except TypeError:
                ai_msg = bound.invoke(messages)
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                final_answer = self._extract_model_text(ai_msg.content) or str(ai_msg.content)
                break

            for call in tool_calls:
                name = str(call.get("name", ""))
                args = call.get("args", {}) or {}
                args_key = json.dumps(args, sort_keys=True, default=str)
                tool_sig = (name, args_key)
                seen_tool_args[tool_sig] += 1

                if name not in tool_map:
                    content = json.dumps({"error": f"unknown tool: {name}"})
                elif allowed_tools and name not in allowed_tools:
                    content = json.dumps({"error": f"tool blocked by policy: {name}"})
                elif name != "read_file" and seen_tool_args[tool_sig] > 2:
                    content = json.dumps(
                        {
                            "error": (
                                f"duplicate tool call blocked: {name}. "
                                "Use a different step (read_file/apply_patch/write_file) instead of repeating."
                            )
                        }
                    )
                elif forced_patch_fallback and name == "apply_patch":
                    content = json.dumps(
                        {
                            "ok": False,
                            "error": (
                                "apply_patch disabled after repeated failures/no-op in this run; "
                                "switch to write_file fallback."
                            ),
                        }
                    )
                elif block_internet and name == "search_internet":
                    content = json.dumps({"error": "search_internet blocked by coding-agent repo-only policy"})
                else:
                    if name == "semantic_search":
                        if search_budget > 0 and tool_counts["semantic_search"] >= search_budget:
                            content = json.dumps({"error": "semantic_search budget exhausted"})
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                        k_val = int(args.get("k", 0) or 0)
                        if k_val > max_semantic_k:
                            content = json.dumps({"error": f"semantic_search k must be <= {max_semantic_k}"})
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                        normalized = self._normalize_search_key(args)
                        search_seen[normalized] += 1
                        if search_seen[normalized] > search_repeat_limit:
                            content = json.dumps(
                                {"error": "duplicate semantic_search intent blocked; move to read_file or edit phase"}
                            )
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                    if name == "read_file":
                        read_args = args if isinstance(args, dict) else {}
                        if (
                            read_budget > 0
                            and disk_read_count >= read_budget
                            and not self._can_serve_read_from_cache(
                                path=str(read_args.get("path", "")),
                                mode=str(read_args.get("mode", "line")),
                                start_line=int(read_args.get("start_line", 1) or 1),
                                end_line=int(read_args.get("end_line", 200) or 200),
                                flow_id=flow_id,
                                ephemeral_read_cache=ephemeral_read_cache,
                                line_window=read_line_window,
                            )
                        ):
                            content = json.dumps({"error": "read_file budget exhausted"})
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                    if name in {"apply_patch", "write_file"} and require_read_files > 0:
                        if len(unique_read_files) < require_read_files:
                            content = json.dumps(
                                {
                                    "error": (
                                        f"mutation blocked by policy: inspect at least {require_read_files} unique files first"
                                    )
                                }
                            )
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                    try:
                        try:
                            content = tool_map[name].invoke(args, config=cfg)
                        except TypeError:
                            content = tool_map[name].invoke(args)
                    except Exception as exc:
                        content = json.dumps({"error": str(exc)})

                if name == "apply_patch":
                    if self._is_apply_patch_failure(content):
                        apply_patch_failures += 1
                        if apply_patch_failures >= 2:
                            forced_patch_fallback = True
                            warning = (
                                "apply_patch disabled after repeated failures in this run; "
                                "switching to write_file fallback."
                            )
                            if warning not in warnings:
                                warnings.append(warning)
                tool_counts[name] += 1
                if name == "read_file":
                    payload = self._coerce_tool_payload(content)
                    if payload is not None:
                        file_path = str(payload.get("file_path", "")).strip()
                        if file_path:
                            unique_read_files.add(file_path)
                        if not bool(payload.get("cache_hit", False)) and not str(payload.get("error", "")).strip():
                            disk_read_count += 1
                if name == "search_internet":
                    detail = self._search_error_detail(content)
                    if detail:
                        warning = f"search_internet failed: {detail}"
                        if warning not in warnings:
                            warnings.append(warning)
                messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))

        if not final_answer:
            if forced_patch_fallback:
                final_answer = (
                    "apply_patch was disabled after repeated failures in this run. "
                    "Tool loop reached the step limit before write_file fallback completed."
                )
            else:
                final_answer = "Tool loop reached the step limit before a final answer."

        deduped_sources = sorted(
            {(item.file_path, item.start_line, item.end_line, item.symbol_name): item for item in sources}.values(),
            key=lambda item: (item.file_path, item.start_line, item.end_line, item.symbol_name),
        )

        result = AskResponseWithTrace(
            answer=final_answer,
            sources=deduped_sources,
            mode="agent-tools",
            trace=traces,
            warnings=warnings,
        )
        if require_read_files > 0 and len(unique_read_files) < require_read_files:
            result.warnings.append(
                f"read-file gate not satisfied: {len(unique_read_files)}/{require_read_files} unique files inspected"
            )

        run_logger = getattr(self, "run_logger", None)
        if run_logger is not None:
            run_logger.log(
                {
                    "flow": "ask-agent",
                    "model": getattr(self, "model", "unknown"),
                    "question_chars": len(question),
                    "question": question,
                    "index_dir": str(self._resolved_index),
                    "index_dirs": [str(item) for item in self._resolved_indexes],
                    "k": k,
                    "max_steps": max_steps,
                    "timeout_seconds": timeout_seconds,
                    "flow_id": flow_id,
                    "read_line_window": read_line_window,
                    "read_cache_hits": int(read_telemetry.get("read_cache_hits", 0)),
                    "read_cache_misses": int(read_telemetry.get("read_cache_misses", 0)),
                    "read_full_mode_used": int(read_telemetry.get("read_full_mode_used", 0)),
                    "read_full_mode_blocked": int(read_telemetry.get("read_full_mode_blocked", 0)),
                    "read_cache_invalidations": int(read_telemetry.get("read_cache_invalidations", 0)),
                    "tool_calls": len(traces),
                    "trace": [item.to_dict() for item in traces],
                    "sources_count": len(result.sources),
                    "sources": [item.to_dict() for item in result.sources],
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                    "answer": result.answer,
                }
            )

        return result

    def run_multi(
        self,
        question: str,
        index_dirs: Sequence[str | Path],
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        callbacks: Sequence[Any] | None = None,
        flow_id: str | None = None,
        **kwargs: Any,
    ):
        """
        Dir-mode entrypoint.

        This implementation:
        1) Searches across multiple indexes (SearchService.search_multi).
        2) Chooses the best single index to run the agent loop against (highest top-hit score).
        3) Calls self.run(...) using that chosen index_dir.
        4) Returns the agent result, but keeps the multi-index sources + warnings.
        """
        if getattr(self, "search_service", None) is None:
            raise RuntimeError("AskAgent.search_service is required for run_multi()")
        tool_policy = kwargs.pop("tool_policy", None)

        resolved_indexes = [Path(p).resolve() for p in index_dirs]
        if not resolved_indexes:
            raise RuntimeError("run_multi(): index_dirs is empty")

        if not hasattr(self.search_service, "search_multi"):
            return self.run(
                question=question,
                index_dir=resolved_indexes[0],
                k=k,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                index_dirs=resolved_indexes,
                callbacks=callbacks,
                tool_policy=tool_policy,
                flow_id=flow_id,
                **kwargs,
            )

        # 1) Retrieve across all indexes
        sources, warnings = self.search_service.search_multi(  # type: ignore[attr-defined]
            index_dirs=resolved_indexes,
            query=question,
            k=k,
        )
        presearch_warnings = list(warnings or [])
        if not sources:
            # Do not short-circuit on empty retrieval: greenfield/bootstrap requests
            # can still be satisfied via tool calls (run_command/write_file/apply_patch).
            presearch_warnings.append(
                "No indexed hits found across indexes; continuing with tool loop."
            )

        # 2) Choose best index based on highest top-hit score per index bucket
        # We infer which index a hit belongs to by matching hit.file_path under index_dir.parent
        best_index = resolved_indexes[0]
        best_score = float("-inf")

        # score_by_index: index_dir -> max_score
        score_by_index: dict[Path, float] = defaultdict(lambda: float("-inf"))
        for hit in sources:
            try:
                hit_path = Path(hit.file_path).resolve()
            except Exception:
                continue

            for idx in resolved_indexes:
                subproject_root = idx.parent
                if hit_path == subproject_root or subproject_root in hit_path.parents:
                    try:
                        score = float(getattr(hit, "score", 0.0))
                    except Exception:
                        score = 0.0
                    if score > score_by_index[idx]:
                        score_by_index[idx] = score

        for idx, sc in score_by_index.items():
            if sc > best_score:
                best_score = sc
                best_index = idx

        # 3) Run the normal agent loop against the chosen index
        result = self.run(
            question=question,
            index_dir=best_index,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            index_dirs=resolved_indexes,
            callbacks=callbacks,
            tool_policy=tool_policy,
            flow_id=flow_id,
            **kwargs,
        )

        # 4) Attach multi-index sources + warnings so CLI can show them consistently
        # (works for AskResponseWithTrace dataclass-like objects)
        try:
            if hasattr(result, "sources"):
                existing_sources = list(getattr(result, "sources", []) or [])
                merged_sources = sorted(
                    {
                        (
                            item.file_path,
                            item.start_line,
                            item.end_line,
                            item.symbol_name,
                        ): item
                        for item in [*existing_sources, *list(sources)]
                    }.values(),
                    key=lambda item: (
                        -float(getattr(item, "score", 0.0)),
                        item.file_path,
                        item.start_line,
                        item.end_line,
                        item.symbol_name,
                    ),
                )
                result.sources = merged_sources
            if hasattr(result, "warnings"):
                result.warnings = list(getattr(result, "warnings", []) or [])
                if presearch_warnings:
                    result.warnings.extend(presearch_warnings)
            if hasattr(result, "mode") and getattr(result, "mode", None):
                # keep existing mode
                pass
            elif hasattr(result, "mode"):
                result.mode = "agent-tools-dir"
        except Exception:
            # If it's a string/dict/etc, just return as-is
            pass

        return result
