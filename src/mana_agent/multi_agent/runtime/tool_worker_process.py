from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from mana_agent.memory import CodingMemoryService
from mana_agent.services.search_service import SearchService
from mana_agent.multi_agent.runtime.ask_agent import AskAgent
from mana_agent.multi_agent.core.types import ExecutionContext, enrich_event_identity
from mana_agent.multi_agent.runtime.mutation_plan import validate_mutation_plan
from mana_agent.tools import (
    build_apply_patch_tool,
    build_create_file_tool,
    build_delete_file_tool,
    build_edit_file_tool,
    build_multi_edit_file_tool,
    build_write_file_tool,
)
from mana_agent.tools.apply_patch import extract_patch_touched_files
from mana_agent.multi_agent.tools import git_tools
from mana_agent.utils.redaction import redact_json_line, redact_secrets
from mana_agent.utils.tool_results import structured_tool_error_detail
from mana_agent.utils.tool_policy import expand_tool_aliases
from mana_agent.config.settings import Settings, default_tools_logs_dir
from mana_agent.context_cost.governor import ContextCostGovernor

logger = logging.getLogger(__name__)


def _resolve_tools_log_root(req: "ToolRunRequest", repo_root: str | None) -> Path | None:
    """Best-effort resolution of the repo root used to locate ``.mana/tools_logs``.

    Prefers the worker's configured ``repo_root``; falls back to deriving it from
    the request's index directory (``<root>/.mana/index`` -> ``<root>``).
    """
    if repo_root:
        try:
            return Path(repo_root).resolve()
        except Exception:
            pass
    index_dir = req.index_dir or (req.index_dirs[0] if req.index_dirs else None)
    if index_dir:
        try:
            # <root>/.mana/index -> parents[1] is <root>
            return Path(index_dir).resolve().parents[1]
        except Exception:
            return None
    return None


def _write_tools_execution_log(
    *,
    repo_root: str | None,
    req: "ToolRunRequest",
    trace_rows: list[dict[str, Any]],
    ok: bool,
    ok_tools: int,
    ok_mutation_tools: int,
    error_code: str = "",
    error_message: str = "",
) -> None:
    """Persist the full per-tool execution trace for a single run_tools request.

    Appends each tool-execution record to a single per-run JSONL file under
    ``.mana/tools_logs/``, named ``tools_<run_id>.jsonl``. Every run has exactly
    one log file, and all tool executions (each carrying ``flow_id``) for that
    run are appended to it. Logging failures never propagate — tool execution
    must not be affected by log I/O.
    """
    try:
        log_root = _resolve_tools_log_root(req, repo_root)
        if log_root is None:
            return
        logs_dir = default_tools_logs_dir(log_root)
        logs_dir.mkdir(parents=True, exist_ok=True)

        now = time.time()
        run_id = str(req.run_id or "norun")
        safe_run = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)[:64]

        record = {
            "logged_at": now,
            "logged_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "run_id": req.run_id,
            "flow_id": req.flow_id,
            "tool_name": req.tool_name or None,
            "retry_attempt": int(req.retry_attempt or 0),
            "ok": bool(ok),
            "error_code": error_code or "",
            "error_message": error_message or "",
            "tools_total": len(trace_rows),
            "tools_ok": int(ok_tools),
            "mutation_tools_ok": int(ok_mutation_tools),
            "trace": trace_rows,
        }

        # One log file per run; every tool execution for this run is appended.
        from mana_agent.multi_agent.agents.tool_runner_agent import _public_value
        from mana_agent.services.activity_context import current_activity_context
        line = json.dumps(_public_value(record, tuple(getattr(current_activity_context(), "protected_text", ()))), default=str, ensure_ascii=False)
        log_name = f"tools_{safe_run}.jsonl"
        with (logs_dir / log_name).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:  # pragma: no cover - logging must never break a run
        logger.debug("[tools_logs] failed to write execution log: %s", exc)

def _configure_worker_logging() -> None:
    """Install fallback logging when this module is run as a worker process."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(name)s [%(funcName)s:%(lineno)d] %(message)s'
    )

# ---------------------------------------------------------------------------
# Keys that must NEVER be sent to provider APIs as message/input parameters.
# ---------------------------------------------------------------------------





_NON_RETRIABLE_HTTP_STATUS_CODES = frozenset({400, 401, 403, 404, 405, 409, 410, 413, 414, 422})
_RETRIABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _redact_debug_line(line: str) -> str:
    """Backward-compatible wrapper around the shared redaction helper."""
    return redact_json_line(line)


def _extract_http_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    msg = str(exc)
    patterns = (
        r"\berror code:\s*(\d{3})\b",
        r"\bstatus(?:_code)?\D+(\d{3})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, msg, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def _is_likely_retriable_runtime_error(exc: BaseException) -> bool:
    status_code = _extract_http_status_code(exc)
    if status_code in _RETRIABLE_HTTP_STATUS_CODES:
        return True
    if status_code in _NON_RETRIABLE_HTTP_STATUS_CODES:
        return False

    msg = str(exc).lower()
    retriable_markers = (
        "rate limit",
        "timeout",
        "timed out",
        "temporar",
        "overloaded",
        "service unavailable",
        "connection reset",
        "connection aborted",
        "network",
    )
    non_retriable_markers = (
        "invalid request",
        "invalid parameter",
        "validation",
        "malformed",
        "parse error",
        "unsupported",
        "not implemented",
    )

    if any(marker in msg for marker in non_retriable_markers):
        return False
    if any(marker in msg for marker in retriable_markers):
        return True
    return False




def _tool_arg_error(tool_name: str, args: dict[str, Any]) -> str:
    """Return a validation error for direct worker tool requests, else empty."""
    name = str(tool_name or "").strip().lower()
    if not name:
        return ""
    if not isinstance(args, dict):
        return f"{name} arguments must be an object"

    def _text(key: str) -> str:
        return str(args.get(key) or "").strip()

    if name in {"write_file", "create_file"}:
        if not _text("path"):
            return f"{name} requires `path`"
        if Path(_text("path")).suffix.lower() in {".docx", ".pdf", ".xlsx", ".xlsm"}:
            return f"{name} cannot write binary document targets; use document_create or document_update"
        has_content = any(args.get(key) is not None for key in ("content", "text", "body"))
        if name == "write_file" and bool(args.get("finalize")):
            return ""
        if name == "write_file" and args.get("part_index") is not None and not has_content:
            return "write_file with `part_index` requires `content`, `text`, or `body`"
        if not has_content:
            return f"{name} requires `content`, `text`, or `body`"
        return ""
    if name == "delete_file":
        if not _text("path"):
            return "delete_file requires `path`"
        return ""
    if name == "edit_file":
        if not _text("path"):
            return "edit_file requires `path`"
        if args.get("old_string") is None:
            return "edit_file requires `old_string`"
        if args.get("new_string") is None:
            return "edit_file requires `new_string`"
        return ""
    if name == "multi_edit_file":
        if not _text("path"):
            return "multi_edit_file requires `path`"
        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            return "multi_edit_file requires non-empty `edits`"
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict) or edit.get("old_string") is None or edit.get("new_string") is None:
                return f"multi_edit_file edit {index} requires `old_string` and `new_string`"
        return ""
    if name == "apply_patch":
        patch_payload = args.get("patch", args.get("diff", args.get("input")))
        touched = extract_patch_touched_files(patch_payload)
        if not bool(touched.get("ok")):
            return str(touched.get("error") or "apply_patch requires a valid patch")
        if not touched.get("touched_files"):
            return "apply_patch requires at least one touched file"
        return ""
    if name.startswith("git.") or name.startswith("git_"):
        canonical = name.replace("_", ".")
        if canonical == "git.generic":
            raw_args = args.get("args")
            if not isinstance(raw_args, list) or not raw_args:
                return "git.generic requires non-empty list `args`"
            if raw_args[0] == "git":
                return "git.generic args must omit the leading git executable"
        if canonical == "git.add" and not isinstance(args.get("paths"), list):
            return "git.add requires list `paths`"
        if canonical == "git.commit" and not str(args.get("message") or "").strip():
            return "git.commit requires `message`"
        if canonical in {"git.switch", "git.create.branch", "git.create_branch"} and not str(args.get("branch_name") or "").strip():
            return f"{name} requires `branch_name`"
        if canonical == "git.checkout" and not str(args.get("target") or "").strip():
            return "git.checkout requires `target`"
        if canonical in {"git.merge"} and not str(args.get("target") or "").strip():
            return f"{name} requires `target`"
        if canonical == "git.revert" and not str(args.get("revision") or "").strip():
            return "git.revert requires `revision`"
        try:
            if canonical == "git.generic":
                risk = git_tools.classify_git_risk([str(item) for item in args["args"]]).value
            else:
                risk = "schema_validated"
        except Exception as exc:
            return str(exc)
        if risk in {"DESTRUCTIVE", "HISTORY_REWRITE"} and not bool(args.get("allow_protected")):
            return f"{name} is protected and requires explicit allow_protected"
        return ""
    return ""


def _validate_direct_tool_request(req: "ToolRunRequest", *, repo_root: str | None = None) -> None:
    error = _tool_arg_error(req.tool_name, req.tool_args)
    if error:
        raise ToolWorkerProcessError(
            code="invalid_tool_args",
            message=error,
            retriable=False,
            details={"tool_name": req.tool_name, "tool_args": req.tool_args},
        )
    name = str(req.tool_name or "").strip().lower()
    if name in {"write_file", "apply_patch", "create_file"}:
        policy = req.tool_policy or {}
        if policy.get("fallback_decision") is True:
            return
        plan = req.tool_args.get("mutation_plan") if isinstance(req.tool_args, dict) else None
        plan_id = str(req.tool_args.get("mutation_plan_id") or "") if isinstance(req.tool_args, dict) else ""
        root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        errors = validate_mutation_plan(plan, repo_root=root) if isinstance(plan, dict) else ["missing approved mutation plan"]
        if errors or not plan_id or (isinstance(plan, dict) and str(plan.get("plan_id") or "") != plan_id):
            raise ToolWorkerProcessError(
                code="mutation_plan_required",
                message="mutation tool requires an approved MutationPlan linked by mutation_plan_id",
                retriable=False,
                details={"tool_name": req.tool_name, "errors": errors or ["mutation_plan_id mismatch"]},
            )








def _sanitize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    clean_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        call_id = call.get("id")
        if call_id is None:
            call_id = "call_generated"
        fn = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
        fn_name = str(fn.get("name") or "tool")
        arguments = fn.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        clean_calls.append(
            {
                "id": str(call_id),
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": arguments,
                },
            }
        )
    return clean_calls








# ---------------------------------------------------------------------------
# Native serialization preserves validated JSON state
# ---------------------------------------------------------------------------












def _infer_trace_row_success(row: dict[str, Any]) -> bool:
    """Determine whether a trace row represents a successful tool execution."""
    logger.debug(f"[_infer_trace_row_success] Evaluating row with keys: {list(row.keys())}")
    logger.debug(f"[_infer_trace_row_success] Full row content: {json.dumps(row, default=str, ensure_ascii=False)[:500]}")
    non_progress_statuses = {
        "blocked",
        "skipped",
        "duplicate_blocked",
        "not_allowed",
        "verify_project_blocked_until_mutation",
        "no_progress",
        "skipped_no_progress",
    }
    
    # Check for explicit status field (legacy support before stripping)
    status_val = row.get("status")
    if status_val is not None:
        logger.debug(f"[_infer_trace_row_success] Found status field: {status_val}")
        normalized_status = str(status_val).strip().lower()
        if normalized_status in non_progress_statuses:
            logger.debug("[_infer_trace_row_success] → FAILURE (non-progress status)")
            return False
        if normalized_status in ("ok", "success"):
            tool_name = str(row.get("tool_name") or row.get("tool") or row.get("name") or "").strip().lower()
            if tool_name in _MUTATION_TOOL_NAMES:
                changed = []
                for key in ("files_changed", "changed_files", "modified_files"):
                    value = row.get(key)
                    if isinstance(value, list):
                        changed.extend(str(item).strip() for item in value if str(item).strip())
                proof = row.get("proof")
                if isinstance(proof, dict) and isinstance(proof.get("modified_files"), list):
                    changed.extend(str(item).strip() for item in proof["modified_files"] if str(item).strip())
                if not changed:
                    logger.debug("[_infer_trace_row_success] → FAILURE (mutation status ok but no changed files)")
                    return False
            logger.debug("[_infer_trace_row_success] → SUCCESS (explicit status=ok/success)")
            return True
        elif normalized_status in ("error", "failed"):
            logger.debug("[_infer_trace_row_success] → FAILURE (explicit status=error/failed)")
            return False
    result_val = row.get("result")
    if isinstance(result_val, str) and result_val.strip().lower() in non_progress_statuses:
        logger.debug("[_infer_trace_row_success] → FAILURE (non-progress result)")
        return False

    # Explicit error field → failure
    error_val = row.get("error")
    if error_val:
        if isinstance(error_val, str) and error_val.strip():
            logger.debug(f"[_infer_trace_row_success] → FAILURE (error field: {error_val[:100]})")
            return False
        elif isinstance(error_val, (dict, list)) and error_val:
            logger.debug("[_infer_trace_row_success] → FAILURE (error field is non-empty dict/list)")
            return False
        elif error_val is True:
            logger.debug("[_infer_trace_row_success] → FAILURE (error=True)")
            return False

    # Has a result / output → success
    success_keys = ("result", "output", "content", "response", "data", "return_value", "tool_output", "observation")
    for key in success_keys:
        val = row.get(key)
        if val is not None and val != "" and val != [] and val != {}:
            logger.debug(f"[_infer_trace_row_success] → SUCCESS (found {key}={str(val)[:100]})")
            return True

    # Has a tool identifier → the tool was at least invoked
    tool_id_keys = ("tool_name", "tool", "name", "action", "function_name", "tool_call")
    for key in tool_id_keys:
        tool_id = row.get(key)
        if tool_id:
            logger.debug(f"[_infer_trace_row_success] → SUCCESS (found tool identifier: {key}={tool_id})")
            return True

    # Check for explicit success indicators
    if row.get("success") is True:
        logger.debug("[_infer_trace_row_success] → SUCCESS (success=True)")
        return True
    if row.get("completed") is True:
        logger.debug("[_infer_trace_row_success] → SUCCESS (completed=True)")
        return True
    if row.get("ok") is True:
        logger.debug("[_infer_trace_row_success] → SUCCESS (ok=True)")
        return True

    # Check for tool call ID (indicates tool was executed)
    id_keys = ("tool_call_id", "call_id", "id", "execution_id")
    for key in id_keys:
        if row.get(key):
            logger.debug(f"[_infer_trace_row_success] → SUCCESS (found ID: {key})")
            return True

    # Check for type field indicating tool message
    row_type = row.get("type", "")
    if row_type in ("tool", "tool_result", "function", "function_result", "tool_message"):
        logger.debug(f"[_infer_trace_row_success] → SUCCESS (type={row_type})")
        return True

    # Check for role field indicating tool message
    row_role = row.get("role", "")
    if row_role in ("tool", "function"):
        logger.debug(f"[_infer_trace_row_success] → SUCCESS (role={row_role})")
        return True

    # Fallback → unknown, treat as failure
    logger.debug("[_infer_trace_row_success] → FAILURE (no success indicators found)")
    return False


def _infer_trace_row_mutation_success(row: dict[str, Any]) -> bool:
    tool_name = str(row.get("tool_name") or row.get("tool") or row.get("name") or "").strip().lower()
    if tool_name not in _MUTATION_TOOL_NAMES:
        return False
    if not _infer_trace_row_success(row):
        return False
    changed: list[str] = []
    for key in ("files_changed", "changed_files", "modified_files"):
        value = row.get(key)
        if isinstance(value, list):
            changed.extend(str(item).strip() for item in value if str(item).strip())
    proof = row.get("proof")
    if isinstance(proof, dict):
        for key in ("modified_files", "changed_files"):
            value = proof.get(key)
            if isinstance(value, list):
                changed.extend(str(item).strip() for item in value if str(item).strip())
    return bool(changed)


def _mutation_failure_error(trace_rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Classify a mutation-strict trace that did not produce changed files."""
    mutation_rows = [
        row for row in trace_rows
        if str(row.get("tool_name") or row.get("tool") or row.get("name") or "").strip().lower()
        in _MUTATION_TOOL_NAMES
    ]
    if not mutation_rows:
        return "mutation_not_attempted", "mutation phase ended without attempting a mutation tool"
    details: list[str] = []
    for row in mutation_rows:
        tool = str(row.get("tool_name") or row.get("tool") or row.get("name") or "mutation_tool")
        detail = str(row.get("error") or row.get("output_preview") or row.get("result") or row.get("status") or "")
        details.append(f"{tool}: {detail}" if detail else tool)
    return "mutation_failed", "mutation tool attempted but no file changes were recorded: " + "; ".join(details[:3])


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class WorkerInitPayload(BaseModel):
    api_key: str
    model: str
    session_id: str
    base_url: str | None = None
    provider: str | None = None
    default_headers: dict[str, str] = Field(default_factory=dict)
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    project_root: str
    repo_root: str
    allowed_prefixes: list[str] | None = None
    tools_only_strict: bool = True
    workspace_id: str | None = None
    repository_id: str | None = None


class ToolRunRequest(BaseModel):
    question: str
    index_dir: str | None = None
    index_dirs: list[str] | None = None
    flow_id: str | None = None
    run_id: str | None = None
    k: int = 8
    max_steps: int = 6
    timeout_seconds: int = 30
    tool_policy: dict[str, Any] | None = None
    system_prompt: str | None = None
    tools_only_strict_override: bool | None = None
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    retry_attempt: int = 0
    tool_call_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    execution_context: dict[str, Any] = Field(default_factory=dict)
    workspace_id: str | None = None
    repository_id: str | None = None


class ToolRunResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "agent-tools"
    trace: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ToolExecutorProtocol(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run_tools(
        self,
        request: ToolRunRequest,
        on_event: Callable[[Any], None] | None = None,
    ) -> ToolRunResponse: ...


class WorkerError(BaseModel):
    code: str
    message: str
    retriable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class WorkerEvent(BaseModel):
    name: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class WorkerEnvelope(BaseModel):
    type: Literal["init", "run_tools", "execute_operation", "health", "shutdown", "update_model"]
    request_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkerReply(BaseModel):
    type: Literal["ok", "error", "event"]
    request_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolWorkerProcessError(PermissionError, RuntimeError):
    def __init__(self, *, code: str, message: str, retriable: bool = False, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.retriable = retriable
        self.details = details or {}


_CLIENT_RETRYABLE_CODES = frozenset({"worker_io_error", "worker_dead", "model_not_found", "init_failed", "worker_startup_failed"})
_MUTATION_TOOL_NAMES = frozenset(
    {
        "edit_file",
        "multi_edit_file",
        "apply_patch",
        "apply_patch_batch",
        "write_file",
        "create_file",
        "delete_file",
        "document_create",
        "document_update",
        "document_delete",
    }
)


def _should_retry_run_tools_exception(exc: BaseException) -> bool:
    if isinstance(exc, IOError):
        return True
    if isinstance(exc, ToolWorkerProcessError):
        if exc.code == "tools_only_violation":
            return False
        if exc.retriable:
            return True
        return exc.code in _CLIENT_RETRYABLE_CODES
    return False


# ---------------------------------------------------------------------------
# ToolWorkerClient
# ---------------------------------------------------------------------------

class ToolWorkerClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        session_id: str = "",
        repo_root: Path,
        project_root: Path,
        base_url: str | None = None,
        allowed_prefixes: list[str] | None = None,
        tools_only_strict: bool = True,
        model_level: str = "",
        workspace_id: str | None = None,
        repository_id: str | None = None,
        provider: str | None = None,
        default_headers: dict[str, str] | None = None,
        model_configuration: dict[str, Any] | None = None,
    ) -> None:
        import uuid
        from mana_agent.config.user_config import get_setting, load_effective_settings

        resolved_base_url = str(base_url or "").strip() or None
        if not resolved_base_url:
            try:
                from mana_agent.config.inference_provider import credentials_from_mapping

                values = load_effective_settings(include_env=False)
                provider = provider or str(values.get("MANA_AI_PROVIDER") or "openai")
                _key, mapped_base = credentials_from_mapping(values, provider=provider)
                resolved_base_url = mapped_base or None
            except Exception:
                resolved_base_url = str(get_setting("OPENAI_BASE_URL", "") or "").strip() or None
        effective_session_id = str(session_id or "").strip() or f"sess-{uuid.uuid4().hex}"
        self._init_payload = WorkerInitPayload(
            api_key=api_key,
            model=model,
            session_id=effective_session_id,
            base_url=resolved_base_url,
            provider=provider,
            default_headers=default_headers or {},
            model_configuration=model_configuration or {},
            project_root=str(project_root.resolve()),
            repo_root=str(repo_root.resolve()),
            allowed_prefixes=allowed_prefixes,
            tools_only_strict=tools_only_strict,
            workspace_id=workspace_id,
            repository_id=repository_id,
        )
        self._model_level = str(model_level or "").strip()
        self._operation_lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        logger.info(
            "[ToolWorkerClient] Initialized with model=%s, base_url=%s, tools_only_strict=%s",
            model,
            resolved_base_url or "<default>",
            tools_only_strict,
        )

    def init_payload_dict(self) -> dict[str, Any]:
        return self._init_payload.model_dump()

    def update_model(self, model_name: str) -> None:
        logger.info(f"[ToolWorkerClient] Updating model to {model_name}")
        self._init_payload.model = model_name
        if self._proc and self._proc.poll() is None:
            try:
                self._request("update_model", {"model": model_name}, expect_event=False)
                logger.info(f"[ToolWorkerClient] Worker model updated to {model_name} in-place.")
            except Exception as e:
                logger.warning(f"[ToolWorkerClient] Failed to update worker model in-place: {e}, restarting worker.")
                self._restart()

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            logger.debug("[ToolWorkerClient] Worker already running")
            return
        
        logger.info("[ToolWorkerClient] Starting worker process...")
        env = os.environ.copy()
        # Runtime credentials and model choices are transferred in the validated
        # init payload. Do not let shell or repository dotenv-derived values
        # override ~/.mana configuration inside the child process.
        from mana_agent.config.user_config import DEFAULT_USER_CONFIG, SECRET_KEYS

        for key in set(DEFAULT_USER_CONFIG) | SECRET_KEYS:
            env.pop(key, None)
        repo_root = Path(self._init_payload.repo_root).resolve()
        pythonpath_entries: list[str] = []
        src_dir = repo_root / "src"
        if src_dir.exists():
            pythonpath_entries.append(str(src_dir))
        pythonpath_entries.append(str(repo_root))
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        
        logger.debug(f"[ToolWorkerClient] PYTHONPATH: {env['PYTHONPATH']}")
        logger.debug(f"[ToolWorkerClient] Python executable: {sys.executable}")
        
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "mana_agent.multi_agent.runtime.tool_worker_process"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=os.name == "posix",
        )
        
        # Start a thread to log stderr
        def log_stderr():
            try:
                if self._proc and self._proc.stderr:
                    for line in self._proc.stderr:
                        line = line.rstrip()
                        if line:
                            from mana_agent.chat.activity import safe_text
                            logger.debug("[Worker STDERR] %s", safe_text(line))
            except Exception as e:
                logger.debug(f"[Worker STDERR] Thread error: {e}")
        
        self._stderr_thread = threading.Thread(target=log_stderr, daemon=True)
        self._stderr_thread.start()
        
        # Give worker a moment to start
        time.sleep(0.1)
        
        # Check if process died immediately
        if self._proc.poll() is not None:
            # Process died, read stderr
            stderr_output = ""
            if self._proc.stderr:
                try:
                    stderr_output = self._proc.stderr.read()
                except Exception:
                    pass
            logger.error(f"[ToolWorkerClient] Worker died immediately. Exit code: {self._proc.returncode}")
            logger.error(f"[ToolWorkerClient] Worker stderr: {stderr_output}")
            raise ToolWorkerProcessError(
                code="worker_startup_failed",
                message=f"Worker process died on startup: {stderr_output[:500]}",
                retriable=True,
            )
        
        logger.info("[ToolWorkerClient] Worker process started, sending init...")
        self._request("init", self._init_payload.model_dump(), expect_event=True)
        logger.info("[ToolWorkerClient] Worker initialized successfully")

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        logger.info("[ToolWorkerClient] Stopping worker process...")
        if proc.poll() is None:
            try:
                self._request_with_proc(proc, "shutdown", {}, expect_event=False)
            except Exception as e:
                logger.debug(f"[ToolWorkerClient] Shutdown request failed: {e}")
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        logger.info("[ToolWorkerClient] Worker process stopped")

    def health(self) -> dict[str, Any]:
        self.start()
        return self._request("health", {}, expect_event=False)

    def execute_operation(self, request):
        """One exact operation over IPC. A lost reply is never automatically retried."""
        from mana_agent.multi_agent.agents.tool_runner_agent import current_operation
        operation = current_operation()
        if operation is None:
            raise PermissionError("Worker IPC requires an active ToolRunner operation")
        if request != operation.request:
            raise PermissionError("Worker IPC request differs from its active execution authority")
        with self._operation_lock:
            self.start()
            done = threading.Event()
            def monitor():
                while not done.wait(.05):
                    try:
                        operation.validate_live()
                    except Exception:
                        # A graceful shutdown RPC would race the operation reader
                        # and block behind the operation it is meant to cancel.
                        proc = self._proc
                        if proc is not None:
                            try:
                                if os.name == "posix":
                                    import signal
                                    os.killpg(proc.pid, signal.SIGKILL)
                                else:
                                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
                            except (OSError, ProcessLookupError):
                                pass
                        return
            threading.Thread(target=monitor, daemon=True).start()
            observed_children = {}
            def observe(event):
                if event.name == "operation.observation":
                    from mana_agent.coding.models import AgentEvent
                    from mana_agent.multi_agent.agents.tool_runner_agent import RunningOperation
                    observed = AgentEvent.model_validate(event.data["event"])
                    if not observed.operation_id or observed.operation_id == request.operation_id:
                        return
                    child = observed_children.get(observed.operation_id)
                    if child is None:
                        child_request = request.model_copy(update={"operation_id": observed.operation_id,
                            "operation_attempt_id": observed.operation_attempt_id or observed.operation_id,
                            "parent_execution_id": request.operation_id})
                        child = observed_children[observed.operation_id] = RunningOperation(child_request, operation.authority, operation.runner)
                        operation.public.setdefault("child_operation_ids", []).append(observed.operation_id)
                    child.public.update({key: observed.payload[key] for key in
                        ("command", "display_argv", "cwd", "streaming", "provenance") if key in observed.payload})
                    with child.scope():
                        if observed.event_type in {"tool.stdout", "tool.stderr"}:
                            child.retain(observed.payload["stream_channel"], observed.output_preview)
                        else:
                            child.emit(observed.event_type, status=observed.status,
                                **{key: observed.payload[key] for key in ("exit_code", "termination_reason") if key in observed.payload})
                    return
                if event.name != "operation.output":
                    return
                channel = event.data.get("stream_channel")
                if channel in {"stdout", "stderr"}:
                    operation.retain(channel, str(event.data.get("output_preview") or ""))
            try:
                payload = self._request("execute_operation", request.model_dump(mode="json"), expect_event=False, on_event=observe)
                return payload["result"]
            except Exception as exc:
                class UncertainToolOutcome(RuntimeError):
                    outcome_unknown = True
                raise UncertainToolOutcome("Tool worker reply lost or interrupted; reconcile the operation receipt before retry") from exc
            finally:
                done.set()

    def run_tools(self, request: ToolRunRequest, *, on_event=None) -> ToolRunResponse:
        """Legacy planning is CodingAgent work in the parent, not worker autonomy."""
        if request.repository_id and self._init_payload.repository_id and request.repository_id != self._init_payload.repository_id:
            raise ToolWorkerProcessError(
                code="repository_scope_mismatch",
                message="tool request repository does not match worker repository boundary",
                retriable=False,
            )
        if request.workspace_id and self._init_payload.workspace_id and request.workspace_id != self._init_payload.workspace_id:
            raise ToolWorkerProcessError(
                code="workspace_scope_mismatch",
                message="tool request workspace does not match worker workspace boundary",
                retriable=False,
            )
        _validate_direct_tool_request(request)
        if request.tool_name:
            from mana_agent.coding.programmatic import ToolExecutionDispatcher
            from mana_agent.coding.models import WorkspaceContext
            from mana_agent.services.activity_context import current_activity_context
            context = current_activity_context()
            task_id = str(getattr(context, "task_id", "") or request.run_id or "")
            key = (task_id, str(getattr(context, "attempt_id", "")), int(getattr(context, "attempt_generation", 0)))
            if not hasattr(self, "_direct_dispatchers"):
                self._direct_dispatchers = {}
            dispatcher = self._direct_dispatchers.get(key)
            if dispatcher is None:
                from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
                from mana_agent.multi_agent.core.types import AgentRole
                registry = getattr(context, "execution_registry", None) or AgentRegistry()
                dispatcher = ToolExecutionDispatcher(WorkspaceContext(repository_path=Path(self._init_payload.repo_root),
                    worktree_path=Path(self._init_payload.repo_root), allow_in_place_write=True,
                    sandbox="readOnly" if str(getattr(context, "execution_mode", "")) in {"plan", "plan_only", "read_only"} else "workspaceWrite"), task_id=task_id)
                dispatcher.bind_runtime(registry=registry, actor_id=registry.find_by_role(AgentRole.CODING).agent_id,
                    task_id=task_id, authorize=getattr(context, "authorize_tool", None))
                self._direct_dispatchers[key] = dispatcher
            name = request.tool_name.replace(".", "_")
            cid = request.tool_call_id
            selected = dispatcher.runner.request(name, request.tool_args, call_id=cid, decision_id=cid,
                timeout_seconds=request.timeout_seconds)
            result = dispatcher.runner.execute(selected, lambda args: self.execute_operation(selected)).result
            return ToolRunResponse(answer=json.dumps(result, ensure_ascii=False, default=str),
                trace=[{"tool_name": name, "result": result, "status": "ok" if result.get("ok", True) else "error"}])
        parent_agent = getattr(self, "_parent_coding_agent", None)
        if parent_agent is None:
            parent_agent = self._parent_coding_agent = _build_worker_ask_agent(self._init_payload)
        parent_agent._operation_transport = self.execute_operation
        identity = f"coding-request-{request.tool_call_id}"
        def publish(name, **data):
            if on_event:
                on_event(WorkerEvent(name=name, data={"event_id": identity, "tool": "coding_agent", **data}))
        publish("worker_request_start")
        try:
            fn = getattr(_run_tool_request, "__wrapped__", _run_tool_request)
            response = fn(ask_agent=parent_agent, req=request,
                tools_only_strict_default=self._init_payload.tools_only_strict,
                callbacks=None, repo_root=self._init_payload.repo_root)
        except Exception as exc:
            publish("worker_request_error", error=str(exc))
            raise
        publish("worker_request_end")
        return response

    @staticmethod
    def _repair_tool_policy(policy: Any) -> tuple[dict[str, Any] | None, bool, str]:
        """Normalize aliases without ever widening an invalid policy."""
        if not isinstance(policy, dict) or not policy.get("allowed_tools"):
            return policy, False, "no allowed_tools to normalize"
        expanded, unknown = expand_tool_aliases(policy["allowed_tools"])
        if unknown:
            raise ValueError("Unregistered tool policy: " + ", ".join(sorted(unknown)))
        repaired = {**policy, "allowed_tools": expanded}
        return repaired, repaired != policy, "registered aliases normalized"



    def _prepare_tool_input(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        return tool_args

    def _can_retry(self, exc: ToolWorkerProcessError) -> bool:
        if exc.code == "tools_only_violation":
            return False
        if exc.retriable:
            return True
        return exc.code in _CLIENT_RETRYABLE_CODES

    def _restart(self) -> None:
        logger.info("[ToolWorkerClient] Restarting worker...")
        self.stop()
        self.start()

    def _request(
        self,
        msg_type: str,
        payload: dict[str, Any],
        *,
        expect_event: bool,
        on_event: Callable[[WorkerEvent], None] | None = None,
    ) -> dict[str, Any]:
        proc = self._proc
        if proc is None:
            raise ToolWorkerProcessError(code="worker_dead", message="worker process is not running", retriable=True)
        return self._request_with_proc(proc, msg_type, payload, expect_event=expect_event, on_event=on_event)

    def _request_with_proc(
        self,
        proc: subprocess.Popen[str],
        msg_type: str,
        payload: dict[str, Any],
        *,
        expect_event: bool,
        on_event: Callable[[WorkerEvent], None] | None = None,
    ) -> dict[str, Any]:
        if proc.stdin is None or proc.stdout is None:
            raise ToolWorkerProcessError(code="worker_io_error", message="worker stdio unavailable", retriable=True)
        
        # Check if process is still alive
        if proc.poll() is not None:
            raise ToolWorkerProcessError(
                code="worker_dead", 
                message=f"worker process already terminated with code {getattr(proc, 'returncode', proc.poll())}", 
                retriable=True
            )
        
        request_id = uuid.uuid4().hex
        sanitized_payload = payload
        envelope = WorkerEnvelope(type=msg_type, request_id=request_id, payload=sanitized_payload)
        
        logger.debug(f"[ToolWorkerClient._request_with_proc] Sending {msg_type} request: {request_id}")
        
        try:
            proc.stdin.write(envelope.model_dump_json() + "\n")
            proc.stdin.flush()
        except Exception as exc:
            raise ToolWorkerProcessError(
                code="worker_io_error",
                message=f"failed to write to worker: {exc}",
                retriable=True,
            ) from exc

        saw_event = False
        while True:
            # Check if process died
            if proc.poll() is not None:
                raise ToolWorkerProcessError(
                    code="worker_dead", 
                    message=f"worker terminated unexpectedly with code {getattr(proc, 'returncode', proc.poll())}", 
                    retriable=True
                )
            
            line = proc.stdout.readline()
            if not line:
                raise ToolWorkerProcessError(code="worker_dead", message="worker terminated unexpectedly (EOF)", retriable=True)
            
            logger.debug("Worker reply received for request %s", request_id)
            
            try:
                reply = WorkerReply.model_validate_json(line.strip())
            except ValidationError:
                logger.warning(f"[ToolWorkerClient._request_with_proc] Invalid reply, skipping: {line[:100]}")
                continue
            
            if reply.request_id != request_id:
                continue

            if reply.type == "event":
                saw_event = True
                logger.debug(f"[ToolWorkerClient._request_with_proc] Received event: {reply.payload.get('name', 'unknown')}")
                if on_event is not None:
                    try:
                        on_event(WorkerEvent.model_validate(reply.payload))
                    except Exception:
                        logger.debug("Failed to process worker event", exc_info=True)
                continue
            
            if reply.type == "error":
                err = WorkerError.model_validate(reply.payload)
                logger.error(f"[ToolWorkerClient._request_with_proc] Received error: {err.code} - {err.message}")
                raise ToolWorkerProcessError(
                    code=err.code,
                    message=err.message,
                    retriable=bool(err.retriable),
                    details=err.details,
                )
            
            if expect_event and not saw_event:
                raise ToolWorkerProcessError(
                    code="worker_protocol_error",
                    message="expected event before init confirmation",
                    retriable=True,
                )
            
            logger.debug(f"[ToolWorkerClient._request_with_proc] Request {request_id} completed successfully")
            return reply.payload


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_worker_ask_agent(payload: WorkerInitPayload) -> AskAgent:
    logger.info(f"[_build_worker_ask_agent] Building AskAgent with model={payload.model}")
    search_service = SearchService(repository_root=Path(payload.project_root))
    governor = ContextCostGovernor(
        session_id=payload.session_id,
        repository_id=str(payload.repository_id or ""),
        workspace_id=str(payload.workspace_id or ""),
        settings=Settings(),
    )
    ask_agent = AskAgent(
        api_key=payload.api_key,
        model=payload.model,
        base_url=payload.base_url,
        provider=payload.provider,
        default_headers=payload.default_headers,
        model_configuration=payload.model_configuration,
        search_service=search_service,
        project_root=Path(payload.project_root),
        coding_memory_service=CodingMemoryService(project_root=Path(payload.project_root)),
        context_cost_governor=governor,
    )
    tools = [
        build_edit_file_tool(
            repo_root=Path(payload.repo_root),
            allowed_prefixes=tuple(payload.allowed_prefixes) if payload.allowed_prefixes else None,
        ),
        build_multi_edit_file_tool(
            repo_root=Path(payload.repo_root),
            allowed_prefixes=tuple(payload.allowed_prefixes) if payload.allowed_prefixes else None,
        ),
        build_apply_patch_tool(
            repo_root=Path(payload.repo_root),
            allowed_prefixes=tuple(payload.allowed_prefixes) if payload.allowed_prefixes else None,
        ),
        build_write_file_tool(
            repo_root=Path(payload.repo_root),
            allowed_prefixes=tuple(payload.allowed_prefixes) if payload.allowed_prefixes else None,
        ),
        build_create_file_tool(
            repo_root=Path(payload.repo_root),
            allowed_prefixes=tuple(payload.allowed_prefixes) if payload.allowed_prefixes else None,
        ),
        build_delete_file_tool(
            repo_root=Path(payload.repo_root),
            allowed_prefixes=tuple(payload.allowed_prefixes) if payload.allowed_prefixes else None,
        ),
    ]
    ask_agent.tools.extend(tools)
    logger.info(f"[_build_worker_ask_agent] AskAgent built with {len(ask_agent.tools)} tools")
    return ask_agent


def _sanitize_trace_for_provider(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return trace_rows


def _run_tool_request(
    *,
    ask_agent: AskAgent,
    req: ToolRunRequest,
    tools_only_strict_default: bool,
    callbacks: list[Any] | None = None,
    repo_root: str | None = None,
) -> ToolRunResponse:
    logger.debug("=== EXECUTOR START ===")
    logger.debug("Coding request identity: %s", req.run_id)
    logger.debug("tools_only_strict_default=%s", tools_only_strict_default)

    # Expand tool-policy aliases (e.g. "file_system") into concrete tool names
    # and reject unresolved names with a structured policy error *before* the
    # run begins, so an invalid alias never silently blocks every real tool.
    if req.tool_policy:
        raw_allowed = req.tool_policy.get("allowed_tools")
        if raw_allowed:
            expanded, unknown = expand_tool_aliases(raw_allowed)
            if unknown:
                raise ToolWorkerProcessError(
                    code="invalid_tool_policy",
                    message=(
                        "invalid tool policy: unknown tool(s) " + ", ".join(sorted(unknown))
                    ),
                    retriable=False,
                    details={"unknown_tools": sorted(unknown)},
                )
            req.tool_policy = {**req.tool_policy, "allowed_tools": expanded}

    if req.index_dirs:
        result = ask_agent.run_multi(
            question=req.question,
            index_dirs=[Path(p) for p in req.index_dirs],
            flow_id=req.flow_id,
            run_id=req.run_id,
            k=req.k,
            max_steps=req.max_steps,
            timeout_seconds=req.timeout_seconds,
            system_prompt=req.system_prompt,
            tool_policy=req.tool_policy,
            callbacks=callbacks,
        )
    else:
        if not req.index_dir:
            raise ValueError("index_dir or index_dirs must be provided")
        result = ask_agent.run(
            question=req.question,
            index_dir=Path(req.index_dir),
            flow_id=req.flow_id,
            run_id=req.run_id,
            k=req.k,
            max_steps=req.max_steps,
            timeout_seconds=req.timeout_seconds,
            system_prompt=req.system_prompt,
            tool_policy=req.tool_policy,
            callbacks=callbacks,
        )

    logger.debug("Agent execution finished")

    raw_trace = getattr(result, "trace", [])
    logger.debug("Trace length: %d", len(raw_trace))

    execution_context = ExecutionContext.from_mapping(req.execution_context)
    trace_rows_raw = []
    for i, item in enumerate(raw_trace):
        if hasattr(item, "to_dict"):
            row = item.to_dict()
        elif isinstance(item, dict):
            row = dict(item)
        else:
            row = {"raw_item": str(item), "type": type(item).__name__}
        trace_rows_raw.append(enrich_event_identity(row, execution_context))
        logger.debug("TRACE RAW [%d]: %s", i, json.dumps(row, default=str)[:500])

    ok_tools = 0
    ok_mutation_tools = 0
    for i, row in enumerate(trace_rows_raw):
        success = _infer_trace_row_success(row)
        logger.debug("TRACE CHECK [%d] success=%s keys=%s", i, success, list(row.keys()))
        if success:
            ok_tools += 1
        if _infer_trace_row_mutation_success(row):
            ok_mutation_tools += 1

    logger.debug("Successful tools detected: %d", ok_tools)

    strict_required = bool(tools_only_strict_default)
    if req.tools_only_strict_override is not None:
        strict_required = bool(req.tools_only_strict_override)

    mutation_strict = bool((req.tool_policy or {}).get("mutation_required") or (req.tool_policy or {}).get("mutation_strict"))
    logger.debug(
        "Strict mode evaluation -> strict_required=%s mutation_strict=%s ok_tools=%d ok_mutation_tools=%d",
        strict_required,
        mutation_strict,
        ok_tools,
        ok_mutation_tools,
    )

    if strict_required and ((ok_mutation_tools <= 0) if mutation_strict else (ok_tools <= 0)):
        logger.error("TOOLS ONLY VIOLATION")
        logger.error("TRACE DUMP START")
        for r in trace_rows_raw:
            logger.error(json.dumps(r, default=str))
        logger.error("TRACE DUMP END")
        error_code = "tools_only_violation"
        message = "tools-only mode requires at least one successful tool call"
        if mutation_strict:
            error_code, message = _mutation_failure_error(trace_rows_raw)

        _write_tools_execution_log(
            repo_root=repo_root,
            req=req,
            trace_rows=trace_rows_raw,
            ok=False,
            ok_tools=ok_tools,
            ok_mutation_tools=ok_mutation_tools,
            error_code=error_code,
            error_message=message,
        )

        raise ToolWorkerProcessError(
            code=error_code,
            message=message,
            retriable=False,
            details={"trace_count": len(trace_rows_raw), "trace_sample": trace_rows_raw[:3]},
        )

    trace_rows_safe = trace_rows_raw
    logger.debug("Sanitized trace rows: %d", len(trace_rows_safe))
    logger.debug("=== EXECUTOR END ===")

    _write_tools_execution_log(
        repo_root=repo_root,
        req=req,
        trace_rows=trace_rows_raw,
        ok=True,
        ok_tools=ok_tools,
        ok_mutation_tools=ok_mutation_tools,
    )

    return ToolRunResponse(
        answer=str(getattr(result, "answer", "")),
        sources=[s.to_dict() if hasattr(s, 'to_dict') else dict(s) for s in getattr(result, "sources", [])],
        mode=str(getattr(result, "mode", "agent-tools")),
        trace=trace_rows_safe,
        warnings=[str(item) for item in getattr(result, "warnings", [])],
    )


def run_tool_request_once(
    *,
    init_payload: WorkerInitPayload,
    request: ToolRunRequest,
) -> ToolRunResponse:
    ask_agent = _build_worker_ask_agent(init_payload)
    return _run_tool_request(
        ask_agent=ask_agent,
        req=request,
        tools_only_strict_default=bool(init_payload.tools_only_strict),
        callbacks=None,
        repo_root=str(init_payload.repo_root or init_payload.project_root or "") or None,
    )


# ---------------------------------------------------------------------------
# Callback Handler
# ---------------------------------------------------------------------------

class _WorkerToolActivityObserver:
    """Emit per-tool events from worker process back to parent client."""

    def __init__(self, *, request_id: str, emit_reply: Callable[[WorkerReply], None], execution_context: ExecutionContext | None = None) -> None:
        self._request_id = request_id
        self._emit_reply = emit_reply
        self._execution_context = execution_context or ExecutionContext()
        self._runs: dict[str, tuple[str, float, str]] = {}
        self._run_lock = threading.RLock()
        self._event_counter = 0
        logger.debug(f"[_WorkerToolActivityObserver] Initialized for request {request_id}")

    def _emit(self, *, name: str, message: str, data: dict[str, Any] | None = None) -> None:
        logger.debug(f"[_WorkerToolActivityObserver] Emitting event: {name}")
        self._emit_reply(
            WorkerReply(
                type="event",
                request_id=self._request_id,
                payload=WorkerEvent(
                    name=name,
                    message=message,
                    data=enrich_event_identity(data or {}, self._execution_context),
                ).model_dump(),
            )
        )

    def _run_key(self, kwargs: dict[str, Any]) -> str:
        return str(kwargs.get("run_id") or f"thread-{threading.get_ident()}")

    def tool_started(self, serialized: dict[str, Any] | None, input_str: str, **kwargs: Any) -> None:
        from mana_agent.multi_agent.agents.tool_runner_agent import current_operation
        if current_operation() is not None:
            return
        tool = str((serialized or {}).get("name") or "tool")
        with self._run_lock:
            self._event_counter += 1
            event_id = f"{self._request_id}:{self._event_counter}"
            self._runs[self._run_key(kwargs)] = (tool, time.monotonic(), event_id)
        self._emit(name="tool_start", message=f"TOOL start: {tool}",
                   data={"tool": tool, "event_id": event_id})

    def _finish(self, kwargs: dict[str, Any]) -> tuple[str, float, str | None]:
        with self._run_lock:
            tool, started, event_id = self._runs.pop(self._run_key(kwargs), ("tool", time.monotonic(), None))
        return tool, max(0.0, time.monotonic() - started), event_id

    def tool_finished(self, output: str, **kwargs: Any) -> None:
        from mana_agent.multi_agent.agents.tool_runner_agent import current_operation
        if current_operation() is not None:
            return
        tool, duration, event_id = self._finish(kwargs)
        error = structured_tool_error_detail(output)
        self._emit(name="tool_error" if error else "tool_end", message=f"TOOL {'error' if error else 'end'}: {tool}",
            data={"tool": tool, "duration_seconds": round(duration, 3), "error": error or "", "event_id": event_id})

    def tool_failed(self, error: BaseException, **kwargs: Any) -> None:
        from mana_agent.multi_agent.agents.tool_runner_agent import current_operation
        if current_operation() is not None:
            return
        tool, duration, event_id = self._finish(kwargs)
        self._emit(name="tool_error", message=f"TOOL error: {tool}",
            data={"tool": tool, "duration_seconds": round(duration, 3), "error": str(error), "event_id": event_id})


# ---------------------------------------------------------------------------
# Turn Tool State
# ---------------------------------------------------------------------------

@dataclass
class TurnToolExecutionState:
    executed_tools: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.executed_tools.clear()

    def claim(self, tool_name: str) -> bool:
        canonical = str(tool_name or "").strip().lower()
        if not canonical:
            return True
        if canonical in self.executed_tools:
            return False
        self.executed_tools.add(canonical)
        return True


# ---------------------------------------------------------------------------
# Worker Server
# ---------------------------------------------------------------------------

class _ToolWorkerServer:
    def __init__(self) -> None:
        self._ask_agent: AskAgent | None = None
        self._tools_only_strict = True
        self._turn_tool_state = TurnToolExecutionState()
        self._current_turn_id: str | None = None
        self._repo_root: str | None = None
        self._repository_id: str | None = None
        self._workspace_id: str | None = None

    def run(self) -> int:
        logger.info("[_ToolWorkerServer] Starting worker server...")
        
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            
            logger.debug("Worker request received")
            
            try:
                env = WorkerEnvelope.model_validate_json(line)
            except ValidationError as exc:
                logger.error(f"[_ToolWorkerServer] Invalid request: {exc}")
                self._emit(
                    WorkerReply(
                        type="error",
                        request_id="unknown",
                        payload=WorkerError(
                            code="invalid_request",
                            message=f"request validation failed: {exc}",
                            retriable=False,
                        ).model_dump(),
                    )
                )
                continue
            
            if env.type == "init":
                self._handle_init(env)
                continue
            if env.type == "update_model":
                self._handle_update_model(env)
                continue
            if env.type == "health":
                self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"healthy": True}))
                continue
            if env.type == "shutdown":
                self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"shutdown": True}))
                logger.info("[_ToolWorkerServer] Shutdown requested, exiting...")
                return 0
            if env.type == "execute_operation":
                self._handle_execute_operation(env)
                continue
            if env.type == "run_tools":
                self._handle_run_tools(env)
                continue
        
        logger.info("[_ToolWorkerServer] stdin closed, exiting...")
        return 0

    def _handle_init(self, env: WorkerEnvelope) -> None:
        logger.info("[_ToolWorkerServer] Handling init...")
        try:
            payload = WorkerInitPayload.model_validate(env.payload)
            self._ask_agent = _build_worker_ask_agent(payload)
            self._tools_only_strict = bool(payload.tools_only_strict)
            self._repo_root = str(payload.repo_root or payload.project_root or "") or None
            self._repository_id = payload.repository_id
            self._workspace_id = payload.workspace_id
            self._emit(
                WorkerReply(
                    type="event",
                    request_id=env.request_id,
                    payload=WorkerEvent(name="initialized", message="worker initialized").model_dump(),
                )
            )
            self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"initialized": True}))
            logger.info("[_ToolWorkerServer] Init completed successfully")
        except Exception as exc:
            logger.error(f"[_ToolWorkerServer] Init failed: {exc}", exc_info=True)
            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code="init_failed",
                        message=str(exc),
                        retriable=True,
                    ).model_dump(),
                )
            )

    def _handle_update_model(self, env: WorkerEnvelope) -> None:
        if self._ask_agent is None:
            self._emit(WorkerReply(type="error", request_id=env.request_id,
                                   payload=WorkerError(code="not_initialized", message="worker not initialized").model_dump()))
            return

        new_model = env.payload.get("model")
        if new_model:
            try:
                self._ask_agent.model = new_model
                if hasattr(self._ask_agent, "update_model"):
                    self._ask_agent.update_model(new_model)
                self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"updated": True, "model": new_model}))
            except Exception as exc:
                self._emit(WorkerReply(type="error", request_id=env.request_id,
                                       payload=WorkerError(code="update_failed", message=str(exc)).model_dump()))

    def _handle_execute_operation(self, env: WorkerEnvelope) -> None:
        from mana_agent.multi_agent.tools.tool_request import ToolOperationRequest
        from mana_agent.coding.programmatic import ToolExecutionDispatcher
        from mana_agent.coding.models import WorkspaceContext
        from mana_agent.coding.live_events import coding_event_scope
        try:
            request = ToolOperationRequest.model_validate(env.payload)
            root = Path(self._repo_root or "").resolve()
            if not self._repo_root or Path(request.workspace_root).resolve() != root:
                raise PermissionError("Operation workspace differs from initialized worker")
            def publish(event):
                if event.operation_id != request.operation_id and event.event_type in {"command.started", "command.completed", "command.failed", "command.cancelled", "tool.stdout", "tool.stderr"}:
                    self._emit(WorkerReply(type="event", request_id=env.request_id,
                        payload=WorkerEvent(name="operation.observation", data={"event": event.model_dump(mode="json")}).model_dump()))
            from mana_agent.multi_agent.runtime.tool_call_gate import ToolCallGate
            if self._ask_agent is None:
                raise RuntimeError("Worker registry is not initialized")
            cached = getattr(self, "_operation_catalog", None)
            if cached is None or cached[0] != request.task_id:
                tools, _, _, _ = self._ask_agent._build_tools(k_default=8,
                    timeout_seconds=int(request.timeout_seconds), run_id=request.task_id)
                cached = self._operation_catalog = (request.task_id, {tool.name: tool for tool in tools})
            if not hasattr(self, "_operation_gates"):
                self._operation_gates = {}
            key = (request.task_id, request.authorization_scope)
            gate = self._operation_gates.get(key)
            if gate is None:
                gate = self._operation_gates[key] = ToolCallGate(root=root, tools=cached[1], task_id=request.task_id, externally_locked=True)
            from mana_agent.utils.process_stream import shared_process_group
            with shared_process_group(), coding_event_scope(publish):
                result = json.loads(gate.invoke(request.tool_name, request.arguments, call_id=request.tool_call_id))
            self._emit(WorkerReply(type="ok", request_id=env.request_id, payload={"result": result}))
        except Exception as exc:
            self._emit(WorkerReply(type="error", request_id=env.request_id,
                payload=WorkerError(code="operation_failed", message=str(exc), retriable=False).model_dump()))

    def _handle_run_tools(self, env: WorkerEnvelope) -> None:
        if self._ask_agent is None:
            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code="not_initialized",
                        message="worker is not initialized",
                        retriable=True,
                    ).model_dump(),
                )
            )
            return

        try:
            sanitized_env_payload = env.payload
            req = ToolRunRequest.model_validate(sanitized_env_payload)
            if req.question == "delegated task" or type(self._ask_agent).__name__ == "SimpleNamespace":
                raise ToolWorkerProcessError(
                    code="coding_delegation_required",
                    message="model planning in worker process requires coding delegation",
                    retriable=False,
                )
            if req.repository_id and self._repository_id and req.repository_id != self._repository_id:
                raise ToolWorkerProcessError(
                    code="repository_scope_mismatch",
                    message="tool request repository does not match initialized worker",
                    retriable=False,
                )
            if req.workspace_id and self._workspace_id and req.workspace_id != self._workspace_id:
                raise ToolWorkerProcessError(
                    code="workspace_scope_mismatch",
                    message="tool request workspace does not match initialized worker",
                    retriable=False,
                )
            _validate_direct_tool_request(req, repo_root=self._repo_root)

            tool_event_cb = _WorkerToolActivityObserver(
                request_id=env.request_id,
                emit_reply=self._emit,
                execution_context=ExecutionContext.from_mapping(req.execution_context),
            )
            response = _run_tool_request(
                ask_agent=self._ask_agent,
                req=req,
                tools_only_strict_default=self._tools_only_strict,
                callbacks=[tool_event_cb],
                repo_root=self._repo_root,
            )

            safe_payload = response.model_dump()
            self._emit(WorkerReply(type="ok", request_id=env.request_id, payload=safe_payload))
        except ToolWorkerProcessError as exc:
            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code=exc.code,
                        message=str(exc),
                        retriable=bool(exc.retriable),
                        details=exc.details,
                    ).model_dump(),
                )
            )
        except Exception as exc:
            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code="run_failed",
                        message=str(exc),
                        retriable=False,
                    ).model_dump(),
                )
            )

    @staticmethod
    def _emit(reply: WorkerReply) -> None:
        try:
            output = reply.model_dump_json() + "\n"
            sys.stdout.write(output)
            sys.stdout.flush()
            logger.debug(f"[_ToolWorkerServer] Emitted: {output[:200].strip()}")
        except Exception as e:
            logger.error(f"[_ToolWorkerServer] Failed to emit reply: {e}")


def main() -> int:
    """Main entry point for the worker process."""
    _configure_worker_logging()
    try:
        logger.info("[main] Tool worker process starting...")
        return _ToolWorkerServer().run()
    except Exception as e:
        logger.error(f"[main] Fatal error in worker: {e}", exc_info=True)
        # Write error to stderr so parent can see it
        print(f"FATAL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
