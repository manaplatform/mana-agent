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

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log
from langchain_core.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field, ValidationError

from mana_agent.services.coding_memory_service import CodingMemoryService
from mana_agent.services.search_service import SearchService
from mana_agent.llm.ask_agent import AskAgent
from mana_agent.llm.mutation_plan import validate_mutation_plan
from mana_agent.vector_store.embeddings import build_embeddings
from mana_agent.tools import (
    build_apply_patch_tool,
    build_create_file_tool,
    build_delete_file_tool,
    build_edit_file_tool,
    build_multi_edit_file_tool,
    build_write_file_tool,
)
from mana_agent.tools.apply_patch import extract_patch_touched_files
from mana_agent.vector_store.faiss_store import FaissStore
from mana_agent.utils.redaction import redact_json_line, redact_secrets
from mana_agent.utils.tool_policy import expand_tool_aliases
from mana_agent.config.settings import default_tools_logs_dir

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
            "question": req.question,
            "tool_name": req.tool_name or None,
            "tool_args": req.tool_args or {},
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
        line = json.dumps(record, default=str, ensure_ascii=False)
        line = redact_secrets(line)
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
_PROVIDER_BANNED_KEYS = frozenset({"status"})

_BANNED_PARAM_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("unknown parameter", "status"),
    ("unrecognized parameter", "status"),
    ("unexpected parameter", "status"),
    ("invalid parameter", "status"),
    ("unknown parameter", "input"),
    ("unrecognized request parameter", ""),
    ("additional properties", "status"),
    ("is not allowed", "status"),
    ("status", "not permitted"),
    ("status", "not allowed"),
    ("status", "is not expected"),
]


def _is_banned_param_provider_error(error_message: str) -> bool:
    msg = error_message.lower()
    if "unknown parameter" in msg and "status" in msg:
        return True
    for pattern_a, pattern_b in _BANNED_PARAM_ERROR_PATTERNS:
        if pattern_a in msg and (not pattern_b or pattern_b in msg):
            return True
    return False


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


def _strip_banned_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_banned_keys(v)
            for k, v in obj.items()
            if k not in _PROVIDER_BANNED_KEYS
        }
    if isinstance(obj, list):
        return [_strip_banned_keys(item) for item in obj]
    return obj


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


def _deep_strip_banned_keys_inplace(obj: Any) -> Any:
    if isinstance(obj, dict):
        keys_to_remove = [k for k in obj if k in _PROVIDER_BANNED_KEYS]
        for k in keys_to_remove:
            del obj[k]
        for v in obj.values():
            _deep_strip_banned_keys_inplace(v)
    elif isinstance(obj, list):
        for item in obj:
            _deep_strip_banned_keys_inplace(item)
    return obj


_CHAT_ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})
_ROLE_ALIASES = {
    "ai": "assistant",
    "assistant_tool": "tool",
    "human": "user",
}


def _sanitize_chat_message_content(content: Any) -> str | list[dict[str, Any]] | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized_parts: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                normalized_parts.append({"type": "text", "text": str(item)})
                continue
            item_type = str(item.get("type", "text") or "text")
            if item_type == "text":
                text_value = item.get("text", "")
                if not isinstance(text_value, str):
                    text_value = json.dumps(text_value, ensure_ascii=False)
                normalized_parts.append({"type": "text", "text": text_value})
                continue
            normalized_parts.append({"type": item_type, "text": json.dumps(item, ensure_ascii=False)})
        return normalized_parts
    if isinstance(content, (dict, tuple, set)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


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


def _sanitize_chat_message(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        if isinstance(message, str):
            return {"role": "user", "content": message}
        return None

    role_raw = str(message.get("role", "user") or "user").strip().lower()
    role = _ROLE_ALIASES.get(role_raw, role_raw)
    if role not in _CHAT_ALLOWED_MESSAGE_ROLES:
        role = "user"

    clean: dict[str, Any] = {"role": role}

    if role == "assistant":
        tool_calls = _sanitize_tool_calls(message.get("tool_calls"))
        if tool_calls:
            clean["tool_calls"] = tool_calls

    if role in {"tool", "function"}:
        tool_call_id = message.get("tool_call_id") or message.get("id")
        if tool_call_id is not None:
            clean["tool_call_id"] = str(tool_call_id)
        elif role == "tool":
            clean["role"] = "assistant"

    content = _sanitize_chat_message_content(message.get("content", ""))
    if content is None and clean.get("tool_calls"):
        content = ""
    if isinstance(content, str):
        clean["content"] = content
    elif isinstance(content, list):
        clean["content"] = content
    else:
        clean["content"] = ""

    name = message.get("name")
    if isinstance(name, str) and name.strip():
        clean["name"] = name.strip()

    return clean


def _sanitize_tools_payload(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    clean_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            continue
        description = fn.get("description", "")
        if not isinstance(description, str):
            description = str(description)
        parameters = fn.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        clean_tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn_name.strip(),
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return clean_tools


def _sanitize_openai_json_payload(payload: Any) -> Any:
    sanitized = _strip_banned_keys(payload)
    if not isinstance(sanitized, dict):
        return sanitized

    if "messages" in sanitized and isinstance(sanitized.get("messages"), list):
        sanitized["messages"] = [
            msg for msg in (_sanitize_chat_message(item) for item in sanitized.get("messages", [])) if msg is not None
        ]
        if "tools" in sanitized:
            sanitized["tools"] = _sanitize_tools_payload(sanitized.get("tools"))

    return sanitized


# ---------------------------------------------------------------------------
# Monkey-patch section
# ---------------------------------------------------------------------------
_PATCHES_APPLIED = False


def _apply_global_patches() -> None:
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    _PATCHES_APPLIED = True
    
    try:
        _patch_langchain_messages()
    except Exception as e:
        logger.debug(f"Could not patch LangChain messages: {e}")
    
    try:
        _patch_openai_client()
    except Exception as e:
        logger.debug(f"Could not patch OpenAI client: {e}")
    
    try:
        _patch_httpx_client()
    except Exception as e:
        logger.debug(f"Could not patch httpx: {e}")
    
    try:
        _patch_litellm()
    except Exception as e:
        logger.debug(f"Could not patch LiteLLM: {e}")
    
    logger.debug("Global patches applied (or skipped if unavailable)")


def _patch_langchain_messages() -> None:
    try:
        from langchain_core.messages import BaseMessage
        
        if hasattr(BaseMessage, 'model_dump'):
            _original_model_dump = BaseMessage.model_dump
            
            def _patched_model_dump(self, *args, **kwargs):
                result = _original_model_dump(self, *args, **kwargs)
                return _strip_banned_keys(result)
            
            BaseMessage.model_dump = _patched_model_dump
        
        if hasattr(BaseMessage, 'dict'):
            _original_dict = BaseMessage.dict
            
            def _patched_dict(self, *args, **kwargs):
                result = _original_dict(self, *args, **kwargs)
                return _strip_banned_keys(result)
            
            BaseMessage.dict = _patched_dict
        
        logger.debug("Patched LangChain BaseMessage serialization")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not patch LangChain messages: {e}")


def _patch_openai_client() -> None:
    try:
        from openai._base_client import SyncAPIClient, AsyncAPIClient
        
        if hasattr(SyncAPIClient, '_build_request'):
            _original_build_request = SyncAPIClient._build_request
            
            def _patched_build_request(self, options, *args, **kwargs):
                if hasattr(options, 'json_data') and options.json_data is not None:
                    options.json_data = _sanitize_openai_json_payload(options.json_data)
                return _original_build_request(self, options, *args, **kwargs)
            
            SyncAPIClient._build_request = _patched_build_request
        
        if hasattr(AsyncAPIClient, '_build_request'):
            _original_async_build_request = AsyncAPIClient._build_request
            
            def _patched_async_build_request(self, options, *args, **kwargs):
                if hasattr(options, 'json_data') and options.json_data is not None:
                    options.json_data = _sanitize_openai_json_payload(options.json_data)
                return _original_async_build_request(self, options, *args, **kwargs)
            
            AsyncAPIClient._build_request = _patched_async_build_request
        
        logger.debug("Patched OpenAI client request building")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not patch OpenAI client: {e}")


def _patch_httpx_client() -> None:
    try:
        import httpx
        
        _original_request = httpx.Client.request
        _original_async_request = httpx.AsyncClient.request
        
        def _sanitize_content(content: Any) -> Any:
            if content is None:
                return None
            try:
                if isinstance(content, bytes):
                    data = json.loads(content.decode('utf-8'))
                    return json.dumps(_sanitize_openai_json_payload(data)).encode('utf-8')
                elif isinstance(content, str):
                    data = json.loads(content)
                    return json.dumps(_sanitize_openai_json_payload(data))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            return content
        
        def _patched_request(self, method, url, *args, **kwargs):
            if 'json' in kwargs and kwargs['json'] is not None:
                kwargs['json'] = _sanitize_openai_json_payload(kwargs['json'])
            if 'content' in kwargs and kwargs['content'] is not None:
                kwargs['content'] = _sanitize_content(kwargs['content'])
            return _original_request(self, method, url, *args, **kwargs)
        
        async def _patched_async_request(self, method, url, *args, **kwargs):
            if 'json' in kwargs and kwargs['json'] is not None:
                kwargs['json'] = _sanitize_openai_json_payload(kwargs['json'])
            if 'content' in kwargs and kwargs['content'] is not None:
                kwargs['content'] = _sanitize_content(kwargs['content'])
            return await _original_async_request(self, method, url, *args, **kwargs)
        
        httpx.Client.request = _patched_request
        httpx.AsyncClient.request = _patched_async_request
        
        logger.debug("Patched httpx client")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not patch httpx: {e}")


def _patch_litellm() -> None:
    try:
        import litellm
        
        _original_completion = litellm.completion
        _original_acompletion = getattr(litellm, 'acompletion', None)
        
        def _patched_completion(*args, **kwargs):
            if 'messages' in kwargs:
                kwargs['messages'] = _strip_banned_keys(kwargs['messages'])
            return _original_completion(*args, **kwargs)
        
        litellm.completion = _patched_completion
        
        if _original_acompletion is not None:
            async def _patched_acompletion(*args, **kwargs):
                if 'messages' in kwargs:
                    kwargs['messages'] = _strip_banned_keys(kwargs['messages'])
                return await _original_acompletion(*args, **kwargs)
            
            litellm.acompletion = _patched_acompletion
        
        logger.debug("Patched LiteLLM")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not patch LiteLLM: {e}")


# Apply patches immediately when module is loaded
_apply_global_patches()


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
            if tool_name in {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file", "create_file", "delete_file"}:
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
    if tool_name not in {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file", "create_file", "delete_file"}:
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
        in {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file", "create_file", "delete_file"}
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
    base_url: str | None = None
    project_root: str
    repo_root: str
    allowed_prefixes: list[str] | None = None
    tools_only_strict: bool = True


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
    type: Literal["init", "run_tools", "health", "shutdown", "update_model"]
    request_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkerReply(BaseModel):
    type: Literal["ok", "error", "event"]
    request_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolWorkerProcessError(RuntimeError):
    def __init__(self, *, code: str, message: str, retriable: bool = False, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.retriable = retriable
        self.details = details or {}


_CLIENT_RETRYABLE_CODES = frozenset({"worker_io_error", "worker_dead", "model_not_found", "init_failed", "worker_startup_failed"})


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
        repo_root: Path,
        project_root: Path,
        base_url: str | None = None,
        allowed_prefixes: list[str] | None = None,
        tools_only_strict: bool = True,
    ) -> None:
        resolved_base_url = str(base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
        self._init_payload = WorkerInitPayload(
            api_key=api_key,
            model=model,
            base_url=resolved_base_url,
            project_root=str(project_root.resolve()),
            repo_root=str(repo_root.resolve()),
            allowed_prefixes=allowed_prefixes,
            tools_only_strict=tools_only_strict,
        )
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
            [sys.executable, "-m", "mana_agent.llm.tool_worker_process"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        
        # Start a thread to log stderr
        def log_stderr():
            try:
                if self._proc and self._proc.stderr:
                    for line in self._proc.stderr:
                        line = line.rstrip()
                        if line:
                            logger.debug(f"[Worker STDERR] {line}")
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(_should_retry_run_tools_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def run_tools(
        self,
        request: ToolRunRequest,
        *,
        on_event: Callable[[WorkerEvent], None] | None = None,
    ) -> ToolRunResponse:
        logger.info(f"[ToolWorkerClient.run_tools] Starting with question: {request.question[:100]}...")
        logger.debug(f"[ToolWorkerClient.run_tools] Full request: {request.model_dump()}")
        _validate_direct_tool_request(request)
        request_started = time.time()
        request_event_id = f"worker-request-{uuid.uuid4().hex}"

        def _emit_request_event(name: str, data: dict[str, Any] | None = None) -> None:
            if on_event is None:
                return
            try:
                on_event(
                    WorkerEvent(
                        name=name,
                        message=name.replace("_", " "),
                        data={"tool": "tool_worker", "event_id": request_event_id, **(data or {})},
                    )
                )
            except Exception:
                logger.debug("Failed to process worker request event", exc_info=True)

        _emit_request_event(
            "worker_request_start",
            {"args": request.question[:160]},
        )
        
        self.start()
        payload_dict = _strip_banned_keys(request.model_dump())
        
        if "tool_args" in payload_dict:
            payload_dict["tool_args"] = self._prepare_tool_input(payload_dict["tool_args"])

        logger.debug(f"[ToolWorkerClient.run_tools] Sanitized payload: {json.dumps(payload_dict, default=str)[:500]}")

        # Bounded auto-repair loop: an invalid tool policy is repaired in place
        # (aliases expanded, unknown names dropped) and retried exactly once
        # before giving up, instead of failing outright or looping uselessly.
        policy_repaired = False
        while True:
            try:
                response_payload = self._request(
                    "run_tools",
                    payload_dict,
                    expect_event=False,
                    on_event=on_event,
                )
                logger.info(f"[ToolWorkerClient.run_tools] Received response with keys: {list(response_payload.keys())}")
                logger.debug(f"[ToolWorkerClient.run_tools] Response trace count: {len(response_payload.get('trace', []))}")
                _emit_request_event(
                    "worker_request_end",
                    {"duration_seconds": round(max(0.0, time.time() - request_started), 3)},
                )
                break

            except ToolWorkerProcessError as exc:
                logger.error(f"[ToolWorkerClient.run_tools] ToolWorkerProcessError: code={exc.code}, message={exc}")

                if exc.code == "invalid_tool_policy" and not policy_repaired:
                    repaired_policy, changed, summary = self._repair_tool_policy(payload_dict.get("tool_policy"))
                    if changed:
                        logger.warning(
                            "[ToolWorkerClient.run_tools] Repairing invalid tool policy and retrying once. %s",
                            summary,
                        )
                        payload_dict = {**payload_dict, "tool_policy": repaired_policy}
                        policy_repaired = True
                        continue
                    # Nothing repairable — surface the structured policy error.
                    _emit_request_event(
                        "worker_request_error",
                        {
                            "duration_seconds": round(max(0.0, time.time() - request_started), 3),
                            "error": f"{exc.code}: {exc}",
                        },
                    )
                    raise

                if self._is_banned_param_error(exc):
                    logger.warning(
                        "[ToolWorkerClient.run_tools] Provider rejected a banned parameter — stripping and retrying. "
                        "Original error: %s",
                        exc,
                    )
                    self._restart()
                    _emit_request_event(
                        "worker_request_error",
                        {
                            "duration_seconds": round(max(0.0, time.time() - request_started), 3),
                            "error": f"{exc.code}: {exc}",
                        },
                    )
                    raise

                if self._can_retry(exc):
                    logger.warning(
                        "[ToolWorkerClient.run_tools] Retrying run_tools due to worker error: %s. Message: %s",
                        exc.code,
                        exc,
                    )
                    self._restart()
                    _emit_request_event(
                        "worker_request_error",
                        {
                            "duration_seconds": round(max(0.0, time.time() - request_started), 3),
                            "error": f"{exc.code}: {exc}",
                        },
                    )
                    raise
                _emit_request_event(
                    "worker_request_error",
                    {
                        "duration_seconds": round(max(0.0, time.time() - request_started), 3),
                        "error": f"{exc.code}: {exc}",
                    },
                )
                raise

        return ToolRunResponse.model_validate(response_payload)

    @staticmethod
    def _repair_tool_policy(policy: Any) -> tuple[dict[str, Any] | None, bool, str]:
        """Expand aliases and drop unknown tool names from a policy.

        Returns ``(repaired_policy, changed, summary)``. When the expanded
        allow-list is empty (every name was unknown), ``allowed_tools`` is
        removed so the worker falls back to allowing all tools rather than
        blocking everything. ``summary`` describes what changed for logging.
        """
        if not isinstance(policy, dict):
            return policy, False, "no tool_policy to repair"
        raw_allowed = policy.get("allowed_tools")
        if not raw_allowed:
            return policy, False, "no allowed_tools to repair"
        expanded, unknown = expand_tool_aliases(raw_allowed)
        if not unknown and list(expanded) == list(raw_allowed):
            return policy, False, "allowed_tools already valid"
        repaired = dict(policy)
        if expanded:
            repaired["allowed_tools"] = expanded
            summary = f"allowed_tools {list(raw_allowed)} -> {expanded}; dropped unknown {sorted(unknown)}"
        else:
            repaired.pop("allowed_tools", None)
            summary = f"all tools unknown {sorted(unknown)}; cleared allowed_tools (allow all)"
        return repaired, True, summary

    @staticmethod
    def _is_banned_param_error(exc: ToolWorkerProcessError) -> bool:
        return _is_banned_param_provider_error(str(exc))

    _is_status_param_error = _is_banned_param_error

    def _prepare_tool_input(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        return _strip_banned_keys(tool_args)

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
        sanitized_payload = _strip_banned_keys(payload)
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
            
            logger.debug(f"[ToolWorkerClient._request_with_proc] Received line: {line[:200].strip()}")
            
            try:
                reply = WorkerReply.model_validate_json(line.strip())
            except ValidationError as exc:
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
    embeddings = build_embeddings(
        api_key=payload.api_key,
        base_url=payload.base_url,
        model=getattr(payload, "embed_model", None),
    )
    search_service = SearchService(store=FaissStore(embeddings))
    ask_agent = AskAgent(
        api_key=payload.api_key,
        model=payload.model,
        base_url=payload.base_url,
        search_service=search_service,
        project_root=Path(payload.project_root),
        coding_memory_service=CodingMemoryService(project_root=Path(payload.project_root)),
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
    return _strip_banned_keys(trace_rows)


def _should_retry_run_tool_request(exc: BaseException) -> bool:
    """Decide whether to retry a worker-side tool run.

    Critically, a ``tools_only_violation`` is NOT retried: re-running with the
    exact same prompt and tool policy when zero tools succeeded changes nothing
    and only burns API calls. Genuinely transient provider errors (rate limits,
    timeouts) are still retried.
    """
    if isinstance(exc, ToolWorkerProcessError):
        if exc.code == "tools_only_violation":
            return False
        if exc.retriable:
            return True
        return exc.code in _CLIENT_RETRYABLE_CODES
    return _is_likely_retriable_runtime_error(exc)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(_should_retry_run_tool_request),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _run_tool_request(
    *,
    ask_agent: AskAgent,
    req: ToolRunRequest,
    tools_only_strict_default: bool,
    callbacks: list[BaseCallbackHandler] | None = None,
    repo_root: str | None = None,
) -> ToolRunResponse:
    logger.debug("=== EXECUTOR START ===")
    logger.debug("Question: %s", req.question)
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

    trace_rows_raw = []
    for i, item in enumerate(raw_trace):
        if hasattr(item, "to_dict"):
            row = item.to_dict()
        elif isinstance(item, dict):
            row = dict(item)
        else:
            row = {"raw_item": str(item), "type": type(item).__name__}
        trace_rows_raw.append(row)
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

    trace_rows_safe = _strip_banned_keys(trace_rows_raw)
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
        sources=[_strip_banned_keys(s.to_dict() if hasattr(s, 'to_dict') else dict(s)) for s in getattr(result, "sources", [])],
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

class _WorkerToolEventCallback(BaseCallbackHandler):
    """Emit per-tool events from worker process back to parent client."""

    def __init__(self, *, request_id: str, emit_reply: Callable[[WorkerReply], None]) -> None:
        self._request_id = request_id
        self._emit_reply = emit_reply
        self._tool: str | None = None
        self._t0: float = 0.0
        self._event_id: str | None = None
        self._event_counter = 0
        logger.debug(f"[_WorkerToolEventCallback] Initialized for request {request_id}")

    def _emit(self, *, name: str, message: str, data: dict[str, Any] | None = None) -> None:
        logger.debug(f"[_WorkerToolEventCallback] Emitting event: {name}")
        self._emit_reply(
            WorkerReply(
                type="event",
                request_id=self._request_id,
                payload=WorkerEvent(name=name, message=message, data=data or {}).model_dump(),
            )
        )

    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str, **kwargs: Any) -> None:
        _ = kwargs
        tool = str((serialized or {}).get("name") or "tool")
        self._tool = tool
        self._t0 = time.time()
        self._event_counter += 1
        self._event_id = f"{self._request_id}:{self._event_counter}"
        args = (input_str or "").strip().replace("\n", " ")
        if len(args) > 160:
            args = args[:160] + "…"
        msg = f"TOOL start: {tool}"
        if args:
            msg += f" | args: {args}"
        self._emit(
            name="tool_start",
            message=msg,
            data={"tool": tool, "args": args, "event_id": self._event_id},
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        _ = (output, kwargs)
        tool = self._tool or "tool"
        event_id = self._event_id
        dt = max(0.0, time.time() - self._t0)
        self._tool = None
        self._event_id = None
        self._emit(
            name="tool_end",
            message=f"TOOL end: {tool} ({dt:0.1f}s)",
            data={"tool": tool, "duration_seconds": round(dt, 3), "event_id": event_id},
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        _ = kwargs
        tool = self._tool or "tool"
        event_id = self._event_id
        self._tool = None
        self._event_id = None
        err = str(error).strip()
        msg = f"TOOL error: {tool}" + (f" - {err}" if err else "")
        self._emit(
            name="tool_error",
            message=msg,
            data={"tool": tool, "error": err, "event_id": event_id},
        )


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

    def run(self) -> int:
        logger.info("[_ToolWorkerServer] Starting worker server...")
        
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            
            logger.debug(f"[_ToolWorkerServer] Received: {_redact_debug_line(line)[:200]}")
            
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

    def _handle_run_tools(self, env: WorkerEnvelope) -> None:
        logger.info(f"[_ToolWorkerServer] Handling run_tools request: {env.request_id}")
        
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
            sanitized_env_payload = _strip_banned_keys(env.payload)
            req = ToolRunRequest.model_validate(sanitized_env_payload)
            _validate_direct_tool_request(req, repo_root=self._repo_root)

            tool_name = str(req.tool_name or "").strip()
            if tool_name:
                retry_attempt = max(0, int(req.retry_attempt or 0))
                # Scope duplicate-detection to a single turn (request_id). A new
                # turn resets the guard; otherwise the first read_file/etc. would
                # claim the tool name for the whole worker-process lifetime and
                # every later work item with the same tool_name would be wrongly
                # rejected as a duplicate. Per-fingerprint dedup already happens
                # in the queue; this only catches a tool repeated within one turn.
                if env.request_id != self._current_turn_id:
                    self._current_turn_id = env.request_id
                    self._turn_tool_state.reset()
                if retry_attempt == 0 and not self._turn_tool_state.claim(tool_name):
                    logger.warning(
                        "Blocking duplicate tool execution within turn: tool=%s turn=%s",
                        tool_name,
                        env.request_id,
                    )
                    self._emit(
                        WorkerReply(
                            type="ok",
                            request_id=env.request_id,
                            payload=ToolRunResponse(
                                answer="Tool already executed in this turn.",
                                sources=[],
                                mode="agent-tools",
                                trace=[
                                    {
                                        "tool_name": tool_name,
                                        "status": "duplicate_blocked",
                                        "result": "duplicate_blocked",
                                        "turn_id": env.request_id,
                                        "message": "Tool already executed in this turn.",
                                    }
                                ],
                                warnings=["Tool already executed in this turn."],
                            ).model_dump(),
                        )
                    )
                    return
                if retry_attempt > 0:
                    logger.info(
                        "Allowing retry tool execution for turn: tool=%s turn=%s retry_attempt=%s",
                        tool_name,
                        env.request_id,
                        retry_attempt,
                    )
                else:
                    logger.info("Registered tool execution for turn: tool=%s turn=%s", tool_name, env.request_id)
            
            tool_event_cb = _WorkerToolEventCallback(request_id=env.request_id, emit_reply=self._emit)
            response = _run_tool_request(
                ask_agent=self._ask_agent,
                req=req,
                tools_only_strict_default=self._tools_only_strict,
                callbacks=[tool_event_cb],
                repo_root=self._repo_root,
            )

            safe_payload = _strip_banned_keys(response.model_dump())
            self._emit(WorkerReply(type="ok", request_id=env.request_id, payload=safe_payload))
            logger.info(f"[_ToolWorkerServer] run_tools completed: {env.request_id}")
            
        except ToolWorkerProcessError as exc:
            logger.error(f"[_ToolWorkerServer] Raw exception type={type(exc)} message={exc}")
            logger.error(f"[_ToolWorkerServer] ToolWorkerProcessError: {exc.code} - {exc}")
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
            logger.error(f"[_ToolWorkerServer] Unexpected error: {exc}", exc_info=True)
            code = "run_failed"
            err_msg = str(exc).lower()
            retriable = _is_likely_retriable_runtime_error(exc)

            if "model_not_found" in err_msg or "404" in err_msg or "not found" in err_msg:
                code = "model_not_found"

            if _is_banned_param_provider_error(err_msg):
                code = "provider_unknown_param"
                logger.error(
                    "Provider rejected a banned parameter (likely 'status'). "
                    "Original error: %s",
                    exc,
                )
                retriable = True

            self._emit(
                WorkerReply(
                    type="error",
                    request_id=env.request_id,
                    payload=WorkerError(
                        code=code,
                        message=str(exc),
                        retriable=retriable,
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
