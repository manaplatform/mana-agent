from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import ast
import re
from time import perf_counter
from typing import Any, Literal, Sequence
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, BaseTool
from mana_agent.multi_agent.runtime.compatibility import create_chat_model
from pydantic import BaseModel, Field

from langchain_core.callbacks.base import BaseCallbackHandler
from mana_agent.analysis.models import AskResponseWithTrace, SearchHit, ToolInvocationTrace
from mana_agent.multi_agent.runtime.prompts import ASK_AGENT_SYSTEM_PROMPT
from mana_agent.analysis.chunker import CodeChunker
from mana_agent.services.structure_service import StructureService
from mana_agent.multi_agent.runtime.run_logger import LlmRunLogger
from mana_agent.config.settings import default_index_dir
from mana_agent.services.coding_memory_service import CodingMemoryService
from mana_agent.services.memory_service import EvidenceMemory
from mana_agent.services.search_service import SearchService
from mana_agent.documents.service import DocumentService
from mana_agent.search.config import SearchConfig
from mana_agent.search.decision import SearchDecisionEngine
from mana_agent.search.router import SearchRouter, SearchRouterResult
from mana_agent.tools import coding_tool_contracts_payload, extract_patch_touched_files
from mana_agent.utils.tool_policy import resolve_allowed_tools
from mana_agent.tools.repository import (
    apply_patch_batch as repo_apply_patch_batch,
    call_graph as repo_call_graph,
    dumps_tool_result,
    find_symbols as repo_find_symbols,
    git_diff as repo_git_diff,
    git_status as repo_git_status,
    repo_batch_read as repo_batch_read_files,
    repo_batch_search as repo_batch_text_search,
    list_files as repo_list_files,
    repo_search as repo_text_search,
    run_script_once as repo_run_script_once,
    verify_project as repo_verify_project,
)
from mana_agent.skills.manager import SkillManager
from mana_agent.multi_agent.tools import git_tools
from mana_agent.mcp.tools import discovered_mcp_langchain_tools

logger = logging.getLogger(__name__)


# Injected on the final step of a mutation-required run that has not yet produced
# a write. It flips the worker from "read forever" to "act": the next turn is
# bound to mutation tools only, so the run ends in a real, project-grounded file
# instead of a natural-language answer that would trip the tools-only gate.
_FORCED_WRITE_INSTRUCTION = (
    "You have gathered enough evidence from the repository. Do NOT read, search, "
    "or list anything else. Right now, in this step, call exactly one mutation "
    "tool (edit_file, multi_edit_file, apply_patch, apply_patch_batch, write_file, create_file, delete_file, document_create, document_update, or document_delete) to apply the required "
    "project-level change with its full, final, project-specific content. Update "
    "all required imports, exports, registries, routers, commands, call sites, tests, "
    "and docs, and remove stale references. Do not answer in prose and do not emit "
    "placeholder or stub content."
)


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

class _RepoSearchInput(BaseModel):
    query: str
    glob: str = "**/*"
    regex: bool = False
    limit: int = 100

class _RepoBatchReadInput(BaseModel):
    files: list[str]

class _RepoBatchSearchInput(BaseModel):
    patterns: list[dict[str, Any]]

class _RunScriptOnceInput(BaseModel):
    script: str
    cwd: str | None = None

class _ApplyPatchBatchInput(BaseModel):
    patches: list[dict[str, Any]]

class _ReadSkillInput(BaseModel):
    skill_name: str

class _ListFilesInput(BaseModel):
    glob: str = "**/*"
    limit: int = 200

class _FindSymbolsInput(BaseModel):
    query: str = ""
    limit: int = 100

class _CallGraphInput(BaseModel):
    query: str = ""
    limit: int = 100

class _GitDiffInput(BaseModel):
    path: str = ""

class _GitGenericInput(BaseModel):
    args: list[str]
    repo_path: str | None = None
    timeout: int | None = None
    allow_protected: bool = False

class _GitHelpInput(BaseModel):
    command: str | None = None
    all: bool = False
    refresh: bool = False
    repo_path: str | None = None
    timeout: int | None = None

class _GitBranchInput(BaseModel):
    all: bool = False

class _GitCreateBranchInput(BaseModel):
    branch_name: str
    switch_to: bool = True

class _GitSwitchInput(BaseModel):
    branch_name: str

class _GitAddInput(BaseModel):
    paths: list[str]

class _GitCommitInput(BaseModel):
    message: str
    amend: bool = False

class _GitPushInput(BaseModel):
    remote: str = ""
    branch_name: str = ""
    set_upstream: bool = False
    force: bool = False

class _GitRemoteInput(BaseModel):
    verbose: bool = True

class _GitLogInput(BaseModel):
    limit: int = 10
    oneline: bool = True

class _VerifyProjectInput(BaseModel):
    quick: bool = False

class _DocumentDetectInput(BaseModel):
    path: str
    mime_type: str | None = None

class _DocumentReadInput(BaseModel):
    path: str
    use_cache: bool = True
    max_chunks: int = 400

class _DocumentAnalyzeInput(BaseModel):
    path: str

class _DocumentQueryInput(BaseModel):
    query: str
    paths: list[str] | None = None
    file_types: list[str] | None = None
    path_filter: str = ""
    sheet: str = ""
    page: int | None = None
    section: str = ""
    limit: int = 10

class _DocumentCreateInput(BaseModel):
    path: str
    content: dict[str, Any]
    file_type: str | None = None
    overwrite: bool = False

class _DocumentUpdateInput(BaseModel):
    path: str
    operation: str
    payload: dict[str, Any]
    backup: bool = True

class _DocumentDeleteInput(BaseModel):
    path: str
    explicit: bool = False
    backup: bool = True


class AskAgent:
    READ_FULL_FILE_MAX_LINES = 5000
    READ_FULL_FILE_MAX_CHARS = 250000

    # Tool-loop progress guards. When this many consecutive tool results add no
    # new evidence (or are blocked duplicates), stop executing tools and
    # synthesize a best-effort final answer from the evidence already collected.
    MAX_STAGNANT_STEPS = 2

    # Tools whose calls are deduplicated semantically (similar queries collapse
    # to the same canonical intent, e.g. ``README`` / ``README.md`` / ``README*``).
    SEARCH_LIKE_TOOLS = frozenset(
        {"semantic_search", "repo_search", "repo_batch_search", "find_symbols", "call_graph", "list_files"}
    )

    _BLOCKED_PATTERNS = [
        "git rm ",
        "git reset --hard",
        "git checkout --",
        "rm -rf",
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
        self.llm = create_chat_model(api_key=api_key, model=model, base_url=base_url)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.search_service = search_service
        self.project_root = Path(project_root).resolve()
        self.coding_memory_service = coding_memory_service
        self._resolved_index = default_index_dir(self.project_root)
        self._resolved_indexes = [self._resolved_index]
        self.run_logger = LlmRunLogger()
        self.search_config = SearchConfig.from_env()

        # ✅ NEW: allow external code to register extra tools (e.g. write_file/apply_patch)
        self.tools: list[BaseTool] = []
        self.mcp_server_overrides: list[str] = []

    def update_model(self, model_name: str) -> None:
        resolved = str(model_name or "").strip()
        if not resolved or resolved == self.model:
            return
        self.llm = create_chat_model(api_key=self.api_key, model=resolved, base_url=self.base_url)
        self.model = resolved

    def _is_blocked_command(self, cmd: str) -> bool:
        lowered = f"{cmd.lower()} "
        for pattern in self._BLOCKED_PATTERNS:
            if "\\" in pattern:
                if re.search(pattern, lowered):
                    return True
            elif pattern in lowered:
                return True
        return False

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
        rewritten = subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)
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
        # Connector results may be prefixed with an untrusted-content warning.
        # Parse the full result first so formatted JSON remains valid, then the
        # payload after that single-line prefix.
        for candidate in (text, text.split("\n", 1)[-1].strip()):
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                pass
            # Some models/tools may return Python dict repr with single quotes.
            try:
                loaded = ast.literal_eval(candidate)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                pass
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

    @staticmethod
    def tool_signature(tool_name: str, args: dict[str, Any] | None) -> str:
        """Stable signature for an exact (tool, args) tuple.

        Uses sorted-key JSON so semantically identical argument dicts always
        produce the same string regardless of key ordering.
        """
        return json.dumps(
            {"tool": tool_name, "args": args or {}},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

    # Doc-ish suffixes stripped during query canonicalization so ``README`` and
    # ``README.md`` collapse together. Embedded wildcards are intentionally left
    # alone (only a trailing wildcard is dropped) so genuinely different patterns
    # like ``a*b`` and ``ab`` are not merged.
    _CANONICAL_QUERY_SUFFIXES = (".markdown", ".rst", ".txt", ".md")

    @classmethod
    def _canonical_search_query(cls, value: Any) -> str:
        """Normalize a search query/glob so near-duplicates collapse together.

        ``README``, ``README.md`` and ``README*`` all canonicalize to ``readme``.
        Only trailing wildcards and a single trailing doc suffix are stripped so
        that distinct patterns are not over-merged.
        """
        q = str(value or "").strip().lower()
        q = re.sub(r"\s+", " ", q)
        # Collapse a trailing glob wildcard only (README* -> README); keep any
        # embedded wildcards intact so different patterns stay distinct.
        q = q.rstrip("*")
        for ext in cls._CANONICAL_QUERY_SUFFIXES:
            if q.endswith(ext):
                q = q[: -len(ext)]
                break
        return q.strip().rstrip(".")

    @classmethod
    def _search_intent_signature(cls, tool_name: str, args: dict[str, Any] | None) -> str:
        """Canonical signature used to dedupe semantically similar searches.

        The signature keys on the canonical primary term *plus* the secondary
        scope arguments (``glob``/``regex`` for ``repo_search``) so two genuinely
        different searches that merely share a query term are not collapsed.
        """
        args = args or {}
        raw = args.get("query")
        if raw is None or str(raw).strip() == "":
            raw = args.get("glob", "")
        payload: dict[str, Any] = {
            "search_tool": tool_name,
            "q": cls._canonical_search_query(raw),
        }
        if tool_name == "repo_search":
            glob = str(args.get("glob", "") or "").strip().lower()
            if glob and glob != "**/*":
                payload["glob"] = glob
            if bool(args.get("regex", False)):
                payload["regex"] = True
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    @classmethod
    def _evidence_fingerprint(cls, content: Any) -> str:
        """Fingerprint a tool result so repeated identical output is detectable."""
        text = re.sub(r"\s+", " ", str(content)).strip().lower()
        if not text:
            return ""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @classmethod
    def _summarize_tool_result(cls, tool_name: str, content: Any) -> str:
        """One-line, human-readable summary of a successful tool result."""
        payload = cls._coerce_tool_payload(content)
        text = ""
        if isinstance(payload, dict):
            for key in ("summary", "content", "result", "stdout", "matches", "files", "hits"):
                value = payload.get(key)
                if value:
                    text = str(value)
                    break
        if not text:
            text = str(content)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 200:
            text = text[:200] + "…"
        return f"{tool_name}: {text}" if text else ""

    def _evidence_digest(
        self,
        *,
        user_query: str,
        observations: list[str],
        trace: list[ToolInvocationTrace],
        sources: list[SearchHit],
        reason: str,
    ) -> str:
        """Deterministic, evidence-only summary used as synthesis input/fallback."""
        parts: list[str] = [
            "I stopped the tool loop before the model produced a final answer "
            f"(reason: {reason}). Here is a best-effort summary from the evidence collected so far."
        ]
        if user_query and user_query.strip():
            parts.append(f"\nQuestion: {user_query.strip()}")

        if observations:
            parts.append("\nKey findings:")
            for obs in observations[:10]:
                parts.append(f"- {obs}")
        elif trace:
            parts.append("\nTools executed:")
            for item in trace[-10:]:
                parts.append(f"- {item.tool_name} ({item.status})")

        if sources:
            parts.append("\nRelevant sources:")
            seen_src: set[str] = set()
            for hit in sources:
                ref = f"{hit.file_path}:{hit.start_line}-{hit.end_line}"
                if ref in seen_src:
                    continue
                seen_src.add(ref)
                parts.append(f"- {ref}")
                if len(seen_src) >= 10:
                    break

        if not observations and not trace and not sources:
            parts.append(
                "\nNo tool evidence was gathered before stopping. "
                "Please refine the question or retry."
            )

        return "\n".join(parts).strip()

    def _synthesize_final_answer(
        self,
        *,
        user_query: str,
        observations: list[str],
        trace: list[ToolInvocationTrace],
        sources: list[SearchHit],
        warnings: list[str],
        reason: str,
    ) -> str:
        """Build a best-effort final answer from the evidence already collected.

        This is used whenever the tool loop stops before the model emits its own
        final answer (max steps reached, no-progress, duplicate loops, or a low
        remaining tool budget) so the user never receives an empty answer or the
        raw step-limit error string.

        A final, tool-free LLM pass turns the collected evidence into a polished
        answer. If that call is unavailable or fails for any reason we fall back
        to the deterministic evidence digest so an answer is always produced.
        """
        digest = self._evidence_digest(
            user_query=user_query,
            observations=observations,
            trace=trace,
            sources=sources,
            reason=reason,
        )

        llm = getattr(self, "llm", None)
        invoke = getattr(llm, "invoke", None)
        if invoke is None:
            return digest

        try:
            messages = [
                SystemMessage(
                    content=(
                        "You are summarizing the result of a partially-completed code "
                        "investigation. The tool loop stopped early, so no further tools "
                        "are available. Using ONLY the evidence provided, write a concise, "
                        "useful answer to the user's question. Do not invent facts beyond "
                        "the evidence; if the evidence is insufficient, say what is known "
                        "and what remains uncertain. Cite file paths/line ranges when given."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User question:\n{user_query.strip()}\n\n"
                        f"Reason the loop stopped: {reason}\n\n"
                        f"Collected evidence:\n{digest}"
                    )
                ),
            ]
            ai_msg = invoke(messages)
            text = self._extract_model_text(getattr(ai_msg, "content", ai_msg))
            if text and text.strip():
                return text.strip()
        except Exception:
            # Any failure (no network, fake LLM in tests, rate limit, …) falls
            # back to the deterministic digest below.
            pass
        return digest

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

    @classmethod
    def _tool_error_detail(cls, content: Any) -> str:
        payload = cls._coerce_tool_payload(content)
        if isinstance(payload, dict):
            return str(payload.get("error", "") or payload.get("error_code", "") or "").strip()
        return ""

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

    def _structured_read_error(self, *, path: str, exc: BaseException) -> dict[str, Any]:
        """Build a structured, machine-inspectable error for read_file failures.

        Returns a payload with an explicit ``error_code`` so recovery logic can
        branch on the failure kind instead of parsing opaque strings (e.g. a
        bare resolved path). ``error`` is kept for backward compatibility with
        existing callers/log inspectors.
        """
        # Best-effort resolved path, even when resolution itself failed.
        try:
            requested = Path(path)
            resolved = requested if requested.is_absolute() else (self.project_root / requested)
            resolved_path = str(resolved.resolve())
        except Exception:
            resolved_path = str(path)

        message = str(exc).strip()
        if isinstance(exc, FileNotFoundError):
            error_code = "file_not_found"
            # _resolve_read_path raises FileNotFoundError(str(resolved)).
            if message:
                resolved_path = message
            message = "File does not exist under repository root."
        elif isinstance(exc, PermissionError):
            error_code = "permission_denied"
            message = message or "Permission denied reading file."
        elif isinstance(exc, ValueError) and "outside project root" in message.lower():
            error_code = "path_outside_repo"
            message = "Path resolves outside the repository root."
        else:
            error_code = "read_failed"
            message = message or "Failed to read file."

        return {
            "ok": False,
            "error_code": error_code,
            "tool": "read_file",
            "path": str(path),
            "resolved_path": resolved_path,
            "error": message,
            "message": message,
        }

    @staticmethod
    def _is_binary_path(path: Path) -> bool:
        try:
            return b"\x00" in path.read_bytes()[:4096]
        except Exception:
            return True

    def _to_project_rel(self, path: str | Path) -> str:
        requested = Path(path)
        resolved = requested if requested.is_absolute() else (self.project_root / requested)
        resolved = resolved.resolve()
        return resolved.relative_to(self.project_root).as_posix()

    def _mutation_unread_targets(
        self,
        *,
        name: str,
        args: dict[str, Any],
        unique_read_files: set[str],
    ) -> list[str]:
        targets: list[str] = []
        if name in {"edit_file", "multi_edit_file", "write_file", "create_file", "delete_file", "document_create", "document_update", "document_delete"}:
            raw_path = str(args.get("path", "")).strip()
            if raw_path:
                try:
                    targets = [self._to_project_rel(raw_path)]
                except Exception:
                    return [raw_path]
        elif name == "apply_patch":
            touched = extract_patch_touched_files(str(args.get("patch", "") or ""))
            if not bool(touched.get("ok")):
                return []
            targets = [str(item) for item in touched.get("touched_files", [])]

        unread: list[str] = []
        for rel in targets:
            target = (self.project_root / rel).resolve()
            if target.exists() and rel not in unique_read_files:
                unread.append(rel)
        return unread

    def _mutation_changed_files(
        self,
        *,
        name: str,
        args: dict[str, Any],
        content: Any,
    ) -> list[str]:
        """Repo-relative files a *successful* mutation tool call changed.

        Mutation tool results (and the ``ToolInvocationTrace`` built from them)
        do not carry an explicit changed-files list, so strict mutation gates
        downstream cannot recognize a real write. We reconstruct it here from the
        tool payload, falling back to the call's own ``path``/``patch`` argument
        when the payload omits a list. Returns ``[]`` for non-mutation tools or
        when the call did not succeed.
        """
        if name not in {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "create_file", "write_file", "delete_file", "document_create", "document_update", "document_delete"}:
            return []
        if name == "apply_patch":
            if self._is_apply_patch_failure(content):
                return []
        else:
            payload = self._coerce_tool_payload(content)
            if isinstance(payload, dict):
                if payload.get("ok") is False:
                    return []
                if str(payload.get("error", "")).strip() or str(payload.get("error_code", "")).strip():
                    return []

        changed: list[str] = []
        payload = self._coerce_tool_payload(content)
        if isinstance(payload, dict):
            for key in ("files_changed", "changed_files", "modified_files", "touched_files"):
                value = payload.get(key)
                if isinstance(value, list):
                    changed.extend(str(item) for item in value if str(item).strip())
            proof = payload.get("proof")
            if isinstance(proof, dict) and isinstance(proof.get("modified_files"), list):
                changed.extend(str(item) for item in proof["modified_files"] if str(item).strip())
        if not changed:
            if name == "apply_patch":
                touched = extract_patch_touched_files(str(args.get("patch", "") or ""))
                if bool(touched.get("ok")):
                    changed.extend(str(item) for item in touched.get("touched_files", []) if str(item).strip())
            else:
                raw_path = str(args.get("path", "")).strip()
                if raw_path:
                    changed.append(raw_path)

        normalized: list[str] = []
        for item in changed:
            try:
                rel = self._to_project_rel(item)
            except Exception:
                rel = str(item).replace("\\", "/").lstrip("./")
            if rel and rel not in normalized:
                normalized.append(rel)
        return normalized

    @staticmethod
    def _document_binary_write_error(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if name not in {"write_file", "create_file"}:
            return None
        path = str(args.get("path", "") or "").strip()
        if Path(path).suffix.lower() not in {".docx", ".pdf", ".xlsx", ".xlsm"}:
            return None
        content = args.get("content", args.get("text", args.get("body", "")))
        reason = (
            "empty binary document content"
            if not str(content or "").strip()
            else "binary document target cannot be written by text file tools"
        )
        return {
            "ok": False,
            "error": reason,
            "error_code": "document_text_tool_blocked",
            "tool": name,
            "path": path,
            "message": (
                "Use document_create or document_update for Word/PDF/Excel artifacts; use run_command only "
                "for temporary helper scripts that are not committed."
            ),
        }

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
        persist_flow_cache: bool = False,
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
        if persist_flow_cache and resolved_flow and service is not None:
            try:
                service.upsert_read_cache_row(
                    flow_id=resolved_flow,
                    file_path=file_path,
                    mode=str(row.get("mode", "")),
                    start_line=int(row.get("start_line", 0) or 0),
                    end_line=int(row.get("end_line", 0) or 0),
                    line_count=int(row.get("line_count", 0) or 0),
                    content_text=str(row.get("content_text", "")),
                    file_size_bytes=int(row.get("file_size_bytes", 0) or 0),
                    file_mtime_ns=int(row.get("file_mtime_ns", 0) or 0),
                )
            except Exception:
                logger.debug("Failed to store persistent read cache row", exc_info=True)

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
                "normalized_path": file_path,
                "mode": "full",
                "start_line": 1,
                "end_line": file_line_count,
                "line_count": file_line_count,
                "content": str(row.get("content_text", "")),
                "cache_hit": True,
                "source": "memory",
                "cache_source": cache_source,
                "cache_invalidated": False,
                "full_file_cached": row_mode == "full",
                "covered_range": [1, file_line_count],
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
            "normalized_path": file_path,
            "mode": "line",
            "start_line": int(start_line),
            "end_line": actual_end,
            "line_count": file_line_count,
            "content": "\n".join(segment),
            "cache_hit": True,
            "source": "memory",
            "cache_source": cache_source,
            "cache_invalidated": False,
            "full_file_cached": row_mode == "full",
            "covered_range": [int(start_line), actual_end],
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
        run_id: str | None = None,
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
        if payload is not None:
            return True
        run_payload, _ = EvidenceMemory(repo_root=self.project_root, run_id=run_id).lookup(
            resolved=resolved,
            mode=requested_mode,
            start_line=normalized_start,
            end_line=normalized_end,
        )
        return run_payload is not None

    def _build_tools(
        self,
        k_default: int,
        timeout_seconds: int,
        read_line_window: int = 400,
        flow_id: str | None = None,
        run_id: str | None = None,
        ephemeral_read_cache: dict[str, list[dict[str, Any]]] | None = None,
        read_telemetry: dict[str, int] | None = None,
        required_mcp_server: str | None = None,
    ) -> tuple[list[BaseTool], list[ToolInvocationTrace], list[SearchHit], list[str]]:
        traces: list[ToolInvocationTrace] = []
        sources: list[SearchHit] = []
        warnings: list[str] = []
        safe_read_line_window = max(200, min(int(read_line_window or 400), 2000))
        evidence_memory = EvidenceMemory(repo_root=self.project_root, run_id=run_id)
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
                if resolved.is_dir():
                    raise ValueError(f"{path} is a directory; use list_files instead of read_file")
                if self._is_binary_path(resolved):
                    raise ValueError("binary files cannot be read by read_file")
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
                run_cached_payload, run_invalidated = evidence_memory.lookup(
                    resolved=resolved,
                    mode=requested_mode,
                    start_line=start,
                    end_line=end,
                )
                if run_invalidated:
                    invalidated = True
                if invalidated and read_telemetry is not None:
                    read_telemetry["read_cache_invalidations"] = int(read_telemetry.get("read_cache_invalidations", 0)) + 1
                if run_cached_payload is not None:
                    if read_telemetry is not None:
                        read_telemetry["read_cache_hits"] = int(read_telemetry.get("read_cache_hits", 0)) + 1
                        if requested_mode == "full":
                            read_telemetry["read_full_mode_used"] = int(read_telemetry.get("read_full_mode_used", 0)) + 1
                    run_cached_payload["cache_invalidated"] = bool(invalidated)
                    encoded = json.dumps(run_cached_payload)
                    output_preview = encoded
                    return encoded
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
                        persist_flow_cache=read_telemetry is not None or ephemeral_read_cache is not None,
                    )
                    evidence_memory.store(
                        original_path=path,
                        resolved=resolved,
                        mode="full",
                        start_line=1,
                        end_line=line_count,
                        line_count=line_count,
                        content=content_text,
                        summary=f"full file read, {line_count} lines",
                    )
                    if read_telemetry is not None:
                        read_telemetry["read_full_mode_used"] = int(read_telemetry.get("read_full_mode_used", 0)) + 1
                    result = {
                        "file_path": str(resolved),
                        "normalized_path": str(resolved),
                        "mode": "full",
                        "start_line": 1,
                        "end_line": line_count,
                        "line_count": line_count,
                        "content": content_text,
                        "cache_hit": False,
                        "source": "tool",
                        "cache_source": "disk",
                        "cache_invalidated": bool(invalidated),
                        "full_file_cached": True,
                        "covered_range": [1, line_count],
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
                    persist_flow_cache=read_telemetry is not None or ephemeral_read_cache is not None,
                )
                segment_text = "\n".join(segment)
                evidence_memory.store(
                    original_path=path,
                    resolved=resolved,
                    mode="line",
                    start_line=start,
                    end_line=actual_end,
                    line_count=line_count,
                    content=segment_text,
                    summary=f"line range read, lines {start}-{actual_end}",
                )
                result = {
                    "file_path": str(resolved),
                    "normalized_path": str(resolved),
                    "mode": "line",
                    "start_line": start,
                    "end_line": actual_end,
                    "line_count": line_count,
                    "content": segment_text,
                    "cache_hit": False,
                    "source": "tool",
                    "cache_source": "disk",
                    "cache_invalidated": bool(invalidated),
                    "full_file_cached": False,
                    "covered_range": [start, actual_end],
                }
                encoded = json.dumps(result)
                output_preview = encoded
                return encoded
            except Exception as exc:
                status = "error"
                payload = self._structured_read_error(path=path, exc=exc)
                encoded = json.dumps(payload)
                output_preview = encoded
                return encoded
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

        def repo_search(query: str, glob: str = "**/*", regex: bool = False, limit: int = 100) -> str:
            return dumps_tool_result(
                repo_text_search(self.project_root, query=query, glob=glob, regex=regex, limit=limit)
            )

        def repo_batch_read(files: list[str]) -> str:
            return dumps_tool_result(repo_batch_read_files(self.project_root, files=files))

        def repo_batch_search(patterns: list[dict[str, Any]]) -> str:
            return dumps_tool_result(repo_batch_text_search(self.project_root, patterns=patterns))

        def run_script_once(script: str, cwd: str | None = None) -> str:
            return dumps_tool_result(repo_run_script_once(self.project_root, script=script, cwd=cwd, timeout=timeout_seconds))

        def apply_patch_batch(patches: list[dict[str, Any]]) -> str:
            return dumps_tool_result(repo_apply_patch_batch(self.project_root, patches=patches))

        skill_manager = SkillManager(project_root=self.project_root)

        def read_skill(skill_name: str) -> str:
            return skill_manager.read_skill(skill_name)

        def list_files(glob: str = "**/*", limit: int = 200) -> str:
            return dumps_tool_result(repo_list_files(self.project_root, glob=glob, limit=limit))

        def find_symbols(query: str = "", limit: int = 100) -> str:
            return dumps_tool_result(repo_find_symbols(self.project_root, query=query, limit=limit))

        def call_graph(query: str = "", limit: int = 100) -> str:
            return dumps_tool_result(repo_call_graph(self.project_root, query=query, limit=limit))

        def git_status() -> str:
            return dumps_tool_result(repo_git_status(self.project_root))

        def git_diff(path: str = "") -> str:
            return dumps_tool_result(repo_git_diff(self.project_root, path=path))

        def git_help(command: str | None = None, all: bool = False, refresh: bool = False, repo_path: str | None = None, timeout: int | None = None) -> str:
            return dumps_tool_result(git_tools.git_help(command=command, all=all, refresh=refresh, repo_path=repo_path or self.project_root, timeout=timeout))

        def git_generic(args: list[str], repo_path: str | None = None, timeout: int | None = None, allow_protected: bool = False) -> str:
            return dumps_tool_result(git_tools.generic(args=args, repo_path=repo_path or self.project_root, timeout=timeout, allow_protected=allow_protected))

        def git_log(limit: int = 10, oneline: bool = True) -> str:
            return dumps_tool_result(git_tools.log(repo_path=self.project_root, limit=limit, oneline=oneline))

        def git_branch(all: bool = False) -> str:
            return dumps_tool_result(git_tools.branch(repo_path=self.project_root, all=all))

        def git_create_branch(branch_name: str, switch_to: bool = True) -> str:
            return dumps_tool_result(git_tools.create_branch(repo_path=self.project_root, branch_name=branch_name, switch_to=switch_to))

        def git_switch(branch_name: str) -> str:
            return dumps_tool_result(git_tools.switch(repo_path=self.project_root, branch_name=branch_name))

        def git_add(paths: list[str]) -> str:
            return dumps_tool_result(git_tools.add(repo_path=self.project_root, paths=paths))

        def git_commit(message: str, amend: bool = False) -> str:
            return dumps_tool_result(git_tools.commit(repo_path=self.project_root, message=message, amend=amend))

        def git_push(remote: str = "", branch_name: str = "", set_upstream: bool = False, force: bool = False) -> str:
            return dumps_tool_result(git_tools.push(repo_path=self.project_root, remote=remote, branch_name=branch_name, set_upstream=set_upstream, force=force))

        def git_remote(verbose: bool = True) -> str:
            return dumps_tool_result(git_tools.remote(repo_path=self.project_root, verbose=verbose))

        def verify_project(quick: bool = False) -> str:
            return dumps_tool_result(repo_verify_project(self.project_root, quick=quick))

        def tool_contracts() -> str:
            return dumps_tool_result(coding_tool_contracts_payload())

        document_service = DocumentService(self.project_root)

        def document_detect(path: str, mime_type: str | None = None) -> str:
            return dumps_tool_result(document_service.detect(path, mime_type=mime_type))

        def document_read(path: str, use_cache: bool = True, max_chunks: int = 400) -> str:
            return dumps_tool_result(document_service.read(path, use_cache=use_cache, max_chunks=max_chunks))

        def document_analyze(path: str) -> str:
            return dumps_tool_result(document_service.analyze(path))

        def document_query(
            query: str,
            paths: list[str] | None = None,
            file_types: list[str] | None = None,
            path_filter: str = "",
            sheet: str = "",
            page: int | None = None,
            section: str = "",
            limit: int = 10,
        ) -> str:
            return dumps_tool_result(
                document_service.query(
                    query,
                    paths=paths,
                    file_types=file_types,
                    path_filter=path_filter,
                    sheet=sheet,
                    page=page,
                    section=section,
                    limit=limit,
                )
            )

        def document_create(path: str, content: dict[str, Any], file_type: str | None = None, overwrite: bool = False) -> str:
            return dumps_tool_result(document_service.create(path, content=content, file_type=file_type, overwrite=overwrite))

        def document_update(path: str, operation: str, payload: dict[str, Any], backup: bool = True) -> str:
            return dumps_tool_result(document_service.update(path, operation=operation, payload=payload, backup=backup))

        def document_delete(path: str, explicit: bool = False, backup: bool = True) -> str:
            return dumps_tool_result(document_service.delete(path, explicit=explicit, backup=backup))

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
            StructuredTool.from_function(
                func=repo_search,
                name="repo_search",
                description="Search repository text with regex or literal matching. Read-only.",
                args_schema=_RepoSearchInput,
            ),
            StructuredTool.from_function(
                func=repo_batch_read,
                name="repo_batch_read",
                description="Read multiple repository files in one call with per-file errors and truncation metadata.",
                args_schema=_RepoBatchReadInput,
            ),
            StructuredTool.from_function(
                func=repo_batch_search,
                name="repo_batch_search",
                description="Run multiple repository text searches in one call and return grouped JSON results.",
                args_schema=_RepoBatchSearchInput,
            ),
            StructuredTool.from_function(
                func=run_script_once,
                name="run_script_once",
                description="Run one grouped, non-destructive shell script in the repository and return exit code/output/duration.",
                args_schema=_RunScriptOnceInput,
            ),
            StructuredTool.from_function(
                func=apply_patch_batch,
                name="apply_patch_batch",
                description="Validate and apply multiple related Codex patch payloads in one call.",
                args_schema=_ApplyPatchBatchInput,
            ),
            StructuredTool.from_function(
                func=read_skill,
                name="read_skill",
                description="Load one full skills/<skill_name>/SKILL.md body on demand after matching the compact skill index.",
                args_schema=_ReadSkillInput,
            ),
            StructuredTool.from_function(
                func=list_files,
                name="list_files",
                description="List repository files relative to the project root. Read-only.",
                args_schema=_ListFilesInput,
            ),
            StructuredTool.from_function(
                func=find_symbols,
                name="find_symbols",
                description="Find Python classes/functions/methods using AST. Read-only.",
                args_schema=_FindSymbolsInput,
            ),
            StructuredTool.from_function(
                func=call_graph,
                name="call_graph",
                description="Inspect Python AST call edges by caller, callee, or file path. Read-only.",
                args_schema=_CallGraphInput,
            ),
            StructuredTool.from_function(
                func=git_status,
                name="git_status",
                description="Return git status --short. Read-only.",
                args_schema=_LsInput,
            ),
            StructuredTool.from_function(
                func=git_diff,
                name="git_diff",
                description="Return git diff, optionally for one project-relative path. Read-only.",
                args_schema=_GitDiffInput,
            ),
            StructuredTool.from_function(
                func=git_help,
                name="git_help",
                description="Return Git help or dynamically discovered commands from git help -a. Use before git_generic for uncommon commands.",
                args_schema=_GitHelpInput,
            ),
            StructuredTool.from_function(
                func=git_generic,
                name="git_generic",
                description="Run a model-selected Git argv list through the shared safe executor. Omit the leading git executable.",
                args_schema=_GitGenericInput,
            ),
            StructuredTool.from_function(
                func=git_log,
                name="git_log",
                description="Inspect recent commit history. Read-only.",
                args_schema=_GitLogInput,
            ),
            StructuredTool.from_function(
                func=git_branch,
                name="git_branch",
                description="List branches. Read-only.",
                args_schema=_GitBranchInput,
            ),
            StructuredTool.from_function(
                func=git_create_branch,
                name="git_create_branch",
                description="Create a safely named branch after the model has inspected repository state.",
                args_schema=_GitCreateBranchInput,
            ),
            StructuredTool.from_function(
                func=git_switch,
                name="git_switch",
                description="Switch to an existing safely named branch.",
                args_schema=_GitSwitchInput,
            ),
            StructuredTool.from_function(
                func=git_add,
                name="git_add",
                description="Stage specific relevant paths only. Do not use for blanket staging unless the model verified all changes are relevant.",
                args_schema=_GitAddInput,
            ),
            StructuredTool.from_function(
                func=git_commit,
                name="git_commit",
                description="Create or amend a commit with a message generated from staged diff, request, files changed, and verification.",
                args_schema=_GitCommitInput,
            ),
            StructuredTool.from_function(
                func=git_push,
                name="git_push",
                description="Push after inspecting status, branch, remotes, and upstream. Force push is protected and blocked by default.",
                args_schema=_GitPushInput,
            ),
            StructuredTool.from_function(
                func=git_remote,
                name="git_remote",
                description="Inspect configured remotes. Read-only.",
                args_schema=_GitRemoteInput,
            ),
            StructuredTool.from_function(
                func=verify_project,
                name="verify_project",
                description="Run standard pytest/ruff/mypy/import/CLI smoke checks and report skipped tools.",
                args_schema=_VerifyProjectInput,
            ),
            StructuredTool.from_function(
                func=tool_contracts,
                name="tool_contracts",
                description="Return strict name/schema/output/error/safety/examples contracts for coding tools.",
                args_schema=_ListToolsInput,
            ),
            StructuredTool.from_function(
                func=document_detect,
                name="document_detect",
                description="Detect supported project document files by path, extension, and MIME when available.",
                args_schema=_DocumentDetectInput,
            ),
            StructuredTool.from_function(
                func=document_read,
                name="document_read",
                description="Read DOCX, PDF, XLSX/XLSM, or CSV files into normalized chunks with citation metadata.",
                args_schema=_DocumentReadInput,
            ),
            StructuredTool.from_function(
                func=document_analyze,
                name="document_analyze",
                description="Analyze document structure, key points, tables, PDF OCR needs, and workbook schemas/formulas.",
                args_schema=_DocumentAnalyzeInput,
            ),
            StructuredTool.from_function(
                func=document_query,
                name="document_query",
                description="Search parsed document chunks with optional file-type, path, sheet, page, or section filters.",
                args_schema=_DocumentQueryInput,
            ),
            StructuredTool.from_function(
                func=document_create,
                name="document_create",
                description="Create DOCX, XLSX/XLSM, CSV, or simple text PDF artifacts without overwriting by default.",
                args_schema=_DocumentCreateInput,
            ),
            StructuredTool.from_function(
                func=document_update,
                name="document_update",
                description="Safely update supported document files with backups unless disabled.",
                args_schema=_DocumentUpdateInput,
            ),
            StructuredTool.from_function(
                func=document_delete,
                name="document_delete",
                description="Delete a supported document file only when explicit delete intent is validated.",
                args_schema=_DocumentDeleteInput,
            ),
        ]

        # Starting a stdio MCP server is an external side effect. Registered
        # providers are not discovered for every chat turn; discovery happens
        # only when the caller explicitly selected an MCP provider.
        mcp_overrides = list(getattr(self, "mcp_server_overrides", []) or [])
        selected_mcp_servers = [str(required_mcp_server).strip()] if str(required_mcp_server or "").strip() else []
        mcp_tools, mcp_warnings = (
            discovered_mcp_langchain_tools(
                overrides=mcp_overrides,
                server_ids=selected_mcp_servers,
            )
            if mcp_overrides or selected_mcp_servers
            else ([], [])
        )
        warnings.extend(mcp_warnings)
        # include externally-registered tools (write_file/apply_patch/etc)
        from mana_agent.connectors.email.runtime_tools import build_email_langchain_tools

        from mana_agent.config.user_config import get_setting

        browser_tools = []
        if bool(get_setting("MANA_BROWSER_ENABLED", True)):
            try:
                from mana_agent.connectors.browser.runtime_tools import build_browser_langchain_tools
            except ImportError:
                pass
            else:
                browser_tools = build_browser_langchain_tools()

        # Account metadata is local; Gmail is contacted only if the model calls
        # one of these explicitly selected tools.
        all_tools = [
            *base_tools,
            *build_email_langchain_tools(),
            *browser_tools,
            *mcp_tools,
            *list(getattr(self, "tools", []) or []),
        ]
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
        run_id: str | None = None,
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
            run_id=run_id,
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
        run_id: str | None = None,
        required_mcp_server: str | None = None,
    ) -> AskResponseWithTrace:
        started = perf_counter()

        self._resolved_index = Path(index_dir).resolve()
        if index_dirs:
            self._resolved_indexes = sorted({Path(item).resolve() for item in index_dirs}, key=lambda item: str(item))
        else:
            self._resolved_indexes = [self._resolved_index]

        policy = dict(tool_policy or {})
        external_search_result = self._prepare_external_search_context(
            question=question,
            system_prompt=system_prompt,
            run_id=run_id,
            disabled=bool(policy.get("disable_external_search")),
        )
        if external_search_result is not None:
            context_block = external_search_result.context_block(
                max_results=self.search_config.max_injected_results,
                max_words=self.search_config.max_summary_words,
            )
            if context_block:
                question = f"{question}\n\n{context_block}"
        # Expand high-level aliases (e.g. "file_system") into concrete tool
        # names before enforcement so an unexpanded alias never blocks real
        # tools like ls/list_files/read_file/repo_search.
        raw_allowed = [str(x) for x in (policy.get("allowed_tools") or []) if str(x).strip()]
        allowed_tools = set(resolve_allowed_tools(raw_allowed, strict=False))
        allowed_tools.update(name for name in raw_allowed if name.startswith(("mcp.", "mcp__")))
        search_budget = int(policy.get("search_budget", 0) or 0)
        read_budget = int(policy.get("read_budget", 0) or 0)
        read_line_window = max(200, min(int(policy.get("read_line_window", 400) or 400), 2000))
        require_read_files = int(policy.get("require_read_files", 0) or 0)
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
        evidence_memory = EvidenceMemory(repo_root=self.project_root, run_id=run_id)

        tools, traces, sources, warnings = self._build_tools(
            k_default=k,
            timeout_seconds=timeout_seconds,
            read_line_window=read_line_window,
            flow_id=flow_id,
            run_id=run_id,
            ephemeral_read_cache=ephemeral_read_cache,
            read_telemetry=read_telemetry,
            required_mcp_server=required_mcp_server,
        )
        pending_external_traces = list(getattr(self, "_pending_external_search_traces", []) or [])
        if pending_external_traces:
            traces.extend(pending_external_traces)
            self._pending_external_search_traces = []
        tool_map = {tool.name: tool for tool in tools}
        # Preserve explicitly model-selected registered tools that are not part
        # of the repository alias registry (for example browser_* connectors).
        # Unknown names remain excluded and never widen the policy.
        allowed_tools.update(name for name in raw_allowed if name in tool_map)

        bound_tools = [tool for tool in tools if not allowed_tools or tool.name in allowed_tools]
        bound = self.llm.bind_tools(bound_tools)
        bound_initial_required = (
            self.llm.bind_tools(bound_tools, tool_choice="required")
            if bool(policy.get("require_initial_tool_call"))
            else None
        )

        # Mutation-required runs (the forced-write deliverable path) must end in a
        # real write. Prepare a mutation-only bound model so that, when the step
        # budget is nearly spent without a successful mutation, the final step can
        # only call a write tool instead of synthesizing a prose answer that would
        # later trip the tools-only gate. tool_choice support varies by provider,
        # so degrade gracefully (required -> any -> plain bind).
        mutation_required = bool(policy.get("mutation_required") or policy.get("mutation_strict"))
        mutation_tool_names = [
            name
            for name in (
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
            )
            if name in tool_map and (not allowed_tools or name in allowed_tools)
        ]
        bound_mutation = None
        if mutation_required and mutation_tool_names:
            mutation_tools = [tool_map[name] for name in mutation_tool_names]
            for choice in ("required", "any"):
                try:
                    bound_mutation = self.llm.bind_tools(mutation_tools, tool_choice=choice)
                    break
                except Exception:
                    bound_mutation = None
            if bound_mutation is None:
                bound_mutation = self.llm.bind_tools(mutation_tools)
        mutation_succeeded = False
        forced_write_done = False

        messages = [
            SystemMessage(content=system_prompt or ASK_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        cfg: dict[str, Any] = {"callbacks": list(callbacks) if callbacks else []}
        seen_tool_args: dict[tuple[str, str], int] = defaultdict(int)
        tool_counts: dict[str, int] = defaultdict(int)
        unique_read_files: set[str] = set()
        unique_read_files.update(evidence_memory.read_files())
        disk_read_count = 0

        # Loop-progress guards.
        seen_tool_calls: set[str] = set()  # exact (tool, args) signatures
        seen_search_intents: set[str] = set()  # canonical search-intent signatures
        evidence_fingerprints: set[str] = set()  # fingerprints of useful results
        observations: list[str] = []  # human-readable evidence collected so far
        stagnant_steps = 0
        force_synthesis_reason: str | None = None

        # Tool-call metrics: distinguish attempted / successful / failed /
        # blocked-by-policy so blocked calls are never hidden as "0 tool calls".
        tool_metrics: dict[str, int] = {
            "tool_calls_attempted": 0,
            "tool_calls_successful": 0,
            "tool_calls_failed": 0,
            "tool_calls_blocked_by_policy": 0,
        }
        tool_errors: list[dict[str, Any]] = []
        # Repeated-failure guard keyed by (tool, path, mode). A missing file is
        # attempted at most twice per flow before we stop and force a fallback.
        REPEATED_FAILURE_LIMIT = 2
        read_failure_counts: dict[tuple[str, str, str], int] = defaultdict(int)

        def _read_failure_key(args: dict[str, Any]) -> tuple[str, str, str]:
            return (
                "read_file",
                str(args.get("path", "")).strip(),
                self._normalize_read_mode(args.get("mode")),
            )

        def record_metrics(tool_name: str, tool_result: Any, status: str) -> None:
            tool_metrics["tool_calls_attempted"] += 1
            if status == "blocked":
                tool_metrics["tool_calls_blocked_by_policy"] += 1
            elif status == "error":
                tool_metrics["tool_calls_failed"] += 1
                payload = self._coerce_tool_payload(tool_result)
                entry: dict[str, Any] = {"tool": tool_name}
                if isinstance(payload, dict):
                    if payload.get("error_code"):
                        entry["error_code"] = str(payload.get("error_code"))
                    if payload.get("path"):
                        entry["path"] = str(payload.get("path"))
                tool_errors.append(entry)
            else:
                tool_metrics["tool_calls_successful"] += 1

        final_answer = ""

        def safe_tool_args(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
            sanitized = dict(tool_args)
            if tool_name == "browser_type" and "value" in sanitized:
                sanitized["value"] = "[REDACTED]"
            if tool_name.startswith("browser_"):
                for key in ("password", "token", "cookie", "authorization"):
                    if key in sanitized:
                        sanitized[key] = "[REDACTED]"
            return sanitized

        def persist_tool_call(tool_name: str, tool_args: dict[str, Any], tool_result: Any, status: str) -> None:
            record_metrics(tool_name, tool_result, status)
            if not flow_id or self.coding_memory_service is None:
                return
            try:
                self.coding_memory_service.record_tool_call(
                    flow_id=flow_id,
                    tool_name=tool_name,
                    arguments=safe_tool_args(tool_name, tool_args),
                    result=tool_result,
                    status=status,
                )
                if tool_name == "verify_project":
                    payload = self._coerce_tool_payload(tool_result)
                    if payload is not None:
                        self.coding_memory_service.record_verification_result(
                            flow_id=flow_id,
                            results=payload,
                            status="passed" if bool(payload.get("ok")) else "failed",
                        )
            except Exception:
                return

        for step_idx in range(max_steps):
            need_forced_write = (
                mutation_required and not mutation_succeeded and bound_mutation is not None
            )
            # When the remaining tool budget is too low to make further progress,
            # stop calling tools and synthesize a final answer from the evidence
            # gathered so far rather than risk ending with no answer at all.
            remaining_steps = max_steps - step_idx
            if remaining_steps <= 1 and step_idx > 0 and not final_answer:
                # A mutation-required run must not bail to a natural-language
                # answer here: that guarantees zero mutations and a downstream
                # tools_only_violation. Spend the final step forcing a write.
                if need_forced_write and not forced_write_done:
                    forced_write_done = True
                    messages.append(HumanMessage(content=_FORCED_WRITE_INSTRUCTION))
                else:
                    force_synthesis_reason = force_synthesis_reason or "remaining_tool_budget_low"
                    break

            # Once a forced write is in flight, restrict the model to mutation
            # tools so it can only act, never read again, until a write lands.
            use_bound = (
                bound_mutation
                if (forced_write_done and not mutation_succeeded and bound_mutation is not None)
                else (bound_initial_required if step_idx == 0 and bound_initial_required is not None else bound)
            )
            try:
                ai_msg = use_bound.invoke(messages, config=cfg)
            except TypeError:
                ai_msg = use_bound.invoke(messages)
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                # The model tried to answer in prose. For a mutation-required run
                # with no write yet, force a mutation-only step instead of
                # accepting a prose answer that cannot satisfy the deliverable.
                if need_forced_write and not forced_write_done:
                    forced_write_done = True
                    messages.append(HumanMessage(content=_FORCED_WRITE_INSTRUCTION))
                    continue
                final_answer = self._extract_model_text(ai_msg.content) or str(ai_msg.content)
                break

            for call in tool_calls:
                name = str(call.get("name", ""))
                args = call.get("args", {}) or {}
                args_key = json.dumps(args, sort_keys=True, default=str)
                tool_sig = (name, args_key)
                seen_tool_args[tool_sig] += 1
                exact_signature = self.tool_signature(name, args if isinstance(args, dict) else {})
                search_signature = (
                    self._search_intent_signature(name, args if isinstance(args, dict) else {})
                    if name in self.SEARCH_LIKE_TOOLS
                    else None
                )
                is_duplicate_call = False

                if name not in tool_map:
                    content = json.dumps({"error": f"unknown tool: {name}"})
                    persist_tool_call(name, args if isinstance(args, dict) else {}, content, "error")
                elif allowed_tools and name not in allowed_tools:
                    content = json.dumps({"error": f"tool blocked by policy: {name}"})
                    persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                elif name != "read_file" and (
                    exact_signature in seen_tool_calls
                    or (search_signature is not None and search_signature in seen_search_intents)
                ):
                    # Exact-duplicate or semantically-equivalent search already run.
                    is_duplicate_call = True
                    content = json.dumps(
                        {
                            "error": (
                                f"Duplicate tool call blocked: {name}. This (or an equivalent) "
                                "call already ran; use a different step or provide the final answer."
                            )
                        }
                    )
                    warning = f"Duplicate tool call blocked: {name}"
                    if warning not in warnings:
                        warnings.append(warning)
                    persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                elif name != "read_file" and seen_tool_args[tool_sig] > 2:
                    content = json.dumps(
                        {
                            "error": (
                                f"duplicate tool call blocked: {name}. "
                                "Use a different step (repo_batch_read/read_file/edit_file/multi_edit_file/apply_patch/apply_patch_batch/write_file/create_file/document_create) instead of repeating."
                            )
                        }
                    )
                    persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                else:
                    # Record signatures so the next identical/equivalent call is
                    # recognized as a duplicate (read_file is intentionally
                    # exempt: re-reads are idempotent and cache-backed).
                    if name != "read_file":
                        seen_tool_calls.add(exact_signature)
                        if search_signature is not None:
                            seen_search_intents.add(search_signature)
                    if name == "semantic_search":
                        if search_budget > 0 and tool_counts["semantic_search"] >= search_budget:
                            content = json.dumps({"error": "semantic_search budget exhausted"})
                            persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                        k_val = int(args.get("k", 0) or 0)
                        if k_val > max_semantic_k:
                            content = json.dumps({"error": f"semantic_search k must be <= {max_semantic_k}"})
                            persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                        normalized = self._normalize_search_key(args)
                        search_seen[normalized] += 1
                        if search_seen[normalized] > search_repeat_limit:
                            content = json.dumps(
                                {"error": "duplicate semantic_search intent blocked; move to read_file or edit phase"}
                            )
                            persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                    if name == "read_file":
                        read_args = args if isinstance(args, dict) else {}
                        failure_key = _read_failure_key(read_args)
                        if read_failure_counts[failure_key] >= REPEATED_FAILURE_LIMIT:
                            target = failure_key[1] or "(unknown path)"
                            content = json.dumps(
                                {
                                    "ok": False,
                                    "error_code": "tool_blocked_by_policy",
                                    "tool": "read_file",
                                    "path": target,
                                    "error": (
                                        f"read_file({target}) failed {read_failure_counts[failure_key]} times; "
                                        "stop retrying it. Use list_files or repo_search to discover the correct "
                                        "path, then move on to the next step."
                                    ),
                                }
                            )
                            persist_tool_call(name, read_args, content, "blocked")
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                        if (
                            read_budget > 0
                            and disk_read_count >= read_budget
                            and not self._can_serve_read_from_cache(
                                path=str(read_args.get("path", "")),
                                mode=str(read_args.get("mode", "line")),
                                start_line=int(read_args.get("start_line", 1) or 1),
                                end_line=int(read_args.get("end_line", 200) or 200),
                                flow_id=flow_id,
                                run_id=run_id,
                                ephemeral_read_cache=ephemeral_read_cache,
                                line_window=read_line_window,
                            )
                        ):
                            content = json.dumps({"error": "read_file budget exhausted"})
                            persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                    if name in {"write_file", "create_file"}:
                        document_write_error = self._document_binary_write_error(
                            name=name,
                            args=args if isinstance(args, dict) else {},
                        )
                        if document_write_error is not None:
                            content = json.dumps(document_write_error)
                            persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                    if name in {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "create_file", "write_file", "delete_file", "document_create", "document_update", "document_delete"} and require_read_files > 0:
                        if len(unique_read_files) < require_read_files:
                            content = json.dumps(
                                {
                                    "error": (
                                        f"mutation blocked by policy: inspect at least {require_read_files} unique files first"
                                    )
                                }
                            )
                            persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                    if name in {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "create_file", "write_file", "delete_file", "document_create", "document_update", "document_delete"}:
                        unread_targets = self._mutation_unread_targets(
                            name=name,
                            args=args if isinstance(args, dict) else {},
                            unique_read_files=unique_read_files,
                        )
                        if unread_targets:
                            content = json.dumps(
                                {
                                    "ok": False,
                                    "error": (
                                        "mutation blocked by policy: target files must be read before editing"
                                    ),
                                    "unread_files": unread_targets,
                                }
                            )
                            persist_tool_call(name, args if isinstance(args, dict) else {}, content, "blocked")
                            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))
                            continue
                    try:
                        trace_count_before = len(traces)
                        tool_started = perf_counter()
                        try:
                            content = tool_map[name].invoke(args, config=cfg)
                        except TypeError:
                            content = tool_map[name].invoke(args)
                        if len(traces) == trace_count_before:
                            traces.append(
                                ToolInvocationTrace(
                                    tool_name=name,
                                    args_summary=json.dumps(safe_tool_args(name, args if isinstance(args, dict) else {}), sort_keys=True, default=str)[:500],
                                    duration_ms=(perf_counter() - tool_started) * 1000,
                                    status="ok" if not self._tool_error_detail(content) else "error",
                                    output_preview=str(content)[:4000],
                                    changed_files=self._mutation_changed_files(
                                        name=name,
                                        args=args if isinstance(args, dict) else {},
                                        content=content,
                                    ),
                                )
                            )
                    except Exception as exc:
                        content = json.dumps({"error": str(exc)})
                        traces.append(
                            ToolInvocationTrace(
                                tool_name=name,
                                args_summary=json.dumps(safe_tool_args(name, args if isinstance(args, dict) else {}), sort_keys=True, default=str)[:500],
                                duration_ms=0.0,
                                status="error",
                                output_preview=str(exc)[:4000],
                            )
                        )
                    persist_tool_call(
                        name,
                        args if isinstance(args, dict) else {},
                        content,
                        "error" if self._tool_error_detail(content) else "ok",
                    )
                    # No-progress detection: a successful result that adds new
                    # evidence resets the stagnation counter; an executed result
                    # that repeats earlier evidence increments it. Errors also
                    # count as no-progress (even when the error *text* changes),
                    # except for read_file, which has a dedicated repeated-failure
                    # limit and would otherwise be cut off early.
                    if self._tool_error_detail(content):
                        if name != "read_file":
                            stagnant_steps += 1
                    else:
                        fingerprint = self._evidence_fingerprint(content)
                        if fingerprint and fingerprint not in evidence_fingerprints:
                            evidence_fingerprints.add(fingerprint)
                            stagnant_steps = 0
                            observation = self._summarize_tool_result(name, content)
                            if observation:
                                observations.append(observation)
                        else:
                            stagnant_steps += 1

                # A blocked duplicate makes no progress either.
                if is_duplicate_call:
                    stagnant_steps += 1

                tool_counts[name] += 1
                if name == "read_file":
                    payload = self._coerce_tool_payload(content)
                    if payload is not None:
                        read_failed = (
                            payload.get("ok") is False
                            or bool(str(payload.get("error_code", "")).strip())
                            or bool(str(payload.get("error", "")).strip())
                        )
                        if read_failed:
                            read_failure_counts[_read_failure_key(args if isinstance(args, dict) else {})] += 1
                        file_path = str(payload.get("file_path", "")).strip()
                        if file_path:
                            try:
                                unique_read_files.add(self._to_project_rel(file_path))
                            except Exception:
                                unique_read_files.add(file_path)
                        if not bool(payload.get("cache_hit", False)) and not read_failed:
                            disk_read_count += 1
                if name in {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "create_file", "write_file", "delete_file", "document_create", "document_update", "document_delete"}:
                    changed_paths = self._mutation_changed_files(
                        name=name,
                        args=args if isinstance(args, dict) else {},
                        content=content,
                    )
                    if changed_paths:
                        evidence_memory.invalidate_many(set(changed_paths))
                        mutation_succeeded = True
                messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))

                if stagnant_steps >= self.MAX_STAGNANT_STEPS and not force_synthesis_reason:
                    force_synthesis_reason = "no_progress"

            # Stop the loop early when no further progress is being made so we
            # synthesize an answer from the evidence collected rather than spin.
            if force_synthesis_reason and not final_answer:
                break

        # Final-answer fallback: never surface the raw step-limit string. Always
        # synthesize a best-effort answer from the collected evidence/trace.
        if not final_answer:
            reason = force_synthesis_reason or "max_steps_reached"
            final_answer = self._synthesize_final_answer(
                user_query=question,
                observations=observations,
                trace=traces,
                sources=sources,
                warnings=warnings,
                reason=reason,
            )
            reason_messages = {
                "max_steps_reached": "Tool loop reached max_steps; returned best-effort final answer.",
                "no_progress": "Tool loop stopped after no-progress detection; returned best-effort final answer.",
                "remaining_tool_budget_low": (
                    "Tool loop stopped due to low remaining tool budget; returned best-effort final answer."
                ),
            }
            warnings.append(reason_messages.get(reason, f"Tool loop stopped ({reason}); returned best-effort final answer."))

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
                    "tool_calls_attempted": int(tool_metrics["tool_calls_attempted"]),
                    "tool_calls_successful": int(tool_metrics["tool_calls_successful"]),
                    "tool_calls_failed": int(tool_metrics["tool_calls_failed"]),
                    "tool_calls_blocked_by_policy": int(tool_metrics["tool_calls_blocked_by_policy"]),
                    "tool_errors": tool_errors,
                    "trace": [item.to_dict() for item in traces],
                    "sources_count": len(result.sources),
                    "sources": [item.to_dict() for item in result.sources],
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                    "answer": result.answer,
                    "external_search": external_search_result.decision.to_dict() if external_search_result else None,
                    "external_search_results": (
                        [item.to_dict() for item in external_search_result.results]
                        if external_search_result
                        else []
                    ),
                    "external_search_memory_hits": (
                        [item.to_dict() for item in external_search_result.memory_hits]
                        if external_search_result
                        else []
                    ),
                }
            )

        return result

    def _prepare_external_search_context(
        self,
        *,
        question: str,
        system_prompt: str | None,
        run_id: str | None,
        disabled: bool = False,
    ) -> SearchRouterResult | None:
        config = getattr(self, "search_config", None) or SearchConfig.from_env()
        self.search_config = config
        if disabled or not config.enable_ask_agent:
            return None
        guardrail = SearchDecisionEngine(llm=None, config=config).decide(
            user_query=question,
            repo_context=system_prompt or "",
            memory_context=f"run_id={run_id or ''}",
        )
        if not guardrail.needs_search:
            return None
        try:
            router = SearchRouter(root=str(self.project_root), llm=getattr(self, "llm", None), config=config)
            result = router.run(
                user_query=question,
                repo_context=system_prompt or "",
                memory_context=f"run_id={run_id or ''}",
                task_id=run_id,
            )
        except Exception as exc:
            logger.debug("external search routing failed: %s", exc, exc_info=True)
            return None
        traces = list(getattr(self, "_pending_external_search_traces", []) or [])
        traces.append(
            ToolInvocationTrace(
                tool_name="🔎 Search decision",
                args_summary=f"{result.decision.mode}: {result.decision.reason}"[:500],
                duration_ms=0.0,
                status="ok",
                output_preview=(
                    f"mode={result.decision.mode} results={len(result.results)} "
                    f"memory_hits={len(result.memory_hits)}"
                ),
            )
        )
        if result.memory_hits:
            traces.append(
                ToolInvocationTrace(
                    tool_name="🧠 Reusing search memory",
                    args_summary=", ".join(item.title for item in result.memory_hits[:3])[:500],
                    duration_ms=0.0,
                    status="ok",
                    output_preview=result.context_block(
                        max_results=config.max_injected_results,
                        max_words=config.max_summary_words,
                    )[:4000],
                )
            )
        if result.results or result.memory_hits:
            result_targets = {item.source_type for item in result.results}
            if result_targets == {"web"}:
                context_tool_name = "🌐 Searching web"
            elif result_targets == {"github"}:
                context_tool_name = "🔎 Searching GitHub"
            elif result_targets:
                context_tool_name = "🔎 External search context"
            else:
                context_tool_name = "External search context"
            traces.append(
                ToolInvocationTrace(
                    tool_name=context_tool_name,
                    args_summary=f"results={len(result.results)} memory_hits={len(result.memory_hits)}",
                    duration_ms=0.0,
                    status="ok",
                    output_preview=result.context_block(
                        max_results=config.max_injected_results,
                        max_words=config.max_summary_words,
                    )[:4000],
                )
            )
        self._pending_external_search_traces = traces
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
        run_id: str | None = None,
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
        required_mcp_server = kwargs.pop("required_mcp_server", None)

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
                run_id=run_id,
                required_mcp_server=required_mcp_server,
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
            run_id=run_id,
            required_mcp_server=required_mcp_server,
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
