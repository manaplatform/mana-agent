"""Machine-readable contracts for coding-agent tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolContract(BaseModel):
    """Strict contract metadata exposed to agents and tests."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    error_format: dict[str, Any]
    safety_rules: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _excel_cell_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "cell": {"type": "string"},
            "value": {},
            "formula": {"type": "string"},
        },
        "required": ["cell"],
        "anyOf": [
            {"required": ["value"]},
            {"required": ["formula"]},
        ],
        "additionalProperties": False,
    }


def _excel_table_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "start_cell": {"type": "string"},
            "columns": {
                "type": "array",
                "items": {},
            },
            "headers": {
                "type": "array",
                "items": {},
            },
            "rows": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {},
                },
            },
            "name": {"type": "string"},
            "style": {"type": "string"},
        },
        "required": ["rows"],
        "additionalProperties": False,
    }


def _excel_sheet_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "cells": {
                "type": "array",
                "items": _excel_cell_schema(),
            },
            "tables": {
                "type": "array",
                "items": _excel_table_schema(),
            },
            "formulas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cell": {"type": "string"},
                        "formula": {"type": "string"},
                    },
                    "required": ["cell", "formula"],
                    "additionalProperties": False,
                },
            },
            "rows": {
                "type": "array",
                "items": {},
            },
        },
        "additionalProperties": False,
    }


def _document_create_content_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "paragraphs": {
                "type": "array",
                "items": {"type": "string"},
            },
            "tables": {
                "type": "array",
                "items": {},
            },
            "text": {"type": "string"},
            "rows": {
                "type": "array",
                "items": {},
            },
            "sheets": {
                "type": "object",
                "description": (
                    "Excel sheets MUST be an object keyed by sheet name. "
                    "Never pass sheets as a list. Correct: "
                    "{'Sheet1': {'cells': [{'cell': 'A1', 'value': 200}]}}."
                ),
                "additionalProperties": _excel_sheet_schema(),
            },
        },
        "additionalProperties": False,
    }


def _git_tool_contracts(common_error: dict[str, Any]) -> list[ToolContract]:
    base_output = _schema(
        {
            "ok": {"type": "boolean"},
            "command": {"type": "array"},
            "repo_root": {"type": "string"},
            "risk_level": {
                "enum": [
                    "READ_ONLY",
                    "LOCAL_SAFE_WRITE",
                    "LOCAL_HISTORY_WRITE",
                    "REMOTE_WRITE",
                    "DESTRUCTIVE",
                    "HISTORY_REWRITE",
                ]
            },
            "returncode": {"type": ["integer", "null"]},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "duration_ms": {"type": "number"},
            "blocked": {"type": "boolean"},
            "state": {"type": "object"},
        }
    )
    safety = [
        "Use only after a model decision selects a Git action.",
        "All commands run as argv lists through subprocess.run with shell=False.",
        "Protected destructive/history-rewrite commands are blocked unless explicit user intent is validated.",
        "Inspect status/diff/staged diff before committing; stage only relevant files.",
        "Inspect status/current branch/remotes/upstream before pushing; never force-push by default.",
    ]
    specs: list[tuple[str, str, dict[str, Any], list[str] | None]] = [
        ("git.status", "Inspect repository status.", {"repo_path": {"type": "string"}, "short": {"type": "boolean"}, "porcelain": {"type": "boolean"}}, None),
        ("git.diff", "Inspect unstaged or staged diff.", {"repo_path": {"type": "string"}, "path": {"type": "string"}, "staged": {"type": "boolean"}}, None),
        ("git.log", "Inspect commit history.", {"repo_path": {"type": "string"}, "limit": {"type": "integer"}, "oneline": {"type": "boolean"}}, None),
        ("git.show", "Inspect a revision.", {"repo_path": {"type": "string"}, "revision": {"type": "string"}, "stat": {"type": "boolean"}}, None),
        ("git.branch", "List local or all branches.", {"repo_path": {"type": "string"}, "all": {"type": "boolean"}}, None),
        ("git.switch", "Switch to an existing branch.", {"repo_path": {"type": "string"}, "branch_name": {"type": "string"}}, ["branch_name"]),
        ("git.checkout", "Checkout a target or create a branch with checkout -b.", {"repo_path": {"type": "string"}, "target": {"type": "string"}, "new_branch": {"type": "boolean"}}, ["target"]),
        ("git.create_branch", "Create a branch, optionally switching to it.", {"repo_path": {"type": "string"}, "branch_name": {"type": "string"}, "switch_to": {"type": "boolean"}}, ["branch_name"]),
        ("git.add", "Stage specific repository paths.", {"repo_path": {"type": "string"}, "paths": {"type": "array"}}, ["paths"]),
        ("git.restore", "Restore specific paths or unstage with --staged.", {"repo_path": {"type": "string"}, "paths": {"type": "array"}, "staged": {"type": "boolean"}}, ["paths"]),
        ("git.stash", "Create a safety stash.", {"repo_path": {"type": "string"}, "message": {"type": "string"}, "include_untracked": {"type": "boolean"}}, None),
        ("git.commit", "Create or amend a local commit with a model-generated message.", {"repo_path": {"type": "string"}, "message": {"type": "string"}, "amend": {"type": "boolean"}}, ["message"]),
        ("git.push", "Push current or selected branch; protected force modes are blocked by default.", {"repo_path": {"type": "string"}, "remote": {"type": "string"}, "branch_name": {"type": "string"}, "set_upstream": {"type": "boolean"}, "force": {"type": "boolean"}}, None),
        ("git.pull", "Pull from upstream, optionally with --rebase.", {"repo_path": {"type": "string"}, "rebase": {"type": "boolean"}}, None),
        ("git.fetch", "Fetch from a remote.", {"repo_path": {"type": "string"}, "remote": {"type": "string"}, "prune": {"type": "boolean"}}, None),
        ("git.remote", "Inspect remotes.", {"repo_path": {"type": "string"}, "verbose": {"type": "boolean"}}, None),
        ("git.tag", "List or create tags.", {"repo_path": {"type": "string"}, "name": {"type": "string"}, "message": {"type": "string"}}, None),
        ("git.merge", "Merge a selected target.", {"repo_path": {"type": "string"}, "target": {"type": "string"}, "no_ff": {"type": "boolean"}}, ["target"]),
        ("git.rebase", "Rebase, continue, or abort according to current Git state.", {"repo_path": {"type": "string"}, "target": {"type": "string"}, "continue_": {"type": "boolean"}, "abort": {"type": "boolean"}}, None),
        ("git.revert", "Revert a selected revision.", {"repo_path": {"type": "string"}, "revision": {"type": "string"}, "no_commit": {"type": "boolean"}}, ["revision"]),
        ("git.reset", "Reset HEAD; --hard is protected and blocked by default.", {"repo_path": {"type": "string"}, "mode": {"type": "string"}, "target": {"type": "string"}, "allow_protected": {"type": "boolean"}}, None),
        ("git.clean", "Clean untracked files; force directory clean is protected and blocked by default.", {"repo_path": {"type": "string"}, "force": {"type": "boolean"}, "directories": {"type": "boolean"}, "allow_protected": {"type": "boolean"}}, None),
        ("git.config", "Inspect or set Git config.", {"repo_path": {"type": "string"}, "key": {"type": "string"}, "value": {"type": "string"}, "get": {"type": "boolean"}}, None),
        ("git.generic", "Run a dynamically discovered Git command as an argv list.", {"repo_path": {"type": "string"}, "args": {"type": "array"}, "timeout": {"type": "integer"}, "allow_protected": {"type": "boolean"}}, ["args"]),
        ("git.help", "Return common help, command help, or dynamically discovered git help -a commands.", {"repo_path": {"type": "string"}, "command": {"type": "string"}, "all": {"type": "boolean"}, "refresh": {"type": "boolean"}, "timeout": {"type": "integer"}}, None),
    ]
    return [
        ToolContract(
            name=name,
            description=description,
            input_schema=_schema(properties, required),
            output_schema=base_output,
            error_format=common_error,
            safety_rules=safety,
            examples=[{"input": {key: value for key, value in ({"args": ["status"]} if name == "git.generic" else {}).items()}}],
        )
        for name, description, properties, required in specs
    ]


def _document_tool_contracts(common_error: dict[str, Any]) -> list[ToolContract]:
    base_output = _schema(
        {
            "ok": {"type": "boolean"},
            "path": {"type": "string"},
            "file_type": {"type": "string"},
            "metadata": {"type": "object"},
            "analysis": {"type": "object"},
            "chunks": {"type": "array"},
            "results": {"type": "array"},
            "warnings": {"type": "array"},
            "error": {"type": "string"},
            "message": {"type": "string"},
            "verification": {"type": "object"},
        }
    )
    safety = [
        "Use only after a model decision selects a document capability.",
        "Reject paths outside the repository root.",
        "Read large files as normalized chunks and do not load entire large documents into prompts.",
        "Report scanned or image-only PDFs as needing OCR; do not invent text.",
        "Create backups before destructive updates unless explicitly disabled.",
        "Require explicit delete intent for document file deletion.",
        "For Excel create operations, content.sheets MUST be an object keyed by sheet name, not a list.",
        "For Excel create operations, use cells with {'cell': 'A1', 'value': ...} or {'cell': 'D1', 'formula': '=SUM(A1:C1)'}.",
        "Never use ExcelJS-style rows/cells payloads such as {'rows': [{'cells': [{'v': 200}]}]}; this project uses openpyxl-style normalized schemas.",
    ]
    specs: list[tuple[str, str, dict[str, Any], list[str] | None, dict[str, Any]]] = [
        (
            "document_detect",
            "Detect whether a project file is a supported Word, PDF, Excel, or CSV document.",
            {"path": {"type": "string"}, "mime_type": {"type": "string"}},
            ["path"],
            {"path": "docs/report.pdf"},
        ),
        (
            "document_read",
            "Read a supported document into normalized chunks with citation metadata.",
            {"path": {"type": "string"}, "use_cache": {"type": "boolean"}, "max_chunks": {"type": "integer"}},
            ["path"],
            {"path": "docs/report.docx", "use_cache": True, "max_chunks": 100},
        ),
        (
            "document_analyze",
            "Summarize document structure, key points, tables, counts, formulas, and OCR warnings.",
            {"path": {"type": "string"}},
            ["path"],
            {"path": "budget.xlsx"},
        ),
        (
            "document_query",
            "Search parsed document chunks with file-type, path, sheet, page, and section filters.",
            {
                "query": {"type": "string"},
                "paths": {"type": "array"},
                "file_types": {"type": "array"},
                "path_filter": {"type": "string"},
                "sheet": {"type": "string"},
                "page": {"type": "integer"},
                "section": {"type": "string"},
                "limit": {"type": "integer"},
            },
            ["query"],
            {"query": "payment terms", "file_types": ["pdf", "docx"], "limit": 5},
        ),
        (
            "document_create",
            (
                "Create a DOCX, XLSX/XLSM, CSV, or simple text PDF artifact without overwriting by default. "
                "For Excel, content.sheets must be an object keyed by sheet name. "
                "Do not pass sheets as a list."
            ),
            {
                "path": {"type": "string"},
                "content": _document_create_content_schema(),
                "file_type": {
                    "type": "string",
                    "description": "Use lowercase values: docx, pdf, xlsx, xlsm, or csv.",
                },
                "overwrite": {"type": "boolean"},
            },
            ["path", "content"],
            {
                "path": "row_sum.xlsx",
                "file_type": "xlsx",
                "overwrite": False,
                "content": {
                    "sheets": {
                        "Sheet1": {
                            "cells": [
                                {"cell": "A1", "value": 200},
                                {"cell": "B1", "value": 300},
                                {"cell": "C1", "value": 400},
                                {"cell": "D1", "formula": "=SUM(A1:C1)"},
                            ]
                        }
                    }
                },
            },
        ),
        (
            "document_update",
            "Safely update DOCX sections/text/tables/metadata, Excel cells/rows/sheets/formulas, or PDF metadata.",
            {"path": {"type": "string"}, "operation": {"type": "string"}, "payload": {"type": "object"}, "backup": {"type": "boolean"}},
            ["path", "operation", "payload"],
            {"path": "budget.xlsx", "operation": "update_cell", "payload": {"sheet": "March", "cell": "B2", "value": 1200}},
        ),
        (
            "document_delete",
            "Delete a supported document file only when explicit delete intent has been validated.",
            {"path": {"type": "string"}, "explicit": {"type": "boolean"}, "backup": {"type": "boolean"}},
            ["path", "explicit"],
            {"path": "docs/generated-report.docx", "explicit": True},
        ),
    ]
    return [
        ToolContract(
            name=name,
            description=description,
            input_schema=_schema(properties, required),
            output_schema=base_output,
            error_format=common_error,
            safety_rules=safety,
            examples=[{"input": example}],
        )
        for name, description, properties, required, example in specs
    ]


def coding_tool_contracts() -> list[ToolContract]:
    """Return contracts for the built-in coding-agent tool surface."""

    from mana_agent.connectors.email.tools import email_tool_contracts

    common_error = {
        "ok": False,
        "error": {"code": "string", "message": "string", "details": "object"},
    }
    return [
        *email_tool_contracts(),
        ToolContract(
            name="semantic_search",
            description="Search indexed code chunks semantically using the local vector index when available.",
            input_schema=_schema({"query": {"type": "string"}, "k": {"type": "integer"}}, ["query"]),
            output_schema=_schema({"results": {"type": "array"}, "warnings": {"type": "array"}}),
            error_format=common_error,
            safety_rules=[
                "Read matching files before editing them.",
                "Do not repeat the same query indefinitely.",
                "Use repo_search, read_file, find_symbols, call_graph, or verify_project when they fit better than semantic retrieval.",
            ],
            examples=[{"input": {"query": "safe_apply_patch path validation", "k": 8}}],
        ),
        ToolContract(
            name="repo_search",
            description="Search repository text with regex or literal matching.",
            input_schema=_schema(
                {
                    "query": {"type": "string"},
                    "glob": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                ["query"],
            ),
            output_schema=_schema({"matches": {"type": "array"}, "truncated": {"type": "boolean"}}),
            error_format=common_error,
            safety_rules=["Search is read-only.", "Binary files and ignored metadata directories are skipped."],
            examples=[{"input": {"query": "class CodingAgent", "glob": "*.py", "regex": False}}],
        ),
        ToolContract(
            name="read_file",
            description="Safely read a repository file by full file or line range, using run evidence memory before disk.",
            input_schema=_schema(
                {
                    "path": {"type": "string"},
                    "mode": {"enum": ["line", "full"]},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                ["path"],
            ),
            output_schema=_schema(
                {
                    "file_path": {"type": "string"},
                    "normalized_path": {"type": "string"},
                    "content": {"type": "string"},
                    "cache_hit": {"type": "boolean"},
                    "source": {"enum": ["memory", "tool"]},
                    "covered_range": {"type": "array"},
                }
            ),
            error_format=common_error,
            safety_rules=[
                "Reject paths outside the project root.",
                "Reject binary files.",
                "Check run-scoped evidence memory before disk reads.",
                "Treat cache_hit=true/source=memory as valid evidence equal to a fresh tool read.",
            ],
            examples=[{"input": {"path": "src/mana_agent/multi_agent/runtime/ask_agent.py", "mode": "full"}}],
        ),
        ToolContract(
            name="read_skill",
            description="Load one full skills/<skill_name>/SKILL.md body on demand after matching the compact skill index.",
            input_schema=_schema({"skill_name": {"type": "string"}}, ["skill_name"]),
            output_schema=_schema({"content": {"type": "string"}}),
            error_format=common_error,
            safety_rules=[
                "Validate skill_name.",
                "Never load all skills eagerly.",
                "Use only when the current task matches the skill trigger.",
            ],
            examples=[{"input": {"skill_name": "django"}}],
        ),
        *_document_tool_contracts(common_error),
        *_git_tool_contracts(common_error),
        ToolContract(
            name="repo_batch_read",
            description="Read multiple repository text files in one call with per-file errors and truncation metadata.",
            input_schema=_schema({"files": {"type": "array"}}, ["files"]),
            output_schema=_schema({"files": {"type": "array"}, "errors": {"type": "array"}, "truncated": {"type": "boolean"}}),
            error_format=common_error,
            safety_rules=[
                "Validate every path is inside the repository.",
                "Return per-file errors instead of failing the whole batch.",
                "Use this instead of multiple read_file calls when reading more than one file.",
            ],
            examples=[{"input": {"files": ["src/a.py", "src/b.py"]}}],
        ),
        ToolContract(
            name="repo_batch_search",
            description="Run multiple repository text searches in one call and return grouped results per query.",
            input_schema=_schema({"patterns": {"type": "array"}}, ["patterns"]),
            output_schema=_schema({"results": {"type": "array"}, "errors": {"type": "array"}}),
            error_format=common_error,
            safety_rules=[
                "Prefer ripgrep-like exact search semantics through repository-local text search.",
                "Use this instead of multiple repo_search calls for independent queries.",
                "Respect per-query limits.",
            ],
            examples=[{"input": {"patterns": [{"query": "Skill", "glob": "**/*.py", "regex": False, "limit": 20}]}}],
        ),
        ToolContract(
            name="apply_patch",
            description="Apply a Codex-style text patch inside the repository.",
            input_schema=_schema(
                {
                    "patch": {"type": "string"},
                    "check_only": {"type": "boolean"},
                },
                ["patch"],
            ),
            output_schema=_schema({"ok": {"type": "boolean"}, "touched_files": {"type": "array"}, "changed_ranges": {"type": "array"}}),
            error_format=common_error,
            safety_rules=[
                "Reject unread existing target files when read tracking is supplied.",
                "Reject traversal, absolute paths, paths outside root, and stale contexts.",
                "Match update hunks by surrounding text, never by generated line numbers.",
                "Store patch preview and result under .mana/logs/ before returning.",
            ],
            examples=[{"input": {"patch": "*** Begin Patch\n*** Update File: a.py\n@@\n-old\n+new\n*** End Patch"}}],
        ),
        ToolContract(
            name="apply_patch_batch",
            description="Validate and apply multiple related Codex patch payloads in one call.",
            input_schema=_schema({"patches": {"type": "array"}}, ["patches"]),
            output_schema=_schema({"ok": {"type": "boolean"}, "results": {"type": "array"}, "changed_files": {"type": "array"}}),
            error_format=common_error,
            safety_rules=[
                "Each patch must be Codex patch text.",
                "Dry-run validate all patches before applying.",
                "Use for related multi-file edits instead of many separate apply_patch calls.",
            ],
            examples=[{"input": {"patches": [{"path": "a.py", "patch": "*** Begin Patch\n*** Update File: a.py\n@@\n-old\n+new\n*** End Patch"}]}}],
        ),
        ToolContract(
            name="edit_file",
            description="Replace one exact old_string in a repository file.",
            input_schema=_schema(
                {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                ["path", "old_string", "new_string"],
            ),
            output_schema=_schema(
                {
                    "ok": {"type": "boolean"},
                    "path": {"type": "string"},
                    "files_changed": {"type": "array"},
                    "before_sha256": {"type": "string"},
                    "after_sha256": {"type": "string"},
                    "changed_ranges": {"type": "array"},
                }
            ),
            error_format=common_error,
            safety_rules=[
                "Re-read the file immediately before editing.",
                "Search old_string exactly; fail on missing or ambiguous matches.",
                "Never use line numbers as the source of truth.",
            ],
            examples=[{"input": {"path": "a.py", "old_string": "old", "new_string": "new", "replace_all": False}}],
        ),
        ToolContract(
            name="multi_edit_file",
            description="Apply several exact-string replacements to one file atomically.",
            input_schema=_schema(
                {
                    "path": {"type": "string"},
                    "edits": {"type": "array"},
                },
                ["path", "edits"],
            ),
            output_schema=_schema(
                {
                    "ok": {"type": "boolean"},
                    "path": {"type": "string"},
                    "files_changed": {"type": "array"},
                    "before_sha256": {"type": "string"},
                    "after_sha256": {"type": "string"},
                    "changed_ranges": {"type": "array"},
                }
            ),
            error_format=common_error,
            safety_rules=[
                "Re-read the file once before editing.",
                "Apply edits sequentially in memory and abort without writing on the first failed edit.",
                "Write once at the end.",
            ],
            examples=[{"input": {"path": "a.py", "edits": [{"old_string": "old", "new_string": "new"}]}}],
        ),
        ToolContract(
            name="create_file",
            description="Create a new repository text file without overwriting an existing target.",
            input_schema=_schema(
                {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "text": {"type": "string"},
                    "body": {"type": "string"},
                },
                ["path"],
            ),
            output_schema=_schema(
                {
                    "ok": {"type": "boolean"},
                    "path": {"type": "string"},
                    "bytes_written": {"type": "integer"},
                    "sha256": {"type": "string"},
                    "error": {"type": "string"},
                }
            ),
            error_format=common_error,
            safety_rules=[
                "Reject traversal, absolute paths, paths outside root, and disallowed prefixes.",
                "Refuse to overwrite an existing target file.",
                "Create parent directories as needed and write atomically.",
            ],
            examples=[{"input": {"path": "docs/new-note.md", "content": "# New note\n"}}],
        ),
        ToolContract(
            name="write_file",
            description="Write full file content, guarded against accidental overwrites of existing files.",
            input_schema=_schema(
                {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "text": {"type": "string"},
                    "body": {"type": "string"},
                    "expected_sha256": {"type": "string"},
                    "force": {"type": "boolean"},
                },
                ["path", "content"],
            ),
            output_schema=_schema(
                {
                    "ok": {"type": "boolean"},
                    "path": {"type": "string"},
                    "bytes_written": {"type": "integer"},
                    "sha256": {"type": "string"},
                    "files_changed": {"type": "array"},
                    "error": {"type": "string"},
                }
            ),
            error_format=common_error,
            safety_rules=[
                "Reject traversal, absolute paths, paths outside root, and disallowed prefixes.",
                "Reject overwriting existing files unless expected_sha256 matches the current content or force=true is supplied.",
                "Use edit_file, multi_edit_file, or apply_patch before whole-file rewrites.",
                "Write atomically.",
            ],
            examples=[{"input": {"path": "docs/note.md", "content": "# Note\n", "expected_sha256": "<sha256-from-read>"}}],
        ),
        ToolContract(
            name="delete_file",
            description="Delete one existing repository file without touching directories or paths outside the repository.",
            input_schema=_schema({"path": {"type": "string"}}, ["path"]),
            output_schema=_schema(
                {
                    "ok": {"type": "boolean"},
                    "path": {"type": "string"},
                    "deleted": {"type": "boolean"},
                    "files_changed": {"type": "array"},
                    "error": {"type": "string"},
                }
            ),
            error_format=common_error,
            safety_rules=[
                "Reject traversal, absolute paths, paths outside root, and disallowed prefixes.",
                "Refuse to delete directories.",
                "Refuse missing targets so accidental no-op deletes do not count as progress.",
            ],
            examples=[{"input": {"path": "docs/obsolete-note.md"}}],
        ),
        ToolContract(
            name="list_files",
            description="List repository files with optional glob filtering.",
            input_schema=_schema({"glob": {"type": "string"}, "limit": {"type": "integer"}}),
            output_schema=_schema({"files": {"type": "array"}, "truncated": {"type": "boolean"}}),
            error_format=common_error,
            safety_rules=["Read-only.", "Skip VCS, cache, virtualenv, and binary-like metadata directories."],
            examples=[{"input": {"glob": "src/**/*.py", "limit": 100}}],
        ),
        ToolContract(
            name="find_symbols",
            description="Find Python functions, classes, and methods by name.",
            input_schema=_schema({"query": {"type": "string"}, "limit": {"type": "integer"}}),
            output_schema=_schema({"symbols": {"type": "array"}, "truncated": {"type": "boolean"}}),
            error_format=common_error,
            safety_rules=["Read-only.", "Parse Python with ast instead of regular expressions where possible."],
            examples=[{"input": {"query": "CodingAgent", "limit": 20}}],
        ),
        ToolContract(
            name="call_graph",
            description="Inspect Python AST call edges by caller, callee, or file path.",
            input_schema=_schema({"query": {"type": "string"}, "limit": {"type": "integer"}}),
            output_schema=_schema({"edges": {"type": "array"}, "truncated": {"type": "boolean"}}),
            error_format=common_error,
            safety_rules=[
                "Read-only.",
                "Use for control-flow/call-site questions; it reports syntactic calls, not runtime dispatch.",
            ],
            examples=[{"input": {"query": "run_tools", "limit": 50}}],
        ),
        ToolContract(
            name="run_command",
            description="Run a non-destructive command in the project root.",
            input_schema=_schema({"cmd": {"type": "string"}}, ["cmd"]),
            output_schema=_schema({"returncode": {"type": "integer"}, "stdout": {"type": "string"}, "stderr": {"type": "string"}}),
            error_format=common_error,
            safety_rules=["Block destructive shell patterns.", "Use project root as cwd.", "Return stdout/stderr and exit code."],
            examples=[{"input": {"cmd": "pytest -q"}}],
        ),
        ToolContract(
            name="run_script_once",
            description="Run one grouped, non-destructive shell script in the repository and return exit code/output/duration.",
            input_schema=_schema({"script": {"type": "string"}, "cwd": {"type": "string"}}, ["script"]),
            output_schema=_schema(
                {
                    "returncode": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "duration_ms": {"type": "number"},
                }
            ),
            error_format=common_error,
            safety_rules=[
                "Prefer one grouped script for multiple safe checks.",
                "Block destructive command patterns.",
                "Return output summary, exit code, and duration.",
            ],
            examples=[{"input": {"script": "python -m compileall src\npytest -q"}}],
        ),
        ToolContract(
            name="verify_project",
            description="Auto-detect and run pytest, ruff, mypy, import, CLI, and git checks.",
            input_schema=_schema({"quick": {"type": "boolean"}}),
            output_schema=_schema({"checks": {"type": "array"}, "ok": {"type": "boolean"}}),
            error_format=common_error,
            safety_rules=["Verification is read-only except normal test caches.", "Missing commands are reported as skipped."],
            examples=[{"input": {"quick": False}}],
        ),
    ]


def coding_tool_contracts_payload() -> dict[str, Any]:
    """JSON-friendly contract payload."""

    return {"tools": [item.model_dump() for item in coding_tool_contracts()]}
