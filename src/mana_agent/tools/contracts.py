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


def coding_tool_contracts() -> list[ToolContract]:
    """Return contracts for the built-in coding-agent tool surface."""

    common_error = {
        "ok": False,
        "error": {"code": "string", "message": "string", "details": "object"},
    }
    return [
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
            examples=[{"input": {"path": "src/mana_agent/llm/ask_agent.py", "mode": "full"}}],
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
