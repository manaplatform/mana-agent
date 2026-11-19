"""Repository evidence operations owned by an admitted Coding task."""

from __future__ import annotations

import json
import threading
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from mana_agent.repository_evidence.models import EvidenceQuery, EvidenceRef, relative_source
from mana_agent.tools.contracts import ToolContract
from mana_agent.utils.redaction import redact_secrets


INVESTIGATE = "investigate_repository"
INTEGRATE = "integrate_repository_evidence"
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ReadFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["read_file"]
    path: Text
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class SearchRepository(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["repo_search"]
    query: Text
    paths: list[Text] = Field(min_length=1, max_length=16)


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: Text
    paths: list[Text] = Field(min_length=1, max_length=16)
    requests: list[Annotated[ReadFile | SearchRepository, Field(discriminator="tool")]] = Field(
        min_length=1, max_length=4,
    )


def scoped_paths(root: Path, paths, *, authority=()) -> tuple[str, ...]:
    """Validate the full batch scope before any repository operation."""
    selected = tuple(sorted({relative_source(path) for path in paths}))
    if not selected or "." in selected:
        raise ValueError("Select specific repository files or directories for this investigation")
    allowed = tuple(Path(path) for path in authority if path != ".")
    for path in selected:
        absolute = root / path
        if absolute.resolve() != absolute:
            raise PermissionError("Investigation targets must stay in the assigned checkout and cannot use symlinks")
        if allowed and not any(Path(path).is_relative_to(scope) for scope in allowed):
            raise PermissionError("Investigation target is outside its authorized paths")
    return selected


class CodingEvidenceTools:
    """Bound tools shared by native dispatch and Codex app-server dispatch."""

    definitions = {
        INVESTIGATE: (InvestigationRequest, "Record a focused question and execute up to four selected repository reads or searches within literal paths. Call this tool on its own."),
        INTEGRATE: (EvidenceQuery, "Filter and rank current repository evidence within the selected paths or existing task. Returns observations and conflicts."),
    }

    def __init__(self, taskboard, task_id, root, *, dispatch, remaining, charge, authorize=None):
        self.taskboard = taskboard
        self.task_id = task_id
        self.root = Path(root).resolve()
        self.dispatch = dispatch
        self.remaining = remaining
        self.charge = charge
        self.authorize = authorize
        self.lock = threading.RLock()

    @classmethod
    def contract(cls, name):
        model, description = cls.definitions[name]
        return ToolContract(name=name, description=description, input_schema=model.model_json_schema(),
            output_schema={}, error_format={}, required_capability="repository_read", effect="read_only",
            repository_operation=True)

    @classmethod
    def codex_specs(cls):
        return [{"name": name, "description": description, "inputSchema": model.model_json_schema(),
                 "deferLoading": False} for name, (model, description) in cls.definitions.items()]

    def invoke(self, name, arguments, *, call_id):
        from mana_agent.chat.activity import safe_text
        from mana_agent.services.activity_context import current_activity_context, emit_activity

        activity = current_activity_context()
        if activity is None or activity.task_id != self.task_id:
            raise PermissionError("Repository evidence requires its active Coding task")
        with self.lock:
            with activity.publication():
                if self.authorize is not None:
                    self.authorize(name)
                task = self.taskboard.get_task(self.task_id)
                if name in task.forbidden_tools or task.allowed_tools and name not in task.allowed_tools:
                    raise PermissionError("Evidence tool is outside the TaskBoard capability scope")
                assigned = task.coding_assignment.get("allowed_scope") or ()
                knowledge = self.taskboard.evidence_board(root=self.root, repository_id=task.primary_repository_id)
            if name == INTEGRATE:
                query = EvidenceQuery.model_validate(arguments)
                paths = scoped_paths(self.root, query.paths, authority=assigned) if query.paths else ()
                if query.task_id and query.task_id != self.task_id:
                    raise PermissionError("Evidence query belongs to a different Coding task")
                return knowledge.query(query.model_copy(update={"include_stale": False, "paths": paths,
                    "task_id": self.task_id if not paths else query.task_id}))
            if name != INVESTIGATE:
                raise ValueError("Unknown Coding evidence tool")
            request = InvestigationRequest.model_validate(arguments)
            paths = scoped_paths(self.root, request.paths, authority=assigned)
            selected = []
            for operation in request.requests:
                if self.authorize is not None:
                    self.authorize(operation.tool)
                if operation.tool in task.forbidden_tools or task.allowed_tools and operation.tool not in task.allowed_tools:
                    raise PermissionError("Repository operation is outside the TaskBoard capability scope")
                args = operation.model_dump(exclude={"tool"}, exclude_none=True)
                if isinstance(operation, ReadFile):
                    args["path"] = scoped_paths(self.root, (operation.path,), authority=paths)[0]
                    end = operation.end_line or operation.start_line + 119
                    if end < operation.start_line:
                        raise ValueError("end_line cannot precede start_line")
                    args["end_line"] = min(end, operation.start_line + 119)
                else:
                    args["paths"] = list(scoped_paths(self.root, operation.paths, authority=paths))
                    args["limit"] = 30
                key = sha256(json.dumps([operation.tool, args], sort_keys=True).encode()).hexdigest()
                if any(previous[0] == key for previous in selected):
                    raise ValueError("Select each read or search once per investigation request")
                selected.append((key, operation.tool, args))

            with activity.publication():
                state = dict(self.taskboard.get_task(self.task_id).mode_workflow)
                saved = state.get("coding_investigation") or {}
                receipts = dict(saved.get("receipts") or {})
                question = {"objective": safe_text(redact_secrets(request.objective), protected=activity.protected_text), "paths": list(paths),
                            "receipts": receipts}
                self.taskboard.update_task(self.task_id, mode_workflow={**state, "coding_investigation": question})

            rows, pending = [], []
            for index, (key, tool, args) in enumerate(selected):
                saved_ref = receipts.get(key)
                if saved_ref:
                    record = knowledge.get(EvidenceRef.model_validate(saved_ref))
                    if record.freshness == "current" and record.validation == "supported" and record.sources:
                        source = record.sources[0]
                        rows.append({"tool": tool, "index": index, "reused": True, "evidence": [saved_ref],
                                     "result": {"ok": True, "path": source.path, "content": source.excerpt,
                                                "start_line": source.start_line, "end_line": source.end_line}})
                        continue
                pending.append((index, key, tool, args))

            available = max(0, self.remaining())
            reused = len(rows)
            if len(pending) > available:
                return {"ok": False, "error_code": "BUDGET_EXCEEDED", "remaining": available,
                        "required": len(pending), "observations": rows, "gap": question["objective"]}
            for index, key, tool, args in pending:
                with activity.publication():
                    self.charge()
                result = self.dispatch(tool, args, call_id=f"{call_id}:read:{index}")
                refs = result.get("repository_evidence_refs") or []
                rows.append({"tool": tool, "index": index, "evidence": refs, "result": result})
                # A source-backed exact read can be reused across attempts. Search results may gain
                # matches when other files change, so the whole search is never treated as current.
                if tool == "read_file" and result.get("ok") and refs:
                    receipts[key] = refs[0]
                    with activity.publication():
                        state = dict(self.taskboard.get_task(self.task_id).mode_workflow)
                        self.taskboard.update_task(self.task_id, mode_workflow={**state, "coding_investigation": {
                            **question, "receipts": dict(list(receipts.items())[-100:]),
                        }})
            remaining = max(0, self.remaining())
            emit_activity("coding.progress", title="Repository investigation returned", status="success",
                operation_id=f"{call_id}:evidence", metadata={"reused_evidence": reused, "tool_budget_remaining": remaining})
            return {"ok": True, "remaining": remaining,
                    "observations": sorted(rows, key=lambda row: row["index"]),
                    "gap": question["objective"] if any(not row["result"].get("ok") for row in rows) else ""}
