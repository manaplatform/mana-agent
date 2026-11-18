"""Source stamps captured by registered read implementations and host ingestion."""

from hashlib import sha256
import json
from pathlib import Path

from mana_agent.repository_evidence.models import EvidenceProvenance, EvidenceSource, InvestigationCitation, relative_source


def read_directory(root: Path, path: str = ".") -> dict:
    from mana_agent.tools.repository import _is_skipped_dir

    relative_source(path)
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_dir():
        raise PermissionError("Directory source escapes its assigned checkout")
    relative_source(target.relative_to(root).as_posix())
    entries = []
    truncated = False
    for entry in target.iterdir():
        if len(entries) >= 500:
            truncated = True
            break
        if entry.name.startswith(".") or _is_skipped_dir(entry.name):
            continue
        try:
            relative_source(entry.relative_to(root).as_posix())
        except ValueError:
            continue
        entries.append((entry.name, "symlink" if entry.is_symlink() else "directory" if entry.is_dir() else "file"))
    entries.sort()
    result = {"ok": True, "path": path, "directories": [name for name, kind in entries if kind == "directory"],
              "files": [name for name, kind in entries if kind == "file"], "truncated": truncated}
    if truncated:
        return result
    encoded = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    source = EvidenceSource(path=path, source_kind="directory", sha256=sha256(encoded.encode()).hexdigest(), excerpt=encoded[:12000])
    return {**result, "content": encoded, "source_evidence": source.model_dump(mode="json")}


def validate_selected_source(runner, source: InvestigationCitation, *, decision_id: str) -> dict:
    from mana_agent.agent.plan_service import FileListEvidence, FileReadEvidence
    if source.source_kind == "directory":
        request = FileListEvidence(tool="list_files", paths=(source.path,), level=1,
                                   reason="Validate the selected directory observation")
        implementation = lambda _: read_directory(runner.authority.workspace.worktree_path, source.path)
    else:
        request = FileReadEvidence(tool="read_file", paths=(source.path,), level=1,
            reason="Validate the selected source citation", start_line=source.start_line or 1, end_line=source.end_line)
        implementation = lambda _: read_source(runner.authority.workspace.worktree_path, source.path,
                                               start_line=source.start_line, end_line=source.end_line)
    schema = runner.authority.schemas.get(request.tool)
    if schema is None:
        raise PermissionError(f"Source citation requires {request.tool}, which is unavailable in the assigned scope. "
                              "Cite an authorized observation or request a scoped investigation.")
    properties = schema["properties"]
    arguments = request.model_dump(mode="json", include=set(properties))
    if "path" in properties:
        arguments["path"] = source.path
    arguments = {key: value for key, value in arguments.items() if value is not None}
    invocation = runner.request(request.tool, arguments,
        call_id=f"{decision_id}:source:{source.source_kind}:{source.path}:{source.start_line}:{source.end_line}", decision_id=decision_id)
    return runner.execute(invocation, implementation).result


def read_source(root: Path, path: str, *, start_line=None, end_line=None, max_chars=None) -> dict:
    relative_source(path)
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise PermissionError("Source escapes assigned checkout")
    relative_source(target.relative_to(root).as_posix())
    with target.open("rb") as stream:
        data = stream.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("Source exceeds the bounded evidence read; select a smaller source artifact")
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    start = max(1, int(start_line or 1))
    end = min(int(end_line or len(lines)), len(lines))
    if end_line is not None and int(end_line) < start:
        raise ValueError("Source range is reversed")
    content = "".join(lines[start - 1:end])
    if max_chars is not None:
        content = content[:max_chars]
    source = EvidenceSource(path=path, sha256=sha256(data).hexdigest(),
                            start_line=start if end >= start else None,
                            end_line=end if end >= start else None, excerpt=content[:12000])
    return {"ok": True, "path": path, "content": content, "start_line": start,
            "end_line": end, "total_lines": len(lines), "source_evidence": source.model_dump(mode="json")}


def ingest_operation(authority, result) -> None:
    """Use host-bound identities, never identity fields from the tool payload."""
    context = authority.activity
    taskboard = authority.taskboard
    if taskboard is None or context is None or not context.attempt_id:
        return
    board = taskboard.evidence_board(root=Path(result.result.get("execution", {}).get("workspace_root")
                                                or authority.workspace.worktree_path))
    digest = sha256(json.dumps(result.result, sort_keys=True, default=str).encode()).hexdigest()
    provenance = EvidenceProvenance(
        task_id=result.task_id, attempt_id=result.attempt_id, attempt_generation=result.attempt_generation,
        decision_id=result.decision_id, invocation_id=result.operation_id, agent_id=result.agent_id,
        backend="native_tool_runner", result_digest=digest,
        result_reference=next((item.get("reference", "") for item in result.output_references.get("result", [])
                               if item.get("reference")), ""),
    )
    sources = ()
    if result.tool_name in {"read_file", "ls", "list_files"} and result.status == "success" and result.result.get("source_evidence"):
        sources = (EvidenceSource.model_validate(result.result["source_evidence"]),)
    observation = {"tool": result.tool_name, "status": result.status, "exit_code": result.exit_code,
                   "operation_id": result.operation_id}
    selected_policy = authority.policies.get(result.tool_name)
    category = "test_results" if selected_policy and selected_policy.capability == "test_execution" else "code_behavior" if sources and sources[0].source_kind == "file" else "error_observation" if result.status != "success" else "repository_structure"
    record = board.record_observation(
        provenance=provenance, category=category,
        statement=f"{result.tool_name} returned {result.status}", sources=sources, observation=observation,
    )
    produced = [record.ref.model_dump(mode="json")]
    task = taskboard.get_task(result.task_id)
    if (task.task_kind in {"investigation", "understanding"} and result.tool_name == "batch_read_files"
            and result.status in {"success", "failed"}):
        from mana_agent.repository_evidence.service import StaleEvidenceSource

        # Each result row has the hash calculated from the bytes actually read.
        # A truncated row has no reliable end-line boundary for its excerpt.
        files = result.result.get("files", [])
        for index, row in enumerate(files[:20]):
            if not row.get("ok") or row.get("truncated") or not row.get("content") or row.get("file_size_bytes", 0) > 2_000_000:
                continue
            start = row.get("start_line", 1)
            end = row.get("end_line", row["total_lines"])
            if end < start:
                continue
            source = EvidenceSource(path=row["path"], sha256=row["sha256"], start_line=start, end_line=end,
                                    excerpt=row["content"][:12000])
            try:
                file_record = board.record_observation(provenance=provenance, category="code_behavior",
                    statement="batch_read_files returned a file", sources=(source,),
                    observation={**observation, "status": "success"})
            except StaleEvidenceSource:
                continue
            ref = file_record.ref.model_dump(mode="json")
            files[index] = {"repository_evidence_refs": [ref], **row}
            produced.append(ref)
    refs = list(task.evidence_refs)
    taskboard.update_task(result.task_id, evidence_refs=[*refs, *(ref for ref in produced if ref not in refs)])
    result.result["repository_evidence_refs"] = produced
    for mutation in result.mutations:
        paths = [path for path in (mutation.path, mutation.old_path) if path]
        board.invalidate_sources(paths, reason="Observed repository mutation", task_id=result.task_id)
