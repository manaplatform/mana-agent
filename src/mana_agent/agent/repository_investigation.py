"""Shared repository read contracts and bounded observation formatting."""

from __future__ import annotations

from mana_agent.agent.plan_service import (
    DirectoryListEvidence, EvidenceRequest, FileListEvidence, FileReadEvidence, InvestigationScope,
    RepositoryDiscoveryEvidence, RepositorySearchEvidence, RepositorySymbolEvidence,
)
from mana_agent.human_inbox.models import canonical_digest

READ_TOOLS = {
    "read_file": ("Read lines from one known file.", FileReadEvidence),
    "ls": ("List immediate repository directories without descending or reading files.", DirectoryListEvidence),
    "repo_search": ("Find the literal query text in the selected repository scope.", RepositorySearchEvidence),
    "list_files": ("List files in the selected scope. At levels 1–3, omitting paths and a targeted glob "
                   "lists only top-level files. Use paths [\".\"] to request that scope explicitly.", FileListEvidence),
    "find_symbols": ("Find symbols in the selected repository scope.", RepositorySymbolEvidence),
    "call_graph": ("Find callers in the selected repository scope.", RepositorySymbolEvidence),
}


def _read_schema(model: type[EvidenceRequest]) -> dict:
    schema = model.model_json_schema()
    for field in {"tool", "initial_repository_scope"} & schema["properties"].keys():
        schema["properties"].pop(field)
        if field in schema.get("required", []):
            schema["required"].remove(field)
    if issubclass(model, RepositoryDiscoveryEvidence):
        for field in ("paths", "glob"):
            schema["properties"][field].pop("default", None)
    return schema


def read_tool_schemas() -> dict[str, dict]:
    return {name: _read_schema(model) for name, (_, model) in READ_TOOLS.items()}


def _observed(row: dict) -> bool:
    return not isinstance(row["result"], dict) or row["result"].get("ok") is not False


def observation_fingerprint(row: dict) -> str | None:
    """Identify useful evidence independent of its execution receipt."""
    result = row["result"]
    if not _observed(row):
        return None
    if not isinstance(result, dict):
        return canonical_digest(result) if result else None
    payload = {key: value for key, value in result.items()
               if key not in {"ok", "execution", "mutations", "limit", "truncated", "repository_evidence_refs"}}
    # Empty results remain available as negative evidence but do not justify more rounds.
    if not any(payload.get(key) for key in ("content", "files", "directories", "matches", "symbols", "edges", "nodes")):
        return None
    return canonical_digest(payload)


def investigation_scope(evidence: list[dict], remaining: int | None = None) -> InvestigationScope:
    return InvestigationScope(
        {row["id"]: row["level"] for row in evidence if _observed(row)},
        {row["fingerprint"]: row["id"] for row in evidence if not row.get("stale")},
        remaining,
    )


def compact_observation(row: dict) -> dict:
    result = row["result"]
    if row["level"] > 0 and isinstance(result, dict):
        result = {key: value for key, value in result.items()
                  if key not in {"execution", "mutations", "source_evidence"}}
        if isinstance(result.get("content"), str) and len(result["content"]) > 6000:
            result["content"] = result["content"][:6000]
            result["content_truncated"] = True
        if isinstance(result.get("directories"), list) and len(result["directories"]) > 50:
            result["directories"] = result["directories"][:50]
            result["directories_truncated"] = True
    return {"id": row["id"], "level": row["level"], "result": result}
