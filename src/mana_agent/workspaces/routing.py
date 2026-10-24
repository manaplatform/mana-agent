from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mana_agent.workspaces.context import WorkspaceContext
from mana_agent.workspaces.models import RepositoryPermission, RepositoryScopeDecision


SCOPE_DECISION_PROMPT = """You select repository scope for a Mana-Agent turn.
Return JSON only. Select the primary repository unless the task genuinely requires other listed repositories.
Never invent repository ids. Cross-repository writes require requires_verification=true.
Apply supplied repository_guidance to repository work, but keep it subordinate to system/safety
rules, permissions, and the user's explicit request.
Schema:
{
  "repository_ids": ["repo_id"],
  "access_by_repository": {"repo_id": "read|write|git|verify"},
  "relationship_depth": 0,
  "requires_multi_repo": false,
  "requires_verification": false,
  "safe_to_continue": true,
  "reason": "short evidence-based reason"
}
"""


class ScopeDecisionError(RuntimeError):
    pass


class RepositoryScopeDecisionEngine:
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def decide(
        self,
        *,
        request: str,
        context: WorkspaceContext,
        repository_guidance: str = "",
    ) -> RepositoryScopeDecision:
        # The cwd-to-primary binding is mechanical, not a routing fallback. An
        # unavailable model therefore remains restricted to one repository.
        if len(context.session.attached_repository_ids) == 1 or self.llm is None or not hasattr(self.llm, "invoke"):
            return context.validate_scope(
                RepositoryScopeDecision(
                    workspace_id=context.workspace.workspace_id,
                    session_id=context.session.session_id,
                    primary_repository_id=context.session.primary_repository_id,
                    repository_ids=[context.session.primary_repository_id],
                    permissions=[RepositoryPermission(repository_id=context.session.primary_repository_id)],
                    safe_to_continue=True,
                    reason="Session is mechanically bound to its cwd repository; no model expansion was available.",
                    source="session_binding",
                )
            )
        payload = {
            "request": request,
            "repository_guidance": repository_guidance,
            "workspace_id": context.workspace.workspace_id,
            "session_id": context.session.session_id,
            "primary_repository_id": context.session.primary_repository_id,
            "repositories": [
                {
                    "repository_id": item.repository_id,
                    "name": item.name,
                    "path": item.canonical_path,
                    "role": item.role,
                    "kind": item.kind,
                    "tags": item.tags,
                }
                for item in context.repositories.values()
            ],
        }
        try:
            response = self.llm.invoke(
                [SystemMessage(content=SCOPE_DECISION_PROMPT), HumanMessage(content=json.dumps(payload, sort_keys=True))]
            )
            raw = getattr(response, "content", response)
            text = " ".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in raw) if isinstance(raw, list) else str(raw)
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start : end + 1])
        except Exception as exc:
            raise ScopeDecisionError(f"Model repository-scope decision failed. No cross-repository action executed. Reason: {exc}") from exc
        ids = [str(item) for item in data.get("repository_ids", [])]
        access = data.get("access_by_repository", {}) if isinstance(data.get("access_by_repository"), dict) else {}
        decision = RepositoryScopeDecision(
            workspace_id=context.workspace.workspace_id,
            session_id=context.session.session_id,
            primary_repository_id=context.session.primary_repository_id,
            repository_ids=ids,
            permissions=[
                RepositoryPermission(repository_id=repository_id, access=str(access.get(repository_id) or "read"))
                for repository_id in ids
            ],
            relationship_depth=int(data.get("relationship_depth", 0) or 0),
            requires_multi_repo=bool(data.get("requires_multi_repo", len(ids) > 1)),
            requires_verification=bool(data.get("requires_verification", False)),
            safe_to_continue=bool(data.get("safe_to_continue", False)),
            reason=str(data.get("reason") or ""),
            source="model",
        )
        if decision.requires_multi_repo != (len(decision.repository_ids) > 1):
            raise ScopeDecisionError("Model repository-scope decision is inconsistent with selected repositories.")
        if any(permission.access in {"write", "git"} for permission in decision.permissions) and not decision.requires_verification:
            raise ScopeDecisionError("Cross-repository write/git scope requires verification.")
        return context.validate_scope(decision)
