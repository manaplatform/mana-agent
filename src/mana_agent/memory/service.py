"""Sole application-facing entry point for all memory backends."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mana_agent.memory.config import MemoryConfig
from mana_agent.memory.contracts import MemoryCapabilities
from mana_agent.memory.errors import MemoryConfigurationError
from mana_agent.memory.factory import create_memory_backend
from mana_agent.memory.models import (
    MemoryHealth,
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemoryUpdateRequest,
    MemoryWriteRequest,
)

logger = logging.getLogger(__name__)


class MemoryService:
    """Own one AI/semantic backend and always-on local system stores.

    External mode selects a hosted provider for conversation and semantic
    memory only. Local system stores remain available for:

    * run evidence (file read cache / artifact continuity)
    * coding-flow checkpoints and turn history
    * multi-agent compatibility records when no external runtime adapter exists

    Semantic AI writes never silently fall back from external to internal.
    """

    def __init__(
        self,
        root: str | Path = ".",
        *,
        project_root: str | Path | None = None,
        config: MemoryConfig | None = None,
        max_turns: int = 5,
        max_tasks: int = 20,
        session_id: str | None = None,
        workspace_id: str | None = None,
        repository_id: str | None = None,
        repository_ids: list[str] | None = None,
        user_id: str | None = None,
        enable_compatibility: bool = True,
        **_: Any,
    ) -> None:
        self.root = Path(project_root or root).resolve()
        self.config = (config or MemoryConfig.load()).validate()
        self.backend = create_memory_backend(self.config, root=self.root)
        from mana_agent.memory.capsules.service import CapsuleService

        self.capsules = CapsuleService(
            self.root,
            config=self.config.capsules,
            provider=self.config.provider,
        )
        self._coding: Any | None = None
        self._multi: Any | None = None
        self._external_runtime: Any | None = None
        self.user_id = str(user_id or "")
        self.session_id = str(session_id or "")
        self.workspace_id = str(workspace_id or "")
        self.repository_id = str(
            repository_id or (repository_ids or [""])[0] or ""
        )
        self.conversation_id = ""
        if enable_compatibility:
            # Coding-flow continuity is system state, not hosted AI memory.
            from mana_agent.services.coding_memory_service import CodingMemoryService as LegacyCodingMemoryService

            self._coding = LegacyCodingMemoryService(
                project_root=self.root,
                max_turns=max_turns,
                max_tasks=max_tasks,
                session_id=session_id,
            )
            if self.config.mode == "internal":
                from mana_agent.services.memory_service import MultiAgentMemoryService as LegacyMultiAgentMemoryService

                self._multi = LegacyMultiAgentMemoryService(
                    root=self.root,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    repository_id=repository_id or (repository_ids or [None])[0],
                    capsule_service=self.capsules,
                    capsules_enabled=self.config.capsules.enabled,
                )
            elif self.config.mode == "external":
                from mana_agent.memory.compatibility import ExternalRuntimeMemory
                from mana_agent.workspaces.paths import repository_id_for_path

                self._external_runtime = ExternalRuntimeMemory(
                    service=self,
                    root=self.root,
                    user_id=str(user_id or ""),
                    workspace_id=str(workspace_id or ""),
                    repository_id=str(
                        repository_id
                        or (repository_ids or [""])[0]
                        or repository_id_for_path(self.root)
                    ),
                    session_id=str(session_id or ""),
                )
        self._capabilities = self._build_capabilities(enable_compatibility=enable_compatibility)
        logger.info(
            "Memory service initialized: mode=%s provider=%s capabilities=%s",
            self.config.mode,
            self.config.provider,
            self._capabilities.as_dict(),
        )

    @staticmethod
    def _build_capabilities(*, enable_compatibility: bool) -> MemoryCapabilities:
        # Semantic domains always come from the selected backend.
        # System domains stay local whenever compatibility stores are enabled.
        return MemoryCapabilities(
            conversation=True,
            semantic_search=True,
            evidence=True,
            checkpoints=bool(enable_compatibility),
            coding_flow=bool(enable_compatibility),
            task_state=bool(enable_compatibility),
            multi_agent_runtime=bool(enable_compatibility),
        )

    @property
    def capabilities(self) -> MemoryCapabilities:
        return self._capabilities

    def require_capability(self, capability: str) -> None:
        """Fail closed when a route needs a domain this service cannot provide."""
        if self._capabilities.supports(capability):
            return
        raise MemoryConfigurationError(
            f"Memory capability {capability!r} is unavailable for mode={self.config.mode!r} "
            f"provider={self.config.provider!r}; no safe backend exists for that domain."
        )

    async def add(self, request: MemoryWriteRequest) -> MemoryRecord:
        self._reject_deleted_scope(request.scope)
        return await self.backend.add(request)

    async def search(self, request: MemorySearchRequest) -> list[MemoryRecord]:
        if self._scope_deleted(request.scope):
            return []
        return await self.backend.search(request)

    async def get(self, memory_id: str, scope: MemoryScope) -> MemoryRecord | None:
        if self._scope_deleted(scope):
            return None
        return await self.backend.get(memory_id, scope)

    async def update(self, memory_id: str, request: MemoryUpdateRequest) -> MemoryRecord:
        return await self.backend.update(memory_id, request)

    async def delete(self, memory_id: str, scope: MemoryScope) -> None:
        await self.backend.delete(memory_id, scope)

    async def clear(self, scope: MemoryScope) -> None:
        await self.backend.clear(scope)

    async def healthcheck(self) -> MemoryHealth:
        return await self.backend.healthcheck()

    async def close(self) -> None:
        await self.backend.close()

    def add_blocking(self, request: MemoryWriteRequest) -> MemoryRecord:
        from mana_agent.memory.compatibility import run_sync

        return run_sync(self.add(request))

    def search_blocking(self, request: MemorySearchRequest) -> list[MemoryRecord]:
        from mana_agent.memory.compatibility import run_sync

        return run_sync(self.search(request))

    def close_blocking(self) -> None:
        from mana_agent.memory.compatibility import run_sync

        run_sync(self.close())

    def bind_scope(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        workspace_id: str | None = None,
        repository_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """Update identities without constructing a backend or creating records."""
        if user_id is not None:
            self.user_id = str(user_id)
        if session_id is not None:
            self.session_id = str(session_id)
        if workspace_id is not None:
            self.workspace_id = str(workspace_id)
        if repository_id is not None:
            self.repository_id = str(repository_id)
        if conversation_id is not None:
            self.conversation_id = str(conversation_id)
        runtime = self._external_runtime
        if runtime is not None:
            runtime.user_id = self.user_id
            runtime.session_id = self.session_id
            runtime.workspace_id = self.workspace_id
            runtime.repository_id = self.repository_id

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "provider": self.config.provider,
            "capabilities": self._capabilities.as_dict(),
        }

    def capsule_diagnostics(self) -> dict[str, Any]:
        """Expose effective non-secret capsule settings to diagnostics."""
        return self.capsules.effective_settings()

    def build_capsule_bundle(self, request: Any) -> list[Any]:
        """Build a compact bundle only from an explicit authorized read request."""
        from mana_agent.memory.capsules.models import CapsuleReadRequest

        if not isinstance(request, CapsuleReadRequest):
            raise MemoryConfigurationError(
                "A validated CapsuleReadRequest is required; no legacy memory bundle was built."
            )
        return self.capsules.query_capsules(request)

    def build_bundle(self, **kwargs: Any) -> Any:
        """Retain legacy bundles only behind the disabled-capsule compatibility flag."""
        if self.config.capsules.enabled:
            raise MemoryConfigurationError(
                "Broad legacy memory bundles are disabled while scoped capsules are enabled; "
                "use build_capsule_bundle with a validated CapsuleReadRequest."
            )
        memory = self._require_runtime_store(self._multi or self._external_runtime, "legacy multi-agent memory bundle")
        return memory.build_bundle(**kwargs)

    @staticmethod
    def _scope_deleted(scope: MemoryScope) -> bool:
        session_id = str(scope.session_id or "").strip()
        if not session_id:
            return False
        from mana_agent.workspaces.paths import mana_home

        return (mana_home() / "runtime" / "session-tombstones" / f"{session_id}.json").is_file()

    def _reject_deleted_scope(self, scope: MemoryScope) -> None:
        if self._scope_deleted(scope):
            raise MemoryConfigurationError("The requested session was deleted; memory writes are permanently disabled for that scope.")

    def session_payload(self) -> dict[str, Any]:
        """Return legacy session evidence through the shared compatibility boundary."""
        if self.config.capsules.enabled:
            raise MemoryConfigurationError(
                "Broad session memory is disabled while scoped capsules are enabled."
            )
        memory = self._require_runtime_store(self._multi or self._external_runtime, "session evidence")
        return {
            "tasks": [asdict(item) for item in memory.task_records.values()],
            "tools": [asdict(item) for item in memory.tool_executions.values()],
            "decisions": [asdict(item) for item in memory.agent_decisions],
            "verifications": [asdict(item) for item in memory.verifications],
        }

    def project_snapshot(self, *, max_chars: int = 1200) -> str:
        if self.config.capsules.enabled:
            raise MemoryConfigurationError(
                "Broad project memory is disabled while scoped capsules are enabled."
            )
        memory = self._require_runtime_store(self._multi or self._external_runtime, "project memory snapshot")
        legacy_reader = getattr(self.backend, "project_snapshot", None)
        legacy = legacy_reader(max_chars=max_chars) if legacy_reader is not None else ""
        if legacy:
            return str(legacy)[:max_chars]
        facts = [str(item.get("fact") or "").strip() for item in memory.project_memory]
        text = "\n".join(item for item in facts if item)
        return text[:max_chars]

    @property
    def coding(self) -> Any:
        self.require_capability("coding_flow")
        return self._require_system_store(self._coding, "coding-flow memory")

    @property
    def multi_agent(self) -> Any:
        self.require_capability("multi_agent_runtime")
        delegate = self._multi if self._multi is not None else self._external_runtime
        return self._require_runtime_store(delegate, "multi-agent compatibility memory")

    def evidence_memory(self, *, run_id: str | None) -> Any:
        """Return the local run-evidence store.

        Run evidence is repository file state under the Mana run directory. It is
        not hosted AI memory and must not depend on external provider mapping.
        """
        self.require_capability("evidence")
        from mana_agent.services.memory_service import EvidenceMemory as LocalEvidenceMemory

        return LocalEvidenceMemory(repo_root=self.root, run_id=run_id)

    def _require_system_store(self, delegate: Any, operation: str) -> Any:
        if delegate is None:
            raise MemoryConfigurationError(
                f"{operation} system store is unavailable; no safe backend exists for that domain."
            )
        return delegate

    def _require_runtime_store(self, delegate: Any, operation: str) -> Any:
        if delegate is None:
            raise MemoryConfigurationError(
                f"{operation} has not yet been mapped to the selected external provider and no "
                "local multi-agent runtime store is available; no action was executed."
            )
        return delegate

    def __getattr__(self, name: str) -> Any:
        # Prefer local coding-flow system state, then multi-agent local store,
        # then external runtime adapter for multi-agent AI-memory-shaped ops.
        for delegate in (self._coding, self._multi, self._external_runtime):
            if delegate is not None and hasattr(delegate, name):
                return getattr(delegate, name)
        if self.config.mode == "external":
            raise MemoryConfigurationError(
                f"{name} is an unmapped memory operation for the selected external provider; "
                "no internal AI-memory fallback was executed."
            )
        raise AttributeError(name)


# Compatibility names intentionally resolve to the shared façade, never to a provider.
CodingMemoryService = MemoryService
MultiAgentMemoryService = MemoryService


class EvidenceMemory:
    """Compatibility façade for local run evidence.

    Always uses the durable local evidence store. External providers never own
    run evidence, so constructing this class does not require multi-agent or
    external provider mapping.
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        run_id: str | None,
        service: MemoryService | None = None,
    ) -> None:
        if service is not None:
            self.service = service
            self._delegate = service.evidence_memory(run_id=run_id)
            return
        # Avoid constructing a full MemoryService (and external backends) just
        # to open local run evidence for an agent turn.
        from mana_agent.services.memory_service import EvidenceMemory as LocalEvidenceMemory

        self.service = None
        self._delegate = LocalEvidenceMemory(repo_root=Path(repo_root).resolve(), run_id=run_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)
