"""
mana_agent.services.ask_service

Central orchestration layer for answering questions over indexed code context.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from mana_agent.analysis.models import (
    AskResponse,
    AskResponseWithTrace,
    SearchHit,
    SourceGroup,
)
from mana_agent.multi_agent.runtime.ask_agent import AskAgent
from mana_agent.multi_agent.runtime.entry_router import EntryRouter, RouteDecisionError, RouteRuntimeState
from mana_agent.multi_agent.runtime.qna_chain import QnAChain
from mana_agent.multi_agent.runtime.route_executor import (
    RouteExecutionContext,
    RouteExecutor,
    available_command_names,
    available_tool_names,
)
from mana_agent.services.search_service import SearchService
from mana_agent.vector_store.faiss_store import FaissStore
from mana_agent.mcp.config import load_mcp_servers

logger = logging.getLogger(__name__)


@runtime_checkable
class AskCallback(Protocol):
    def on_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        ...


class AskService:
    def __init__(
        self,
        store: FaissStore,
        qna_chain: QnAChain,
        ask_agent: AskAgent | None = None,
        search_service: SearchService | None = None,
        project_root: str | Path | None = None,
        entry_router: EntryRouter | None = None,
        route_executor: RouteExecutor | None = None,
    ) -> None:
        self.store = store
        self.qna_chain = qna_chain
        self.ask_agent = ask_agent
        self.search_service = search_service
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self.entry_router = entry_router or EntryRouter(llm=getattr(qna_chain, "llm", None))
        self.route_executor = route_executor or RouteExecutor(
            router=self.entry_router,
            store=store,
            qna_chain=qna_chain,
            ask_agent=ask_agent,
            search_service=search_service,
            project_root=self.project_root,
        )

    @staticmethod
    def _requested_mcp_server(question: str) -> str | None:
        """Return an explicitly named configured MCP provider, if any.

        This is a user-supplied execution constraint, not a workflow router:
        the model still chooses the concrete tool from that provider.
        """
        text = str(question or "")
        try:
            server_ids = [server.id for server in load_mcp_servers()]
        except Exception:
            return None
        matches = [
            server_id
            for server_id in server_ids
            if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(server_id)}(?![A-Za-z0-9_-])", text, re.IGNORECASE)
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _render_context(sources: list[SearchHit]) -> str:
        blocks: list[str] = []
        for src in sources:
            blocks.append(
                "\n".join(
                    [
                        f"source: {src.file_path}:{src.start_line}-{src.end_line}",
                        f"symbol: {src.symbol_name}",
                        "snippet:",
                        src.snippet,
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)

    # ==========================================================
    # Classic ask
    # ==========================================================

    def ask(self, index_dir: str | Path, question: str, k: int) -> AskResponse:
        resolved_index = Path(index_dir).resolve()
        required_mcp_server = self._requested_mcp_server(question)
        logger.info("Running model-routed ask flow: index_dir=%s k=%d", resolved_index, k)
        decision = self._route_or_error(
            question=question,
            index_dir=resolved_index,
            runtime_state=RouteRuntimeState(
                index_available=self._index_available(resolved_index),
                dir_mode=False,
                required_mcp_server=required_mcp_server,
            ),
        )
        if isinstance(decision, AskResponseWithTrace):
            return decision
        return self.route_executor.execute(
            decision,
            RouteExecutionContext(
                question=question,
                index_dir=resolved_index,
                project_root=self.project_root,
                k=k,
                index_available=self._index_available(resolved_index),
                required_mcp_server=required_mcp_server,
            ),
        )

    # ==========================================================
    # Agent tools (single index)
    # ==========================================================

    def ask_with_tools(
        self,
        index_dir: str | Path,
        question: str,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        callbacks: Sequence[Any] | None = None,
    ) -> AskResponseWithTrace:

        resolved_index = Path(index_dir).resolve()
        required_mcp_server = self._requested_mcp_server(question)
        decision = self._route_or_error(
            question=question,
            index_dir=resolved_index,
            runtime_state=RouteRuntimeState(
                index_available=self._index_available(resolved_index),
                dir_mode=False,
                required_mcp_server=required_mcp_server,
            ),
        )
        if isinstance(decision, AskResponseWithTrace):
            return decision
        try:
            return self.route_executor.execute(
                decision,
                RouteExecutionContext(
                    question=question,
                    index_dir=resolved_index,
                    project_root=self.project_root,
                    k=k,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    callbacks=callbacks,
                    index_available=self._index_available(resolved_index),
                    required_mcp_server=required_mcp_server,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - no alternate route is selected here
            logger.exception("model-selected ask route failed")
            return AskResponseWithTrace(
                answer=f"Selected route failed: {exc}",
                sources=[],
                mode="route-error",
                trace=[],
                warnings=[str(exc)],
                route_trace={
                    "route_kind": decision.kind,
                    "router_model": self.entry_router.router_model,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "validation": "execution_failed",
                    "executed_tools": [],
                },
            )

    # ==========================================================
    # Dir mode
    # ==========================================================

    @staticmethod
    def _group_sources_by_index(
        sources: list[SearchHit],
        index_dirs: list[Path],
    ) -> list[SourceGroup]:

        import json

        grouped: dict[Path, list[SearchHit]] = {item.resolve(): [] for item in index_dirs}
        metadata: dict[Path, dict[str, str]] = {}
        for index_dir in grouped:
            try:
                manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            legacy_root = index_dir.parent.parent if index_dir.parent.name == ".mana" else index_dir.parent
            metadata[index_dir] = {
                "repository_id": str(manifest.get("repository_id") or ""),
                "repository_name": str(manifest.get("repository_name") or legacy_root.name),
                "repository_root": str(manifest.get("repository_root") or legacy_root),
            }

        for source in sources:
            for index_dir in grouped.keys():
                meta = metadata[index_dir]
                if source.repository_id and source.repository_id == meta["repository_id"]:
                    grouped[index_dir].append(source)
                    break
                if meta["repository_root"]:
                    source_path = Path(source.file_path).resolve()
                    repository_root = Path(meta["repository_root"]).resolve()
                    if source_path == repository_root or repository_root in source_path.parents:
                        grouped[index_dir].append(source)
                        break

        result: list[SourceGroup] = []
        for index_dir, hits in grouped.items():
            if hits:
                result.append(
                    SourceGroup(
                        index_dir=str(index_dir),
                        subproject_root=metadata[index_dir]["repository_root"],
                        sources=hits,
                        repository_id=metadata[index_dir]["repository_id"],
                        repository_name=metadata[index_dir]["repository_name"],
                    )
                )
        return result

    def ask_dir_mode(
        self,
        index_dirs: list[str | Path],
        question: str,
        k: int,
        root_dir: str | Path,
    ) -> AskResponse:

        if self.search_service is None:
            raise RuntimeError("search service is not configured")

        resolved = sorted({Path(p).resolve() for p in index_dirs})

        if not resolved:
            root = Path(root_dir).resolve()
            msg = f"No usable indexes found under {root}"
            return AskResponse(answer=msg, sources=[], warnings=[msg])

        sources, warnings = self.search_service.search_multi(
            index_dirs=resolved,
            query=question,
            k=k,
        )

        if not sources:
            return AskResponse(
                answer=f"The selected semantic route found no indexed context under {Path(root_dir).resolve()}.",
                sources=[],
                warnings=warnings,
            )

        context = self._render_context(sources)
        answer = self.qna_chain.run(question=question, context=context)

        return AskResponse(
            answer=answer,
            sources=sources,
            source_groups=self._group_sources_by_index(sources, resolved),
            warnings=warnings,
        )

    def ask_with_tools_dir_mode(
        self,
        index_dirs: list[str | Path],
        question: str,
        k: int,
        max_steps: int = 6,
        timeout_seconds: int = 30,
        root_dir: str | Path | None = None,
        callbacks: Sequence[Any] | None = None,
    ) -> AskResponseWithTrace:

        resolved = sorted({Path(p).resolve() for p in index_dirs})
        root = Path(root_dir or self.project_root).resolve()
        required_mcp_server = self._requested_mcp_server(question)
        decision = self._route_or_error(
            question=question,
            index_dir=resolved[0] if resolved else None,
            runtime_state=RouteRuntimeState(
                index_available=bool(resolved),
                dir_mode=True,
                required_mcp_server=required_mcp_server,
            ),
        )
        if isinstance(decision, AskResponseWithTrace):
            return decision
        try:
            result = self.route_executor.execute(
                decision,
                RouteExecutionContext(
                    question=question,
                    index_dirs=resolved,
                    index_dir=resolved[0] if resolved else None,
                    project_root=self.project_root,
                    k=k,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    root_dir=root,
                    callbacks=callbacks,
                    dir_mode=True,
                    index_available=bool(resolved),
                    required_mcp_server=required_mcp_server,
                )
            )
            result.source_groups = self._group_sources_by_index(
                result.sources,
                resolved,
            )
            return result

        except Exception as exc:  # noqa: BLE001 - no alternate route is selected here
            logger.exception("model-selected dir-mode route failed")
            return AskResponseWithTrace(
                answer=f"Selected route failed: {exc}",
                sources=[],
                source_groups=[],
                warnings=[str(exc)],
                mode="route-error",
                trace=[],
                route_trace={
                    "route_kind": decision.kind,
                    "router_model": self.entry_router.router_model,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "validation": "execution_failed",
                    "executed_tools": [],
                },
            )

    def _route_or_error(
        self,
        *,
        question: str,
        index_dir: Path | None,
        runtime_state: RouteRuntimeState,
    ) -> Any:
        try:
            return self.entry_router.route(
                question=question,
                index_dir=index_dir,
                project_root=self.project_root,
                available_commands=available_command_names(self.project_root),
                available_tools=available_tool_names(
                    required_mcp_server=runtime_state.required_mcp_server,
                ),
                runtime_state=runtime_state,
            )
        except RouteDecisionError as exc:
            return AskResponseWithTrace(
                answer=str(exc),
                sources=[],
                warnings=[str(exc)],
                mode="route-error",
                trace=[],
                route_trace={
                    "route_kind": "unsupported",
                    "router_model": self.entry_router.router_model,
                    "confidence": 0.0,
                    "reason": str(exc),
                    "validation": "router_error",
                    "executed_tools": [],
                },
            )

    def _index_available(self, resolved_index: Path) -> bool:
        if isinstance(self.store, FaissStore):
            return (resolved_index / "faiss").exists()
        return True
