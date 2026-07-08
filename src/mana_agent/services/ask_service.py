"""
mana_agent.services.ask_service

Central orchestration layer for answering questions over indexed code context.
"""

from __future__ import annotations

import logging
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
        logger.info("Running model-routed ask flow: index_dir=%s k=%d", resolved_index, k)
        decision = self._route_or_error(
            question=question,
            index_dir=resolved_index,
            runtime_state=RouteRuntimeState(
                index_available=self._index_available(resolved_index),
                dir_mode=False,
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
        decision = self._route_or_error(
            question=question,
            index_dir=resolved_index,
            runtime_state=RouteRuntimeState(
                index_available=self._index_available(resolved_index),
                dir_mode=False,
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

        grouped: dict[Path, list[SearchHit]] = {
            item.resolve(): [] for item in index_dirs
        }

        for source in sources:
            source_path = Path(source.file_path).resolve()
            for index_dir in grouped.keys():
                subproject_root = index_dir.parent.parent if index_dir.parent.name == ".mana" else index_dir.parent
                if subproject_root in source_path.parents:
                    grouped[index_dir].append(source)

        result: list[SourceGroup] = []
        for index_dir, hits in grouped.items():
            if hits:
                result.append(
                    SourceGroup(
                        index_dir=str(index_dir),
                        subproject_root=str(index_dir.parent.parent if index_dir.parent.name == ".mana" else index_dir.parent),
                        sources=hits,
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
        decision = self._route_or_error(
            question=question,
            index_dir=resolved[0] if resolved else None,
            runtime_state=RouteRuntimeState(
                index_available=bool(resolved),
                dir_mode=True,
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
                available_tools=available_tool_names(),
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
