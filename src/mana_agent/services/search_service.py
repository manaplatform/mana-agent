from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mana_agent.analysis.models import SearchHit
from mana_agent.utils.io import read_jsonl
from mana_agent.vector_store.faiss_store import FaissStore
from mana_agent.multi_agent.runtime.prompts import TOOL_FIRST

logger = logging.getLogger(__name__)



class SearchService:
    def __init__(self, store: FaissStore) -> None:
        self.store = store
        self._executor_workers = max(1, min(8, os.cpu_count() or 4))

    # ---------------------------------------
    # Main search APIs
    # ---------------------------------------
    def search(self, index_dir: str | Path, query: str, k: int) -> list[SearchHit]:
        resolved_index = Path(index_dir).resolve()
        logger.info("Running semantic search: index_dir=%s k=%d", resolved_index, k)
        logger.debug("Search query: %s", query)

        try:
            hits = self.store.search(resolved_index, query=query, k=k)
        except Exception as exc:
            logger.warning("Vector search failed for %s: %s", resolved_index, exc)
            hits = []

        # If vectors are missing, fall back to lexical chunk search
        if not hits and not (resolved_index / "faiss").exists():
            logger.info("Falling back to lexical chunk search: index_dir=%s", resolved_index)
            hits = self._lexical_search(resolved_index, query=query, k=k)

        logger.info("Semantic search completed: hits=%d", len(hits))
        return hits

    def search_multi(self, index_dirs: list[Path], query: str, k: int) -> tuple[list[SearchHit], list[str]]:
        resolved_indexes = sorted({Path(item).resolve() for item in index_dirs}, key=lambda item: str(item))
        logger.info(
            "Running multi-index semantic search: indexes=%d k=%d",
            len(resolved_indexes),
            k,
        )
        logger.debug("Search query: %s", query)

        warnings: list[str] = []
        merged: list[SearchHit] = []
        worker_count = min(len(resolved_indexes), self._executor_workers)

        logger.info(
            "Executing concurrent multi-index search",
            extra={
                "requested_indexes": len(resolved_indexes),
                "worker_count": worker_count,
            },
        )

        if worker_count <= 1:
            for index_dir in resolved_indexes:
                self._collect_search_results(index_dir, query, k, merged, warnings)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_index = {
                    executor.submit(self.search, index_dir=index_dir, query=query, k=k): index_dir
                    for index_dir in resolved_indexes
                }
                for future in as_completed(future_to_index):
                    index_dir = future_to_index[future]
                    try:
                        hits = future.result()
                    except Exception as exc:
                        warning = f"Concurrent search skipped {index_dir}: {exc}"
                        logger.warning(warning)
                        warnings.append(warning)
                        continue
                    merged.extend(hits)

        deduped = sorted(
            {(item.file_path, item.start_line, item.end_line, item.symbol_name): item for item in merged}.values(),
            key=lambda item: (-item.score, item.file_path, item.start_line, item.end_line, item.symbol_name),
        )
        return deduped[:k], warnings

    # ---------------------------------------
    # Lexical fallback
    # ---------------------------------------
    @staticmethod
    def _tokenize(value: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", value.lower()))

    def _lexical_search(self, index_dir: Path, query: str, k: int) -> list[SearchHit]:
        chunk_rows = read_jsonl(index_dir / "chunks.jsonl")
        if not chunk_rows:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scored: list[SearchHit] = []
        for row in chunk_rows:
            text = str(row.get("text", ""))
            if not text:
                continue
            tokens = self._tokenize(text)
            if not tokens:
                continue
            overlap = q_tokens & tokens
            if not overlap:
                continue

            score = len(overlap) / max(len(q_tokens), 1)
            scored.append(
                SearchHit(
                    score=float(score),
                    file_path=str(row.get("file_path", "")),
                    start_line=int(row.get("start_line", 1)),
                    end_line=int(row.get("end_line", 1)),
                    symbol_name=str(row.get("symbol_name", "unknown")),
                    snippet=text[:500],
                )
            )

        return sorted(
            scored,
            key=lambda item: (-item.score, item.file_path, item.start_line, item.end_line, item.symbol_name),
        )[:k]

    def _collect_search_results(
        self,
        index_dir: Path,
        query: str,
        k: int,
        merged: list[SearchHit],
        warnings: list[str],
    ) -> None:
        try:
            hits = self.search(index_dir=index_dir, query=query, k=k)
            merged.extend(hits)
        except Exception as exc:
            warning = f"Skipped unusable index {index_dir}: {exc}"
            logger.warning(warning)
            warnings.append(warning)

    # ---------------------------------------
    # Tool-friendly wrappers
    # ---------------------------------------
    def tool_semantic_search(self, index_dir: str | Path, query: str, k: int = 5) -> dict:
        """
        Tool wrapper for agents: run search() and return JSON-friendly payload.
        """
        results = self.search(index_dir=index_dir, query=query, k=k)
        return {
            "index_dir": str(Path(index_dir).resolve()),
            "query": query,
            "k": k,
            "results": [
                {
                    "file": r.file_path,
                    "start": r.start_line,
                    "end": r.end_line,
                    "score": float(r.score),
                    "symbol": r.symbol_name,
                    "snippet": (r.snippet or "")[:500],
                }
                for r in results
            ],
        }

    def tool_semantic_search_multi(self, index_dirs: list[str | Path], query: str, k: int = 5) -> dict:
        """
        Tool wrapper for agents: run search_multi() and return JSON-friendly payload.
        """
        dirs = [Path(p).resolve() for p in index_dirs]
        results, warnings = self.search_multi(index_dirs=dirs, query=query, k=k)
        return {
            "index_dirs": [str(p) for p in dirs],
            "query": query,
            "k": k,
            "warnings": warnings,
            "results": [
                {
                    "file": r.file_path,
                    "start": r.start_line,
                    "end": r.end_line,
                    "score": float(r.score),
                    "symbol": r.symbol_name,
                    "snippet": (r.snippet or "")[:500],
                }
                for r in results
            ],
        }

    # ---------------------------------------
    # Tool-first prompt access
    # ---------------------------------------
    @staticmethod
    def tool_first_system_prompt() -> str:
        """
        Returns the canonical tool-first system prompt string.
        Use this in AskAgent / QnAChain system messages to force tool usage.
        """
        return TOOL_FIRST