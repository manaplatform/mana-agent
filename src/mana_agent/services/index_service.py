from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

from mana_agent.analysis.chunker import CodeChunker
from mana_agent.analysis.models import CodeChunk
from mana_agent.analysis.models import CodeSymbol
from mana_agent.parsers.python_parser import PythonParser
from mana_agent.utils.io import (
    ensure_dir,
    iter_source_files,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.workspaces.paths import mana_home
from mana_agent.workspaces.store import atomic_write_json

logger = logging.getLogger(__name__)

CHUNK_SCHEMA_VERSION = 2


class IndexService:
    def __init__(self, parser: PythonParser, chunker: CodeChunker) -> None:
        self.parser = parser
        self.chunker = chunker
        self._executor_workers = max(1, min(8, os.cpu_count() or 4))

    @staticmethod
    def _manifest_path(index_dir: Path) -> Path:
        return index_dir / "manifest.json"

    @staticmethod
    def _chunks_path(index_dir: Path) -> Path:
        return index_dir / "chunks.jsonl"

    def _load_manifest(self, index_dir: Path) -> dict:
        logger.debug("Loading manifest from %s", self._manifest_path(index_dir))
        payload = read_json(self._manifest_path(index_dir))
        if not payload:
            payload = {"files": {}}
        payload.setdefault("files", {})
        return payload

    def _load_chunks(self, index_dir: Path) -> dict[str, CodeChunk]:
        logger.debug("Loading chunks from %s", self._chunks_path(index_dir))
        rows = read_jsonl(self._chunks_path(index_dir))
        chunks: dict[str, CodeChunk] = {}
        for row in rows:
            chunk = CodeChunk(**row)
            chunks[chunk.id] = chunk
        return chunks

    def _build_chunks_for_file(self, file_path: str) -> tuple[str, list[CodeChunk]]:
        logger.debug("Gathering chunks for %s via thread pool", file_path)
        try:
            digest = sha256_file(Path(file_path))
            parser_key = self.parser.__class__.__name__.lower()
            cache_path = mana_home() / "cache" / "parsed-chunks" / f"{parser_key}-{digest}.json"
            symbols: list[CodeSymbol]
            if cache_path.exists():
                rows = read_json(cache_path).get("symbols", [])
                symbols = [CodeSymbol(**{**row, "file_path": file_path}) for row in rows]
            else:
                symbols = self.parser.parse_file(file_path)
                atomic_write_json(
                    cache_path,
                    {
                        "schema_version": 1,
                        "content_hash": digest,
                        "parser": parser_key,
                        "symbols": [symbol.to_dict() for symbol in symbols],
                    },
                )
            file_chunks = self.chunker.build_chunks(symbols)
            return file_path, file_chunks
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("Failed to build chunks for %s: %s", file_path, exc)
            return file_path, []

    def index(
        self,
        target_path: str | Path,
        index_dir: str | Path,
        rebuild: bool = False,
        repository_id: str | None = None,
        repository_name: str | None = None,
    ) -> dict:
        target = Path(target_path).resolve()
        registered = WorkspaceService().register_repository(target)
        repository_id = repository_id or registered.repository_id
        repository_name = repository_name or registered.name
        index_root = ensure_dir(index_dir)
        logger.info(
            "Starting index run: target=%s index_dir=%s rebuild=%s",
            target,
            index_root,
            rebuild,
        )

        if rebuild:
            manifest = {"files": {}}
            chunk_map: dict[str, CodeChunk] = {}
        else:
            manifest = self._load_manifest(index_root)
            chunk_map = self._load_chunks(index_root)

        current_files = iter_source_files(target)
        logger.info("Discovered %d source files", len(current_files))
        current_hashes = {str(path): sha256_file(path) for path in current_files}
        chunk_schema_changed = manifest.get("chunk_schema_version") != CHUNK_SCHEMA_VERSION
        manifest.update(
            {
                "schema_version": 2,
                "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                "repository_id": repository_id,
                "repository_name": repository_name,
                "repository_root": str(target),
            }
        )
        known_files = set(manifest["files"].keys())
        existing_files = set(current_hashes.keys())

        changed_files = {
            path for path, digest in current_hashes.items() if manifest["files"].get(path, {}).get("sha256") != digest
        }
        if chunk_schema_changed:
            changed_files = set(existing_files)
        deleted_files = known_files - existing_files
        logger.info(
            "Index delta computed: changed=%d deleted=%d unchanged=%d",
            len(changed_files),
            len(deleted_files),
            len(existing_files) - len(changed_files),
        )

        remove_chunk_ids: list[str] = []
        for file_path in sorted(changed_files | deleted_files):
            old = manifest["files"].get(file_path, {})
            remove_chunk_ids.extend(old.get("chunk_ids", []))
            for chunk_id in old.get("chunk_ids", []):
                chunk_map.pop(chunk_id, None)
            manifest["files"].pop(file_path, None)

        new_chunks: list[CodeChunk] = []
        if changed_files:
            sorted_changed = sorted(changed_files)
            worker_count = min(len(sorted_changed), self._executor_workers)
            logger.info(
                "Processing %d changed files using %d worker(s)",
                len(sorted_changed),
                worker_count,
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for file_path, file_chunks in executor.map(self._build_chunks_for_file, sorted_changed):
                    if not file_chunks:
                        logger.warning("No chunks built for %s", file_path)
                        continue
                    relative_path = Path(file_path).resolve().relative_to(target).as_posix()
                    for chunk in file_chunks:
                        chunk.repository_id = str(repository_id or "")
                        chunk.repository_name = str(repository_name or "")
                        chunk.relative_path = relative_path
                        chunk.qualified_path = f"{repository_name}::{relative_path}"
                        chunk.id = f"{repository_id}:{chunk.id}"
                    new_chunks.extend(file_chunks)
                    for chunk in file_chunks:
                        chunk_map[chunk.id] = chunk
                    manifest["files"][file_path] = {
                        "sha256": current_hashes[file_path],
                        "last_indexed_at": datetime.now(timezone.utc).isoformat(),
                        "chunk_ids": [chunk.id for chunk in file_chunks],
                    }
                    logger.debug("Prepared %d chunks for %s", len(file_chunks), file_path)

        # Persist local analysis metadata; repository search reads live files.
        write_json(self._manifest_path(index_root), manifest)
        write_jsonl(self._chunks_path(index_root), [chunk.to_dict() for chunk in chunk_map.values()])
        logger.debug(
            "Persisted manifest and chunk catalog: files=%d chunks=%d",
            len(manifest["files"]),
            len(chunk_map),
        )

        return {
            "indexed_files": len(changed_files),
            "deleted_files": len(deleted_files),
            "total_files": len(existing_files),
            "new_chunks": len(new_chunks),
            "removed_chunks": len(remove_chunk_ids),
            "index_dir": str(index_root),
            "repository_id": repository_id,
            "repository_name": repository_name,
        }
