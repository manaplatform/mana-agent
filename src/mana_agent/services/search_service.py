"""Repository search through the same bounded native tools used by agents."""
from __future__ import annotations
import json
from pathlib import Path
from mana_agent.analysis.models import SearchHit
from mana_agent.tools.repository import repo_search

class SearchService:
    def __init__(self, repository_root: str | Path | None = None) -> None:
        self.repository_root = Path(repository_root).resolve() if repository_root is not None else None

    def _root(self, reference: str | Path | None) -> Path:
        if reference is None:
            if self.repository_root is None:
                raise ValueError("Repository search requires an explicit repository root")
            return self.repository_root
        path = Path(reference).resolve()
        if path == self.repository_root:
            return path
        manifest = path / "manifest.json"
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            root = Path(metadata["repository_root"]).resolve()
            from mana_agent.workspaces.store import WorkspaceStore
            known = {Path(item.canonical_path).resolve() for item in WorkspaceStore().list_repositories()}
            if root != self.repository_root and root not in known:
                raise PermissionError("Analysis metadata references an unregistered repository")
            return root
        if self.repository_root is not None:
            if path == self.repository_root or path.is_relative_to(self.repository_root / ".mana"):
                return self.repository_root
            from mana_agent.workspaces.paths import repository_index_dir, repository_id_for_path
            if path == repository_index_dir(repository_id_for_path(self.repository_root)):
                return self.repository_root
        if path.is_dir():
            from mana_agent.workspaces.store import WorkspaceStore
            if self.repository_root is None or path.is_relative_to(self.repository_root):
                return path
            if any(Path(item.canonical_path).resolve() == path for item in WorkspaceStore().list_repositories()):
                return path
        raise ValueError("Unknown repository scope; select a repository root instead of a missing index")

    def search(self, index_dir: str | Path | None = None, query: str = "", k: int = 5, *, repository_root: str | Path | None = None) -> list[SearchHit]:
        root = self._root(repository_root if repository_root is not None else index_dir)
        return self._search_root(root, query=query, k=k)

    @staticmethod
    def _search_root(root: Path, *, query: str, k: int) -> list[SearchHit]:
        payload = repo_search(root, query=query, limit=k)
        if not payload.get("ok"):
            raise ValueError(payload.get("error", "Repository search failed"))
        return [SearchHit(score=1.0, file_path=str(root / item["file"]), start_line=item["line"],
            end_line=item["line"], symbol_name="", snippet=item["text"], repository_name=root.name,
            relative_path=item["file"], qualified_path=f"{root.name}::{item['file']}") for item in payload["matches"]]

    def search_multi(self, index_dirs: list[Path], query: str, k: int) -> tuple[list[SearchHit], list[str]]:
        roots = sorted({self._root(reference) for reference in index_dirs} if index_dirs else {self._root(None)})
        hits = []
        for root in roots:
            hits.extend(self._search_root(root, query=query, k=k))
        return hits[:k], []
