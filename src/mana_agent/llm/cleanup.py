"""Confined removal of retired Mana repository-vector artifacts. Never unpickles."""
from __future__ import annotations
import os
from pathlib import Path


def purge_repository_vectors(project_roots: list[Path], mana_home: Path) -> dict[str, list[str]]:
    removed, refused = [], []
    roots = [root.absolute() / ".mana" / "index" for root in project_roots]
    roots += [root.absolute() / ".mana_index" for root in project_roots]
    repositories = mana_home.absolute() / "repositories"
    if repositories.is_dir() and not repositories.is_symlink():
        roots.extend(path / "index" for path in repositories.iterdir() if path.is_dir() and not path.is_symlink())
    def safe(path: Path, boundary: Path) -> bool:
        current = path
        while current != boundary:
            if current.is_symlink() or current.parent == current:
                return False
            current = current.parent
        return not boundary.is_symlink() and path.resolve().is_relative_to(boundary.resolve())
    for root in roots:
        # A symlink above the recognized root is also an escape, even if the
        # final component itself is a regular directory.
        if any(parent.is_symlink() for parent in (root, *root.parents)):
            refused.append(str(root))
            continue
        directories = [root / "faiss"]
        if (root / "index.faiss").exists() or (root / "index.faiss").is_symlink():
            directories.append(root)
        for directory in directories:
            _purge_index_directory(directory, root.parent.parent, removed, refused, safe, remove_directory=directory != root)
    cache = mana_home.absolute() / "cache" / "embeddings"
    if cache.exists() or cache.is_symlink():
        if any(parent.is_symlink() for parent in (cache, *cache.parents)) or not safe(cache, mana_home.absolute()) or not cache.is_dir():
            refused.append(str(cache))
        else:
            for directory, subdirs, files in os.walk(cache, followlinks=False, topdown=False):
                for name in files:
                    item = Path(directory) / name
                    if item.is_symlink():
                        refused.append(str(item))
                    else:
                        item.unlink()
                        removed.append(str(item))
                for name in subdirs:
                    item = Path(directory) / name
                    if item.is_symlink():
                        refused.append(str(item))
                    elif not list(item.iterdir()):
                        item.rmdir()
            if not list(cache.iterdir()):
                cache.rmdir()
    return {"removed": removed, "refused": refused}


def _purge_index_directory(directory, boundary, removed, refused, safe, *, remove_directory):
    if not directory.exists() and not directory.is_symlink():
        return
    if not safe(directory, boundary) or not directory.is_dir():
        refused.append(str(directory))
        return
    for name in ("index.faiss", "index.pkl"):
        artifact = directory / name
        if artifact.is_symlink():
            refused.append(str(artifact))
        elif artifact.is_file():
            artifact.unlink()
            removed.append(str(artifact))
    if remove_directory and not list(directory.iterdir()):
        directory.rmdir()
