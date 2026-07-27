from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess

from mana_agent.model_routing.models import RepositoryMetadata


_LANGUAGE_SUFFIXES = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".rs": "rust",
    ".java": "java", ".kt": "kotlin", ".php": "php", ".rb": "ruby",
    ".cs": "csharp", ".cpp": "cpp", ".c": "c", ".swift": "swift",
}
_SENSITIVE_PARTS = {"auth", "security", "migration", "deploy", "database", "persistence", "concurrency", "api"}


@dataclass(slots=True)
class _CacheEntry:
    fingerprint: str
    metadata: RepositoryMetadata


class RepositoryMetadataInspector:
    """A bounded, fingerprinted repository inventory reused across requests."""

    def __init__(self) -> None:
        self._cache: dict[Path, _CacheEntry] = {}

    def inspect(self, root: Path) -> RepositoryMetadata:
        resolved = root.resolve()
        fingerprint = self._fingerprint(resolved)
        cached = self._cache.get(resolved)
        if cached and cached.fingerprint == fingerprint:
            return cached.metadata
        files = self._tracked_files(resolved)
        languages = sorted({_LANGUAGE_SUFFIXES[path.suffix.lower()] for path in files if path.suffix.lower() in _LANGUAGE_SUFFIXES})
        names = {path.name for path in files}
        frameworks: set[str] = set()
        build_systems: set[str] = set()
        if "pyproject.toml" in names:
            build_systems.add("pyproject")
            text = self._small_read(resolved / "pyproject.toml")
            frameworks.update(name for name in ("django", "fastapi", "flask") if name in text.lower())
        if "package.json" in names:
            build_systems.add("npm")
            text = self._small_read(resolved / "package.json")
            frameworks.update(name for name in ("react", "next", "vue", "nestjs") if name in text.lower())
        if "Cargo.toml" in names:
            build_systems.add("cargo")
        if "go.mod" in names:
            build_systems.add("go-modules")
        changed = self._git_lines(resolved, ["status", "--porcelain=v1"])
        changed_files = tuple(sorted({line[3:].strip() for line in changed if len(line) > 3}))
        sensitive = tuple(sorted({part for path in changed_files for part in _SENSITIVE_PARTS if part in path.lower().replace("-", "_").split("/")}))
        tests = sum(1 for path in files if "test" in path.name.lower() or "tests" in path.parts)
        final_fingerprint = self._fingerprint(resolved)
        metadata = RepositoryMetadata(
            languages=tuple(languages), frameworks=tuple(sorted(frameworks)), build_systems=tuple(sorted(build_systems)),
            file_count=len(files), test_file_count=tests, changed_files=changed_files, sensitive_areas=sensitive,
            fingerprint=final_fingerprint,
        )
        self._cache[resolved] = _CacheEntry(final_fingerprint, metadata)
        return metadata

    def _fingerprint(self, root: Path) -> str:
        values = [f"status:{line}" for line in self._git_lines(root, ["status", "--porcelain=v1"])]
        values.extend(f"index:{line}" for line in self._git_lines(root, ["write-tree"]))
        values.extend(f"head:{line}" for line in self._git_lines(root, ["rev-parse", "HEAD"]))
        for relative in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod"):
            path = root / relative
            try:
                stat = path.stat()
                values.append(f"{relative}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                continue
        return hashlib.sha256("|".join(values).encode()).hexdigest()[:20]

    @staticmethod
    def _git_lines(root: Path, args: list[str]) -> list[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            capture_output=True,
            check=False,
        )
        return result.stdout.splitlines() if result.returncode == 0 else []

    def _tracked_files(self, root: Path) -> list[Path]:
        lines = self._git_lines(root, ["ls-files"])
        return [Path(line) for line in lines[:100_000] if line]

    @staticmethod
    def _small_read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")[:100_000]
        except OSError:
            return ""
