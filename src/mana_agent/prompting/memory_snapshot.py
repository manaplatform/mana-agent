from __future__ import annotations

from pathlib import Path


_MEMORY_CANDIDATES = (
    ".mana/memory.md",
    ".mana/project_memory.md",
    "MEMORY.md",
)


def _compact_text(text: str, *, max_chars: int) -> str:
    lines: list[str] = []
    for raw in str(text or "").splitlines():
        cleaned = " ".join(raw.strip().split())
        if cleaned:
            lines.append(cleaned)
        if sum(len(line) + 1 for line in lines) >= max_chars:
            break
    compact = "\n".join(lines)
    if len(compact) > max_chars:
        return compact[: max_chars - 3].rstrip() + "..."
    return compact


def render_memory_snapshot(*, repo_root: str | Path | None = None, max_chars: int = 1200) -> str:
    root = Path(repo_root or Path.cwd()).expanduser().resolve()
    for relative in _MEMORY_CANDIDATES:
        path = root / relative
        if path.is_file():
            compact = _compact_text(path.read_text(encoding="utf-8"), max_chars=max_chars)
            return f"Project Memory Snapshot\n- source: {relative}\n{compact}"
    return "Project Memory Snapshot\n- none available"

