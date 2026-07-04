from __future__ import annotations

from pathlib import Path


def _compact_text(text: str, *, max_chars: int) -> str:
    lines: list[str] = []
    total = 0
    for raw in str(text or "").splitlines():
        cleaned = " ".join(raw.strip().split())
        if not cleaned:
            continue
        lines.append(cleaned)
        total += len(cleaned) + 1
        if total >= max_chars:
            break
    compact = "\n".join(lines)
    if len(compact) > max_chars:
        return compact[: max_chars - 3].rstrip() + "..."
    return compact


def render_repo_rules(*, repo_root: str | Path | None = None, max_chars: int = 2400) -> str:
    root = Path(repo_root or Path.cwd()).expanduser().resolve()
    agents = root / "AGENTS.md"
    lines = ["Repository Rules"]
    if agents.is_file():
        lines.append("- source: AGENTS.md")
        lines.append(_compact_text(agents.read_text(encoding="utf-8"), max_chars=max_chars))
    else:
        lines.append("- source: default")
        lines.append("- Inspect repository state before editing.")
        lines.append("- Preserve user changes and avoid unrelated rewrites.")
        lines.append("- Run relevant verification when code changes.")
    return "\n".join(lines)
