from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from mana_agent.config.settings import default_llm_logs_dir, mana_home
from mana_agent.config.user_config import get_setting
from mana_agent.utils.io import ensure_dir
from mana_agent.utils.path_safety import safe_cwd, safe_resolve


class LlmRunLogger:
    def __init__(self, log_file: str | Path | None = None) -> None:
        configured_path = str(get_setting("MANA_LLM_LOG_FILE", "") or "").strip()

        # Never crash when the process CWD was deleted under a live agent
        # (SWE-bench worktree thrash, concurrent runners, etc.).
        project_root = safe_cwd(fallback=mana_home())
        project_name = project_root.name or "project"
        date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # 1️⃣ Explicit argument
        if log_file:
            resolved = Path(log_file)

        # 2️⃣ User-level Mana configuration
        elif configured_path:
            resolved = Path(configured_path)

        # 3️⃣ Default location (safe)
        else:
            resolved = default_llm_logs_dir(project_root) / f"{date_tag}-{project_name}-runs.jsonl"

        # safe_resolve: Windows realpath always calls getcwd().
        resolved = safe_resolve(resolved)

        # ✅ FIX: if path is directory → generate file inside it
        if resolved.exists() and resolved.is_dir():
            resolved = (
                resolved
                / f"{date_tag}-{project_name}-runs.jsonl"
            )

        # ✅ Also handle case where path ends with slash but doesn't exist yet
        if not resolved.suffix:
            # no file extension → treat as directory
            resolved = (
                resolved
                / f"{date_tag}-{project_name}-runs.jsonl"
            )

        self.log_file = resolved

    def log(self, payload: dict[str, Any]) -> None:
        ensure_dir(self.log_file.parent)

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }

        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
