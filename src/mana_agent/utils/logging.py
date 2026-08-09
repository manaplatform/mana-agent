from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys

from mana_agent.config.settings import default_logs_dir, mana_home
from mana_agent.utils.path_safety import safe_cwd


def setup_logging(verbose: bool = False, log_dir: str | Path | None = None) -> Path:
    level = logging.DEBUG if verbose else logging.INFO
    project_root = safe_cwd(fallback=mana_home())
    project_name = project_root.name or "project"
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if log_dir:
        from mana_agent.utils.path_safety import safe_resolve

        root = safe_resolve(log_dir)
    else:
        root = default_logs_dir(project_root)
    root.mkdir(parents=True, exist_ok=True)
    log_file = root / f"{date_tag}-{project_name}.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    if verbose:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    root_logger.propagate = False
    return log_file
