"""Explicit removal of retired Mana-owned repository vectors (standard library only)."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--mana-home", type=Path, default=Path(os.environ.get("MANA_HOME") or Path.home() / ".mana"))
    options = parser.parse_args()
    roots = list(options.root)
    metadata_root = options.mana_home / "repositories"
    if metadata_root.is_dir() and not metadata_root.is_symlink():
        for entry in metadata_root.glob("*/repository.json"):
            if entry.is_symlink() or entry.parent.is_symlink():
                continue
            record = json.loads(entry.read_text())
            canonical = record.get("canonical_path")
            if canonical and Path(canonical).is_absolute() and Path(canonical).is_dir():
                roots.append(Path(canonical))
    source = Path(__file__).resolve().parents[1] / "src/mana_agent/llm/cleanup.py"
    spec = importlib.util.spec_from_file_location("mana_vector_cleanup", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps(module.purge_repository_vectors(list(dict.fromkeys(roots)), options.mana_home), indent=2))
