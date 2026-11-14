from __future__ import annotations
import asyncio
from pathlib import Path
import pytest
from mana_agent.llm.cleanup import purge_repository_vectors
from mana_agent.llm.probes import run_capability_probe

def test_vector_cleanup_preserves_analysis_and_unrelated_files(tmp_path):
    root, home = tmp_path / "repo", tmp_path / "home"
    directory = root / ".mana/index/faiss"
    directory.mkdir(parents=True)
    (directory / "index.faiss").write_bytes(b"vectors")
    (directory / "index.pkl").write_bytes(b"never deserialize")
    (directory / "notes.txt").write_text("keep")
    (directory.parent / "chunks.jsonl").write_text("keep analysis")
    result = purge_repository_vectors([root], home)
    assert len(result["removed"]) == 2
    assert (directory / "notes.txt").exists()
    assert (directory.parent / "chunks.jsonl").exists()

def test_cleanup_rejects_symlink_escape(tmp_path):
    root, outside = tmp_path / "repo", tmp_path / "outside"
    outside.mkdir()
    (outside / "index.pkl").write_bytes(b"keep")
    (root / ".mana/index").mkdir(parents=True)
    (root / ".mana/index/faiss").symlink_to(outside, target_is_directory=True)
    result = purge_repository_vectors([root], tmp_path / "home")
    assert result["refused"]
    assert (outside / "index.pkl").exists()

@pytest.mark.parametrize("initiated,consent", [(False, False), (False, True), (True, False)])
def test_probes_require_explicit_settings_initiation(initiated, consent):
    with pytest.raises(PermissionError):
        asyncio.run(run_capability_probe(None, "tool_calling", user_initiated=initiated, consent=consent))
