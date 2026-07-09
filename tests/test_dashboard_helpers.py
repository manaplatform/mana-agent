"""Minimal tests for dashboard helpers (Grok Build).

These verify that the optional UI bridge loads without side effects
on core, and that artifact loading is safe (no crash on missing .mana).
"""
from pathlib import Path
import tempfile

import pytest

from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    get_index_stats,
    get_metrics_summary,
    list_analysis_artifacts,
    load_automations,
    load_taskboard_state,
    load_recent_traces,
    safe_read_json,
    trigger_automation,
)


def test_safe_read_json_missing_is_none():
    assert safe_read_json(Path("/nonexistent/xyz.json")) is None


def test_loads_gracefully_without_mana_dir():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        tb = load_taskboard_state(root)
        assert isinstance(tb, dict)
        assert "status" in tb or "tasks" in tb
        traces = load_recent_traces(root)
        assert isinstance(traces, list)
        idx = get_index_stats(root)
        assert isinstance(idx, dict)


def test_find_mana_root_defaults_to_cwd():
    # Should return a path (does not need to be perfect in temp env)
    p = find_mana_root()
    assert isinstance(p, Path)


def test_new_helpers_graceful(tmp_path):
    root = tmp_path
    m = get_metrics_summary(root)
    assert isinstance(m, dict)
    assert "sessions" in m and "success_rate" in m

    arts = list_analysis_artifacts(root)
    assert isinstance(arts, list)

    autos = load_automations(root)
    assert "automations" in autos and "runs" in autos

    # trigger is safe even without data
    r = trigger_automation("noop", root=root)
    assert isinstance(r, dict)
    assert "ok" in r or "action" in r
