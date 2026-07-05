from __future__ import annotations

from mana_agent.multi_agent.runtime.repo_chain import RepositoryMultiChain


def test_compact_dependency_report_limits_large_lists() -> None:
    dependency_report = {
        "project_root": "/tmp/repo",
        "package_managers": [f"pm-{i}" for i in range(40)],
        "frameworks": [f"fw-{i}" for i in range(40)],
        "technologies": [f"tech-{i}" for i in range(40)],
        "runtime_dependencies": [f"dep-{i}" for i in range(500)],
        "dev_dependencies": [f"dev-{i}" for i in range(500)],
        "manifests": [f"manifest-{i}" for i in range(500)],
        "languages": [f"lang-{i}" for i in range(40)],
        "module_edges": [{"source": f"a{i}", "target": f"b{i}", "kind": "module", "file_path": f"f{i}"} for i in range(800)],
        "dependency_edges": [{"source": f"c{i}", "target": f"d{i}", "kind": "external", "file_path": f"g{i}"} for i in range(800)],
    }

    compact = RepositoryMultiChain._compact_dependency_report(dependency_report)

    assert len(compact["package_managers"]) == RepositoryMultiChain._MAX_FRAMEWORKS
    assert len(compact["runtime_dependencies"]) == RepositoryMultiChain._MAX_DEPENDENCIES
    assert len(compact["module_edges"]) == RepositoryMultiChain._MAX_EDGES
    assert set(compact["module_edges"][0]) == {"source", "target", "kind"}


def test_compact_file_summaries_limits_entries_and_text() -> None:
    summaries = [
        {
            "file_path": f"f{i}.py",
            "language": "python",
            "symbols": [f"s{j}" for j in range(100)],
            "summary": "x" * 2000,
        }
        for i in range(40)
    ]

    compact = RepositoryMultiChain._compact_file_summaries(summaries)

    assert len(compact) == RepositoryMultiChain._MAX_FILE_SUMMARIES
    assert len(compact[0]["symbols"]) == RepositoryMultiChain._MAX_SYMBOLS_PER_FILE
    assert len(compact[0]["summary"]) <= RepositoryMultiChain._MAX_SUMMARY_CHARS
