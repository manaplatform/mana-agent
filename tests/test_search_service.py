from mana_agent.services.search_service import SearchService

def test_repository_search_requires_no_index_or_credentials(tmp_path):
    (tmp_path / "sample.py").write_text("native_engine = True\n")
    results = SearchService(repository_root=tmp_path).search(query="native_engine",k=5)
    assert len(results) == 1
    assert results[0].start_line == 1
    assert results[0].relative_path == "sample.py"


def test_directory_chat_groups_live_matches_by_repository_root(tmp_path):
    from mana_agent.services.ask_service import AskService

    project = tmp_path / "component"
    project.mkdir()
    (project / "sample.py").write_text("native_engine = True\n")
    hits, warnings = SearchService(repository_root=tmp_path).search_multi([project], query="native_engine", k=5)

    assert not warnings
    group, = AskService._group_sources_by_index(hits, [project])
    assert group.subproject_root == str(project.resolve())
    assert group.sources[0].relative_path == "sample.py"
