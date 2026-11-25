from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.tools.repository import apply_patch_batch, list_files, repo_batch_read, repo_batch_search, run_script_once


def test_brace_globs_find_typescript_declarations_in_scope(tmp_path: Path) -> None:
    from mana_agent.tools.repository import execute_repository_discovery

    source = tmp_path / "src"
    source.mkdir()
    (source / "types.ts").write_text("export interface StorageAreaUsage {}\n", encoding="utf-8")
    (source / "view.tsx").write_text("export interface StorageAreaUsage {}\n", encoding="utf-8")
    (source / "other.js").write_text("export interface StorageAreaUsage {}\n", encoding="utf-8")
    result = execute_repository_discovery(tmp_path, "repo_search", {
        "query": "StorageAreaUsage", "paths": ["src"], "glob": "**/*.{ts,tsx}",
    })
    assert result["ok"]
    assert [row["file"] for row in result["matches"]] == ["src/types.ts", "src/view.tsx"]
    assert list_files(tmp_path, glob="src/**/*.{ts,tsx}")["files"] == ["src/types.ts", "src/view.tsx"]
    batch = repo_batch_search(tmp_path, patterns=[{"query": "StorageAreaUsage", "glob": "**/*.{ts,tsx}"}])
    assert [row["file"] for row in batch["results"][0]["matches"]] == ["src/types.ts", "src/view.tsx"]


def test_invalid_brace_glob_returns_search_error(tmp_path: Path) -> None:
    from mana_agent.tools.repository import execute_repository_discovery

    for pattern in ("**/*.{ts,tsx", "**/*.{ts}", "**/*.{ts,}", "**/*.[ts"):
        result = execute_repository_discovery(tmp_path, "repo_search", {
            "query": "value", "paths": ["src"], "glob": pattern,
        })
        assert not result["ok"]
        assert result["error_code"] == "INVALID_DISCOVERY_SELECTION"


def test_list_files_flat_and_recursive_globs_include_existing_files(tmp_path: Path) -> None:
    skill_dir = tmp_path / "src" / "mana_agent" / "default_skills"
    nested = skill_dir / "nested"
    nested.mkdir(parents=True)
    (skill_dir / "cli.md").write_text("# CLI\n", encoding="utf-8")
    (skill_dir / "reactjs.md").write_text("# ReactJS\n", encoding="utf-8")
    (nested / "extra.md").write_text("# Extra\n", encoding="utf-8")

    flat = list_files(tmp_path, glob="src/mana_agent/default_skills/*.md")["files"]
    recursive_star = list_files(tmp_path, glob="src/mana_agent/default_skills/**/*")["files"]
    recursive_dir = list_files(tmp_path, glob="src/mana_agent/default_skills/**")["files"]

    assert flat == [
        "src/mana_agent/default_skills/cli.md",
        "src/mana_agent/default_skills/reactjs.md",
    ]
    assert "src/mana_agent/default_skills/cli.md" in recursive_star
    assert "src/mana_agent/default_skills/nested/extra.md" in recursive_star
    assert recursive_star == recursive_dir


def test_repo_batch_read_returns_multiple_files_and_errors(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("beta\n", encoding="utf-8")

    result = repo_batch_read(tmp_path, files=["src/a.py", "src/b.py", "../escape.py", "missing.py"])

    assert result["ok"] is False
    ok_files = [item for item in result["files"] if item["ok"]]
    assert [item["path"] for item in ok_files] == ["src/a.py", "src/b.py"]
    assert ok_files[0]["content"] == "alpha\n"
    assert {item["error"] for item in result["errors"]} == {"path_outside_repo", "file_not_found"}


def test_repo_batch_search_groups_results_per_query(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("class Package:\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("SkillIndexItem = object\n", encoding="utf-8")

    result = repo_batch_search(
        tmp_path,
        patterns=[
            {"query": "class Package", "glob": "**/*.py", "regex": False, "limit": 10},
            {"query": "SkillIndexItem", "glob": "**/*.py", "regex": False, "limit": 10},
        ],
    )

    assert result["ok"] is True
    assert len(result["results"]) == 2
    assert result["results"][0]["matches"][0]["file"] == "src/a.py"
    assert result["results"][1]["matches"][0]["file"] == "src/b.py"


def test_run_script_once_returns_exit_code_output_and_duration(tmp_path: Path) -> None:
    result = run_script_once(tmp_path, script="printf 'hello'")

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["stdout"] == "hello"
    assert result["duration_ms"] >= 0


def test_apply_patch_batch_validates_and_applies_multiple_patches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("old a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("old b\n", encoding="utf-8")

    result = apply_patch_batch(
        tmp_path,
        patches=[
            {"path": "a.txt", "patch": "*** Begin Patch\n*** Update File: a.txt\n@@\n-old a\n+new a\n*** End Patch"},
            {"path": "b.txt", "patch": "*** Begin Patch\n*** Update File: b.txt\n@@\n-old b\n+new b\n*** End Patch"},
        ],
    )

    assert result["ok"] is True
    assert result["changed_files"] == ["a.txt", "b.txt"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "new a\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "new b\n"


def test_repository_search_ignores_external_symlinks_and_local_credential_files(tmp_path):
    from mana_agent.tools.repository import repo_search
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "protected.txt"
    external.write_text("fixture marker\n")
    (root / ".env").write_text("fixture marker\n")
    (root / "visible.txt").write_text("fixture marker\n")
    try:
        (root / "outside.txt").symlink_to(external)
    except OSError:
        pytest.skip("creating symlinks requires local OS support")
    result = repo_search(root, query="fixture marker")
    assert [item["file"] for item in result["matches"]] == ["visible.txt"]


@pytest.mark.parametrize("backend", ["rg", "python"])
def test_search_backends_preserve_filters_case_context_and_scope(tmp_path, monkeypatch, backend):
    from mana_agent.tools.repository import execute_repository_discovery
    from mana_agent.tools import repository_search

    if backend == "python":
        monkeypatch.setattr(repository_search, "_executable", lambda _: None)
    elif repository_search._executable("rg") is None:
        pytest.skip("native ripgrep is unavailable")
    for name in ("types.ts", "view.tsx", "script.js", "ignored/private.ts", ".env/credentials.ts"):
        file = tmp_path / "src" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("first\nexport interface StorageAreaUsage {}\nlast\n", encoding="utf-8")
    options = {"paths": ["src"], "globs": ["**/*.ts", "**/*.tsx"], "exclude_dirs": ["src/ignored"]}
    listing = execute_repository_discovery(tmp_path, "list_files", options)
    assert listing["files"] == ["src/types.ts", "src/view.tsx"]
    result = execute_repository_discovery(tmp_path, "repo_search", {
        **options, "query": "INTERFACE StorageAreaUsage", "case_sensitive": False, "context": 1,
    })
    assert result["backend"] == backend
    assert result["search_status"] == "matches"
    assert [row["file"] for row in result["matches"]] == listing["files"]
    assert [(row["line"], row["text"]) for row in result["context"]] == [(1, "first"), (3, "last")] * 2
    if backend == "python":
        assert result["unavailable_executables"] == ["rg"]


def test_portable_regex_uses_existing_cancellable_command_runner(tmp_path, monkeypatch):
    from mana_agent.tools import repository_search
    from mana_agent.tools.repository import repo_search

    monkeypatch.setattr(repository_search, "_executable", lambda _: None)
    (tmp_path / "types.ts").write_text("export interface StorageAreaUsage {}\n", encoding="utf-8")
    result = repo_search(tmp_path, query=r"(?<=interface )Storage\w+", regex=True)
    assert result["backend"] == "python"
    assert result["returncode"] == 0
    assert result["matches"][0]["file"] == "types.ts"
    miss = repo_search(tmp_path, query="(", regex=True, scope=("missing-dir",))
    assert miss["search_status"] == "invalid_pattern"


def test_git_grep_fallback_searches_scoped_tracked_and_untracked_files(tmp_path, monkeypatch):
    import subprocess
    from mana_agent.tools import repository_search
    from mana_agent.tools.repository import repo_search

    git = repository_search._executable("git")
    if not git:
        pytest.skip("git is unavailable")
    subprocess.run([git, "init", "-q"], cwd=tmp_path, check=True)
    names = ("new.tsx", "tracked type.ts")
    for name in names:
        (tmp_path / name).write_text("export interface StorageAreaUsage {}\n", encoding="utf-8")
    subprocess.run([git, "add", "--", names[1]], cwd=tmp_path, check=True)
    monkeypatch.setattr(repository_search, "_executable", lambda name: git if name == "git" else None)
    outcome = repo_search(tmp_path, query="StorageAreaUsage", scope=names, glob="**/*.{ts,tsx}")
    assert outcome["search_status"] == "matches"
    assert outcome["backend"] == "git"
    assert outcome["unavailable_executables"] == ["rg"]
    assert [(row["file"], row["line"]) for row in outcome["matches"]] == [(name, 1) for name in names]


@pytest.mark.parametrize(("code", "stderr", "status"), [
    (1, "", "no_matches"),
    (2, "regex parse error", "invalid_pattern"),
    (2, "src: Permission denied (os error 13)", "permission_denied"),
    (2, "unexpected I/O error", "execution_failed"),
])
def test_native_exit_status_is_a_search_observation(tmp_path, monkeypatch, code, stderr, status):
    import subprocess
    from mana_agent.tools import repository_search
    from mana_agent.tools.repository import repo_search

    monkeypatch.setattr(repository_search, "_executable", lambda _: "rg")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, code, "", stderr)

    monkeypatch.setattr(repository_search, "run_streaming_process", run)
    result = repo_search(tmp_path, query="-e ; literal $HOME")
    assert result["search_status"] == status
    assert result["returncode"] == code
    assert result["ok"] is (code == 1)
    argv, options = calls[0]
    assert len(calls) == 1
    assert argv[argv.index("-e") + 1] == "-e ; literal $HOME"
    assert argv[-2:] == ["--", "."]
    assert options["cwd"] == tmp_path.resolve()


@pytest.mark.parametrize("failure", ["missing", "timeout", "permission"])
def test_native_launch_failures_distinguish_fallback_from_failed_execution(tmp_path, monkeypatch, failure):
    import subprocess
    from mana_agent.tools import repository_search
    from mana_agent.tools.repository import repo_search

    monkeypatch.setattr(repository_search, "_executable", lambda _: "rg")
    (tmp_path / "types.ts").write_text("StorageAreaUsage\n", encoding="utf-8")

    def run(argv, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("rg disappeared")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        raise PermissionError("Cannot launch rg")

    monkeypatch.setattr(repository_search, "run_streaming_process", run)
    result = repo_search(tmp_path, query="StorageAreaUsage")
    expected = {"missing": "matches", "timeout": "timeout", "permission": "permission_denied"}
    assert result["search_status"] == expected[failure]
    if failure == "missing":
        assert result["backend"] == "python"
        assert result["unavailable_executables"] == ["rg"]


def test_equivalent_searches_skip_execution_until_source_changes(tmp_path, monkeypatch):
    from mana_agent.coding.programmatic import ToolExecutionDispatcher
    from mana_agent.tools import repository

    (tmp_path / "src").mkdir()
    actual = repository.repo_search
    calls = []

    def search(*args, **kwargs):
        calls.append(kwargs["query"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(repository, "repo_search", search)
    dispatcher = ToolExecutionDispatcher.for_coding_task(repo_root=tmp_path, task_id="repeat-search")
    args = {"query": "StorageAreaUsage", "paths": ["src"], "glob": "**/*.{ts,tsx}"}
    assert dispatcher.dispatch("repo_search", args)["search_status"] == "no_matches"
    alternative = {"query": args["query"], "paths": ["./src"], "globs": ["**/*.tsx", "**/*.ts"],
                   "regex": False, "case_sensitive": True, "context": 0, "limit": 100}
    assert dispatcher.dispatch("repo_search", alternative)["error_code"] == "REPEATED_SEARCH"
    assert calls == ["StorageAreaUsage"]
    assert dispatcher.dispatch("repo_search", {**args, "query": "StorageArea"})["ok"]
    assert dispatcher.dispatch("write_file", {"path": "src/types.ts", "content": "interface StorageAreaUsage {}"})["ok"]
    assert dispatcher.dispatch("repo_search", alternative)["matches"][0]["file"] == "src/types.ts"


def test_search_rejects_explicit_escape_and_listing_filters_protected_paths(tmp_path):
    from mana_agent.tools.repository import execute_repository_discovery

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (tmp_path / "escape.ts").write_text("StorageAreaUsage\n", encoding="utf-8")
    assert not execute_repository_discovery(checkout, "repo_search", {
        "query": "StorageAreaUsage", "paths": ["../escape.ts"],
    })["ok"]
    (checkout / ".env.ts").write_text("StorageAreaUsage\n", encoding="utf-8")
    assert execute_repository_discovery(checkout, "list_files", {"paths": ["."]})["files"] == []
