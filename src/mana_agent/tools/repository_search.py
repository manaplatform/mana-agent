"""Native and portable backends for the structured repository search tools."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from mana_agent.utils.process_stream import run_streaming_process
from mana_agent.utils.redaction import redact_secrets

SEARCH_TIMEOUT_SECONDS = 20
_CAPTURE_BYTES = 1_000_000
_PROTECTED_NAMES = (".env*", ".git", ".mana", ".ssh", "secrets.toml", "credentials.json", "id_rsa", "id_ed25519")


class InvalidSearchPattern(ValueError):
    """The selected file filter or text expression is invalid."""


def file_filters(glob: str, globs: list[str] | None = None) -> tuple[str, ...]:
    from mana_agent.tools.repository import _expand_file_glob

    if globs is not None and (not isinstance(globs, list) or not all(isinstance(value, str) for value in globs)):
        raise InvalidSearchPattern("globs must be an array of file glob strings")
    if globs and len(globs) > 64:
        raise InvalidSearchPattern("Search supports at most 64 file filters")
    values = [glob] if not globs or glob not in {"**", "**/*"} else []
    values.extend(globs or [])
    expanded = set()
    for value in values:
        if not isinstance(value, str):
            raise InvalidSearchPattern("Select a file glob string")
        pattern = value.replace("\\", "/").removeprefix("./")
        if (not pattern or pattern.startswith(("/", "!")) or pattern.endswith("/")
                or any(part in {"", ".", ".."} for part in pattern.split("/"))
                or ":" in pattern or any(ord(char) < 32 for char in pattern)):
            raise InvalidSearchPattern("Select a repository-relative file glob. Use exclude_dirs for directory exclusions.")
        try:
            for item in _expand_file_glob(pattern):
                # Reject invalid or unsupported bracket expressions consistently
                # rather than letting the portable matcher treat them literally.
                classes = re.findall(r"\[([^\[\]]+)\]", item)
                if len(classes) != item.count("["):
                    raise InvalidSearchPattern("Invalid or unsupported glob character class")
                for chars in classes:
                    re.compile("[" + ("^" + chars[1:] if chars.startswith("!") else chars) + "]")
                expanded.add("**/*" if item == "**" else item)
        except (ValueError, re.error) as exc:
            raise InvalidSearchPattern(str(exc)) from exc
    if len(expanded) > 64:
        raise InvalidSearchPattern("Search supports at most 64 expanded file filters")
    return tuple(sorted(expanded))


def search_selection_key(tool: str, args: dict[str, Any]) -> tuple | None:
    """Identify equivalent structured searches after validating their selected scope."""
    from mana_agent.tools.repository import _selected_discovery_scope, _text_search_options

    try:
        if tool in {"repo_batch_search", "batch_search_files"}:
            default_limit = 50 if tool == "batch_search_files" else 20
            keys = [search_selection_key("repo_search", {"limit": default_limit, **entry})
                    for entry in (args.get("patterns") or [])[:50]]
            return ("text_batch", frozenset(keys)) if keys and None not in keys else None
        if tool not in {"repo_search", "list_files"}:
            return None
        listing = tool == "list_files"
        scope = _selected_discovery_scope(args, allow_root_listing=listing)
        filters = file_filters(str(args.get("glob") or "**/*"), args.get("globs"))
        directories = directory_filters(args.get("exclude_dirs"))
        limit = max(1, min(int(args.get("limit") or (200 if listing else 100)), 5000 if listing else 1000))
        options = _text_search_options(args)
        return (tool, tuple(sorted(scope)) if scope is not None else None, filters, directories, limit,
                None if listing else (str(args.get("query") or ""), options["regex"],
                                      options["case_sensitive"], options["context"]))
    except (TypeError, ValueError):
        return None


def directory_filters(values: list[str] | None) -> tuple[str, ...]:
    from mana_agent.tools.repository import normalize_discovery_path

    if values is not None and (not isinstance(values, list) or not all(isinstance(value, str) for value in values)):
        raise InvalidSearchPattern("exclude_dirs must be an array of literal directory names")
    return tuple(sorted({normalize_discovery_path(value) for value in values or []}))


def _excluded(relative: str, exclusions: tuple[str, ...], *, directory: bool = False) -> bool:
    from mana_agent.tools.repository import _is_skipped_dir

    parts = relative.split("/")
    folded = [part.casefold() for part in parts]
    if any(part.startswith(".env") or part in _PROTECTED_NAMES for part in folded):
        return True
    directories = parts if directory else parts[:-1]
    if any(_is_skipped_dir(part.casefold()) for part in directories):
        return True
    joined = "/".join(directories)
    return any(value in directories if "/" not in value else
               joined == value or joined.startswith(value + "/") for value in exclusions)


def _relative(root: Path, name: str, exclusions: tuple[str, ...]) -> str | None:
    path = Path(name)
    if path.anchor or ".." in path.parts:
        return None
    relative = path.as_posix()
    candidate = root / path
    if _excluded(relative, exclusions) or candidate.is_symlink() or candidate.resolve() != candidate:
        return None
    return relative


def _targets(root: Path, scope: tuple[str, ...] | None, exclusions: tuple[str, ...]) -> list[str]:
    from mana_agent.tools.repository import normalize_discovery_path

    if scope is None or scope == (".",):
        return ["."]
    targets = []
    for raw in sorted(set(scope)):
        name = normalize_discovery_path(raw, allow_root_listing=True)
        path = root / name
        if name == ".":
            targets.append(name)
            continue
        if path.is_symlink() or path.resolve() != path:
            raise PermissionError(f"Search target crosses a symbolic link or repository boundary: {name}")
        try:
            path.stat()
        except FileNotFoundError:
            continue
        if (path.is_dir() or path.is_file()) and not _excluded(name, exclusions, directory=path.is_dir()):
            if not any(path.is_relative_to(root / target) for target in targets if target != "."):
                targets.append(name)
    return targets


def _search_environment() -> dict[str, str]:
    # Option injection and external Git indexes would invalidate workspace confinement.
    return {key: value for key, value in os.environ.items()
            if not key.upper().startswith("GIT_") and key.upper() != "RIPGREP_CONFIG_PATH"}


def _rg_arguments(filters: tuple[str, ...], exclusions: tuple[str, ...]) -> list[str]:
    from mana_agent.tools.repository import _SKIP_DIRS

    argv = ["--no-config", "--no-follow", "--no-ignore", "--hidden", "--sort", "path"]
    for pattern in filters:
        # A leading slash anchors slash-containing patterns at the command cwd.
        argv.extend(["--glob", "/" + pattern if "/" in pattern else pattern])
    directories = "{" + ",".join((*sorted(_SKIP_DIRS), ".venv*", "venv*", "__*__")) + "}"
    protected = "{" + ",".join(_PROTECTED_NAMES) + "}"
    argv.extend(["--iglob", f"!**/{directories}/", "--iglob", f"!**/{protected}",
                 "--iglob", f"!**/{protected}/**"])
    for directory in exclusions:
        literal = "".join(f"\\{char}" if char in "{}!]" else char for char in directory)
        prefix = "/" if "/" in directory else "**/"
        argv.extend(["--glob", f"!{prefix}{literal}/"])
    return argv


def _command(argv: list[str], root: Path, deadline: float) -> subprocess.CompletedProcess | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Repository search exceeded its time limit")
    try:
        return run_streaming_process(argv, cwd=root, env=_search_environment(), timeout=remaining,
                                     capture_limit_bytes=_CAPTURE_BYTES, stream_stdout=False,
                                     stop_on_capture_limit=True, success_exit_codes=(0, 1))
    except FileNotFoundError:
        # A discovered executable can disappear between discovery and launch.
        return None


def _executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable and (os.name != "nt" or Path(executable).suffix.lower() in {".exe", ".com"}):
        return executable
    return None


def _row(relative: str, number: int, line: str) -> dict[str, Any]:
    return {"file": relative, "line": number, "text": redact_secrets(line.rstrip("\r\n")[:500])}


def _rg_text(value: dict[str, str]) -> str:
    if "text" in value:
        return value["text"]
    return os.fsdecode(base64.b64decode(value["bytes"], validate=True))


def _native_records(backend: str, stdout: str, *, listing: bool, truncated: bool):
    if listing:
        names = stdout.split("\0")[:-1]
        if stdout and not stdout.endswith("\0") and not truncated:
            raise ValueError("Invalid native file-list output")
        yield from ((name, None, "", False) for name in names)
    elif backend in {"rg", "python"}:
        lines = stdout.split("\n")
        for line in lines[:-1]:
            record = json.loads(line)
            if record["type"] in {"match", "context"}:
                data = record["data"]
                yield (_rg_text(data["path"]), data["line_number"], _rg_text(data["lines"]),
                       record["type"] == "context")
        if lines[-1] and not truncated:
            raise ValueError("Incomplete ripgrep output")
    else:
        # -z separates the filename and line number with NUL. Parse the name
        # before the newline-delimited content so newlines in names survive.
        position = 0
        while position < len(stdout):
            first = stdout.find("\0", position)
            second = stdout.find("\0", first + 1) if first >= 0 else -1
            end = stdout.find("\n", second + 1) if second >= 0 else -1
            if end < 0:
                if truncated:
                    break
                raise ValueError("Incomplete git grep output")
            yield stdout[position:first], int(stdout[first + 1:second]), stdout[second + 1:end], False
            position = end + 1


def _portable(root, targets, shallow, filters, exclusions, matcher, context, check_active):
    from mana_agent.tools.repository import _matches_file_glob

    def raise_walk_error(exc):
        raise exc

    def walk(target):
        path = root / target
        if path.is_file():
            yield path
            return
        for directory, dirs, files in os.walk(path, onerror=raise_walk_error):
            check_active()
            base = Path(directory)
            dirs[:] = [] if shallow and target == "." else sorted(
                name for name in dirs
                if not _excluded((base / name).relative_to(root).as_posix(), exclusions, directory=True)
                and not (base / name).is_symlink()
                and (base / name).resolve() == base / name
            )
            yield from (base / name for name in sorted(files))

    seen = set()
    for target in targets:
        for path in walk(target):
            check_active()
            relative = _relative(root, path.relative_to(root).as_posix(), exclusions)
            if relative is None or relative in seen or not path.is_file():
                continue
            if not any(_matches_file_glob(relative, pattern) for pattern in filters):
                continue
            seen.add(relative)
            if matcher is None:
                yield relative, None, "", False
                continue
            with path.open("rb") as source:
                if b"\0" in source.read(4096):
                    continue
                source.seek(0)
                previous = deque(maxlen=context)
                reported = 0
                context_until = 0
                for number, raw_line in enumerate(source, 1):
                    check_active()
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if matcher(line):
                        for prior_number, prior_line in previous:
                            if prior_number > reported:
                                yield relative, prior_number, prior_line, True
                        yield relative, number, line, False
                        reported = number
                        context_until = number + context
                    elif number <= context_until:
                        yield relative, number, line, True
                        reported = number
                    previous.append((number, line[:500]))


def search_repository(
    root: Path, *, filters: tuple[str, ...], exclusions: tuple[str, ...], limit: int,
    scope: tuple[str, ...] | None, query: str | None = None, regex: bool = False,
    case_sensitive: bool = True, context: int = 0, deadline: float | None = None, count_all: bool = False,
) -> dict[str, Any]:
    from mana_agent.multi_agent.agents.tool_runner_agent import current_operation
    from mana_agent.tools.repository import _glob_root, _matches_file_glob

    started = time.monotonic()
    deadline = min(deadline or started + SEARCH_TIMEOUT_SECONDS, started + SEARCH_TIMEOUT_SECONDS)
    operation = current_operation()
    listing = query is None
    key = "files" if listing else "matches"
    metadata: dict[str, Any] = {}

    def finish(status, **fields):
        rows = fields.get(key, [])
        summary = fields.get("error") or f"{len(rows)} {'files' if listing else 'matching lines'}" + (
            " (truncated)" if fields.get("truncated") else "")
        if status == "no_matches" and not listing:
            fields["guidance"] = "Try a related identifier, relax a file filter within the selected scope, or read a likely file."
        return {key: [], **({"total_matches": 0, "count_complete": False} if count_all else {}),
                "limit": limit, "truncated": False, "search_status": status,
                "ok": status in {"matches", "no_matches"}, "summary": summary,
                "duration_ms": round((time.monotonic() - started) * 1000), **metadata, **fields}

    def check_active():
        if operation is not None:
            operation.check_active()
        if time.monotonic() >= deadline:
            raise TimeoutError("Repository search exceeded its time limit")

    try:
        check_active()
        matcher = None
        if query is not None:
            if not query or any(char in query for char in "\0\r\n"):
                raise InvalidSearchPattern("Search requires a nonempty single-line query")
            try:
                pattern = re.compile(query if regex else re.escape(query), 0 if case_sensitive else re.IGNORECASE)
            except re.error as exc:
                raise InvalidSearchPattern(str(exc)) from exc
            matcher = pattern.search

        prefixes = [_glob_root(pattern) for pattern in filters]
        selected = scope
        if prefixes and all(prefixes):
            if scope is None:
                selected = tuple(prefixes)
            elif "." not in scope:
                selected = tuple(prefix if prefix.startswith(path + "/") else path
                                 for path in scope for prefix in prefixes
                                 if path == prefix or prefix.startswith(path + "/") or path.startswith(prefix + "/"))
        targets = _targets(root, selected, exclusions)
        shallow = scope is not None and "." in scope
        targets = [name for name in targets if not (root / name).is_file()
                   or any(_matches_file_glob(name, pattern) for pattern in filters)]
        if not targets:
            return finish("no_matches", **({"count_complete": True} if count_all else {}))

        # Python-specific regex constructs keep their existing semantics.
        native_regex = not regex or not re.search(r"\(\?|\\(?:[1-9uUNZAbBsSwWdD]|[0-7])", query or "")
        native_supported = native_regex and (not shallow or targets == ["."])
        rg = _executable("rg") if native_supported else None
        argv = []
        if rg:
            argv = [rg, "--files", "--null"] if listing else [rg, "--json", "-e", query]
            argv.extend(_rg_arguments(filters, exclusions))
            if shallow:
                argv.extend(["--max-depth", "1"])
            if not listing:
                argv.extend(["--no-binary", "--max-count", str(limit + 1), "--case-sensitive" if case_sensitive
                             else "--ignore-case", "--context", str(context)])
                if not regex:
                    argv.append("--fixed-strings")
                else:
                    argv.append("--crlf")
            argv.append("--")
            argv.extend("." if target == "." else "./" + target for target in targets)

        completed = None
        backend = "rg" if argv else "python"
        unavailable = [] if rg or not native_supported else ["rg"]
        if argv:
            metadata["backend"] = "rg"
            completed = _command(argv, root, deadline)
            if completed is None:
                unavailable.append("rg")
        if (completed is None and not listing and not regex and case_sensitive and not context
                and len(targets) <= 64 and (root / ".git").exists()
                and all((root / name).is_file() for name in targets)):
            git = _executable("git")
            if git:
                backend = "git"
                metadata["backend"] = backend
                argv = [git, "--work-tree=.", "--literal-pathspecs", "grep", "--untracked", "--no-exclude-standard",
                        "--no-recurse-submodules", "--no-textconv", "--color=never", "-InzF", "-e", query, "--", *targets]
                completed = _command(argv, root, deadline)
            if completed is None:
                unavailable.append("git")

        if unavailable:
            metadata["unavailable_executables"] = unavailable
        if completed is None and regex:
            # A Python expression can spend its entire budget evaluating one line.
            # Run it through the existing cancellable command runtime.
            backend = "python"
            metadata["backend"] = backend
            bootstrap = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from mana_agent.tools.repository_search import _portable_regex_main; _portable_regex_main()"
            )
            specification = json.dumps([targets, shallow, filters, exclusions, query, case_sensitive, context])
            completed = _command([sys.executable, "-I", "-c", bootstrap,
                                  str(Path(__file__).resolve().parents[2]), specification], root, deadline)
            if completed is None:
                return finish("missing_executable", error="The portable search interpreter could not be launched",
                              error_code="SEARCH_EXECUTABLE_MISSING")
        if completed is None:
            metadata["backend"] = "python"
            records = _portable(root, targets, shallow, filters, exclusions, matcher, context, check_active)
        else:
            metadata.update(backend=backend, returncode=completed.returncode)
            stderr = (completed.stderr or "").replace("\r\n", "\n").strip()
            if backend == "rg" and completed.returncode == 1 and stderr == (
                "rg: No files were searched, which means ripgrep probably applied a filter you didn't expect.\n"
                "Running with --debug will show why files are being skipped."
            ):
                stderr = ""
            if ((completed.returncode not in (0, 1) and not getattr(completed, "stopped_at_capture_limit", False))
                    or stderr):
                reason = stderr or f"{backend} exited with code {completed.returncode}"
                if re.search(r"permission denied|access is denied|operation not permitted", reason, re.I):
                    status = "permission_denied"
                elif re.search(r"(?:regex|glob).*(?:error|invalid)|error parsing", reason, re.I):
                    status = "invalid_pattern"
                else:
                    status = "execution_failed"
                return finish(status, error=redact_secrets(reason[:1000]), error_code=status.upper())
            records = _native_records(backend, completed.stdout, listing=listing,
                                      truncated=getattr(completed, "capture_truncated", False))

        rows: list[Any] = []
        context_rows = []
        remaining_chars = _CAPTURE_BYTES
        total_matches = 0
        count_complete = not (completed and getattr(completed, "capture_truncated", False))
        truncated = bool(completed and getattr(completed, "capture_truncated", False))
        last_name = None
        relative = None
        for name, number, line, is_context in records:
            check_active()
            if name != last_name:
                relative = _relative(root, name, exclusions)
                last_name = name
            if relative is None:
                continue
            remaining_chars -= len(relative) + min(500, len(line)) + 64
            if remaining_chars < 0:
                truncated = True
                count_complete = False
                break
            if is_context:
                if len(context_rows) < 2 * context * limit:
                    context_rows.append(_row(relative, number, line))
                continue
            total_matches += 1
            if len(rows) >= limit:
                truncated = True
                if count_all:
                    continue
                break
            rows.append(relative if listing else _row(relative, number, line))
        if listing:
            rows.sort()
        else:
            rows.sort(key=lambda row: (row["file"], row["line"]))
        extra = {}
        if context:
            nearby = {(row["file"], number) for row in rows
                      for number in range(row["line"] - context, row["line"] + context + 1)}
            extra["context"] = [row for row in context_rows if (row["file"], row["line"]) in nearby]
        if count_all:
            extra.update(total_matches=total_matches, count_complete=count_complete)
        # Output capture may end before the first complete record.
        status = "matches" if rows else "output_limited" if truncated else "no_matches"
        if status == "output_limited":
            extra.update(error="Search output limit reached before a complete result. Select a narrower search.",
                         error_code="SEARCH_OUTPUT_LIMIT")
        return finish(status, **{key: rows}, truncated=truncated, **extra)
    except InvalidSearchPattern as exc:
        return finish("invalid_pattern", error=str(exc), error_code="INVALID_DISCOVERY_SELECTION")
    except PermissionError as exc:
        return finish("permission_denied", error=str(exc), error_code="SEARCH_PERMISSION_DENIED")
    except (subprocess.TimeoutExpired, TimeoutError):
        return finish("timeout", error="Repository search exceeded its time limit",
                      error_code="SEARCH_TIMEOUT", timed_out=True)
    except InterruptedError:
        raise
    except (OSError, ValueError, KeyError, binascii.Error) as exc:
        return finish("execution_failed", error=str(exc), error_code="SEARCH_EXECUTION_FAILED")


def _portable_regex_main() -> None:
    """Private entry point launched via the shared command runtime."""
    targets, shallow, filters, exclusions, query, case_sensitive, context = json.loads(sys.argv[2])
    matcher = re.compile(query, 0 if case_sensitive else re.IGNORECASE).search
    for file, number, line, surrounding in _portable(
        Path.cwd().resolve(), targets, shallow, filters, exclusions, matcher, context, lambda: None,
    ):
        print(json.dumps({"type": "context" if surrounding else "match",
                          "data": {"path": {"text": file}, "lines": {"text": line[:500]}, "line_number": number}}))
