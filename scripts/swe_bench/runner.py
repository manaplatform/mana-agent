#!/usr/bin/env python3
"""Generate SWE-bench Verified predictions.jsonl using mana-agent.

Loads ``princeton-nlp/SWE-bench_Verified``, checks out each instance at
``base_commit`` into an isolated git worktree, runs mana-agent once on the
issue text, captures the final git diff as ``model_patch``, and writes one
JSONL prediction per instance in the official harness format:

    {"instance_id": "...", "model_name_or_path": "mana-agent", "model_patch": "..."}

Scope: prediction generation only (smoke grading is documented separately).
Does not run the full 500-instance suite, SWE-bench Pro, Terminal-Bench,
pass@k, or leaderboard submission.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

LOG = logging.getLogger("mana_agent.swe_bench")

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL_NAME_OR_PATH = "mana-agent"
# Cheap/fast default for smoke and cost-controlled initial runs.
DEFAULT_AGENT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_OUTPUT = "predictions.jsonl"
DEFAULT_WORK_DIR = ".swe-bench"
GITHUB_URL = "https://github.com/{repo}.git"

SAFE_INSTANCE_ID = re.compile(r"[^A-Za-z0-9._-]+")


class SweBenchRunnerError(RuntimeError):
    """Raised for hard runner failures that stop the whole run."""


@dataclass
class InstanceResult:
    instance_id: str
    model_patch: str = ""
    status: str = "ok"
    error: str = ""
    duration_seconds: float = 0.0
    worktree: str = ""
    empty_patch: bool = True


@dataclass
class RunnerConfig:
    dataset_name: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    limit: int | None = None
    instance_ids: list[str] = field(default_factory=list)
    output: Path = Path(DEFAULT_OUTPUT)
    work_dir: Path = Path(DEFAULT_WORK_DIR)
    agent_model: str = DEFAULT_AGENT_MODEL
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    mana_bin: str | None = None
    retain_worktrees: bool = False
    skip_agent: bool = False
    max_prompt_chars: int = 48_000
    continue_on_error: bool = True


def _sanitize_id(instance_id: str) -> str:
    cleaned = SAFE_INSTANCE_ID.sub("_", instance_id.strip())
    return cleaned or "instance"


def _run(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    check: bool = False,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    LOG.debug("run: %s (cwd=%s)", " ".join(shlex.quote(a) for a in args), cwd)
    return subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        check=check,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
    )


def _git(args: Sequence[str], *, cwd: Path, timeout: int | None = 120) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd, timeout=timeout)


def resolve_mana_bin(explicit: str | None = None) -> list[str]:
    """Return argv prefix that invokes mana-agent non-interactively."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return [str(path.resolve())]
        return shlex.split(explicit)

    which = shutil_which("mana-agent")
    if which:
        return [which]

    # Fall back to the module entrypoint from the current interpreter.
    return [sys.executable, "-m", "mana_agent"]


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def load_instances(
    *,
    dataset_name: str,
    split: str,
    limit: int | None,
    instance_ids: Sequence[str],
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment specific
        raise SweBenchRunnerError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from exc

    LOG.info("Loading dataset %s split=%s", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    rows: list[dict[str, Any]] = [dict(row) for row in ds]

    if instance_ids:
        wanted = {i.strip() for i in instance_ids if i.strip()}
        rows = [r for r in rows if str(r.get("instance_id", "")) in wanted]
        missing = wanted - {str(r.get("instance_id", "")) for r in rows}
        if missing:
            LOG.warning("Requested instance_ids not found in dataset: %s", sorted(missing))

    if limit is not None:
        if limit < 0:
            raise SweBenchRunnerError("--limit must be >= 0")
        rows = rows[:limit]

    LOG.info("Selected %d instance(s)", len(rows))
    return rows


def ensure_repo_clone(repo: str, repos_dir: Path) -> Path:
    """Clone or update a GitHub mirror under repos_dir."""
    repos_dir.mkdir(parents=True, exist_ok=True)
    dest = repos_dir / repo.replace("/", "__")
    url = GITHUB_URL.format(repo=repo)

    if dest.exists() and (dest / ".git").exists():
        LOG.info("Using cached clone %s", dest)
        # Best-effort fetch so rare base commits are available.
        fetch = _git(["fetch", "--all", "--tags", "--prune"], cwd=dest, timeout=600)
        if fetch.returncode != 0:
            LOG.warning("git fetch failed for %s: %s", dest, (fetch.stderr or "").strip())
        return dest

    if dest.exists():
        raise SweBenchRunnerError(f"Clone path exists but is not a git repo: {dest}")

    LOG.info("Cloning %s -> %s", url, dest)
    clone = _run(
        ["git", "clone", "--filter=blob:none", url, str(dest)],
        timeout=1800,
    )
    if clone.returncode != 0:
        raise SweBenchRunnerError(
            f"Failed to clone {repo}: {(clone.stderr or clone.stdout or '').strip()}"
        )
    return dest


def ensure_base_commit(repo_path: Path, base_commit: str) -> str:
    """Ensure base_commit is present locally; return the resolved SHA."""
    probe = _git(["rev-parse", "--verify", f"{base_commit}^{{commit}}"], cwd=repo_path)
    if probe.returncode == 0:
        return probe.stdout.strip()

    LOG.info("Fetching base_commit %s in %s", base_commit, repo_path)
    # Prefer fetching the exact commit; fall back to a broader fetch.
    for args in (
        ["fetch", "--depth", "1", "origin", base_commit],
        ["fetch", "origin", base_commit],
        ["fetch", "--unshallow"],
        ["fetch", "--all"],
    ):
        result = _git(args, cwd=repo_path, timeout=1800)
        if result.returncode == 0:
            probe = _git(["rev-parse", "--verify", f"{base_commit}^{{commit}}"], cwd=repo_path)
            if probe.returncode == 0:
                return probe.stdout.strip()

    raise SweBenchRunnerError(
        f"Unable to resolve base_commit {base_commit!r} in {repo_path}"
    )


def worktree_is_dirty(path: Path) -> bool:
    status = _git(["status", "--porcelain"], cwd=path)
    if status.returncode != 0:
        return True
    return bool(status.stdout.strip())


def create_isolated_worktree(
    repo_path: Path,
    *,
    base_commit: str,
    worktrees_dir: Path,
    instance_id: str,
) -> Path:
    """Create a clean detached worktree at base_commit."""
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    slug = _sanitize_id(instance_id)
    target = worktrees_dir / slug

    if target.exists():
        # Leftover from a prior failed run — remove before recreating.
        remove_worktree(repo_path, target, force=True)

    resolved = ensure_base_commit(repo_path, base_commit)
    add = _git(
        ["worktree", "add", "--detach", str(target), resolved],
        cwd=repo_path,
        timeout=300,
    )
    if add.returncode != 0:
        raise SweBenchRunnerError(
            f"Checkout failed for {instance_id} @ {base_commit}: "
            f"{(add.stderr or add.stdout or '').strip()}"
        )

    if not target.is_dir():
        raise SweBenchRunnerError(f"Worktree path missing after create: {target}")

    if worktree_is_dirty(target):
        remove_worktree(repo_path, target, force=True)
        raise SweBenchRunnerError(
            f"Worktree for {instance_id} was dirty immediately after checkout"
        )

    # Confirm HEAD matches the requested commit.
    head = _git(["rev-parse", "HEAD"], cwd=target)
    if head.returncode != 0 or head.stdout.strip() != resolved:
        remove_worktree(repo_path, target, force=True)
        raise SweBenchRunnerError(
            f"Worktree HEAD mismatch for {instance_id}: "
            f"expected {resolved}, got {(head.stdout or '').strip()}"
        )

    return target


def remove_worktree(repo_path: Path, worktree: Path, *, force: bool = True) -> None:
    if not worktree.exists():
        return
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(worktree))
    result = _git(args, cwd=repo_path, timeout=300)
    if result.returncode != 0 and worktree.exists():
        # Last resort: prune + delete directory (still safe for runner-owned trees).
        _git(["worktree", "prune"], cwd=repo_path, timeout=60)
        try:
            import shutil

            shutil.rmtree(worktree, ignore_errors=True)
        except Exception as exc:  # pragma: no cover
            LOG.warning("Failed to remove worktree %s: %s", worktree, exc)


def build_prompt(instance: dict[str, Any], *, worktree: Path, max_chars: int) -> str:
    instance_id = str(instance.get("instance_id") or "")
    repo = str(instance.get("repo") or "")
    problem = str(instance.get("problem_statement") or "").strip()
    hints = str(instance.get("hints_text") or "").strip()

    parts = [
        "You are solving a single SWE-bench issue inside an isolated git checkout.",
        f"Repository: {repo}",
        f"Instance ID: {instance_id}",
        f"Working directory (repository root): {worktree}",
        "",
        "Task:",
        "- Read the issue carefully.",
        "- Inspect only the files needed to implement a minimal correct fix.",
        "- Apply the fix directly in this repository (edit source files).",
        "- Do not modify tests solely to make them pass.",
        "- Do not commit, push, rebase, or rewrite git history.",
        "- Do not interact with the user; complete the coding task in one pass.",
        "- When finished, leave the patch as uncommitted working-tree changes.",
        "",
        "Issue / problem statement:",
        problem or "(empty problem statement)",
    ]
    if hints:
        parts.extend(["", "Hints:", hints])

    prompt = "\n".join(parts).strip() + "\n"
    if len(prompt) > max_chars:
        LOG.warning(
            "Truncating prompt for %s from %d to %d chars",
            instance_id,
            len(prompt),
            max_chars,
        )
        prompt = prompt[: max_chars - 32].rstrip() + "\n\n[truncated]\n"
    return prompt


def capture_model_patch(worktree: Path) -> str:
    """Capture a unified diff of all changes, including new/untracked files."""
    # Stage everything in the index without committing so new files appear.
    add = _git(["add", "-A"], cwd=worktree, timeout=120)
    if add.returncode != 0:
        LOG.warning("git add -A failed in %s: %s", worktree, (add.stderr or "").strip())

    diff = _git(
        ["diff", "--cached", "--binary", "--no-ext-diff", "--no-color"],
        cwd=worktree,
        timeout=120,
    )
    if diff.returncode != 0:
        LOG.warning("git diff --cached failed in %s: %s", worktree, (diff.stderr or "").strip())
        # Fall back to unstaged diff only.
        fallback = _git(
            ["diff", "--binary", "--no-ext-diff", "--no-color"],
            cwd=worktree,
            timeout=120,
        )
        return fallback.stdout if fallback.returncode == 0 else ""

    return diff.stdout or ""


def prediction_record(
    *,
    instance_id: str,
    model_name_or_path: str,
    model_patch: str,
) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name_or_path,
        "model_patch": model_patch or "",
    }


def write_prediction_line(output: Path, record: dict[str, str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _seed_isolated_mana_home(target: Path) -> Path:
    """Create a per-instance MANA_HOME that still has user credentials/config.

    Isolation keeps run/session state out of the operator's primary ~/.mana while
    copying config.toml and secrets.toml so non-interactive chat can authenticate.
    """
    import shutil

    target.mkdir(parents=True, exist_ok=True)
    configured = str(os.getenv("MANA_HOME") or "").strip()
    source = (
        Path(configured).expanduser().resolve()
        if configured
        else (Path.home() / ".mana").resolve()
    )
    if source.is_dir() and source != target.resolve():
        for name in ("config.toml", "secrets.toml"):
            src = source / name
            dst = target / name
            if src.is_file() and not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except OSError as exc:
                    LOG.warning("Could not copy %s into isolated MANA_HOME: %s", src, exc)
    return target


def run_mana_agent(
    *,
    worktree: Path,
    prompt: str,
    agent_model: str,
    timeout_seconds: int,
    mana_argv: Sequence[str],
    run_dir: Path,
) -> tuple[int, str, str]:
    """Invoke mana-agent once, non-interactively, with a hard wall-clock timeout.

    Uses the chat CLI with a single prompt and closed stdin so the session
    exits after the coding turn (EOF), leaving patches in the worktree.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "mana_stdout.log"
    stderr_path = run_dir / "mana_stderr.log"
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    # Isolate mana state for this instance to avoid cross-run contamination, but
    # seed credentials from the operator's real MANA_HOME / ~/.mana.
    mana_home = _seed_isolated_mana_home(run_dir / "mana_home")

    env = os.environ.copy()
    env["MANA_HOME"] = str(mana_home)
    # Force the cheap/fast model for this smoke-oriented runner path.
    env["OPENAI_CHAT_MODEL"] = agent_model
    env["MANA_PRIMARY_MODEL"] = agent_model
    # Prefer non-interactive defaults.
    env.setdefault("TERM", "dumb")
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        *mana_argv,
        "chat",
        "--no-tui",
        "--root-dir",
        str(worktree),
        "--model",
        agent_model,
        "--full-auto",
        "--ephemeral-index",
        "--auto-continue",
        "--execution-profile",
        "full-auto",
        "--auto-execute-max-passes",
        "10",
        # Single coding task prompt (positional).
        prompt,
    ]

    LOG.info(
        "Starting mana-agent (timeout=%ss, model=%s, root=%s)",
        timeout_seconds,
        agent_model,
        worktree,
    )
    LOG.debug("Command: %s", " ".join(shlex.quote(c) for c in cmd))

    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open(
        "w", encoding="utf-8"
    ) as err_f:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(worktree),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise SweBenchRunnerError(
                f"mana-agent executable not found: {mana_argv[0]!r}"
            ) from exc

        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            LOG.error(
                "mana-agent timed out after %ss for worktree %s — killing process group",
                timeout_seconds,
                worktree,
            )
            _kill_process_group(proc)
            returncode = proc.returncode if proc.returncode is not None else 124

    elapsed = time.monotonic() - started
    stdout_tail = _tail_text(stdout_path, max_chars=4000)
    stderr_tail = _tail_text(stderr_path, max_chars=4000)
    summary = (
        f"returncode={returncode} elapsed={elapsed:.1f}s timed_out={timed_out}\n"
        f"--- stdout (tail) ---\n{stdout_tail}\n"
        f"--- stderr (tail) ---\n{stderr_tail}\n"
    )
    (run_dir / "mana_summary.txt").write_text(summary, encoding="utf-8")
    return returncode, stdout_tail, stderr_tail


def _tail_text(path: Path, *, max_chars: int) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def process_instance(
    instance: dict[str, Any],
    *,
    cfg: RunnerConfig,
    repos_dir: Path,
    worktrees_dir: Path,
    logs_dir: Path,
    mana_argv: Sequence[str],
) -> InstanceResult:
    instance_id = str(instance.get("instance_id") or "").strip()
    if not instance_id:
        return InstanceResult(
            instance_id="(missing)",
            status="error",
            error="instance_id missing from dataset row",
        )

    repo = str(instance.get("repo") or "").strip()
    base_commit = str(instance.get("base_commit") or "").strip()
    if not repo or not base_commit:
        return InstanceResult(
            instance_id=instance_id,
            status="error",
            error="repo or base_commit missing",
        )

    started = time.monotonic()
    run_dir = logs_dir / _sanitize_id(instance_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "instance.json").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": base_commit,
                "difficulty": instance.get("difficulty"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    repo_path: Path | None = None
    worktree: Path | None = None
    try:
        repo_path = ensure_repo_clone(repo, repos_dir)
        worktree = create_isolated_worktree(
            repo_path,
            base_commit=base_commit,
            worktrees_dir=worktrees_dir,
            instance_id=instance_id,
        )

        if worktree_is_dirty(worktree):
            raise SweBenchRunnerError(f"Worktree dirty before agent run: {worktree}")

        if cfg.skip_agent:
            LOG.warning("Skipping agent for %s (--skip-agent)", instance_id)
            patch = ""
            status = "skipped_agent"
            error = "agent skipped"
        else:
            prompt = build_prompt(
                instance, worktree=worktree, max_chars=cfg.max_prompt_chars
            )
            try:
                rc, _out, err = run_mana_agent(
                    worktree=worktree,
                    prompt=prompt,
                    agent_model=cfg.agent_model,
                    timeout_seconds=cfg.timeout_seconds,
                    mana_argv=mana_argv,
                    run_dir=run_dir,
                )
            except subprocess.TimeoutExpired:
                rc, err = 124, "timeout"
            if rc == 124:
                status = "timeout"
                error = f"mana-agent exceeded hard timeout of {cfg.timeout_seconds}s"
            elif rc != 0:
                status = "agent_error"
                error = f"mana-agent exited with code {rc}"
                if err:
                    error = f"{error}: {err[:500]}"
            else:
                status = "ok"
                error = ""

            patch = capture_model_patch(worktree)

        empty = not bool(patch.strip())
        if empty:
            LOG.warning(
                "Empty model_patch for %s (status=%s). Writing empty prediction.",
                instance_id,
                status,
            )
            if status == "ok":
                status = "empty_patch"
                error = error or "agent finished without producing a patch"

        return InstanceResult(
            instance_id=instance_id,
            model_patch=patch,
            status=status,
            error=error,
            duration_seconds=time.monotonic() - started,
            worktree=str(worktree),
            empty_patch=empty,
        )
    except SweBenchRunnerError as exc:
        LOG.error("Instance %s failed: %s", instance_id, exc)
        return InstanceResult(
            instance_id=instance_id,
            model_patch="",
            status="error",
            error=str(exc),
            duration_seconds=time.monotonic() - started,
            worktree=str(worktree) if worktree else "",
            empty_patch=True,
        )
    except Exception as exc:  # noqa: BLE001 — isolate per-instance failures
        LOG.exception("Unexpected failure for %s", instance_id)
        return InstanceResult(
            instance_id=instance_id,
            model_patch="",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=time.monotonic() - started,
            worktree=str(worktree) if worktree else "",
            empty_patch=True,
        )
    finally:
        if worktree is not None and repo_path is not None and not cfg.retain_worktrees:
            try:
                remove_worktree(repo_path, worktree, force=True)
            except Exception as cleanup_exc:  # pragma: no cover
                LOG.warning("Cleanup failed for %s: %s", worktree, cleanup_exc)


def parse_instance_ids(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    ids: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                ids.append(part)
    return ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SWE-bench Verified predictions.jsonl using mana-agent. "
            "Each line is "
            '{"instance_id","model_name_or_path","model_patch"}.'
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of instances to run (after --instance-ids filter).",
    )
    parser.add_argument(
        "--instance-ids",
        action="append",
        default=[],
        help=(
            "Only run these instance_id values. Repeatable, or comma-separated. "
            "Example: --instance-ids astropy__astropy-12907"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Predictions JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(DEFAULT_WORK_DIR),
        help=f"Cache/work directory for clones, worktrees, logs (default: {DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_AGENT_MODEL,
        help=(
            "LLM id forced for mana-agent during the run "
            f"(default: {DEFAULT_AGENT_MODEL}, cheap/fast for smoke)."
        ),
    )
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help=(
            "Value written into predictions as model_name_or_path "
            f"(default: {DEFAULT_MODEL_NAME_OR_PATH})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Hard per-instance wall-clock timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--mana-bin",
        default=None,
        help="Path or command for mana-agent (default: PATH lookup, else python -m mana_agent).",
    )
    parser.add_argument(
        "--retain-worktrees",
        action="store_true",
        help="Keep per-instance worktrees under work-dir for debugging.",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help=(
            "Skip mana-agent invocation and write empty model_patch values. "
            "Useful for harness format smoke tests without API cost."
        ),
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=48_000,
        help="Truncate issue prompts longer than this many characters.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the run on the first hard instance failure (default: continue).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run(cfg: RunnerConfig) -> list[InstanceResult]:
    work_dir = cfg.work_dir.expanduser().resolve()
    repos_dir = work_dir / "repos"
    worktrees_dir = work_dir / "worktrees"
    logs_dir = work_dir / "logs"
    for path in (repos_dir, worktrees_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    output = cfg.output.expanduser().resolve()
    if output.exists():
        LOG.info("Removing existing predictions file %s", output)
        output.unlink()

    mana_argv = resolve_mana_bin(cfg.mana_bin)
    LOG.info("mana-agent argv: %s", mana_argv)
    LOG.info("Forced agent model: %s", cfg.agent_model)
    LOG.info("Per-instance timeout: %ss", cfg.timeout_seconds)

    instances = load_instances(
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        limit=cfg.limit,
        instance_ids=cfg.instance_ids,
    )
    if not instances:
        raise SweBenchRunnerError("No instances selected. Check --limit / --instance-ids.")

    results: list[InstanceResult] = []
    for index, instance in enumerate(instances, start=1):
        instance_id = str(instance.get("instance_id") or f"row-{index}")
        LOG.info("[%d/%d] Starting %s", index, len(instances), instance_id)
        result = process_instance(
            instance,
            cfg=cfg,
            repos_dir=repos_dir,
            worktrees_dir=worktrees_dir,
            logs_dir=logs_dir,
            mana_argv=mana_argv,
        )
        record = prediction_record(
            instance_id=result.instance_id,
            model_name_or_path=cfg.model_name_or_path,
            model_patch=result.model_patch,
        )
        write_prediction_line(output, record)
        (logs_dir / _sanitize_id(result.instance_id) / "result.json").write_text(
            json.dumps(
                {
                    "instance_id": result.instance_id,
                    "status": result.status,
                    "error": result.error,
                    "duration_seconds": result.duration_seconds,
                    "empty_patch": result.empty_patch,
                    "patch_chars": len(result.model_patch or ""),
                    "worktree": result.worktree,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        results.append(result)
        LOG.info(
            "[%d/%d] Finished %s status=%s empty_patch=%s duration=%.1fs",
            index,
            len(instances),
            result.instance_id,
            result.status,
            result.empty_patch,
            result.duration_seconds,
        )

        if not cfg.continue_on_error and result.status in {"error", "timeout"}:
            raise SweBenchRunnerError(
                f"Fail-fast: {result.instance_id} failed with {result.status}: {result.error}"
            )

    _write_run_summary(work_dir / "run_summary.json", results, output)
    LOG.info("Wrote %d prediction(s) to %s", len(results), output)
    return results


def _write_run_summary(path: Path, results: Iterable[InstanceResult], output: Path) -> None:
    rows = list(results)
    summary = {
        "predictions_path": str(output),
        "count": len(rows),
        "empty_patches": sum(1 for r in rows if r.empty_patch),
        "statuses": {},
        "instances": [
            {
                "instance_id": r.instance_id,
                "status": r.status,
                "empty_patch": r.empty_patch,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
            }
            for r in rows
        ],
    }
    for r in rows:
        summary["statuses"][r.status] = summary["statuses"].get(r.status, 0) + 1
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(bool(args.verbose))

    cfg = RunnerConfig(
        dataset_name=str(args.dataset),
        split=str(args.split),
        limit=args.limit,
        instance_ids=parse_instance_ids(args.instance_ids),
        output=Path(args.output),
        work_dir=Path(args.work_dir),
        agent_model=str(args.model),
        model_name_or_path=str(args.model_name_or_path),
        timeout_seconds=max(30, int(args.timeout)),
        mana_bin=args.mana_bin,
        retain_worktrees=bool(args.retain_worktrees),
        skip_agent=bool(args.skip_agent),
        max_prompt_chars=max(2000, int(args.max_prompt_chars)),
        continue_on_error=not bool(args.fail_fast),
    )

    try:
        results = run(cfg)
    except SweBenchRunnerError as exc:
        LOG.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOG.error("Interrupted")
        return 130

    # Non-zero only when every instance hard-failed (still wrote predictions).
    if results and all(r.status in {"error", "timeout"} for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
