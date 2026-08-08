#!/usr/bin/env python3
"""Generate SWE-bench Verified predictions.jsonl using mana-agent.

Loads ``princeton-nlp/SWE-bench_Verified``, checks out each instance at
``base_commit`` into an isolated git worktree, runs mana-agent once on the
issue text, captures the final git diff as ``model_patch``, and writes one
JSONL prediction per instance in the official harness format:

    {
      "instance_id": "...",
      "model_name_or_path": "mana-agent__nvidia__deepseek-ai__deepseek-v4-flash",
      "agent_name": "mana-agent",
      "agent_provider": "nvidia",
      "agent_model": "deepseek-ai/deepseek-v4-flash",
      "model_patch": "..."
    }

``model_name_or_path`` identifies the **run system** (agent + LLM). The harness
uses it for report filenames. ``agent_name`` is the coding agent (always
``mana-agent`` unless overridden). Provider/model default from
``~/.mana/config.toml`` when ``--provider`` / ``--model`` are omitted.

Instance selection (explicit contract):

* **No** ``--instance-ids`` (and no ``--instance-ids-file``) → load **all**
  instance ids from the SWE-bench dataset split and run them (full Verified
  suite is ~500 rows). Optional ``--limit N`` caps after that selection.
* **With** ``--instance-ids`` / ``--instance-ids-file`` → run **only** those
  ids (still subject to ``--limit``).

Scope: prediction generation (smoke grading is documented separately).
Does not cover SWE-bench Pro, Terminal-Bench, pass@k packaging, or automatic
leaderboard upload.
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
# Coding agent identity written into every prediction row.
DEFAULT_AGENT_NAME = "mana-agent"
# Last-resort LLM only when ~/.mana/config.toml has no model and CLI omits --model.
DEFAULT_AGENT_MODEL = "gpt-4o-mini"
DEFAULT_AGENT_PROVIDER = "openai"
DEFAULT_TIMEOUT_SECONDS = 600
# 0 / negative CLI or env timeout means no runner wall-clock kill.
UNLIMITED_TIMEOUT_SENTINEL = 0
DEFAULT_OUTPUT = "predictions.jsonl"
DEFAULT_WORK_DIR = ".swe-bench"
GITHUB_URL = "https://github.com/{repo}.git"
# Mana provider ids that may appear as a MANA_PRIMARY_MODEL prefix.
_KNOWN_MANA_PROVIDERS = frozenset({"openai", "nvidia", "openrouter", "custom"})
# Optional env overrides when CLI --timeout is omitted (same-line flags preferred).
_TIMEOUT_ENV_KEYS = ("MANA_SWE_BENCH_TIMEOUT", "SWE_BENCH_TIMEOUT")

SAFE_INSTANCE_ID = re.compile(r"[^A-Za-z0-9._-]+")
# Paths that look like test code. SWE-bench applies the official ``test_patch``
# after ``model_patch``; agent edits under these paths often make apply fail
# (status ``failed`` instead of resolved/unresolved).
_TEST_PATH_MARKERS = (
    "/tests/",
    "/test/",
    "/testing/",
    "/__tests__/",
    "/spec/",
    "/specs/",
)
_TEST_FILE_RE = re.compile(
    r"(^|/)("
    r"tests?|"
    r"testing|"
    r"conftest|"
    r"test_[^/]+|"
    r"[^/]+_test|"
    r"[^/]+_tests|"
    r"[^/]+\.test|"
    r"[^/]+\.spec"
    r")(\.[^/]+)?$",
    re.IGNORECASE,
)


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


@dataclass(frozen=True, slots=True)
class OperatorInferenceDefaults:
    """Model/provider resolved from the operator's Mana home config."""

    provider: str
    model: str
    primary_model: str
    config_path: str
    source: str


@dataclass
class RunnerConfig:
    dataset_name: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    limit: int | None = None
    instance_ids: list[str] = field(default_factory=list)
    output: Path = Path(DEFAULT_OUTPUT)
    work_dir: Path = Path(DEFAULT_WORK_DIR)
    agent_name: str = DEFAULT_AGENT_NAME
    agent_provider: str = DEFAULT_AGENT_PROVIDER
    agent_model: str = DEFAULT_AGENT_MODEL
    # None → derive from agent_name + agent_model at write time.
    model_name_or_path: str | None = None
    # Human-readable origin for logs (config path, cli, built-in default).
    model_source: str = "built-in"
    provider_source: str = "built-in"
    # 0 means unlimited (no runner wall-clock kill).
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    timeout_source: str = "built-in"
    mana_bin: str | None = None
    retain_worktrees: bool = False
    skip_agent: bool = False
    # Drop test-file hunks from model_patch (recommended for harness grading).
    exclude_test_files: bool = True
    max_prompt_chars: int = 48_000
    continue_on_error: bool = True


def sanitize_model_token(value: str) -> str:
    """Make a model/agent token safe for harness report paths and run ids."""
    cleaned = str(value or "").strip()
    cleaned = cleaned.replace("/", "__").replace(" ", "-")
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", cleaned)
    return cleaned.strip("._") or "unknown"


def compose_model_name_or_path(*, agent_name: str, agent_model: str) -> str:
    """Default harness id: ``{agent}__{llm}`` (never agent-only or llm-only)."""
    agent = sanitize_model_token(agent_name)
    model = sanitize_model_token(agent_model)
    return f"{agent}__{model}"


def resolve_model_name_or_path(cfg: RunnerConfig) -> str:
    explicit = (cfg.model_name_or_path or "").strip()
    if explicit:
        return sanitize_model_token(explicit)
    # Include provider for multi-host operators so nvidia/deepseek… does not
    # collide with an OpenAI run of the same bare model id.
    model_label = cfg.agent_model
    provider = str(cfg.agent_provider or "").strip().lower()
    if provider and provider not in {"", "openai"} and not model_label.lower().startswith(
        f"{provider}/"
    ):
        model_label = f"{provider}/{model_label}"
    return compose_model_name_or_path(
        agent_name=cfg.agent_name,
        agent_model=model_label,
    )


def operator_mana_home() -> Path:
    """Return the operator Mana home used for credentials and default model."""
    configured = str(os.getenv("MANA_HOME") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".mana").resolve()


def _load_toml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
            import tomli as tomllib  # type: ignore
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        LOG.warning("Could not parse Mana config %s: %s", path, exc)
        return {}


def _strip_provider_prefix(model: str, provider: str) -> str:
    """Return upstream model id without a leading Mana provider segment."""
    text = str(model or "").strip()
    provider_id = str(provider or "").strip().lower()
    if not text:
        return ""
    if provider_id and text.lower().startswith(provider_id + "/"):
        return text[len(provider_id) + 1 :]
    head, _, rest = text.partition("/")
    if rest and head.lower() in _KNOWN_MANA_PROVIDERS:
        # e.g. nvidia/deepseek-ai/... when provider was already resolved.
        if provider_id and head.lower() == provider_id:
            return rest
    return text


def load_operator_inference_defaults(
    mana_home: Path | None = None,
) -> OperatorInferenceDefaults:
    """Read provider + model defaults from ``~/.mana/config.toml`` (or MANA_HOME)."""
    home = mana_home if mana_home is not None else operator_mana_home()
    config_path = home / "config.toml"
    data = _load_toml_mapping(config_path)

    provider = str(data.get("MANA_AI_PROVIDER") or "").strip().lower()
    primary = str(data.get("MANA_PRIMARY_MODEL") or "").strip()
    chat = str(
        data.get("OPENAI_CHAT_MODEL")
        or data.get("LLM_MODEL")
        or data.get("MANA_MODEL_CODING")
        or ""
    ).strip()
    # Role aliases like MODEL_LEVEL_2_CODING are not resolved here; prefer concrete ids.
    if chat.startswith("MODEL_LEVEL_"):
        chat = str(
            data.get(chat)
            or data.get("OPENAI_CHAT_MODEL")
            or data.get("LLM_MODEL")
            or ""
        ).strip()

    if not provider and primary:
        head = primary.split("/", 1)[0].strip().lower()
        if head in _KNOWN_MANA_PROVIDERS:
            provider = head
    if not provider:
        provider = DEFAULT_AGENT_PROVIDER

    model = ""
    source = f"missing ({config_path})"
    if primary:
        model = _strip_provider_prefix(primary, provider)
        source = f"MANA_PRIMARY_MODEL in {config_path}"
    if not model and chat:
        model = _strip_provider_prefix(chat, provider)
        source = f"OPENAI_CHAT_MODEL/LLM_MODEL in {config_path}"
    if not model:
        model = DEFAULT_AGENT_MODEL
        source = f"built-in default ({DEFAULT_AGENT_MODEL}); no model in {config_path}"

    primary_model = primary or f"{provider}/{model}"
    return OperatorInferenceDefaults(
        provider=provider,
        model=model,
        primary_model=primary_model,
        config_path=str(config_path),
        source=source,
    )


def resolve_agent_inference(
    *,
    cli_model: str | None,
    cli_provider: str | None,
    mana_home: Path | None = None,
) -> tuple[str, str, str, str]:
    """Resolve ``(provider, model, provider_source, model_source)``.

    Policy:
    * Provider defaults to ``MANA_AI_PROVIDER`` in ``~/.mana/config.toml``.
    * Model defaults to ``MANA_PRIMARY_MODEL`` / chat model in that config.
    * CLI ``--model`` overrides only the model id; provider still comes from
      config (or CLI ``--provider`` when set).
    * CLI ``--provider`` overrides the configured provider.
    """
    defaults = load_operator_inference_defaults(mana_home)

    cli_provider_text = str(cli_provider or "").strip().lower()
    if cli_provider_text:
        provider = cli_provider_text
        provider_source = "cli --provider"
    else:
        provider = defaults.provider
        provider_source = f"MANA_AI_PROVIDER in {defaults.config_path}"

    cli_model_text = str(cli_model or "").strip()
    if cli_model_text:
        # Allow provider-qualified CLI models (nvidia/deepseek-ai/...).
        head, _, rest = cli_model_text.partition("/")
        if rest and head.lower() in _KNOWN_MANA_PROVIDERS:
            if not cli_provider_text:
                provider = head.lower()
                provider_source = "cli --model prefix"
            model = rest if head.lower() == provider else cli_model_text
            if head.lower() != provider:
                # Different known prefix than selected provider: keep full string
                # as the upstream model id (OpenRouter-style openai/gpt-...).
                model = cli_model_text
            else:
                model = rest
        else:
            model = cli_model_text
        model_source = "cli --model"
    else:
        model_source = defaults.source
        # Prefer primary (may be provider-qualified) under the selected provider.
        candidate = defaults.primary_model or defaults.model
        model = _strip_provider_prefix(candidate, provider) or defaults.model

    if not model:
        model = DEFAULT_AGENT_MODEL
        model_source = f"built-in default ({DEFAULT_AGENT_MODEL})"
    if not provider:
        provider = DEFAULT_AGENT_PROVIDER
        provider_source = "built-in default"

    return provider, model, provider_source, model_source


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


def load_dataset_rows(*, dataset_name: str, split: str) -> list[dict[str, Any]]:
    """Load every row from a HuggingFace SWE-bench dataset split."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment specific
        raise SweBenchRunnerError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from exc

    LOG.info("Loading dataset %s split=%s", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]


def dataset_instance_ids(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Return ordered instance_id values from dataset rows (non-empty only)."""
    ids: list[str] = []
    for row in rows:
        iid = str(row.get("instance_id") or "").strip()
        if iid:
            ids.append(iid)
    return ids


def select_instances(
    rows: Sequence[dict[str, Any]],
    *,
    instance_ids: Sequence[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select dataset rows by optional id filter and limit.

    Selection contract:

    * Empty ``instance_ids`` → keep **all** rows from the loaded SWE-bench split.
    * Non-empty ``instance_ids`` → keep **only** matching rows (warn on missing).
    * ``limit`` (if set) is applied **after** the id filter.
    """
    selected: list[dict[str, Any]] = list(rows)
    all_ids = dataset_instance_ids(selected)

    wanted = [i.strip() for i in instance_ids if str(i).strip()]
    if wanted:
        wanted_set = set(wanted)
        # Preserve dataset order; allow duplicate flags without duplicating rows.
        selected = [
            r for r in selected if str(r.get("instance_id", "")).strip() in wanted_set
        ]
        found = {str(r.get("instance_id", "")).strip() for r in selected}
        missing = sorted(wanted_set - found)
        if missing:
            LOG.warning(
                "Requested instance_ids not found in dataset (%d): %s",
                len(missing),
                missing[:20] + (["..."] if len(missing) > 20 else []),
            )
        LOG.info(
            "Instance filter: explicit --instance-ids (%d requested) → %d match(es) "
            "from %d dataset id(s)",
            len(wanted_set),
            len(selected),
            len(all_ids),
        )
    else:
        LOG.info(
            "Instance filter: no --instance-ids provided → selecting all %d "
            "instance id(s) from SWE-bench dataset",
            len(all_ids),
        )

    if limit is not None:
        if limit < 0:
            raise SweBenchRunnerError("--limit must be >= 0")
        if len(selected) > limit:
            LOG.info("Applying --limit %d (was %d selected)", limit, len(selected))
        selected = selected[:limit]

    LOG.info("Selected %d instance(s) to run", len(selected))
    if selected and LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "Selected ids: %s",
            [str(r.get("instance_id", "")) for r in selected],
        )
    return selected


def load_instances(
    *,
    dataset_name: str,
    split: str,
    limit: int | None,
    instance_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Load SWE-bench rows and apply the instance selection contract."""
    rows = load_dataset_rows(dataset_name=dataset_name, split=split)
    return select_instances(rows, instance_ids=instance_ids, limit=limit)


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


def _worktree_lock_path(worktrees_dir: Path, instance_id: str) -> Path:
    return worktrees_dir / f".{_sanitize_id(instance_id)}.lock"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is not owned by us — treat as live.
        return True
    except OSError:
        return False
    return True


def worktree_lock_holder(worktrees_dir: Path, instance_id: str) -> int | None:
    """Return the live PID holding the instance worktree lock, if any."""
    lock_path = _worktree_lock_path(worktrees_dir, instance_id)
    if not lock_path.is_file():
        return None
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
        pid = int(raw.splitlines()[0].strip())
    except (OSError, ValueError, IndexError):
        return None
    return pid if _pid_is_alive(pid) else None


def acquire_worktree_lock(worktrees_dir: Path, instance_id: str) -> Path:
    """Create an exclusive lock for one instance worktree.

    Prevents concurrent runner processes from deleting a live worktree under an
    active mana-agent (which surfaces as getcwd/Codex ENOENT failures).
    """
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _worktree_lock_path(worktrees_dir, instance_id)
    holder = worktree_lock_holder(worktrees_dir, instance_id)
    if holder is not None and holder != os.getpid():
        raise SweBenchRunnerError(
            f"Worktree for {instance_id} is locked by live process pid={holder} "
            f"({lock_path}). Wait for that run to finish or remove the stale lock."
        )
    lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return lock_path


def release_worktree_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        if not lock_path.is_file():
            return
        raw = lock_path.read_text(encoding="utf-8").strip()
        pid = int(raw.splitlines()[0].strip())
        if pid not in {0, os.getpid()} and _pid_is_alive(pid):
            return
        lock_path.unlink(missing_ok=True)
    except (OSError, ValueError, IndexError):
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


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

    holder = worktree_lock_holder(worktrees_dir, instance_id)
    if holder is not None and holder != os.getpid():
        raise SweBenchRunnerError(
            f"Refusing to recreate worktree for {instance_id}: locked by pid={holder}. "
            "A concurrent SWE-bench run is still using this checkout."
        )

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
    if not (target / ".git").exists():
        remove_worktree(repo_path, target, force=True)
        raise SweBenchRunnerError(
            f"Worktree for {instance_id} is missing .git after checkout: {target}"
        )

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
    # Sibling lock lives next to the worktree directory.
    lock_path = worktree.parent / f".{worktree.name}.lock"
    holder = None
    if lock_path.is_file():
        try:
            holder = int(lock_path.read_text(encoding="utf-8").strip().splitlines()[0])
        except (OSError, ValueError, IndexError):
            holder = None
    if holder is not None and holder != os.getpid() and _pid_is_alive(holder):
        raise SweBenchRunnerError(
            f"Refusing to remove worktree {worktree}: locked by live pid={holder}"
        )
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
        "- Prefer reading source files and applying edits over running the package.",
        "- The repository is a source checkout and may not be installed or importable;",
        "  do not spend the turn diagnosing import/runtime environment failures.",
        "- If you run Python, always use `python3` (never bare `python`, which may be 2.x).",
        "- Apply the fix directly in this repository (edit production source files only).",
        "- Do not add, edit, delete, or rename test files or test helpers.",
        "- Do not modify tests solely to make them pass.",
        "- Official evaluation tests are applied separately; test changes cause harness failures.",
        "- Do not commit, push, rebase, or rewrite git history.",
        "- Do not interact with the user; complete the coding task in one pass.",
        "- Success means production-source edits left as uncommitted working-tree changes.",
        "- Do not finish with only analysis or chat text when a code fix is required.",
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


def _path_looks_like_test(path: str) -> bool:
    """Return True when a repo-relative path is almost certainly a test asset."""
    normalized = path.replace("\\", "/").lstrip("./")
    if not normalized:
        return False
    lower = f"/{normalized.lower()}"
    if any(marker in lower for marker in _TEST_PATH_MARKERS):
        return True
    basename = normalized.rsplit("/", 1)[-1]
    return bool(_TEST_FILE_RE.search(basename))


def is_test_file_path(path: str) -> bool:
    """Public helper: whether a git path should be excluded from model_patch."""
    return _path_looks_like_test(path)


def filter_test_files_from_patch(patch: str) -> tuple[str, list[str]]:
    """Remove unified-diff file sections that touch test paths.

    Returns (filtered_patch, removed_paths). Empty string when nothing remains.
    """
    if not (patch or "").strip():
        return "", []

    # Split on file headers while keeping the delimiter.
    chunks = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    removed: list[str] = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        if not chunk.startswith("diff --git "):
            # Preamble / non-standard content: keep unless it is only whitespace.
            kept.append(chunk)
            continue
        header = chunk.splitlines()[0]
        # diff --git a/path b/path
        match = re.match(r"^diff --git a/(.+?) b/(.+)$", header)
        if not match:
            kept.append(chunk)
            continue
        path_a, path_b = match.group(1), match.group(2)
        if _path_looks_like_test(path_a) or _path_looks_like_test(path_b):
            removed.append(path_b if path_b != "/dev/null" else path_a)
            continue
        kept.append(chunk)

    filtered = "".join(kept)
    # Ensure trailing newline when non-empty (git style).
    if filtered and not filtered.endswith("\n"):
        filtered += "\n"
    return filtered, removed


@dataclass(frozen=True, slots=True)
class WorktreeChangeSummary:
    """Counts of porcelain status paths before model_patch capture."""

    modified: int = 0
    added: int = 0
    deleted: int = 0
    other: int = 0
    paths: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return self.modified + self.added + self.deleted + self.other

    @property
    def is_mass_delete_only(self) -> bool:
        """True when the tree is dominated by accidental mass deletions."""
        # Intentional SWE-bench fixes almost never delete dozens of unrelated
        # files with zero content edits. Those runs produce harness-breaking
        # empty or destructive patches (seen on corrupted worktrees).
        return self.deleted >= 20 and self.modified == 0 and self.added == 0


def summarize_worktree_changes(worktree: Path) -> WorktreeChangeSummary:
    """Parse ``git status --porcelain`` into coarse change counts."""
    status = _git(["status", "--porcelain"], cwd=worktree, timeout=120)
    if status.returncode != 0:
        return WorktreeChangeSummary()
    modified = added = deleted = other = 0
    paths: list[str] = []
    for raw_line in (status.stdout or "").splitlines():
        if not raw_line.strip():
            continue
        # porcelain v1: XY<path> with optional rename " -> "
        code = raw_line[:2] if len(raw_line) >= 2 else "  "
        path = raw_line[3:].strip() if len(raw_line) > 3 else raw_line.strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[-1].strip()
        paths.append(path)
        xy = code.replace(" ", "")
        if "D" in code:
            deleted += 1
        elif "A" in code or "?" in code:
            added += 1
        elif "M" in code or "T" in code or "R" in code or "C" in code:
            modified += 1
        elif xy:
            other += 1
        else:
            other += 1
    return WorktreeChangeSummary(
        modified=modified,
        added=added,
        deleted=deleted,
        other=other,
        paths=tuple(paths[:200]),
    )


def resolve_python3_executable() -> str:
    """Return an absolute Python 3 interpreter path for agent shells."""
    # Prefer the runner's interpreter (always 3.x when this module loads).
    if getattr(sys, "executable", None) and Path(sys.executable).exists():
        return str(Path(sys.executable).resolve())
    for name in ("python3", "python3.12", "python3.11", "python3.10"):
        found = shutil_which(name)
        if found:
            return str(Path(found).resolve())
    raise SweBenchRunnerError(
        "Could not locate a Python 3 interpreter for the SWE-bench agent PATH shim."
    )


def prepare_agent_python_path(*, run_dir: Path, env: dict[str, str]) -> Path:
    """Prepend a run-local bin dir so bare ``python`` invokes Python 3.

    Hosts often expose Python 2.7 as ``python`` early on PATH (macOS Frameworks).
    Agents that run ``python -c '...'`` then hit SyntaxError on f-strings and
    derail into empty patches. A tiny shim keeps the rest of PATH intact.

    On POSIX the shim is an executable shell script named ``python`` /
    ``python3``. On Windows, PATHEXT only resolves ``.exe`` / ``.cmd`` /
    ``.bat`` (not extensionless scripts), so ``python.cmd`` / ``python3.cmd``
    are written instead. ``chmod`` execute bits are not meaningful on NTFS the
    same way and are only applied on POSIX.
    """
    bin_dir = (run_dir / "agent_bin").resolve()
    bin_dir.mkdir(parents=True, exist_ok=True)
    python3 = resolve_python3_executable()
    if os.name == "nt":
        # Windows: cmd shims so `python` / `python3` resolve via PATHEXT.
        for name in ("python", "python3"):
            shim = bin_dir / f"{name}.cmd"
            # Quote the interpreter; forward all args with %*.
            script = f'@echo off\r\n"{python3}" %*\r\n'
            shim.write_text(script, encoding="utf-8", newline="\r\n")
    else:
        for name in ("python", "python3"):
            shim = bin_dir / name
            script = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f'exec "{python3}" "$@"\n'
            )
            shim.write_text(script, encoding="utf-8")
            shim.chmod(0o755)
    existing = str(env.get("PATH") or os.environ.get("PATH") or "")
    env["PATH"] = f"{bin_dir}{os.pathsep}{existing}" if existing else str(bin_dir)
    # Hint for tools that honor PYTHON / VIRTUAL_ENV conventions without PATH.
    env["PYTHON"] = python3
    env.setdefault("PYTHONUTF8", "1")
    LOG.info("Agent PATH shim: python -> %s (bin=%s)", python3, bin_dir)
    return bin_dir


def capture_model_patch(
    worktree: Path,
    *,
    exclude_test_files: bool = True,
    reject_mass_delete: bool = True,
) -> tuple[str, str]:
    """Capture a unified diff of all changes, including new/untracked files.

    Returns ``(patch, rejection_reason)``. ``rejection_reason`` is non-empty
    when the patch is discarded as unsafe (e.g. mass-delete-only worktree).
    """
    summary = summarize_worktree_changes(worktree)
    if reject_mass_delete and summary.is_mass_delete_only:
        reason = (
            f"refusing mass-delete-only worktree "
            f"(deleted={summary.deleted}, modified={summary.modified}, "
            f"added={summary.added}); not emitting destructive model_patch"
        )
        LOG.error("%s in %s", reason, worktree)
        return "", reason

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
        raw = fallback.stdout if fallback.returncode == 0 else ""
    else:
        raw = diff.stdout or ""

    if not exclude_test_files:
        return raw, ""

    filtered, removed = filter_test_files_from_patch(raw)
    if removed:
        LOG.warning(
            "Stripped %d test-file path(s) from model_patch (SWE-bench applies "
            "official test_patch separately): %s",
            len(removed),
            ", ".join(removed[:12]) + ("..." if len(removed) > 12 else ""),
        )
    return filtered, ""


def prediction_record(
    *,
    instance_id: str,
    model_name_or_path: str,
    agent_name: str,
    model_patch: str,
    agent_model: str | None = None,
    agent_provider: str | None = None,
) -> dict[str, str]:
    """Build one harness prediction row with agent + model identity fields."""
    record: dict[str, str] = {
        "instance_id": instance_id,
        # Harness-required: identifies this system for reports / run folders.
        "model_name_or_path": model_name_or_path,
        # Explicit agent identity (extra field; ignored by older harnesses).
        "agent_name": agent_name,
        "model_patch": model_patch or "",
    }
    if agent_model:
        # Optional metadata for operators; not required by the harness.
        record["agent_model"] = agent_model
    if agent_provider:
        record["agent_provider"] = agent_provider
    return record


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


def _toml_quote(value: str) -> str:
    """Quote a scalar for simple TOML assignment lines."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _render_toml_assignment(key: str, value: str | bool | int) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, int):
        rendered = str(value)
    else:
        rendered = _toml_quote(str(value))
    return f"{key} = {rendered}"


def _first_toml_table_index(lines: list[str]) -> int | None:
    """Return the index of the first ``[table]`` header, if any."""
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and not stripped.startswith("[["):
            return index
    return None


def _upsert_toml_keys(path: Path, updates: dict[str, str | bool | int]) -> None:
    """Insert or replace top-level KEY = value lines in a Mana config.toml.

    Operator configs often mix flat keys with nested tables (``[media]``,
    ``[telegram.attachments]``, …). Missing isolation keys must be inserted
    *before* the first table header; appending at EOF nests them under the last
    table and Mana never sees the override (Settings stays at defaults).
    """
    if not updates:
        return
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = original.splitlines()
    table_start = _first_toml_table_index(lines)
    # Only rewrite KEY = value lines that appear in the top-level section so we
    # never "update" a nested table key that happens to share the same name.
    top_end = table_start if table_start is not None else len(lines)

    seen: set[str] = set()
    out: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        replaced = False
        if (
            index < top_end
            and stripped
            and not stripped.startswith("#")
            and "=" in stripped
        ):
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(_render_toml_assignment(key, updates[key]))
                seen.add(key)
                replaced = True
        if not replaced:
            out.append(line)

    missing = [key for key in updates if key not in seen]
    if missing:
        block = ["# SWE-bench runner isolation overrides"]
        block.extend(_render_toml_assignment(key, updates[key]) for key in missing)
        insert_at = _first_toml_table_index(out)
        if insert_at is None:
            if out and out[-1].strip():
                out.append("")
            out.extend(block)
        else:
            # Insert before first [table], with blank lines around the block.
            if insert_at > 0 and out[insert_at - 1].strip():
                block = [""] + block
            if insert_at < len(out) and out[insert_at].strip():
                block = block + [""]
            out[insert_at:insert_at] = block

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")


def _parse_timeout_int(raw: str, *, label: str) -> int:
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise SweBenchRunnerError(
            f"Invalid {label}={raw!r}; expected integer seconds (0 = unlimited)."
        ) from exc


def load_runner_file_timeout(work_dir: Path) -> tuple[int, str] | None:
    """Optional timeout from ``<work-dir>/runner.toml`` or ``runner.env``.

    runner.toml example::

        timeout = 0

    runner.env example::

        MANA_SWE_BENCH_TIMEOUT=0
    """
    toml_path = work_dir / "runner.toml"
    if toml_path.is_file():
        data = _load_toml_mapping(toml_path)
        if "timeout" in data and data["timeout"] is not None:
            return int(data["timeout"]), f"file {toml_path}"
        # Allow nested [runner] timeout = ...
        nested = data.get("runner")
        if isinstance(nested, dict) and nested.get("timeout") is not None:
            return int(nested["timeout"]), f"file {toml_path} [runner]"

    env_path = work_dir / "runner.env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key in _TIMEOUT_ENV_KEYS or key == "TIMEOUT":
                return _parse_timeout_int(value, label=f"{env_path}:{key}"), f"file {env_path}"
    return None


def resolve_runner_timeout_seconds(
    cli_timeout: int | None,
    *,
    env: dict[str, str] | None = None,
    work_dir: Path | None = None,
) -> tuple[int, str]:
    """Resolve per-instance wall-clock timeout.

    Priority: CLI ``--timeout`` → process env → ``<work-dir>/runner.toml|env`` →
    built-in default (600).

    Returns ``(seconds, source)``. ``seconds <= 0`` means unlimited (no kill).

    **Shell trap:** multi-line commands need a trailing ``\\`` after ``runner.py``
    **or put all flags on one line**. Without that, the shell runs the script
    with defaults and drops ``--timeout`` (you will see
    ``source: built-in default``). Prefer::

        MANA_SWE_BENCH_TIMEOUT=0 python scripts/swe_bench/runner.py --output predictions.jsonl

    or write ``timeout = 0`` into ``.swe-bench/runner.toml``.
    """
    if cli_timeout is not None:
        return int(cli_timeout), "cli --timeout"
    environ = env if env is not None else os.environ
    for key in _TIMEOUT_ENV_KEYS:
        raw = str(environ.get(key) or "").strip()
        if not raw:
            continue
        return _parse_timeout_int(raw, label=key), f"env {key}"
    if work_dir is not None:
        from_file = load_runner_file_timeout(Path(work_dir).expanduser())
        if from_file is not None:
            return from_file
    return int(DEFAULT_TIMEOUT_SECONDS), f"built-in default ({DEFAULT_TIMEOUT_SECONDS}s)"


def format_timeout_label(timeout_seconds: int) -> str:
    if int(timeout_seconds) <= 0:
        return "unlimited"
    return f"{int(timeout_seconds)}s"


def warn_if_timeout_likely_dropped(
    *,
    timeout_source: str,
    argv: Sequence[str],
) -> None:
    """Log a loud warning when timeout fell back to the built-in default."""
    if not str(timeout_source).startswith("built-in"):
        return
    # If the user thought they passed --timeout but shell line-break ate it,
    # argv will not contain --timeout.
    has_timeout_flag = any(
        arg == "--timeout" or arg.startswith("--timeout=") for arg in argv
    )
    LOG.warning(
        "Using built-in default timeout (%ss). "
        "CLI argv %s --timeout. "
        "If you intended unlimited/long runs, use ONE of: "
        "(1) single-line: python scripts/swe_bench/runner.py --timeout 0 --output predictions.jsonl ; "
        "(2) env: MANA_SWE_BENCH_TIMEOUT=0 python scripts/swe_bench/runner.py --output predictions.jsonl ; "
        "(3) file: echo 'timeout = 0' > .swe-bench/runner.toml ; "
        "(4) wrapper: bash scripts/swe_bench/run_unlimited.sh. "
        "Broken multi-line (no \\\\ after runner.py) silently drops flags.",
        DEFAULT_TIMEOUT_SECONDS,
        "included" if has_timeout_flag else "did not include",
    )


def _benchmark_config_overrides(
    agent_model: str,
    *,
    agent_provider: str = DEFAULT_AGENT_PROVIDER,
    timeout_seconds: int | None = None,
) -> dict[str, str | bool | int]:
    """Settings that must be forced for reliable non-interactive SWE-bench runs."""
    model = str(agent_model).strip()
    provider = str(agent_provider or DEFAULT_AGENT_PROVIDER).strip().lower() or DEFAULT_AGENT_PROVIDER
    primary = model if model.lower().startswith(provider + "/") else f"{provider}/{model}"
    overrides: dict[str, str | bool | int] = {
        # External supermemory (or other hosted memory) serializes many HTTP
        # calls during chat startup and can look like a hang; use local memory.
        # MANA_MEMORY_FALLBACK_TO_INTERNAL must stay false: true is a hard error
        # in MemoryConfig.validate (external→internal fallback is not implemented).
        "MANA_MEMORY_MODE": "internal",
        "MANA_MEMORY_PROVIDER": "mana",
        "MANA_MEMORY_FALLBACK_TO_INTERNAL": False,
        "MANA_MEMORY_SECRET_REF": "",
        # Pin provider + every common model role so operator MODEL_LEVEL_* /
        # MANA_MODEL_* preferences cannot rewrite the measured model mid-run.
        "MANA_AI_PROVIDER": provider,
        "OPENAI_CHAT_MODEL": model,
        "LLM_MODEL": model,
        "MANA_PRIMARY_MODEL": primary,
        "OPENAI_TOOL_WORKER_MODEL": model,
        "OPENAI_CODING_PLANNER_MODEL": model,
        "MODEL_LEVEL_1_FAST_TOOL": model,
        "MODEL_LEVEL_2_CODING": model,
        "MODEL_LEVEL_3_HIGH_REASONING": model,
        "MANA_MODEL_MAIN": model,
        "MANA_MODEL_HEAD_DECISION": model,
        "MANA_MODEL_PLANNER": model,
        "MANA_MODEL_CODING": model,
        "MANA_MODEL_VERIFIER": model,
        "MANA_MODEL_REVIEWER": model,
        "MANA_MODEL_TOOL": model,
        "MANA_MODEL_TOOL_WORKER": model,
        "MANA_MODEL_SUMMARIZER": model,
        # Coding-only surface: operator ~/.mana often enables desktop/server
        # integrations that dump 100+ irrelevant tools into SWE-bench logs and
        # dilute the coding agent.
        "MANA_BROWSER_ENABLED": False,
        "MANA_COMPUTER_CONTROL_ENABLED": False,
        "MANA_CANVAS_ENABLED": False,
        "MANA_SEARCH_ENABLE_WEB": False,
        "MANA_SEARCH_ENABLE_GITHUB": False,
        "MANA_ACP_ENABLED": False,
        "MANA_A2A_SERVER_ENABLED": False,
        "MANA_FLEET_ENABLED": False,
        "MANA_WORKER_GATEWAY_ENABLED": False,
        # Bench isolation: edit the SWE worktree in place (not a nested managed
        # worktree under ~/.mana/repositories) and auto-allow shell/git
        # transactional REQUIRE_APPROVAL outcomes so non-interactive runs do not
        # stall on human inbox grants. DENY (secrets, workspace escapes, etc.)
        # remains deny.
        "MANA_MANAGED_WORKTREES_ENABLED": False,
        "MANA_CODEX_WORKTREE_ISOLATION": False,
        "MANA_TRANSACTIONAL_ALWAYS_APPROVE": True,
    }
    if timeout_seconds is not None:
        # Keep Codex task timeout aligned with the runner wall clock.
        # Unlimited runner timeout → long finite Codex ceiling (no silent 1800s).
        overrides["MANA_CODEX_TASK_TIMEOUT_SECONDS"] = (
            int(timeout_seconds)
            if int(timeout_seconds) > 0
            else 7 * 24 * 60 * 60
        )
    return overrides


def _set_toml_table_key(path: Path, table: str, key: str, value: bool | str | int) -> None:
    """Set ``key`` inside a top-level ``[table]`` section (create section if missing)."""
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    header = f"[{table}]"
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, int):
        rendered = str(value)
    else:
        rendered = _toml_quote(str(value))
    assignment = f"{key} = {rendered}"
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(header)
        lines.append(assignment)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = j
            break
    replaced = False
    for j in range(start + 1, end):
        stripped = lines[j].strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        existing_key = stripped.split("=", 1)[0].strip()
        if existing_key == key:
            lines[j] = assignment
            replaced = True
            break
    if not replaced:
        lines.insert(end, assignment)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _disable_non_coding_integrations(config_path: Path) -> None:
    """Force nested integration tables off for SWE-bench isolated MANA_HOME."""
    for table in (
        "computer_control",
        "telegram",
        "teach",
        "experience_to_skill",
    ):
        _set_toml_table_key(config_path, table, "enabled", False)
    for table in ("media.image", "media.voice", "media.video"):
        # Nested dotted tables may appear as [media.image] in Mana config.
        _set_toml_table_key(config_path, table, "enabled", False)


def _seed_isolated_mana_home(
    target: Path,
    *,
    agent_model: str,
    agent_provider: str = DEFAULT_AGENT_PROVIDER,
    timeout_seconds: int | None = None,
) -> Path:
    """Create a per-instance MANA_HOME that still has user credentials/config.

    Isolation keeps run/session state out of the operator's primary ~/.mana while
    copying config.toml and secrets.toml so non-interactive chat can authenticate.

    Mana loads explicit file settings over process env, so this also rewrites the
    isolated config for internal memory, the selected provider, and the model.
    """
    import shutil

    target.mkdir(parents=True, exist_ok=True)
    source = operator_mana_home()
    if source.is_dir() and source != target.resolve():
        for name in ("config.toml", "secrets.toml"):
            src = source / name
            dst = target / name
            if src.is_file() and not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except OSError as exc:
                    LOG.warning("Could not copy %s into isolated MANA_HOME: %s", src, exc)
    config_path = target / "config.toml"
    try:
        _upsert_toml_keys(
            config_path,
            _benchmark_config_overrides(
                agent_model,
                agent_provider=agent_provider,
                timeout_seconds=timeout_seconds,
            ),
        )
        _disable_non_coding_integrations(config_path)
    except OSError as exc:
        LOG.warning("Could not apply SWE-bench isolation overrides to %s: %s", config_path, exc)
    return target


def _wait_with_heartbeat(
    proc: subprocess.Popen[str],
    *,
    timeout_seconds: int,
    stdout_path: Path,
    stderr_path: Path,
    heartbeat_seconds: int = 30,
    worktree: Path | None = None,
) -> tuple[int, bool]:
    """Wait for proc with periodic progress logs; return (returncode, timed_out).

    ``timeout_seconds <= 0`` means unlimited (no wall-clock kill).
    """
    started = time.monotonic()
    unlimited = int(timeout_seconds) <= 0
    deadline = None if unlimited else started + max(1, int(timeout_seconds))
    beat = max(5, int(heartbeat_seconds))
    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return (proc.returncode if proc.returncode is not None else 124), True
            wait_for = min(beat, remaining)
        else:
            wait_for = beat
        try:
            returncode = proc.wait(timeout=wait_for)
            return returncode, False
        except subprocess.TimeoutExpired:
            if worktree is not None and not worktree.is_dir():
                LOG.error(
                    "Worktree disappeared under live mana-agent (pid=%s, path=%s); "
                    "killing process group to avoid getcwd/Codex hang",
                    proc.pid,
                    worktree,
                )
                _kill_process_group(proc)
                return (
                    proc.returncode if proc.returncode is not None else 125
                ), False
            elapsed = time.monotonic() - started
            out_size = stdout_path.stat().st_size if stdout_path.exists() else 0
            err_size = stderr_path.stat().st_size if stderr_path.exists() else 0
            err_tail = _tail_text(stderr_path, max_chars=400).replace("\n", " | ")
            LOG.info(
                "mana-agent still running (elapsed=%.0fs / %s, pid=%s, stdout=%dB, stderr=%dB)%s",
                elapsed,
                format_timeout_label(timeout_seconds),
                proc.pid,
                out_size,
                err_size,
                f" stderr_tail={err_tail!r}" if err_tail.strip() else "",
            )


def run_mana_agent(
    *,
    worktree: Path,
    prompt: str,
    agent_model: str,
    agent_provider: str = DEFAULT_AGENT_PROVIDER,
    timeout_seconds: int,
    mana_argv: Sequence[str],
    run_dir: Path,
) -> tuple[int, str, str]:
    """Invoke mana-agent once, non-interactively, with a hard wall-clock timeout.

    Uses the chat CLI with a single prompt and closed stdin so the session
    exits after the coding turn (EOF / single-shot non-TTY), leaving patches in
    the worktree.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "mana_stdout.log"
    stderr_path = run_dir / "mana_stderr.log"
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    # Isolate mana state for this instance to avoid cross-run contamination, but
    # seed credentials from the operator's real MANA_HOME / ~/.mana and pin
    # provider + model + internal memory in the isolated config file.
    mana_home = _seed_isolated_mana_home(
        run_dir / "mana_home",
        agent_model=agent_model,
        agent_provider=agent_provider,
        timeout_seconds=timeout_seconds,
    )

    env = os.environ.copy()
    env["MANA_HOME"] = str(mana_home)
    # Also set env (fills gaps if a key is missing from the rewritten config).
    for key, value in _benchmark_config_overrides(
        agent_model,
        agent_provider=agent_provider,
        timeout_seconds=timeout_seconds,
    ).items():
        if isinstance(value, bool):
            env[key] = "true" if value else "false"
        else:
            env[key] = str(value)
    # Prefer non-interactive defaults.
    env.setdefault("TERM", "dumb")
    env["PYTHONUNBUFFERED"] = "1"
    # Keep chat console quieter; runner heartbeats + log files show progress.
    env.setdefault("MANA_CHAT_ANIMATION", "0")
    # Skip the 179-tool catalog dump that pollutes SWE-bench mana_stdout.log.
    env["MANA_CHAT_QUIET"] = "1"
    env.setdefault("MANA_CHAT_UI", "plain")
    # Ensure bare `python` in agent shell tools is Python 3 (not host Python 2.7).
    prepare_agent_python_path(run_dir=run_dir, env=env)

    # Per-step agent timeout should leave headroom under the hard wall clock.
    # 0 = unlimited (mana-agent normalizes to a long finite ceiling; no 600s cap).
    if int(timeout_seconds) <= 0:
        agent_step_timeout = 0
    else:
        agent_step_timeout = max(
            30, min(int(timeout_seconds), max(30, int(timeout_seconds) - 30))
        )

    cmd = [
        *mana_argv,
        "--no-interactive",
        "--no-banner",
        "chat",
        "--no-tui",
        "--root-dir",
        str(worktree),
        "--model",
        agent_model,
        "--full-auto",
        # Do NOT use --ephemeral-index: it builds a full semantic index
        # synchronously and freezes for large SWE-bench repos (e.g. astropy).
        # Skip auto-index so chat starts immediately with direct project search.
        "--no-auto-index-missing",
        "--no-coding-memory",
        "--auto-continue",
        "--execution-profile",
        "full-auto",
        "--auto-execute-max-passes",
        "10",
        "--agent-timeout-seconds",
        str(agent_step_timeout),
        # Single coding task prompt (positional).
        prompt,
    ]

    LOG.info(
        "Starting mana-agent (timeout=%s, provider=%s, model=%s, root=%s, mana_home=%s, agent_timeout=%s)",
        format_timeout_label(timeout_seconds),
        agent_provider,
        agent_model,
        worktree,
        mana_home,
        format_timeout_label(agent_step_timeout),
    )
    LOG.debug("Command: %s", " ".join(shlex.quote(c) for c in cmd))
    (run_dir / "mana_cmd.txt").write_text(
        " ".join(shlex.quote(c) for c in cmd) + "\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    # Line-buffered logs so partial progress is visible while the agent runs.
    with stdout_path.open("w", encoding="utf-8", buffering=1) as out_f, stderr_path.open(
        "w", encoding="utf-8", buffering=1
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

        LOG.info("mana-agent spawned pid=%s", proc.pid)
        returncode, timed_out = _wait_with_heartbeat(
            proc,
            timeout_seconds=timeout_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            heartbeat_seconds=30,
            worktree=worktree,
        )
        if timed_out:
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
    LOG.info(
        "mana-agent finished returncode=%s elapsed=%.1fs timed_out=%s",
        returncode,
        elapsed,
        timed_out,
    )
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
    worktree_lock: Path | None = None
    try:
        worktree_lock = acquire_worktree_lock(worktrees_dir, instance_id)
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
                    agent_provider=cfg.agent_provider,
                    timeout_seconds=cfg.timeout_seconds,
                    mana_argv=mana_argv,
                    run_dir=run_dir,
                )
            except subprocess.TimeoutExpired:
                rc, err = 124, "timeout"
            if rc == 124:
                status = "timeout"
                error = f"mana-agent exceeded hard timeout of {cfg.timeout_seconds}s"
            elif rc == 125:
                status = "agent_error"
                error = (
                    f"mana-agent worktree disappeared during the run: {worktree}"
                )
            elif rc != 0:
                status = "agent_error"
                error = f"mana-agent exited with code {rc}"
                if err:
                    error = f"{error}: {err[:500]}"
            else:
                status = "ok"
                error = ""

            if worktree is not None and worktree.is_dir():
                change_summary = summarize_worktree_changes(worktree)
                patch, reject_reason = capture_model_patch(
                    worktree, exclude_test_files=cfg.exclude_test_files
                )
            else:
                change_summary = WorktreeChangeSummary()
                patch, reject_reason = "", "worktree missing after agent run"
            if reject_reason:
                status = "destructive_patch"
                error = reject_reason
                patch = ""
            elif status == "ok" and change_summary.total == 0:
                # Explicit diagnostics when agent exits 0 with a clean tree.
                LOG.warning(
                    "No worktree changes after agent for %s "
                    "(deleted=%s modified=%s added=%s)",
                    instance_id,
                    change_summary.deleted,
                    change_summary.modified,
                    change_summary.added,
                )

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
        release_worktree_lock(worktree_lock)


def parse_instance_ids(values: Sequence[str] | None) -> list[str]:
    """Parse CLI ``--instance-ids`` values (repeatable and/or comma-separated)."""
    if not values:
        return []
    ids: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                ids.append(part)
    return ids


def load_instance_ids_file(path: Path) -> list[str]:
    """Load instance ids from a text or JSONL file.

    Accepted formats:

    * One ``instance_id`` per line (``#`` comments and blank lines ignored).
    * JSONL rows with an ``instance_id`` field (e.g. a predictions file).
    * A JSON array of strings.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SweBenchRunnerError(f"--instance-ids-file not found: {resolved}")
    text = resolved.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []

    # JSON array of strings.
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SweBenchRunnerError(
                f"Invalid JSON array in --instance-ids-file {path}: {exc}"
            ) from exc
        if not isinstance(data, list):
            raise SweBenchRunnerError(
                f"--instance-ids-file {path}: expected a JSON array of strings"
            )
        return [str(item).strip() for item in data if str(item).strip()]

    ids: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SweBenchRunnerError(
                    f"Invalid JSONL on line {line_no} of {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise SweBenchRunnerError(
                    f"--instance-ids-file {path} line {line_no}: expected a JSON object"
                )
            iid = str(row.get("instance_id") or "").strip()
            if iid:
                ids.append(iid)
            continue
        # Plain id (allow optional trailing comma from copied lists).
        ids.append(line.rstrip(",").strip())
    return [i for i in ids if i]


def resolve_requested_instance_ids(
    *,
    cli_values: Sequence[str] | None,
    ids_file: Path | None,
) -> list[str]:
    """Merge CLI and file-sourced instance ids (empty → run all dataset ids)."""
    ids = parse_instance_ids(cli_values)
    if ids_file is not None:
        file_ids = load_instance_ids_file(ids_file)
        LOG.info(
            "Loaded %d instance id(s) from --instance-ids-file %s",
            len(file_ids),
            ids_file,
        )
        ids.extend(file_ids)
    # De-dupe while preserving first-seen order.
    seen: set[str] = set()
    ordered: list[str] = []
    for iid in ids:
        if iid not in seen:
            seen.add(iid)
            ordered.append(iid)
    return ordered


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SWE-bench Verified predictions.jsonl using mana-agent. "
            "With no --instance-ids, loads and runs ALL ids from the dataset "
            "split (full Verified suite). With --instance-ids, runs only those. "
            "Each prediction line includes instance_id, model_name_or_path "
            f"(default {DEFAULT_AGENT_NAME}__<llm>), agent_name "
            f"(default {DEFAULT_AGENT_NAME}), optional agent_model, and model_patch."
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
        help=(
            "Maximum number of instances to run after selection. "
            "Applied after --instance-ids / --instance-ids-file. "
            "Omit to run every selected id (all dataset ids when no id filter)."
        ),
    )
    parser.add_argument(
        "--instance-ids",
        action="append",
        default=None,
        help=(
            "Run only these instance_id values (repeatable or comma-separated). "
            "If omitted (and --instance-ids-file is also omitted), every id from "
            "the SWE-bench dataset split is selected. "
            "Example: --instance-ids astropy__astropy-12907"
        ),
    )
    parser.add_argument(
        "--instance-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional file of instance ids: one id per line, a JSON string array, "
            "or JSONL rows with instance_id (e.g. predictions.jsonl). "
            "Combined with --instance-ids. When neither is set, all dataset ids run."
        ),
    )
    parser.add_argument(
        "--list-instance-ids",
        action="store_true",
        help=(
            "Print the selected instance ids (one per line) after applying "
            "--instance-ids / --instance-ids-file / --limit, then exit without "
            "running mana-agent. With no id filter, prints all dataset ids."
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
        default=None,
        help=(
            "LLM id for mana-agent during the run. When omitted, uses "
            "MANA_PRIMARY_MODEL / OPENAI_CHAT_MODEL / LLM_MODEL from "
            "~/.mana/config.toml (or $MANA_HOME/config.toml). "
            "Also used to compose default model_name_or_path."
        ),
    )
    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "Inference provider for mana-agent (e.g. nvidia, openai, openrouter). "
            "When omitted, uses MANA_AI_PROVIDER from ~/.mana/config.toml. "
            "If you pass --model without --provider, the configured provider is used."
        ),
    )
    parser.add_argument(
        "--agent-name",
        default=DEFAULT_AGENT_NAME,
        help=(
            "Agent identity written as agent_name and used in the default "
            f"model_name_or_path (default: {DEFAULT_AGENT_NAME})."
        ),
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help=(
            "Value written into predictions as model_name_or_path. "
            f"Default: {{agent_name}}__{{provider}}__{{model}} when provider is not "
            f"openai, else {{agent_name}}__{{model}}. "
            "Do not set this to the agent name alone."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            f"Hard per-instance wall-clock timeout in seconds "
            f"(default: {DEFAULT_TIMEOUT_SECONDS}, or MANA_SWE_BENCH_TIMEOUT / "
            "SWE_BENCH_TIMEOUT). Use 0 for unlimited. Put flags on the same line "
            "as runner.py (or end the line with \\) so the shell does not drop them."
        ),
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
        "--keep-test-files",
        action="store_true",
        help=(
            "Keep test-file hunks in model_patch. Default strips them because "
            "SWE-bench applies the official test_patch after model_patch; agent "
            "test edits often produce harness status 'failed'."
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
    model_name_or_path = resolve_model_name_or_path(cfg)
    LOG.info("mana-agent argv: %s", mana_argv)
    LOG.info("Agent name: %s", cfg.agent_name)
    LOG.info(
        "Agent provider: %s (source: %s)",
        cfg.agent_provider,
        cfg.provider_source,
    )
    LOG.info(
        "Agent model (LLM): %s (source: %s)",
        cfg.agent_model,
        cfg.model_source,
    )
    LOG.info("Predictions model_name_or_path: %s", model_name_or_path)
    LOG.info("Exclude test files from model_patch: %s", cfg.exclude_test_files)
    LOG.info(
        "Per-instance timeout: %s (source: %s)",
        format_timeout_label(cfg.timeout_seconds),
        cfg.timeout_source,
    )

    instances = load_instances(
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        limit=cfg.limit,
        instance_ids=cfg.instance_ids,
    )
    if not instances:
        raise SweBenchRunnerError(
            "No instances selected. Check --limit / --instance-ids / "
            "--instance-ids-file (omit id filters to run all dataset ids)."
        )

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
            model_name_or_path=model_name_or_path,
            agent_name=cfg.agent_name,
            agent_model=cfg.agent_model,
            agent_provider=cfg.agent_provider,
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
                    "agent_name": cfg.agent_name,
                    "agent_provider": cfg.agent_provider,
                    "agent_model": cfg.agent_model,
                    "model_name_or_path": model_name_or_path,
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

    _write_run_summary(
        work_dir / "run_summary.json",
        results,
        output,
        agent_name=cfg.agent_name,
        agent_provider=cfg.agent_provider,
        agent_model=cfg.agent_model,
        model_name_or_path=model_name_or_path,
    )
    LOG.info("Wrote %d prediction(s) to %s", len(results), output)
    LOG.info(
        "Prediction identity: agent_name=%s agent_provider=%s agent_model=%s "
        "model_name_or_path=%s",
        cfg.agent_name,
        cfg.agent_provider,
        cfg.agent_model,
        model_name_or_path,
    )
    return results


def _write_run_summary(
    path: Path,
    results: Iterable[InstanceResult],
    output: Path,
    *,
    agent_name: str,
    agent_provider: str,
    agent_model: str,
    model_name_or_path: str,
) -> None:
    rows = list(results)
    summary = {
        "predictions_path": str(output),
        "agent_name": agent_name,
        "agent_provider": agent_provider,
        "agent_model": agent_model,
        "model_name_or_path": model_name_or_path,
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
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)
    configure_logging(bool(args.verbose))
    # Always show what the process actually received (catches broken multi-line shells).
    LOG.info("Process argv: %s", " ".join(shlex.quote(a) for a in raw_argv) or "(empty)")

    try:
        requested_ids = resolve_requested_instance_ids(
            cli_values=args.instance_ids,
            ids_file=args.instance_ids_file,
        )
    except SweBenchRunnerError as exc:
        LOG.error("%s", exc)
        return 2

    if not requested_ids:
        LOG.info(
            "No instance ids entered via --instance-ids / --instance-ids-file; "
            "will load all ids from SWE-bench dataset %s split=%s",
            args.dataset,
            args.split,
        )
    else:
        LOG.info(
            "Using %d explicit instance id(s); only those will run",
            len(requested_ids),
        )

    if bool(args.list_instance_ids):
        try:
            instances = load_instances(
                dataset_name=str(args.dataset),
                split=str(args.split),
                limit=args.limit,
                instance_ids=requested_ids,
            )
        except SweBenchRunnerError as exc:
            LOG.error("%s", exc)
            return 2
        for row in instances:
            iid = str(row.get("instance_id") or "").strip()
            if iid:
                print(iid)
        return 0 if instances else 2

    explicit_model_name = getattr(args, "model_name_or_path", None)
    provider, model, provider_source, model_source = resolve_agent_inference(
        cli_model=getattr(args, "model", None),
        cli_provider=getattr(args, "provider", None),
    )
    work_dir = Path(args.work_dir)
    try:
        timeout_seconds, timeout_source = resolve_runner_timeout_seconds(
            getattr(args, "timeout", None),
            work_dir=work_dir,
        )
    except SweBenchRunnerError as exc:
        LOG.error("%s", exc)
        return 2
    warn_if_timeout_likely_dropped(timeout_source=timeout_source, argv=raw_argv)
    cfg = RunnerConfig(
        dataset_name=str(args.dataset),
        split=str(args.split),
        limit=args.limit,
        instance_ids=requested_ids,
        output=Path(args.output),
        work_dir=Path(args.work_dir),
        agent_name=str(args.agent_name).strip() or DEFAULT_AGENT_NAME,
        agent_provider=provider,
        agent_model=model,
        model_source=model_source,
        provider_source=provider_source,
        model_name_or_path=(
            str(explicit_model_name).strip() if explicit_model_name else None
        ),
        timeout_seconds=timeout_seconds,
        timeout_source=timeout_source,
        mana_bin=args.mana_bin,
        retain_worktrees=bool(args.retain_worktrees),
        skip_agent=bool(args.skip_agent),
        exclude_test_files=not bool(args.keep_test_files),
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
