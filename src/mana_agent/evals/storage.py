from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import EvalRun, RunStatus, SCHEMA_VERSION
from .redaction import redact, redact_text

if os.name == "nt":  # pragma: no cover
    import msvcrt
else:  # pragma: no cover
    import fcntl


class EvalStorageError(RuntimeError):
    pass


class IncompatibleSchemaError(EvalStorageError):
    pass


class CompletedRunImmutableError(EvalStorageError):
    pass


def atomic_write(path: Path, content: str) -> None:
    content = redact_text(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_json(run: EvalRun) -> str:
    return json.dumps(redact(run.model_dump(mode="json")), indent=2, sort_keys=True, default=str) + "\n"


class EvalStorage:
    """Canonical portable artifacts plus a disposable, locked SQLite index."""

    def __init__(self, root: str | Path = ".mana/evals") -> None:
        self.root = Path(root).expanduser().resolve()
        self.runs_dir = self.root / "runs"
        self.experiments_dir = self.root / "experiments"
        self.reports_dir = self.root / "reports"
        self.baselines_dir = self.root / "baselines"
        self.index_path = self.root / "evals.sqlite3"
        self.lock_path = self.root / "evals.lock"
        self._thread_lock = threading.RLock()
        for directory in (self.runs_dir, self.experiments_dir, self.reports_dir, self.baselines_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._initialize_index()

    def run_dir(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise EvalStorageError("invalid run id")
        return self.runs_dir / run_id

    @contextmanager
    def locked_index(self) -> Iterator[sqlite3.Connection]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, self.lock_path.open("a+b") as guard:
            if os.name == "nt":  # pragma: no cover
                guard.seek(0)
                if not guard.read(1):
                    guard.write(b"0")
                    guard.flush()
                guard.seek(0)
                msvcrt.locking(guard.fileno(), msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
            connection = sqlite3.connect(self.index_path)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
                connection.commit()
            finally:
                connection.close()
                if os.name == "nt":  # pragma: no cover
                    guard.seek(0)
                    msvcrt.locking(guard.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover
                    fcntl.flock(guard.fileno(), fcntl.LOCK_UN)

    def _initialize_index(self) -> None:
        with self.locked_index() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_fingerprint TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    suite_name TEXT NOT NULL DEFAULT '',
                    suite_version TEXT NOT NULL DEFAULT '',
                    score REAL,
                    task_success INTEGER,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    artifact_path TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_runs_fingerprint ON runs(run_fingerprint, status);
                """
            )
            row = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if row and int(row[0]) != SCHEMA_VERSION:
                raise IncompatibleSchemaError(
                    f"evaluation index schema {row[0]} is incompatible with {SCHEMA_VERSION}; rebuild the disposable index"
                )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def create_run(self, run: EvalRun, *, config: dict[str, Any], environment: dict[str, Any]) -> Path:
        run_dir = self.run_dir(run.run_id)
        if run_dir.exists():
            raise EvalStorageError(f"run already exists: {run.run_id}")
        run_dir.mkdir(parents=True)
        atomic_write(run_dir / "run.json", _run_json(run))
        atomic_write(run_dir / "config.yaml", _yaml_dump(redact(config)))
        atomic_write(run_dir / "environment.json", json.dumps(redact(environment), indent=2, sort_keys=True, default=str) + "\n")
        (run_dir / "events.jsonl").touch(exist_ok=False)
        self.index_run(run)
        return run_dir

    def finalize_run(self, run: EvalRun, artifacts: dict[str, Any] | None = None) -> Path:
        run_dir = self.run_dir(run.run_id)
        existing = self.load_run(run.run_id, allow_incomplete=True)
        if existing.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            raise CompletedRunImmutableError(f"completed run is immutable: {run.run_id}")
        for name, value in (artifacts or {}).items():
            target = run_dir / name
            if target.exists():
                raise CompletedRunImmutableError(f"artifact already exists: {target}")
            if isinstance(value, str):
                content = redact_text(value)
            else:
                content = json.dumps(redact(value), indent=2, sort_keys=True, default=str) + "\n"
            atomic_write(target, content)
        atomic_write(run_dir / "run.json", _run_json(run))
        atomic_write(run_dir / ".complete", f"schema_version={SCHEMA_VERSION}\n")
        self.index_run(run)
        return run_dir

    def load_run(self, run_id: str, *, allow_incomplete: bool = False) -> EvalRun:
        run_dir = self.run_dir(run_id)
        path = run_dir / "run.json"
        if not path.exists():
            raise EvalStorageError(f"run not found: {run_id}")
        run = EvalRun.model_validate_json(path.read_text(encoding="utf-8"))
        if run.schema_version != SCHEMA_VERSION:
            raise IncompatibleSchemaError(f"run schema {run.schema_version} is incompatible with {SCHEMA_VERSION}")
        if not allow_incomplete and not (run_dir / ".complete").exists():
            raise EvalStorageError(f"run is incomplete: {run_id}")
        return run

    def list_runs(self, *, suite: str | None = None, experiment_id: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if suite:
            clauses.append("suite_name = ?")
            values.append(suite)
        if experiment_id:
            clauses.append("experiment_id = ?")
            values.append(experiment_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.locked_index() as connection:
            rows = connection.execute(f"SELECT * FROM runs{where} ORDER BY started_at DESC", values).fetchall()
        return [dict(row) for row in rows]

    def successful_fingerprint(self, fingerprint: str) -> EvalRun | None:
        with self.locked_index() as connection:
            row = connection.execute(
                "SELECT run_id FROM runs WHERE run_fingerprint=? AND status='completed' AND task_success=1 ORDER BY completed_at DESC LIMIT 1",
                (fingerprint,),
            ).fetchone()
        return self.load_run(str(row[0])) if row else None

    def index_run(self, run: EvalRun, *, suite_name: str = "", suite_version: str = "") -> None:
        outcome = run.outcome
        with self.locked_index() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO runs(
                    run_id, run_fingerprint, experiment_id, task_id, variant_id, trial_number,
                    status, suite_name, suite_version, score, task_success, started_at,
                    completed_at, artifact_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id, run.run_fingerprint, run.experiment_id, redact_text(run.task_id), redact_text(run.variant_id),
                    run.trial_number, run.status.value, redact_text(suite_name), redact_text(suite_version),
                    outcome.normalized_score if outcome else None,
                    int(outcome.task_success) if outcome else None,
                    run.started_at.isoformat(), run.completed_at.isoformat() if run.completed_at else None,
                    str(self.run_dir(run.run_id)),
                ),
            )

    def recover_incomplete(self) -> list[str]:
        incomplete: list[str] = []
        for directory in self.runs_dir.iterdir():
            if directory.is_dir() and (directory / "run.json").exists() and not (directory / ".complete").exists():
                incomplete.append(directory.name)
        return sorted(incomplete)


def _yaml_dump(value: Any) -> str:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise EvalStorageError("PyYAML is required to persist evaluation configuration") from exc
    return yaml.safe_dump(value, sort_keys=True, allow_unicode=True)
