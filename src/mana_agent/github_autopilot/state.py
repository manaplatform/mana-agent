from __future__ import annotations

import threading
import uuid
import hashlib
from contextlib import contextmanager
from pathlib import Path

from mana_agent.workspaces.paths import mana_home
from mana_agent.workspaces.store import atomic_write_json

from .models import DeliveryReceipt, GitHubJob, JobState, now_iso


class GitHubAutopilotStore:
    """Durable file-backed delivery/job store with atomic process-local updates."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or mana_home() / "github-autopilot").resolve()
        self.deliveries = self.root / "deliveries"
        self.jobs = self.root / "jobs"
        self.sessions = self.root / "sessions"
        self.locks = self.root / "locks"
        for path in (self.deliveries, self.jobs, self.sessions, self.locks):
            path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @contextmanager
    def _delivery_lock(self):
        lock_path = self.locks / "deliveries.lock"
        with lock_path.open("a+") as handle:
            try:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass
            try:
                yield
            finally:
                try:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    pass

    @contextmanager
    def subject_lock(self, semantic_subject: str):
        digest = hashlib.sha256(semantic_subject.encode("utf-8")).hexdigest()
        lock_path = self.locks / f"subject-{digest}.lock"
        with lock_path.open("a+") as handle:
            try:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass
            try:
                yield
            finally:
                try:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    pass

    @staticmethod
    def _safe(value: str) -> str:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in value):
            raise ValueError("invalid persistent identifier")
        return value

    def delivery_path(self, delivery_id: str) -> Path:
        return self.deliveries / f"{self._safe(delivery_id)}.json"

    def job_path(self, job_id: str) -> Path:
        return self.jobs / f"{self._safe(job_id)}.json"

    def get_delivery(self, delivery_id: str) -> DeliveryReceipt | None:
        path = self.delivery_path(delivery_id)
        return DeliveryReceipt.model_validate_json(path.read_text()) if path.exists() else None

    def accept(self, receipt: DeliveryReceipt, job: GitHubJob | None) -> DeliveryReceipt:
        with self._lock:
            with self._delivery_lock():
                existing = self.get_delivery(receipt.delivery_id)
                if existing is not None:
                    return existing
                if job is not None:
                    self.save_job(job)
                atomic_write_json(self.delivery_path(receipt.delivery_id), receipt.model_dump(mode="json"))
                return receipt

    def save_job(self, job: GitHubJob) -> GitHubJob:
        job.updated_at = now_iso()
        atomic_write_json(self.job_path(job.job_id), job.model_dump(mode="json"))
        return job

    def create_job_id(self) -> str:
        return f"ghjob_{uuid.uuid4().hex[:20]}"

    def get_job(self, job_id: str) -> GitHubJob:
        return GitHubJob.model_validate_json(self.job_path(job_id).read_text())

    def list_jobs(self) -> list[GitHubJob]:
        rows: list[GitHubJob] = []
        for path in self.jobs.glob("*.json"):
            try:
                rows.append(GitHubJob.model_validate_json(path.read_text()))
            except Exception:
                continue
        return sorted(rows, key=lambda item: item.created_at, reverse=True)

    def find_by_semantic_key(self, key: str) -> GitHubJob | None:
        return next((job for job in self.list_jobs() if job.semantic_key == key and job.state not in {JobState.CANCELLED}), None)

    def find_session_job(self, session_id: str) -> GitHubJob | None:
        candidates = [job for job in self.list_jobs() if job.session_id == session_id]
        return candidates[0] if candidates else None

    def update_state(self, job_id: str, state: JobState, *, failure_type: str = "", detail: str = "") -> GitHubJob:
        with self._lock:
            job = self.get_job(job_id)
            job.state = state
            job.failure_type = failure_type
            job.failure_detail = detail[:4000]
            return self.save_job(job)

    def queued_jobs(self) -> list[GitHubJob]:
        return [job for job in self.list_jobs() if job.state == JobState.QUEUED]
