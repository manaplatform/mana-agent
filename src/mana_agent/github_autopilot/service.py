from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from mana_agent.config.settings import Settings
from mana_agent.integrations.codex import CodexCodingAgentShim
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.health import check_codex_health
from mana_agent.workspaces.paths import mana_home

from .config import GitHubAutopilotSettings
from .github_client import GitHubApiError, GitHubClient
from .installation_auth import InstallationAuthenticator, create_app_jwt
from .models import DeliveryReceipt, GitHubJob, JobState, now_iso
from .prompts import build_task_prompt, pull_request_body
from .repository import RepositoryManager
from .router import route_event
from .security import sanitize_event_context
from .state import GitHubAutopilotStore

logger = logging.getLogger(__name__)
_PERMISSION_RANK = {"none": 0, "read": 1, "triage": 2, "write": 3, "maintain": 4, "admin": 5}
_TRANSIENT = {0, 429, 500, 502, 503, 504}


class GitHubAutopilotService:
    def __init__(self, settings: GitHubAutopilotSettings, *, store: GitHubAutopilotStore | None = None, client: GitHubClient | None = None, repository_manager: RepositoryManager | None = None, codex_factory: Any | None = None) -> None:
        self.settings = settings
        self.store = store or GitHubAutopilotStore()
        self.client = client or GitHubClient(settings.api_url)
        self.auth = InstallationAuthenticator(settings, self.client)
        self.repositories = repository_manager or RepositoryManager()
        self.codex_factory = codex_factory or self._codex
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._subject_locks: dict[str, asyncio.Lock] = {}
        self.metrics: Counter[str] = Counter()

    async def start(self) -> None:
        if self._workers:
            return
        for job in self.store.queued_jobs():
            await self.queue.put(job.job_id)
        self._workers = [asyncio.create_task(self._worker(index), name=f"github-autopilot-{index}") for index in range(self.settings.worker_concurrency)]

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def accept(self, delivery_id: str, event_name: str, payload: dict[str, Any]) -> DeliveryReceipt:
        existing = self.store.get_delivery(delivery_id)
        if existing is not None:
            self.metrics["delivery.duplicate"] += 1
            return existing
        decision = route_event(event_name, payload, self.settings, delivery_id)
        if not decision.execute:
            receipt = DeliveryReceipt(delivery_id=delivery_id, event_name=event_name, accepted=True, result="ignored", reason=decision.reason)
            self.store.accept(receipt, None)
            self.metrics[f"delivery.ignored.{decision.reason}"] += 1
            return receipt
        repository = payload.get("repository") or {}
        installation = payload.get("installation") or {}
        if not installation.get("id") or not repository.get("id") or not repository.get("full_name"):
            receipt = DeliveryReceipt(delivery_id=delivery_id, event_name=event_name, accepted=True, result="ignored", reason="missing_required_identity")
            self.store.accept(receipt, None)
            return receipt
        action = str(payload.get("action") or "")
        trigger_id = self._trigger_id(payload)
        semantic = self._semantic_key(int(installation["id"]), int(repository["id"]), decision.subject_type, decision.subject_number, action, trigger_id, self._target_sha(payload))
        session_id = f"github:{installation['id']}:{repository['id']}:{decision.subject_type}:{decision.subject_number}"
        prior = None
        if decision.subject_type == "pull_request" and decision.subject_number is not None:
            prior = next((item for item in self.store.list_jobs() if item.pull_request_number == decision.subject_number), None)
        prior = prior or self.store.find_session_job(session_id)
        if prior and decision.subject_type == "pull_request":
            feedback = sanitize_event_context(self._event_context(event_name, payload), secret_alert=False)
            active = prior.state in {JobState.RUNNING, JobState.INVESTIGATING, JobState.IMPLEMENTING, JobState.VERIFYING}
            if active:
                job = prior.model_copy(deep=True, update={"job_id": self.store.create_job_id(), "delivery_id": delivery_id, "semantic_key": semantic, "event_name": event_name, "action": action, "sender_login": str((payload.get("sender") or {}).get("login") or ""), "sender_type": str((payload.get("sender") or {}).get("type") or ""), "subject_type": decision.subject_type, "subject_number": decision.subject_number, "trigger_id": trigger_id, "base_branch": self._base_branch(payload), "target_sha": self._target_sha(payload), "route_decision": decision, "state": JobState.QUEUED, "failure_type": "", "failure_detail": "", "feedback": [*prior.feedback, feedback], "context": {**prior.context, "latest_review_feedback": feedback}, "created_at": now_iso(), "updated_at": now_iso()})
            else:
                job = prior
                job.feedback.append(feedback)
                job.context["latest_review_feedback"] = feedback
                job.state = JobState.QUEUED
                job.delivery_id = delivery_id
                job.semantic_key = semantic
                job.event_name = event_name
                job.action = action
                job.sender_login = str((payload.get("sender") or {}).get("login") or "")
                job.sender_type = str((payload.get("sender") or {}).get("type") or "")
                job.trigger_id = trigger_id
                job.route_decision = decision
                job.subject_type = decision.subject_type
                job.subject_number = decision.subject_number
                job.base_branch = self._base_branch(payload)
                job.target_sha = self._target_sha(payload)
        else:
            same = self.store.find_by_semantic_key(semantic)
            if same:
                same.feedback.append(sanitize_event_context(self._event_context(event_name, payload)))
                same.state = JobState.QUEUED
                job = same
            else:
                context = sanitize_event_context(self._event_context(event_name, payload), secret_alert=event_name == "secret_scanning_alert")
                job = GitHubJob(job_id=self.store.create_job_id(), delivery_id=delivery_id, semantic_key=semantic, event_name=event_name, action=action, installation_id=int(installation["id"]), repository_id=int(repository["id"]), repository_full_name=str(repository["full_name"]), sender_login=str((payload.get("sender") or {}).get("login") or ""), sender_type=str((payload.get("sender") or {}).get("type") or ""), subject_type=decision.subject_type, subject_number=decision.subject_number, trigger_id=trigger_id, base_branch=self._base_branch(payload), target_sha=self._target_sha(payload), context=context, route_decision=decision, session_id=session_id, state=JobState.QUEUED)
        receipt = DeliveryReceipt(delivery_id=delivery_id, event_name=event_name, accepted=True, job_id=job.job_id, result="queued")
        accepted = self.store.accept(receipt, job)
        if accepted.job_id == job.job_id:
            await self.queue.put(job.job_id)
        self.metrics[f"delivery.queued.{event_name}"] += 1
        return accepted

    async def retry(self, job_id: str) -> GitHubJob:
        job = self.store.get_job(job_id)
        if job.state != JobState.FAILED or job.failure_type not in {"rate_limited", "repository_unavailable", "agent_unavailable", "push_failure", "pull_request_failure"}:
            raise ValueError("only transiently failed jobs may be retried")
        if job.retry_count >= 3:
            raise ValueError("job retry limit reached")
        job.retry_count += 1
        job.state = JobState.QUEUED
        self.store.save_job(job)
        await self.queue.put(job_id)
        return job

    def cancel(self, job_id: str) -> GitHubJob:
        job = self.store.get_job(job_id)
        if job.state in {JobState.COMPLETED, JobState.CANCELLED}:
            return job
        return self.store.update_state(job_id, JobState.CANCELLED, failure_type="cancelled", detail="Cancelled by an authorized operator")

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "enabled": self.settings.enabled, "queue_depth": self.queue.qsize(), "workers": len([task for task in self._workers if not task.done()])}

    def readiness(self) -> dict[str, Any]:
        errors = self.settings.startup_errors()
        if not self.store.root.is_dir():
            errors.append("queue persistence is unavailable")
        return {"ready": not errors and bool(self._workers), "errors": errors, **self.health()}

    def metrics_snapshot(self) -> dict[str, Any]:
        states = Counter(job.state.value for job in self.store.list_jobs())
        return {"deliveries": dict(self.metrics), "jobs_by_state": dict(states), "queue_depth": self.queue.qsize(), "active_repository_locks": sum(lock.locked() for lock in self._subject_locks.values())}

    def doctor(self, *, authenticate: bool = True) -> dict[str, Any]:
        codex_settings = CodexSettings.from_mana_settings(Settings())
        try:
            with tempfile.NamedTemporaryFile(dir=self.store.root, prefix=".doctor-", delete=True):
                queue_writable = True
        except OSError:
            queue_writable = False
        checks: dict[str, Any] = {"configuration": self.settings.startup_errors(), "queue_persistence": queue_writable, "worker_configuration": self.settings.worker_concurrency > 0, "repository_storage": mana_home().is_dir(), "worktree_support": subprocess.run(["git", "--version"], capture_output=True, timeout=5, check=False).returncode == 0, "codex": check_codex_health(codex_settings, mana_home()).model_dump(mode="json")}
        if authenticate and not checks["configuration"] and self.settings.private_key_path:
            try:
                jwt = create_app_jwt(self.settings.app_id, self.settings.private_key_path.read_bytes())
                app = self.client.app(jwt)
                checks["github_authentication"] = {"ok": True, "slug": app.get("slug", "")}
                checks["permissions"] = app.get("permissions", {})
                required = {"contents": "write", "issues": "write", "pull_requests": "write", "actions": "read", "checks": "read", "dependabot_alerts": "read", "code_scanning_alerts": "read", "secret_scanning_alerts": "read"}
                rank = {"none": 0, "read": 1, "write": 2}
                checks["missing_permissions"] = [name for name, level in required.items() if rank.get(str(app.get("permissions", {}).get(name) or "none"), 0) < rank[level]]
            except Exception as exc:
                checks["github_authentication"] = {"ok": False, "error": type(exc).__name__}
        checks["ok"] = not checks["configuration"] and checks["queue_persistence"] and checks["repository_storage"] and checks["worktree_support"] and checks["codex"]["healthy"] and not checks.get("missing_permissions", []) and checks.get("github_authentication", {"ok": True}).get("ok", False)
        return checks

    async def _worker(self, _index: int) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                job = self.store.get_job(job_id)
                if job.state != JobState.QUEUED:
                    continue
                if job.route_decision.trigger == "cancellation":
                    await asyncio.wait_for(asyncio.to_thread(self._process, job_id), timeout=self.settings.maximum_job_runtime)
                else:
                    lock = self._subject_locks.setdefault(job.session_id, asyncio.Lock())
                    async with lock:
                        await asyncio.wait_for(asyncio.to_thread(self._process_locked, job_id), timeout=self.settings.maximum_job_runtime)
                finished = self.store.get_job(job_id)
                if finished.state == JobState.FAILED and finished.failure_type in {"rate_limited", "repository_unavailable", "agent_unavailable", "push_failure", "pull_request_failure"} and finished.retry_count < 3:
                    finished.retry_count += 1
                    finished.state = JobState.QUEUED
                    self.store.save_job(finished)
                    delay = min(60, 2 ** finished.retry_count)
                    asyncio.get_running_loop().call_later(delay, self.queue.put_nowait, job_id)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                self.store.update_state(job_id, JobState.FAILED, failure_type="agent_unavailable", detail="Maximum job runtime exceeded")
            except Exception as exc:
                logger.error("GitHub Autopilot job failed", extra={"job_id": job_id, "error_type": type(exc).__name__})
            finally:
                self.queue.task_done()

    def _process(self, job_id: str) -> None:
        job = self.store.update_state(job_id, JobState.RUNNING)
        try:
            token = self.auth.token(job.installation_id, job.repository_id)
            self._authorize(job, token)
            if job.route_decision.trigger == "cancellation":
                for candidate in self.store.list_jobs():
                    if candidate.job_id != job_id and candidate.session_id == job.session_id and candidate.state not in {JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED}:
                        self.store.update_state(candidate.job_id, JobState.CANCELLED, failure_type="cancelled", detail=f"Cancelled by delivery {job.delivery_id}")
                self.store.update_state(job_id, JobState.COMPLETED)
                return
            if job.subject_type in {"issue", "pull_request"} and job.subject_number:
                try:
                    self.client.comment(job.repository_full_name, job.subject_number, token, f"Mana accepted this task (`{job.job_id}`) and queued an isolated Codex coding session.")
                except GitHubApiError as exc:
                    if exc.status not in _TRANSIENT:
                        raise
            job = self.store.update_state(job_id, JobState.INVESTIGATING)
            self._collect_evidence(job, token)
            self._ensure_not_cancelled(job_id)
            try:
                worktree, branch = self.repositories.prepare(job, token)
            except Exception as exc:
                raise JobFailure("worktree_failure", str(exc)) from exc
            job.worktree_path, job.branch_name = str(worktree), branch
            job.target_sha = self._git(worktree, "rev-parse", "HEAD") or job.target_sha
            self.store.save_job(job)
            self.store.update_state(job_id, JobState.IMPLEMENTING)
            resume_thread_id = str(job.result.get("thread_id") or "")
            agent = self.codex_factory(worktree, job.session_id, resume_thread_id)
            result = agent.generate_auto_execute(build_task_prompt(job))
            self._ensure_not_cancelled(job_id)
            if result.get("status") != "completed":
                raise JobFailure("implementation_failure", str(result.get("answer") or "Codex did not complete the task"))
            self.store.update_state(job_id, JobState.VERIFYING)
            self._validate_result(worktree, result)
            self._commit(worktree, job, result)
            try:
                self.repositories.push(worktree, job.repository_full_name, branch, token)
            except Exception as exc:
                raise JobFailure("push_failure", str(exc)) from exc
            body = pull_request_body(job, result)
            try:
                if job.pull_request_number is None:
                    existing_pr = self.client.find_open_pull_request(job.repository_full_name, token, branch, job.session_id)
                    if existing_pr is not None:
                        job.pull_request_number = int(existing_pr.get("number") or 0) or None
                pr = self.client.create_or_update_pr(job.repository_full_name, token, number=job.pull_request_number, title=f"[Mana] {job.context.get('title') or job.route_decision.trigger}", head=branch, base=job.base_branch, body=body, draft=self.settings.draft_pr_only)
            except Exception as exc:
                raise JobFailure("pull_request_failure", str(exc)) from exc
            job = self.store.get_job(job_id)
            job.pull_request_number = int(pr.get("number") or job.pull_request_number or 0) or None
            job.result = sanitize_event_context(result)
            job.state = JobState.AWAITING_REVIEW
            self.store.save_job(job)
            if job.subject_type in {"issue", "pull_request"} and job.subject_number:
                self.client.comment(job.repository_full_name, job.subject_number, token, f"Mana completed the requested changes in draft PR #{job.pull_request_number}. Verification status: {'passed' if result.get('tests_passed') is True else 'not fully passed' }.")
        except JobFailure as exc:
            self.store.update_state(job_id, JobState.CANCELLED if exc.kind == "cancelled" else JobState.FAILED, failure_type=exc.kind, detail=exc.detail)
            self._notify_failure(job_id, locals().get("token"))
        except GitHubApiError as exc:
            if exc.status == 401:
                current = self.store.get_job(job_id)
                self.auth.invalidate(current.installation_id, current.repository_id)
            kind = "rate_limited" if exc.status == 429 or "rate limit" in str(exc).lower() else "missing_permission" if exc.status in {401, 403, 404} else "repository_unavailable" if exc.status in _TRANSIENT else "pull_request_failure"
            self.store.update_state(job_id, JobState.FAILED, failure_type=kind, detail=str(exc))
            self._notify_failure(job_id, locals().get("token"))
        except Exception as exc:
            self.store.update_state(job_id, JobState.FAILED, failure_type="implementation_failure", detail=f"{type(exc).__name__}: {exc}")
            self._notify_failure(job_id, locals().get("token"))

    def _process_locked(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        with self.store.subject_lock(job.session_id):
            if self.store.get_job(job_id).state == JobState.QUEUED:
                self._process(job_id)

    def _notify_failure(self, job_id: str, token: object) -> None:
        job = self.store.get_job(job_id)
        if not isinstance(token, str) or not token or job.failure_type in {"unauthorized_sender", "cancelled"}:
            return
        if job.subject_type not in {"issue", "pull_request"} or not job.subject_number:
            return
        detail = str(sanitize_event_context(job.failure_detail))[:800]
        try:
            self.client.comment(job.repository_full_name, job.subject_number, token, f"Mana could not complete task `{job.job_id}` (`{job.failure_type}`). {detail}")
        except Exception:
            logger.warning("Unable to post sanitized GitHub task failure", extra={"job_id": job_id, "failure_type": job.failure_type})

    def _authorize(self, job: GitHubJob, token: str) -> None:
        decision = job.route_decision
        if not decision.supported or not decision.execute or not decision.safe_to_continue:
            raise JobFailure("unsupported_event", "Validated route decision does not authorize execution")
        if not decision.requires_human_authorization:
            if not self.settings.security_events_enabled and "alert" in job.subject_type:
                raise JobFailure("unauthorized_sender", "Trusted security events are disabled by repository policy")
            return
        if not job.sender_login:
            raise JobFailure("unauthorized_sender", "A human actor is required")
        permission = self.client.permission(job.repository_full_name, job.sender_login, token)
        job.sender_permission = permission
        self.store.save_job(job)
        allowlisted = self._actor_allowed(job, token)
        if not allowlisted or _PERMISSION_RANK.get(permission, 0) < _PERMISSION_RANK[self.settings.minimum_actor_permission]:
            raise JobFailure("unauthorized_sender", f"Actor permission {permission!r} does not satisfy repository policy")

    def _actor_allowed(self, job: GitHubJob, token: str) -> bool:
        if not self.settings.actor_allowlist:
            return True
        if job.sender_login.lower() in self.settings.actor_allowlist:
            return True
        for entry in self.settings.actor_allowlist:
            if not entry.startswith("team:") or "/" not in entry:
                continue
            organization, team = entry.removeprefix("team:").split("/", 1)
            try:
                if self.client.team_membership(organization, team, job.sender_login, token):
                    return True
            except GitHubApiError:
                continue
        return False

    def _collect_evidence(self, job: GitHubJob, token: str) -> None:
        """Collect event-specific evidence after auth and before any model call."""
        evidence: dict[str, Any] = {}
        if job.subject_type == "workflow_run" and job.subject_number is not None:
            evidence = self.client.workflow_evidence(job.repository_full_name, job.subject_number, token)
            failed = []
            for item in evidence.get("jobs") or []:
                if item.get("conclusion") != "success":
                    failed.append({"id": item.get("id"), "name": item.get("name"), "conclusion": item.get("conclusion"), "steps": [step for step in item.get("steps") or [] if step.get("conclusion") != "success"]})
            evidence = {"failed_jobs_and_steps": failed, "workflow_run": evidence.get("run", {}), "check_annotations": evidence.get("annotations", []), "redacted_log_excerpts": evidence.get("logs", {}), "recent_commits": evidence.get("recent_commits", [])}
        elif job.subject_type == "pull_request" and job.subject_number is not None:
            evidence = self.client.review_evidence(job.repository_full_name, job.subject_number, token)
            current_sha = str((evidence.get("pull_request") or {}).get("head", {}).get("sha") or "")
            comments = evidence.get("review_comments") or []
            evidence["applicable_review_comments"] = [item for item in comments if not item.get("commit_id") or item.get("commit_id") == current_sha]
        elif job.subject_type == "dependabot_alert":
            dependency = ((job.context.get("alert") or {}).get("dependency") or {}) if isinstance(job.context.get("alert"), dict) else {}
            package = str((dependency.get("package") or {}).get("name") or "")
            if package:
                evidence["existing_dependency_pull_requests"] = self.client.open_dependency_pull_requests(job.repository_full_name, package, token)
        if evidence:
            job.context["collected_evidence"] = sanitize_event_context(evidence, secret_alert=job.subject_type == "secret_scanning_alert")
            self.store.save_job(job)

    def _validate_result(self, worktree: Path, result: dict[str, Any]) -> None:
        changed = self._changed_paths(worktree)
        declared = [str(item) for item in result.get("changed_files") or []]
        paths = list(dict.fromkeys([*changed, *declared]))
        if len(paths) > self.settings.maximum_changed_files:
            raise JobFailure("implementation_failure", f"Changed-file limit exceeded ({len(paths)} > {self.settings.maximum_changed_files})")
        if not self.settings.workflow_files_write_enabled and any(path.startswith(".github/workflows/") for path in paths):
            raise JobFailure("missing_permission", "Workflow-file changes are disabled by repository policy")
        if result.get("tests_passed") is False:
            raise JobFailure("verification_failure", "Codex reported failing verification")
        if not result.get("commands_run") and not result.get("tests_run"):
            raise JobFailure("verification_failure", "No command-backed verification result was returned")

    def _ensure_not_cancelled(self, job_id: str) -> None:
        if self.store.get_job(job_id).state == JobState.CANCELLED:
            raise JobFailure("cancelled", "Task was cancelled before repository changes were published")

    def _commit(self, worktree: Path, job: GitHubJob, result: dict[str, Any]) -> None:
        changed = self._changed_paths(worktree)
        if not changed:
            if not self._git(worktree, "log", "-1", "--format=%H", f"origin/{job.base_branch}..HEAD"):
                raise JobFailure("implementation_failure", "Codex completed without repository changes")
            return
        for rel in changed:
            candidate = (worktree / rel).resolve()
            if worktree.resolve() not in candidate.parents and candidate != worktree.resolve():
                raise JobFailure("implementation_failure", "Codex produced an unsafe changed path")
        subprocess.run(["git", "add", "--", *changed], cwd=worktree, check=True, timeout=30)
        staged = self._git(worktree, "diff", "--cached")
        if any(marker in staged for marker in ("github_pat_", "AKIA", "-----BEGIN PRIVATE KEY-----")):
            raise JobFailure("implementation_failure", "Potential secret material detected in the staged diff")
        subprocess.run(["git", "-c", "user.name=mana-agent[bot]", "-c", "user.email=mana-agent[bot]@users.noreply.github.com", "commit", "-m", f"fix: address {job.route_decision.trigger} #{job.subject_number}"], cwd=worktree, check=True, timeout=60, capture_output=True, text=True)

    @staticmethod
    def _changed_paths(worktree: Path) -> list[str]:
        tracked = subprocess.run(["git", "diff", "--name-only", "-z", "HEAD"], cwd=worktree, capture_output=True, timeout=30, check=True).stdout.decode("utf-8", errors="surrogateescape").split("\0")
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=worktree, capture_output=True, timeout=30, check=True).stdout.decode("utf-8", errors="surrogateescape").split("\0")
        return list(dict.fromkeys(path for path in [*tracked, *untracked] if path))

    @staticmethod
    def _git(path: Path, *args: str) -> str:
        result = subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, timeout=30, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    def _codex(self, worktree: Path, session_id: str, resume_thread_id: str = "") -> CodexCodingAgentShim:
        settings = CodexSettings.from_mana_settings(Settings()).model_copy(update={"worktree_isolation": True, "allow_network": False, "task_timeout_seconds": max(1, self.settings.maximum_job_runtime - 5)})
        if not settings.enabled:
            raise JobFailure("agent_unavailable", "Codex integration is disabled; no legacy coding fallback was executed")
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=worktree, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        git_common = (worktree / common).resolve() if not Path(common).is_absolute() else Path(common).resolve()
        source_repo = git_common.parent
        return CodexCodingAgentShim(repo_root=source_repo, codex_settings=settings, repository_id=f"github_{session_id.split(':')[2]}", session_id=session_id, workspace_task_id=session_id, resume_thread_id=resume_thread_id)

    @staticmethod
    def _event_context(event: str, payload: dict[str, Any]) -> dict[str, Any]:
        keys = {"action", "issue", "comment", "pull_request", "review", "workflow_run", "alert", "repository", "sender"}
        subject = payload.get("issue") or payload.get("pull_request") or payload.get("workflow_run") or payload.get("alert") or {}
        dependency = subject.get("dependency") if isinstance(subject, dict) else {}
        package = dependency.get("package") if isinstance(dependency, dict) else {}
        return {"event": event, "title": subject.get("title") or subject.get("name") or (package.get("name") if isinstance(package, dict) else "") or "", **{key: payload[key] for key in keys if key in payload}}

    @staticmethod
    def _trigger_id(payload: dict[str, Any]) -> int | None:
        for key in ("comment", "review", "workflow_run", "alert"):
            value = payload.get(key) or {}
            candidate = value.get("id") if key != "alert" else value.get("number")
            if candidate is not None:
                return int(candidate)
        return None

    @staticmethod
    def _base_branch(payload: dict[str, Any]) -> str:
        return str((payload.get("pull_request") or {}).get("base", {}).get("ref") or (payload.get("workflow_run") or {}).get("head_branch") or (payload.get("repository") or {}).get("default_branch") or "main")

    @staticmethod
    def _target_sha(payload: dict[str, Any]) -> str:
        return str((payload.get("pull_request") or {}).get("head", {}).get("sha") or (payload.get("workflow_run") or {}).get("head_sha") or "")

    @staticmethod
    def _semantic_key(installation: int, repository: int, subject_type: str, subject: int | None, action: str, trigger_id: int | None, sha: str) -> str:
        raw = f"{installation}:{repository}:{subject_type}:{subject}:{action}:{trigger_id}:{sha}"
        return "ghtask_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


class JobFailure(RuntimeError):
    def __init__(self, kind: str, detail: str) -> None:
        self.kind, self.detail = kind, detail
        super().__init__(detail)
