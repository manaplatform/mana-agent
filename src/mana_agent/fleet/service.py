"""Fleet orchestration over the authoritative ExecutionManager boundary."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from mana_agent.execution.errors import CleanupError, ExecutionTimeoutError
from mana_agent.execution.manager import ExecutionManager
from mana_agent.execution.models import (
    ArtifactRequest, ExecutionRequest, RoutingRequest, SandboxSpec,
)
from mana_agent.utils.redaction import redact_secrets
from mana_agent.config.settings import Settings
from mana_agent.execution_supervisor import (
    CompletionContract,
    CompletionContractType,
    ExecutionState as SupervisorState,
    ExecutionSupervisor,
    ExecutionSupervisorConfig,
    SideEffectClassification,
)
from mana_agent.execution_supervisor.errors import ExecutionSupervisorError

from .config import FleetConfig
from .errors import FleetDisabledError, FleetSelectionError, FleetStateError
from .events import FleetEvent
from .models import (
    ArtifactReference, FailureClassification, FleetJob, FleetJobResult, FleetJobState,
    FleetOutcome, FleetRun, FleetRunSummary, FleetSelectionRequest,
    FleetVerificationPlan, PlatformResult, VerificationCell, WorkspaceState, utc_now,
)
from .registry import FleetRegistry
from .selector import select_workers
from .store import FleetStore

EventSink = Callable[[FleetEvent], None]


class FleetService:
    def __init__(
        self, *, config: FleetConfig, registry: FleetRegistry,
        execution_manager: ExecutionManager, store: FleetStore,
        event_sink: EventSink | None = None,
        execution_supervisor: ExecutionSupervisor | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.execution_manager = execution_manager
        self.store = store
        self.event_sink = event_sink
        self.execution_supervisor = execution_supervisor or ExecutionSupervisor(
            ExecutionSupervisorConfig.from_settings(Settings())
        )
        events = store.events()
        self._sequence = events[-1].sequence if events else 0
        self._cancelled_jobs: set[str] = set()

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise FleetDisabledError(
                "Mana Fleet is disabled. Set MANA_FLEET_ENABLED=true after configuring trusted workers."
            )

    def _emit(self, kind: str, *, run: FleetRun | None = None, job: FleetJob | None = None, **data: Any) -> None:
        self._sequence += 1
        safe_data = redact_secrets(data)
        encoded = json.dumps(safe_data, default=str, separators=(",", ":"))
        if len(encoded.encode()) > 64 * 1024:
            safe_data = {
                "truncated": True,
                "preview": encoded[:32 * 1024],
            }
        event = FleetEvent(
            sequence=self._sequence, kind=kind,
            fleet_run_id=job.fleet_run_id if job else (run.fleet_run_id if run else ""),
            job_id=job.job_id if job else "",
            task_id=job.task_id if job else (run.plan.task_id if run else ""),
            session_id=job.session_id if job else (run.plan.session_id if run else ""),
            workspace_id=job.workspace_id if job else (run.plan.workspace_id if run else ""),
            repository_id=job.repository_id if job else (run.plan.repository_id if run else ""),
            worker_id=job.worker_id if job else "",
            execution_provider=job.execution_provider if job else "",
            data=safe_data,
        )
        self.store.append_event(event)
        if self.event_sink:
            self.event_sink(event)

    def decide(self, request: FleetSelectionRequest):
        self._require_enabled()
        self._emit("fleet.selection.requested", decision_id=request.decision_id)
        decision = select_workers(request, self.registry.list(), self.config)
        self._emit(
            "fleet.selection.decided",
            decision_id=decision.decision_id,
            selected_workers=[item.worker_id for item in decision.selected_workers],
            platform_coverage=sorted(decision.platform_coverage),
        )
        return decision

    def create_plan(
        self, *, request: FleetSelectionRequest, repository_path: str | Path,
        repository_commit: str, commands: list[list[str]],
        transfer_mode: str = "git-bundle", artifact_paths: list[str] | None = None,
        retain_workspaces: bool = False,
    ) -> FleetVerificationPlan:
        if not commands or any(not command for command in commands):
            raise FleetSelectionError("fleet verification requires at least one validated argv command")
        repository = Path(repository_path).expanduser().resolve()
        commit = self._validate_repository(repository, repository_commit)
        decision = self.decide(request)
        workers = {item.worker_id: self.registry.require(item.worker_id) for item in decision.selected_workers}
        cells: list[VerificationCell] = []
        target_platforms = request.required_platforms or decision.platform_coverage
        python_versions = sorted(request.runtime.python) or [""]
        node_versions = sorted(request.runtime.node) or [""]
        for platform_name in sorted(target_platforms):
            compatible = [
                worker for worker in workers.values()
                if worker.capabilities.platform == platform_name
            ]
            if not compatible:
                raise FleetSelectionError(
                    f"selected decision does not cover required platform {platform_name}"
                )
            for python_version in python_versions:
                for node_version in node_versions:
                    cells.append(VerificationCell(
                        platform=platform_name,
                        architecture=compatible[0].capabilities.architecture,
                        python_version=python_version,
                        node_version=node_version,
                        commands=tuple(tuple(item) for item in commands),
                        artifact_paths=tuple(artifact_paths or ()),
                    ))
        return FleetVerificationPlan(
            decision=decision,
            task_id=request.task_id, session_id=request.session_id,
            workspace_id=request.workspace_id, repository_id=request.repository_id,
            repository_path=str(repository), repository_commit=commit,
            transfer_mode=transfer_mode,  # Pydantic validates this explicit choice.
            cells=tuple(cells), timeout_seconds=request.timeout_seconds,
            retain_workspaces=retain_workspaces,
            mutation_intent=request.intent == "mutation",
            monetary_budget=request.budget,
        )

    @staticmethod
    def _validate_repository(repository: Path, commit: str) -> str:
        if not repository.is_dir():
            raise FleetSelectionError(f"repository does not exist: {repository}")
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
            cwd=repository, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise FleetSelectionError(f"repository commit is invalid: {commit}")
        return result.stdout.strip()

    def create_run(self, plan: FleetVerificationPlan) -> FleetRun:
        assignments: dict[str, list[str]] = defaultdict(list)
        for selected in plan.decision.selected_workers:
            worker = self.registry.require(selected.worker_id)
            assignments[worker.capabilities.platform].append(selected.worker_id)
        jobs: list[FleetJob] = []
        provider_by_worker = {
            item.worker_id: item.execution_provider for item in plan.decision.selected_workers
        }
        for index, cell in enumerate(plan.cells):
            worker_ids = assignments[cell.platform]
            worker_id = worker_ids[index % len(worker_ids)]
            jobs.append(FleetJob(
                fleet_run_id=plan.fleet_run_id, task_id=plan.task_id,
                session_id=plan.session_id, workspace_id=plan.workspace_id,
                repository_id=plan.repository_id, worker_id=worker_id,
                execution_provider=provider_by_worker[worker_id], cell=cell,
            ))
        run = FleetRun(fleet_run_id=plan.fleet_run_id, plan=plan, jobs=tuple(jobs))
        upstream_task_id = (
            plan.task_id
            if plan.task_id
            and plan.task_id != plan.fleet_run_id
            and self.execution_supervisor.store.get_task_or_none(plan.task_id) is not None
            else None
        )
        self.execution_supervisor.create_task(
            task_id=plan.fleet_run_id,
            parent_task_id=upstream_task_id,
            task_type="fleet_run",
            assigned_agent="fleet",
            runtime_provider="fleet",
            workspace_path=plan.repository_path,
            routing_decision_id=plan.decision.decision_id,
            side_effect_classification=(
                SideEffectClassification.UNKNOWN
                if plan.mutation_intent
                else SideEffectClassification.READ_ONLY
            ),
            completion_contract=[CompletionContract(
                contract_type=CompletionContractType.STRUCTURED_RESULT_VALID,
                metadata={
                    "required_keys": ["fleet_run_id", "summary_present"],
                    "expected_values": {
                        "fleet_run_id": plan.fleet_run_id,
                        "summary_present": True,
                    },
                },
            )],
            estimated_cost=plan.decision.estimated_cost or 0.0,
            monetary_budget=plan.monetary_budget,
        )
        self.execution_supervisor.queue(plan.fleet_run_id)
        per_job_estimated_cost = (
            (plan.decision.estimated_cost or 0.0) / max(1, len(jobs))
        )
        for job in jobs:
            self.execution_supervisor.create_task(
                task_id=job.job_id,
                parent_task_id=plan.fleet_run_id,
                task_type="fleet_verification",
                assigned_agent="fleet",
                runtime_provider=job.execution_provider,
                workspace_path=plan.repository_path,
                routing_decision_id=plan.decision.decision_id,
                side_effect_classification=(
                    SideEffectClassification.UNKNOWN
                    if plan.mutation_intent
                    else SideEffectClassification.READ_ONLY
                ),
                completion_contract=[CompletionContract(
                    contract_type=CompletionContractType.COMMAND_SUCCEEDED,
                )],
                estimated_cost=per_job_estimated_cost,
                monetary_budget=plan.monetary_budget,
            )
            self.execution_supervisor.queue(job.job_id)
        self.store.save_run(run)
        self._emit("fleet.run.created", run=run, job_count=len(jobs))
        for job in jobs:
            self._emit("fleet.job.queued", job=job)
        return run

    async def execute(self, run: FleetRun) -> FleetRun:
        self._require_enabled()
        supervised_run = self.execution_supervisor.store.get_task(run.fleet_run_id)
        if supervised_run.state == SupervisorState.RETRY_SCHEDULED:
            supervised_run = self.execution_supervisor.release_retry(run.fleet_run_id)
        supervised_run, run_lease_token = self.execution_supervisor.acquire_lease(
            run.fleet_run_id,
            owner="fleet:coordinator",
            worker="fleet:coordinator",
        )
        self.execution_supervisor.start(
            run.fleet_run_id,
            attempt_id=supervised_run.attempt_id,
            lease_token=run_lease_token,
        )
        run_heartbeat_stop = asyncio.Event()
        run_heartbeat_errors: list[str] = []

        async def renew_run_lease() -> None:
            while True:
                try:
                    await asyncio.wait_for(
                        run_heartbeat_stop.wait(),
                        timeout=self.execution_supervisor.config.heartbeat_seconds,
                    )
                    return
                except asyncio.TimeoutError:
                    try:
                        self.execution_supervisor.heartbeat(
                            run.fleet_run_id,
                            attempt_id=supervised_run.attempt_id,
                            lease_token=run_lease_token,
                        )
                    except Exception as exc:
                        run_heartbeat_errors.append(str(exc))
                        return

        run_heartbeat = asyncio.create_task(renew_run_lease())
        semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)
        worker_semaphores = {
            job.worker_id: asyncio.Semaphore(max(
                1,
                self.registry.require(job.worker_id).health.concurrency_limit
                - self.registry.require(job.worker_id).health.active_job_count,
            ))
            for job in run.jobs
        }
        results: list[FleetJobResult] = list(run.results)
        completed_ids = {item.job_id for item in results}

        async def execute_job(job: FleetJob) -> None:
            if job.job_id in completed_ids:
                return
            if run.plan.mutation_intent and not job.revalidated:
                job.state = FleetJobState.REVALIDATION_REQUIRED
                job.updated_at = utc_now()
                supervised = self.execution_supervisor.store.get_task_or_none(job.job_id)
                if supervised is not None and supervised.state == SupervisorState.QUEUED:
                    self.execution_supervisor.transition(
                        job.job_id,
                        SupervisorState.WAITING,
                        recovery_reason="mutation requires explicit Fleet revalidation",
                    )
                self.store.save_run(run.model_copy(update={"jobs": tuple(run.jobs)}))
                return
            async with semaphore, worker_semaphores[job.worker_id]:
                result = await self._execute_job(run, job)
                results.append(result)
                updated = run.model_copy(update={"results": tuple(results), "updated_at": utc_now()})
                self.store.save_run(updated)

        try:
            await asyncio.gather(*(execute_job(job) for job in run.jobs))
            for escrow in self.execution_supervisor.store.unacknowledged_results(
                run.fleet_run_id
            ):
                child = self.execution_supervisor.store.get_task(escrow.task_id)
                if child.state == SupervisorState.COMPLETED:
                    self.execution_supervisor.acknowledge_result(
                        escrow.result_id,
                        parent_task_id=run.fleet_run_id,
                    )
            summary = summarize_run(run.plan, results, run.jobs)
        except BaseException as exc:
            current = self.execution_supervisor.store.get_task(run.fleet_run_id)
            if current.state not in {SupervisorState.FAILED, SupervisorState.CANCELLED}:
                self.execution_supervisor.transition(
                    run.fleet_run_id,
                    SupervisorState.FAILED,
                    reason=f"Fleet run coordination failed: {exc}",
                )
            raise
        finally:
            run_heartbeat_stop.set()
            await run_heartbeat
        if run_heartbeat_errors:
            current = self.execution_supervisor.store.get_task(run.fleet_run_id)
            if current.state not in {SupervisorState.FAILED, SupervisorState.CANCELLED}:
                self.execution_supervisor.transition(
                    run.fleet_run_id,
                    SupervisorState.FAILED,
                    reason=f"Fleet run heartbeat failed: {run_heartbeat_errors[-1]}",
                )
            supervisor_state = SupervisorState.FAILED
            supervision_error = run_heartbeat_errors[-1]
        else:
            supervision_error = ""
            try:
                supervised_run = self.execution_supervisor.submit_result(
                    run.fleet_run_id,
                    attempt_id=supervised_run.attempt_id,
                    lease_token=run_lease_token,
                    payload={
                        "fleet_run_id": run.fleet_run_id,
                        "summary_present": True,
                        "summary": summary.model_dump(mode="json"),
                    },
                )
            except ExecutionSupervisorError as exc:
                supervision_error = str(exc)
                supervisor_state = self.execution_supervisor.store.get_task(
                    run.fleet_run_id
                ).state
            else:
                supervisor_state = supervised_run.state
        completed = run.model_copy(
            update={
                "results": tuple(results),
                "summary": summary,
                "updated_at": utc_now(),
                "metadata": {
                    **run.metadata,
                    "execution_supervisor_state": supervisor_state.value,
                    "execution_supervisor_error": supervision_error,
                },
            }
        )
        self.store.save_run(completed)
        self._emit(
            "fleet.comparison.completed"
            if supervisor_state == SupervisorState.COMPLETED
            else "fleet.comparison.failed",
            run=completed,
            outcome=summary.outcome.value,
            supervisor_state=supervisor_state.value,
            error=supervision_error,
        )
        if supervisor_state != SupervisorState.COMPLETED:
            raise FleetStateError(
                "Fleet results were persisted, but durable run completion was not verified: "
                f"{supervision_error or supervisor_state.value}"
            )
        return completed

    async def _execute_job(self, run: FleetRun, job: FleetJob) -> FleetJobResult:
        started = time.monotonic()
        self.registry.reserve(job.worker_id)
        job.state = FleetJobState.ASSIGNED
        job.started_at = utc_now()
        job.updated_at = utc_now()
        self._emit("fleet.job.assigned", job=job)
        context = None
        local_workspace: Path | None = None
        cleanup_failure = ""
        failure: FailureClassification | None = None
        exit_code: int | None = None
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        artifacts: list[ArtifactReference] = []
        final_state = FleetJobState.FAILED
        supervisor_attempt_id = ""
        supervisor_token = ""
        heartbeat_stop = asyncio.Event()
        heartbeat_task: asyncio.Task[None] | None = None
        heartbeat_failures: list[str] = []

        async def renew_lease() -> None:
            while True:
                try:
                    await asyncio.wait_for(
                        heartbeat_stop.wait(),
                        timeout=self.execution_supervisor.config.heartbeat_seconds,
                    )
                    return
                except asyncio.TimeoutError:
                    try:
                        self.execution_supervisor.heartbeat(
                            job.job_id,
                            attempt_id=supervisor_attempt_id,
                            lease_token=supervisor_token,
                        )
                    except Exception as exc:
                        heartbeat_failures.append(str(exc))
                        return
        try:
            supervised = self.execution_supervisor.store.get_task(job.job_id)
            if supervised.state == SupervisorState.RETRY_SCHEDULED:
                supervised = self.execution_supervisor.release_retry(job.job_id)
            if supervised.state == SupervisorState.WAITING:
                supervised = self.execution_supervisor.queue(job.job_id)
            if supervised.state == SupervisorState.CREATED:
                supervised = self.execution_supervisor.queue(job.job_id)
            supervised, supervisor_token = self.execution_supervisor.acquire_lease(
                job.job_id,
                owner=f"fleet:{job.worker_id}",
                worker=job.worker_id,
            )
            supervisor_attempt_id = supervised.attempt_id
            self.execution_supervisor.start(
                job.job_id,
                attempt_id=supervisor_attempt_id,
                lease_token=supervisor_token,
            )
            heartbeat_task = asyncio.create_task(renew_lease())
            if self._is_cancelled(job.job_id):
                raise asyncio.CancelledError
            local_workspace = self._create_isolated_source(run.plan, job)
            job.workspace_state = WorkspaceState.PROVISIONING
            job.state = FleetJobState.PROVISIONING
            self._emit("fleet.workspace.provisioning", job=job)
            spec = SandboxSpec(
                provider_override=job.execution_provider,
                repository_source=local_workspace,
                artifact_paths=list(job.cell.artifact_paths),
                execution_timeout_seconds=run.plan.timeout_seconds,
                max_lifetime_seconds=self.config.workspace_max_lifetime_seconds,
                cleanup_policy="retain" if run.plan.retain_workspaces else "always",
                task_id=job.task_id, session_id=job.session_id,
                workspace_id=job.workspace_id,
                execution_id=job.job_id,
                root_task_id=job.fleet_run_id,
                attempt_id=supervisor_attempt_id,
                checkpoint_id=supervised.checkpoint_id,
                repository_id=job.repository_id,
                labels={"fleet_run_id": job.fleet_run_id, "fleet_job_id": job.job_id, "worker_id": job.worker_id},
            )
            routing = RoutingRequest(
                decision_id=run.plan.decision.decision_id,
                explicit_provider=job.execution_provider,
                trust_level="trusted", risk_level="medium" if run.plan.mutation_intent else "low",
                required_capabilities=frozenset({"artifact_streaming"}) if job.cell.artifact_paths else frozenset(),
                expected_duration_seconds=run.plan.timeout_seconds,
            )
            context = await self.execution_manager.create(spec, routing)
            job.workspace_state = WorkspaceState.READY
            self._emit("fleet.workspace.ready", job=job)
            commit_result = await self.execution_manager.execute(
                context,
                ExecutionRequest(
                    argv=["git", "rev-parse", "HEAD"], timeout_seconds=30,
                    execution_id=job.job_id, task_id=job.task_id,
                    root_task_id=job.fleet_run_id, attempt_id=supervisor_attempt_id,
                    checkpoint_id=supervised.checkpoint_id, session_id=job.session_id,
                    workspace_id=job.workspace_id, repository_id=job.repository_id,
                ),
            )
            if commit_result.exit_code != 0 or commit_result.stdout.strip() != run.plan.repository_commit:
                failure = FailureClassification.REPOSITORY_TRANSFER_FAILURE
                raise FleetStateError("remote workspace commit identity does not match the verification plan")
            status_result = await self.execution_manager.execute(
                context,
                ExecutionRequest(
                    argv=["git", "status", "--porcelain"], timeout_seconds=30,
                    execution_id=job.job_id, task_id=job.task_id,
                    root_task_id=job.fleet_run_id, attempt_id=supervisor_attempt_id,
                    checkpoint_id=supervised.checkpoint_id, session_id=job.session_id,
                    workspace_id=job.workspace_id, repository_id=job.repository_id,
                ),
            )
            if status_result.exit_code != 0 or status_result.stdout.strip():
                failure = FailureClassification.REPOSITORY_TRANSFER_FAILURE
                raise FleetStateError("remote workspace did not start clean")
            self._emit("fleet.repository.synced", job=job, commit=run.plan.repository_commit)
            job.state = FleetJobState.RUNNING
            job.workspace_state = WorkspaceState.RUNNING
            for argv in job.cell.commands:
                if self._is_cancelled(job.job_id):
                    raise asyncio.CancelledError
                self._emit("fleet.command.started", job=job, argv0=argv[0])
                result = await self.execution_manager.execute(
                    context,
                    ExecutionRequest(
                        argv=list(argv), timeout_seconds=run.plan.timeout_seconds,
                        capture_limit_bytes=self.config.max_log_bytes,
                        execution_id=job.job_id, task_id=job.task_id,
                        root_task_id=job.fleet_run_id, attempt_id=supervisor_attempt_id,
                        checkpoint_id=supervised.checkpoint_id, session_id=job.session_id,
                        workspace_id=job.workspace_id, repository_id=job.repository_id,
                    ),
                )
                stdout_parts.append(result.stdout)
                stderr_parts.append(result.stderr)
                exit_code = result.exit_code
                self._emit("fleet.command.completed", job=job, exit_code=result.exit_code)
                if result.exit_code != 0:
                    failure = FailureClassification.TEST_FAILURE
                    break
            if job.cell.artifact_paths:
                job.state = FleetJobState.COLLECTING
                job.workspace_state = WorkspaceState.COLLECTING
                provider = self.execution_manager.registry.get(context.handle.provider)
                collected = await provider.download_artifacts(
                    context.handle,
                    ArtifactRequest(
                        paths=list(job.cell.artifact_paths),
                        destination=self.config.root / "artifacts" / job.job_id,
                        max_total_bytes=self.config.max_artifact_bytes,
                        missing_ok=False,
                    ),
                )
                artifacts = [
                    ArtifactReference(
                        path=item.source_path, sha256=item.sha256,
                        size_bytes=item.size_bytes, reference=item.reference,
                    )
                    for item in collected
                ]
                for item in artifacts:
                    self._emit("fleet.artifact.collected", job=job, path=item.path, sha256=item.sha256)
            final_state = FleetJobState.COMPLETED if failure is None else FleetJobState.FAILED
        except asyncio.CancelledError:
            final_state = FleetJobState.CANCELLED
            failure = None
            self._emit("fleet.job.cancelled", job=job)
        except ExecutionTimeoutError as exc:
            final_state = FleetJobState.TIMED_OUT
            failure = FailureClassification.TIMEOUT
            stderr_parts.append(str(exc))
        except Exception as exc:
            final_state = FleetJobState.FAILED
            failure = failure or FailureClassification.PROVIDER_FAILURE
            stderr_parts.append(str(exc))
        finally:
            if context is not None and not run.plan.retain_workspaces:
                try:
                    await self.execution_manager.terminate_and_cleanup(context)
                    self._emit("fleet.cleanup.completed", job=job)
                except CleanupError as exc:
                    cleanup_failure = str(exc)[:1000]
                    self._emit("fleet.cleanup.failed", job=job, error=cleanup_failure)
            if local_workspace is not None and local_workspace.exists() and not run.plan.retain_workspaces:
                self._remove_isolated_source(Path(run.plan.repository_path), local_workspace)
            self.registry.release(
                job.worker_id, success=final_state is FleetJobState.COMPLETED,
                failure=(stderr_parts[-1] if stderr_parts else ""),
            )
            heartbeat_stop.set()
            if heartbeat_task is not None:
                await heartbeat_task
        if cleanup_failure and failure is None:
            failure = FailureClassification.CLEANUP_FAILURE
            final_state = FleetJobState.FAILED
        if heartbeat_failures and final_state is not FleetJobState.CANCELLED:
            failure = FailureClassification.WORKER_DISCONNECT
            final_state = FleetJobState.FAILED
            stderr_parts.append(f"execution supervisor heartbeat failed: {heartbeat_failures[-1]}")
        job.state = final_state
        job.workspace_state = (
            WorkspaceState.RETAINED if run.plan.retain_workspaces else
            WorkspaceState.CLEANED if not cleanup_failure else WorkspaceState.FAILED
        )
        job.completed_at = utc_now()
        job.updated_at = job.completed_at
        result = FleetJobResult(
            fleet_run_id=job.fleet_run_id, job_id=job.job_id,
            task_id=job.task_id, session_id=job.session_id,
            workspace_id=job.workspace_id, repository_id=job.repository_id,
            worker_id=job.worker_id, execution_provider=job.execution_provider,
            state=final_state, exit_code=exit_code,
            duration_seconds=time.monotonic() - started,
            stdout=str(redact_secrets("\n".join(stdout_parts)))[-self.config.max_log_bytes:],
            stderr=str(redact_secrets("\n".join(stderr_parts)))[-self.config.max_log_bytes:],
            artifacts=tuple(artifacts), failure_classification=failure,
            cleanup_failure=cleanup_failure,
        )
        supervised = self.execution_supervisor.store.get_task_or_none(job.job_id)
        if supervised is not None and supervisor_attempt_id:
            if final_state is FleetJobState.COMPLETED:
                try:
                    verified = self.execution_supervisor.submit_result(
                        job.job_id,
                        attempt_id=supervisor_attempt_id,
                        lease_token=supervisor_token,
                        payload={
                            "exit_code": exit_code,
                            "worker_id": job.worker_id,
                            "artifacts": [item.model_dump(mode="json") for item in artifacts],
                        },
                    )
                except ExecutionSupervisorError as exc:
                    final_state = FleetJobState.FAILED
                    job.state = final_state
                    result = result.model_copy(update={
                        "state": final_state,
                        "failure_classification": FailureClassification.ARTIFACT_COLLECTION_FAILURE,
                        "stderr": f"{result.stderr}\nCompletion supervision failed: {exc}".strip(),
                    })
                else:
                    if verified.state != SupervisorState.COMPLETED:
                        final_state = FleetJobState.FAILED
                        job.state = final_state
                        result = result.model_copy(update={
                            "state": final_state,
                            "failure_classification": FailureClassification.ARTIFACT_COLLECTION_FAILURE,
                        })
            elif final_state is FleetJobState.CANCELLED:
                self.execution_supervisor.cancel(job.job_id, reason="fleet job cancelled", propagate=False)
            elif supervised.state not in {SupervisorState.FAILED, SupervisorState.CANCELLED}:
                self.execution_supervisor.transition(
                    job.job_id,
                    SupervisorState.FAILED,
                    reason=(stderr_parts[-1] if stderr_parts else final_state.value),
                )
        self._emit(
            "fleet.job.completed" if final_state is FleetJobState.COMPLETED else "fleet.job.failed",
            job=job, failure_classification=failure.value if failure else "",
        )
        self.store.clear_cancellation(job.job_id)
        return result

    def _create_isolated_source(self, plan: FleetVerificationPlan, job: FleetJob) -> Path:
        root = (self.config.root / "workspaces").resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = (root / job.job_id).resolve()
        if path.parent != root:
            raise FleetStateError("fleet workspace path escapes the managed root")
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(path), plan.repository_commit],
            cwd=plan.repository_path, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise FleetStateError(result.stderr.strip() or "isolated fleet workspace creation failed")
        return path

    @staticmethod
    def _remove_isolated_source(repository: Path, workspace: Path) -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(workspace)],
            cwd=repository, capture_output=True, text=True, check=False,
        )

    def cancel(self, job_id: str) -> None:
        selected = next(
            (
                (run, job)
                for run in self.store.list_runs()
                for job in run.jobs
                if job_id == job.job_id
            ),
            None,
        )
        if selected is None:
            raise FleetStateError(f"fleet job not found: {job_id}")
        run, _job = selected
        if any(result.job_id == job_id for result in run.results):
            raise FleetStateError("completed fleet jobs cannot be cancelled or re-executed")
        self._cancelled_jobs.add(job_id)
        self.store.request_cancellation(job_id)
        supervised = self.execution_supervisor.store.get_task_or_none(job_id)
        if supervised is not None and supervised.state not in {
            SupervisorState.COMPLETED, SupervisorState.FAILED, SupervisorState.CANCELLED,
        }:
            self.execution_supervisor.cancel(job_id, reason="fleet cancellation requested", propagate=False)

    def _is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled_jobs or self.store.cancellation_requested(job_id)

    def recover(self) -> list[FleetRun]:
        self.execution_supervisor.reconnect_tree()
        self.execution_supervisor.recover()
        recovered: list[FleetRun] = []
        terminal = {
            FleetJobState.COMPLETED, FleetJobState.FAILED, FleetJobState.CANCELLED,
            FleetJobState.TIMED_OUT, FleetJobState.WORKER_DISCONNECTED,
            FleetJobState.REVALIDATION_REQUIRED,
        }
        for run in self.store.list_runs():
            changed = False
            for job in run.jobs:
                if job.state not in terminal:
                    supervised = self.execution_supervisor.store.get_task_or_none(job.job_id)
                    job.state = (
                        FleetJobState.QUEUED
                        if supervised is not None
                        and supervised.state == SupervisorState.RETRY_SCHEDULED
                        else FleetJobState.REVALIDATION_REQUIRED
                        if run.plan.mutation_intent
                        else FleetJobState.FAILED
                    )
                    job.updated_at = utc_now()
                    changed = True
            if changed:
                updated = run.model_copy(update={"jobs": tuple(run.jobs), "updated_at": utc_now()})
                self.store.save_run(updated)
                recovered.append(updated)
        return recovered


def summarize_run(
    plan: FleetVerificationPlan,
    results: list[FleetJobResult],
    jobs: tuple[FleetJob, ...] | list[FleetJob] | None = None,
) -> FleetRunSummary:
    required = frozenset(cell.platform for cell in plan.cells)
    job_by_id = {job.job_id: job.cell.platform for job in (jobs or ())}
    # Keep the aggregation independent from result ordering and provider details.
    # Legacy callers without jobs retain stable positional aggregation.
    if not job_by_id:
        for index, result in enumerate(results):
            if index < len(plan.cells):
                job_by_id[result.job_id] = plan.cells[index].platform
    tested = frozenset(
        job_by_id[result.job_id]
        for result in results
        if result.job_id in job_by_id
        and result.state in {FleetJobState.COMPLETED, FleetJobState.FAILED}
        and result.failure_classification not in {
            FailureClassification.PROVIDER_FAILURE,
            FailureClassification.WORKER_DISCONNECT,
            FailureClassification.PERMISSION_DENIAL,
            FailureClassification.CAPABILITY_MISMATCH,
            FailureClassification.TIMEOUT,
            FailureClassification.REPOSITORY_TRANSFER_FAILURE,
            FailureClassification.ARTIFACT_COLLECTION_FAILURE,
            FailureClassification.MODEL_ROUTING_FAILURE,
        }
    )
    counts = Counter(
        result.failure_classification.value
        for result in results if result.failure_classification is not None
    )
    counts[FailureClassification.CLEANUP_FAILURE.value] += sum(
        bool(result.cleanup_failure)
        and result.failure_classification is not FailureClassification.CLEANUP_FAILURE
        for result in results
    )
    counts += Counter()  # Drop zero-valued cleanup entries.
    cancelled = any(item.state is FleetJobState.CANCELLED for item in results)
    infrastructure = any(
        item.failure_classification not in {None, FailureClassification.TEST_FAILURE, FailureClassification.SETUP_FAILURE}
        or bool(item.cleanup_failure)
        for item in results
    )
    test_failures = any(
        item.failure_classification in {FailureClassification.TEST_FAILURE, FailureClassification.SETUP_FAILURE}
        for item in results
    )
    if cancelled:
        outcome = FleetOutcome.CANCELLED
    elif infrastructure or not required.issubset(tested):
        outcome = FleetOutcome.INFRASTRUCTURE_INCOMPLETE
    elif test_failures:
        outcome = FleetOutcome.FAILED_VERIFICATION
    elif len(results) < len(plan.cells):
        outcome = FleetOutcome.PARTIALLY_VERIFIED
    else:
        outcome = FleetOutcome.FULLY_VERIFIED
    platform_rows: list[PlatformResult] = []
    for platform_name in sorted(required):
        platform_results = [
            result for result in results if job_by_id.get(result.job_id) == platform_name
        ]
        platform_rows.append(PlatformResult(
            platform=platform_name, jobs=len(platform_results),
            passed=sum(item.state is FleetJobState.COMPLETED for item in platform_results),
            failed=sum(item.failure_classification is FailureClassification.TEST_FAILURE for item in platform_results),
            infrastructure_failures=sum(
                item.failure_classification not in {None, FailureClassification.TEST_FAILURE, FailureClassification.SETUP_FAILURE}
                for item in platform_results
            ),
        ))
    return FleetRunSummary(
        fleet_run_id=plan.fleet_run_id, outcome=outcome,
        required_platforms=required, tested_platforms=tested,
        platform_results=tuple(platform_rows),
        failures_by_classification=dict(sorted(counts.items())),
        completed_jobs=len(results), total_jobs=len(plan.cells),
    )
