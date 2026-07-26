"""Fail-closed deterministic validation of model-produced Fleet requests."""

from __future__ import annotations

from .config import FleetConfig
from .errors import FleetSelectionError
from .health import effective_status
from .models import (
    CapabilityMismatch, FleetSelectionDecision, FleetSelectionRequest, FleetWorker,
    SelectedWorker, WorkerStatus,
)


def _mismatches(worker: FleetWorker, request: FleetSelectionRequest, config: FleetConfig) -> list[str]:
    capabilities = worker.capabilities
    status = effective_status(
        worker,
        heartbeat_timeout_seconds=config.heartbeat_timeout_seconds,
        capability_ttl_seconds=config.capability_ttl_seconds,
    )
    reasons: list[str] = []
    if status is not WorkerStatus.CONNECTED:
        reasons.append(f"worker status is {status.value}")
    if request.allowed_platforms and capabilities.platform not in request.allowed_platforms:
        reasons.append(f"platform {capabilities.platform} is not allowed")
    if request.required_architectures and capabilities.architecture not in request.required_architectures:
        reasons.append(f"architecture {capabilities.architecture} is incompatible")
    missing_python = request.runtime.python - set(capabilities.python_versions)
    missing_node = request.runtime.node - set(capabilities.node_versions)
    missing_tools = request.required_tools - capabilities.available_tools
    missing_providers = request.required_provider_capabilities - capabilities.execution_providers
    if missing_python:
        reasons.append(f"missing Python runtimes: {', '.join(sorted(missing_python))}")
    if missing_node:
        reasons.append(f"missing Node runtimes: {', '.join(sorted(missing_node))}")
    if missing_tools:
        reasons.append(f"missing tools: {', '.join(sorted(missing_tools))}")
    if missing_providers:
        reasons.append(f"missing execution providers: {', '.join(sorted(missing_providers))}")
    if not capabilities.execution_providers:
        reasons.append("worker has no authenticated execution provider")
    if request.forbidden_labels & capabilities.labels.values:
        reasons.append("worker has a forbidden label")
    if config.require_trusted_label and "trusted" not in capabilities.labels.values:
        reasons.append("worker does not have the required trusted label")
    if request.intent == "mutation" and "remote-write" not in capabilities.labels.values:
        reasons.append("worker is not authorised for remote mutation")
    return reasons


def select_workers(
    request: FleetSelectionRequest, workers: list[FleetWorker], config: FleetConfig,
) -> FleetSelectionDecision:
    candidates: list[tuple[float, FleetWorker]] = []
    rejected: list[CapabilityMismatch] = []
    for worker in sorted(workers, key=lambda item: item.worker_id):
        reasons = _mismatches(worker, request, config)
        if reasons:
            rejected.append(CapabilityMismatch(worker_id=worker.worker_id, reasons=tuple(reasons)))
            continue
        labels = worker.capabilities.labels.values
        score = (
            100.0
            + 10.0 * len(request.preferred_labels & labels)
            - 5.0 * worker.health.active_job_count
            - 2.0 * len(worker.health.recent_failures)
        )
        candidates.append((score, worker))
    candidates.sort(key=lambda item: (-item[0], item[1].worker_id))

    chosen: list[tuple[float, FleetWorker]] = []
    for platform_name in sorted(request.required_platforms):
        match = next((item for item in candidates if item[1].capabilities.platform == platform_name and item not in chosen), None)
        if match is None:
            details = "; ".join(
                f"{item.worker_id}: {', '.join(item.reasons)}" for item in rejected
            )
            raise FleetSelectionError(
                f"required fleet platform coverage is unavailable: {platform_name}. "
                f"No local fallback was executed. {details}".strip()
            )
        chosen.append(match)
    for item in candidates:
        if len(chosen) >= min(request.maximum_workers, config.max_workers_per_run):
            break
        if item not in chosen:
            chosen.append(item)
    if len(chosen) < request.minimum_workers:
        raise FleetSelectionError(
            f"fleet selection requires {request.minimum_workers} workers but only {len(chosen)} "
            "compatible workers are available. No local fallback was executed."
        )
    selected = tuple(
        SelectedWorker(
            worker_id=worker.worker_id,
            execution_provider=sorted(worker.capabilities.execution_providers)[0],
            score=score,
            reasons=("hard requirements satisfied",),
        )
        for score, worker in chosen
    )
    platforms = frozenset(worker.capabilities.platform for _, worker in chosen)
    return FleetSelectionDecision(
        decision_id=request.decision_id,
        selected_workers=selected,
        rejected_workers=tuple(rejected),
        platform_coverage=platforms,
        runtime_coverage={
            "python": tuple(sorted(set().union(*(set(worker.capabilities.python_versions) for _, worker in chosen)))),
            "node": tuple(sorted(set().union(*(set(worker.capabilities.node_versions) for _, worker in chosen)))),
        },
        estimated_concurrency=sum(
            max(0, worker.health.concurrency_limit - worker.health.active_job_count)
            for _, worker in chosen
        ),
        verification_policy="strict-required-coverage/no-provider-fallback",
    )
