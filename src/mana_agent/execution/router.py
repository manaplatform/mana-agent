"""Capability-aware routing over validated model-produced requirements."""

from __future__ import annotations

from mana_agent.execution.config import ExecutionConfig
from mana_agent.execution.errors import CapabilityMismatchError, ProviderUnavailableError
from mana_agent.execution.models import EnforcementStrength, RoutingDecision, RoutingRequest, SandboxCapabilities
from mana_agent.execution.registry import ProviderRegistry


_CAPABILITY_FIELDS = {
    "snapshots": "snapshots", "native_suspend_resume": "native_suspend_resume",
    "emulated_suspend_resume": "emulated_suspend_resume", "gpu_execution": "gpu_execution",
    "secret_files": "secret_files", "secret_environment_variables": "secret_environment_variables",
    "persistent_volumes": "persistent_volumes", "artifact_streaming": "artifact_streaming",
    "parallel_execution": "parallel_execution",
}


class ExecutionRouter:
    def __init__(self, registry: ProviderRegistry, config: ExecutionConfig) -> None:
        self.registry = registry
        self.config = config

    def _candidate_order(self, request: RoutingRequest) -> tuple[list[str], str, bool]:
        if request.explicit_provider:
            return [request.explicit_provider], "explicit-provider", True
        if request.organization_provider:
            return [request.organization_provider], "organization-policy", True
        if request.resources.gpu_count:
            preferred, rule = self.config.routing.gpu_provider, "gpu-workload"
        elif request.risk_level == "high" or request.trust_level == "untrusted":
            preferred, rule = self.config.routing.high_risk_provider, "high-risk-isolation"
        elif request.parallelism > 1 or request.expected_duration_seconds > 1800:
            preferred, rule = self.config.routing.expensive_provider, "parallel-or-expensive"
        else:
            preferred, rule = self.config.default_provider, "trusted-low-cost"
        # Alternate candidates are considered only before provisioning and only
        # when they satisfy the exact same requirements. Runtime failures never
        # cross this boundary.
        order = [preferred, *[name for name in self.config.allowed_providers if name != preferred]]
        return order, rule, False

    def _mismatch(self, request: RoutingRequest, capabilities: SandboxCapabilities) -> str:
        missing = sorted(name for name in request.required_capabilities if not bool(getattr(capabilities, _CAPABILITY_FIELDS.get(name, ""), False)))
        if missing:
            return f"missing capabilities: {', '.join(missing)}"
        if not capabilities.resource_isolation.satisfies(request.required_resource_enforcement):
            return f"resource enforcement is {capabilities.resource_isolation.value}"
        if not capabilities.network_isolation.satisfies(request.network.required_enforcement):
            return f"network enforcement is {capabilities.network_isolation.value}"
        if request.resources.gpu_count and not capabilities.gpu_execution:
            return "GPU execution unavailable"
        if (request.risk_level == "high" or request.trust_level == "untrusted") and capabilities.resource_isolation != EnforcementStrength.ENFORCED:
            return "untrusted/high-risk execution requires enforced resource isolation"
        if request.parallelism > 1 and not capabilities.parallel_execution:
            return "parallel execution unavailable"
        return ""

    async def route(self, request: RoutingRequest) -> RoutingDecision:
        order, rule, explicit = self._candidate_order(request)
        rejected: dict[str, str] = {}
        for name in order:
            if name not in self.config.allowed_providers:
                rejected[name] = "provider denied by configuration"
                continue
            if not self.config.providers[name].enabled:
                rejected[name] = "provider disabled by configuration"
                if explicit:
                    break
                continue
            try:
                provider = self.registry.get(name)
            except ProviderUnavailableError:
                rejected[name] = "provider disabled or not configured"
                if explicit:
                    break
                continue
            health = await provider.healthcheck()
            if not health.available:
                rejected[name] = health.message or "provider unavailable"
                if explicit:
                    break
                continue
            mismatch = self._mismatch(request, await provider.capabilities())
            if mismatch:
                rejected[name] = mismatch
                if explicit:
                    break
                continue
            return RoutingDecision(
                decision_id=request.decision_id, selected_provider=name, policy_rule=rule,
                requirements_considered=[
                    f"trust={request.trust_level}", f"risk={request.risk_level}",
                    f"network={request.network.mode.value}/{request.network.required_enforcement.value}",
                    f"resource_enforcement={request.required_resource_enforcement.value}",
                    f"gpu={request.resources.gpu_count}", f"parallelism={request.parallelism}",
                    *sorted(request.required_capabilities),
                ], rejected_providers=rejected, explicit=explicit,
            )
        message = "; ".join(f"{name}: {reason}" for name, reason in rejected.items()) or "no providers configured"
        raise CapabilityMismatchError(f"no provider satisfies routing decision {request.decision_id}: {message}")
