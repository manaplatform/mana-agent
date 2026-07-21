"""Structured remote execution fabric errors."""

from __future__ import annotations


class ExecutionFabricError(RuntimeError):
    code = "execution_fabric_error"

    def __init__(self, message: str, *, provider: str = "", diagnostics: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.diagnostics = diagnostics


class ProviderUnavailableError(ExecutionFabricError):
    code = "provider_unavailable"


class ProviderConfigurationError(ExecutionFabricError):
    code = "provider_configuration"


class CapabilityMismatchError(ExecutionFabricError):
    code = "capability_mismatch"


class ProvisioningError(ExecutionFabricError):
    code = "provisioning"


class ExecutionTimeoutError(ExecutionFabricError):
    code = "execution_timeout"


class ExecutionFailedError(ExecutionFabricError):
    code = "execution_failed"


class SnapshotError(ExecutionFabricError):
    code = "snapshot"


class ArtifactError(ExecutionFabricError):
    code = "artifact"


class PolicyEnforcementError(ExecutionFabricError):
    code = "policy_enforcement"


class CleanupError(ExecutionFabricError):
    code = "cleanup"


class LifecycleTransitionError(ExecutionFabricError):
    code = "lifecycle_transition"
