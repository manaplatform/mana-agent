"""Shared worker service lifecycle errors."""


class WorkerServiceError(RuntimeError):
    """Raised when an installed worker service cannot be controlled."""
