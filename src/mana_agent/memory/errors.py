"""Typed failures raised by the provider-neutral memory layer."""


class MemoryError(RuntimeError):
    """Base class for memory failures safe to expose to callers."""


class MemoryConfigurationError(MemoryError):
    pass


class MemoryDependencyError(MemoryError):
    pass


class MemoryAuthenticationError(MemoryError):
    pass


class MemoryNetworkError(MemoryError):
    pass


class MemoryProviderError(MemoryError):
    pass


class MemoryStorageError(MemoryError):
    pass


class MemoryNotFoundError(MemoryError):
    pass
