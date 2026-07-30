from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MediaError(RuntimeError):
    """Safe provider-neutral media error.

    ``detail`` is suitable for user interfaces and must never contain provider
    response bodies, credentials, or signed download URLs.
    """

    code: str
    detail: str
    retryable: bool = False

    def __str__(self) -> str:
        return self.detail


class MediaConfigurationError(MediaError):
    pass


class MediaCapabilityError(MediaError):
    pass


class MediaValidationError(MediaError):
    pass


class MediaProviderError(MediaError):
    pass


class MediaArtifactError(MediaError):
    pass
