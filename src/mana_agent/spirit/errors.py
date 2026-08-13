from __future__ import annotations


class SpiritResolutionError(ValueError):
    """Raised when a Spirit identifier cannot be resolved.

    Callers must stop. No substitute Spirit is selected.
    """

    def __init__(self, message: str, *, spirit_id: str = "", spirit_version: int | None = None) -> None:
        super().__init__(message)
        self.spirit_id = spirit_id
        self.spirit_version = spirit_version
