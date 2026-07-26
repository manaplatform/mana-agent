"""Actionable, fail-closed Fleet errors."""


class FleetError(RuntimeError):
    """Base error for fleet operations."""


class FleetDisabledError(FleetError):
    pass


class FleetDecisionError(FleetError):
    pass


class FleetCapabilityError(FleetError):
    pass


class FleetSelectionError(FleetDecisionError):
    pass


class FleetStateError(FleetError):
    pass


class FleetPersistenceError(FleetError):
    pass
