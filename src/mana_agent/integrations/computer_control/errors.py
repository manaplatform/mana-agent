"""User-facing computer-control failures."""

from __future__ import annotations


class ComputerControlError(RuntimeError):
    code = "computer_control_error"

    def __init__(self, message: str, *, corrective_action: str = "") -> None:
        super().__init__(message)
        self.corrective_action = corrective_action

    def payload(self) -> dict[str, str]:
        return {"error_code": self.code, "message": str(self), "corrective_action": self.corrective_action}


class ComputerControlDisabled(ComputerControlError):
    code = "computer_control_disabled"


class UnsupportedPlatform(ComputerControlError):
    code = "unsupported_platform"


class CapabilityUnavailable(ComputerControlError):
    code = "capability_unavailable"


class ApplicationNotInstalled(ComputerControlError):
    code = "application_not_installed"


class AdapterUnavailable(ComputerControlError):
    code = "application_adapter_unavailable"


class OperatingSystemPermissionDenied(ComputerControlError):
    code = "os_permission_denied"


class ManaPermissionDenied(ComputerControlError):
    code = "mana_permission_denied"


class PermissionApprovalRequired(ComputerControlError):
    code = "permission_required"

    def __init__(
        self,
        message: str,
        *,
        permission_request_id: str,
        permission_scope: str,
        preview: str,
        execution_id: str,
    ) -> None:
        super().__init__(message)
        self.permission_request_id = permission_request_id
        self.permission_scope = permission_scope
        self.preview = preview
        self.execution_id = execution_id

    def payload(self) -> dict[str, str]:
        return {
            **super().payload(),
            "permission_request_id": self.permission_request_id,
            "permission_scope": self.permission_scope,
            "preview": self.preview,
            "execution_id": self.execution_id,
        }


class ConfirmationRequired(ComputerControlError):
    code = "confirmation_required"

    def __init__(self, message: str, *, preview: str, confirmation_request_id: str) -> None:
        super().__init__(message)
        self.preview = preview
        self.confirmation_request_id = confirmation_request_id

    def payload(self) -> dict[str, str]:
        return {
            **super().payload(),
            "preview": self.preview,
            "confirmation_request_id": self.confirmation_request_id,
        }


class InvalidConfirmation(ComputerControlError):
    code = "invalid_confirmation"


class ApplicationNotResponding(ComputerControlError):
    code = "application_not_responding"


class UIElementNotFound(ComputerControlError):
    code = "ui_element_not_found"


class BrowserPageInaccessible(ComputerControlError):
    code = "browser_page_inaccessible"


class UnsavedWorkDetected(ComputerControlError):
    code = "unsaved_work_detected"


class ActionTimedOut(ComputerControlError):
    code = "action_timed_out"


class ActionCancelled(ComputerControlError):
    code = "action_cancelled"


class DesktopSessionUnavailable(ComputerControlError):
    code = "desktop_session_unavailable"


class HeadlessEnvironment(ComputerControlError):
    code = "headless_environment"


class PlatformLimitation(ComputerControlError):
    code = "platform_limitation"


class InvalidActionDecision(ComputerControlError):
    code = "invalid_action_decision"


class RemoteControlDenied(ComputerControlError):
    code = "remote_control_denied"
