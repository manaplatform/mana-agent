"""Machine-readable contracts for the narrow computer tool surface."""

from __future__ import annotations

from mana_agent.tools.contracts import ToolContract


_TOOLS: tuple[tuple[str, str], ...] = (
    ("computer_capabilities", "Discover desktop capability groups; application details require apps-read permission."),
    (
        "computer_permission_status",
        "Read one permission decision without creating a prompt; `ask` means submit the exact selected action to create its in-chat request.",
    ),
    ("computer_list_apps", "List discovered installed application metadata."),
    ("computer_open_app", "Open one validated installed application."),
    ("computer_close_app", "Close one validated application; unsaved work may be affected."),
    ("computer_active_app", "Read the active application identifier."),
    ("calendar_list_events", "Read private calendar event metadata with explicit permission."),
    ("calendar_create_event", "Create a calendar event after permission and preview checks."),
    ("calendar_update_event", "Update a calendar event after permission and preview checks."),
    ("calendar_delete_event", "Delete one calendar event after exact-action confirmation."),
    ("media_get_status", "Read active playback metadata."),
    ("media_play", "Play model-selected media in an available adapter."),
    ("media_pause", "Pause active media playback."),
    ("media_next", "Advance active media playback."),
    ("media_previous", "Return to the previous media item."),
    ("media_set_volume", "Set media volume between zero and one."),
    ("notes_search", "Search private notes with explicit notes-read permission."),
    ("notes_read", "Read one private note with explicit permission."),
    ("notes_create", "Create a note with notes-write permission."),
    ("notes_update", "Update one note with notes-write permission."),
    ("notes_delete", "Delete one note after exact-action confirmation."),
    ("browser_get_active_page", "Read active browser URL/title, never private browser storage."),
    ("browser_read_page", "Read accessible active-page text with explicit permission."),
    ("browser_list_tabs", "List browser tab titles and URLs with explicit permission."),
    ("browser_open_url", "Open one validated absolute HTTP(S) URL."),
    ("browser_activate_tab", "Activate one validated browser tab."),
    ("browser_close_tab", "Close one validated browser tab."),
    ("clipboard_read", "Read sensitive clipboard content with explicit permission."),
    ("clipboard_write", "Replace clipboard text with explicit permission."),
    ("computer_open_path", "Open one file/folder inside configured filesystem roots."),
    ("computer_reveal_path", "Reveal one allowed path in the native file manager."),
    ("computer_file_metadata", "Read metadata for one allowed path."),
    ("computer_copy_path", "Copy one allowed path to another allowed path."),
    ("computer_move_path", "Move one allowed path to another allowed path."),
    ("computer_rename_path", "Rename one allowed path."),
    ("computer_create_directory", "Create one directory inside an allowed root."),
    ("computer_trash_path", "Move one allowed path to OS Trash/Recycle Bin after exact confirmation."),
    ("computer_take_screenshot", "Capture visible screen content with explicit permission."),
    ("computer_send_notification", "Display a local notification."),
    ("computer_get_system_status", "Read non-content volume, battery, and display state."),
    ("computer_set_system_volume", "Change system volume or mute state."),
    ("computer_control_system", "Open settings or perform confirmed lock/power operations."),
)


def computer_tool_contracts() -> list[ToolContract]:
    input_schema = {
        "type": "object",
        "properties": {
            "source_decision_id": {"type": "string", "minLength": 1},
            "execution_id": {"type": ["string", "null"]},
            "confirmation_token": {"type": ["string", "null"]},
        },
        "required": ["source_decision_id"],
        "additionalProperties": True,
    }
    output_schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}, "result": {}, "error_code": {"type": "string"}, "message": {"type": "string"}},
        "required": ["ok"],
    }
    error = {
        "type": "object",
        "properties": {"ok": {"const": False}, "error_code": {"type": "string"}, "message": {"type": "string"}, "corrective_action": {"type": "string"}},
        "required": ["ok", "error_code", "message"],
    }
    safety = [
        "Execute only after a typed validated model decision selects this exact tool and arguments.",
        "Never accept raw shell, AppleScript, PowerShell, D-Bus, COM, or accessibility commands.",
        "High and critical actions require a short-lived exact-action confirmation token.",
        "Never expose passwords, cookies, authentication tokens, payment data, or private browser storage.",
    ]
    return [
        ToolContract(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            error_format=error,
            safety_rules=safety,
        )
        for name, description in _TOOLS
    ]
