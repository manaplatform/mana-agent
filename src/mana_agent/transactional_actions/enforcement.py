from __future__ import annotations


ROUTED_SIDE_EFFECT_TOOLS = frozenset({
    "edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file",
    "create_file", "delete_file", "document_create", "document_update", "document_delete",
    "run_command", "run_script_once", "git_generic", "git_create_branch", "git_switch",
    "git_add", "git_commit", "git_push", "api_request_execute",
})

TRANSACTIONAL_COMPUTER_TOOLS = frozenset({
    "computer_capabilities", "computer_permission_status", "computer_list_apps",
    "computer_open_app", "computer_close_app", "computer_active_app",
    "calendar_list_events", "calendar_create_event", "calendar_update_event",
    "calendar_delete_event", "media_get_status", "media_play", "media_pause",
    "media_next", "media_previous", "media_set_volume", "notes_search",
    "notes_read", "notes_create", "notes_update", "notes_delete",
    "browser_get_active_page", "browser_read_page", "browser_list_tabs",
    "browser_open_url", "browser_activate_tab", "browser_close_tab",
    "clipboard_read", "clipboard_write", "computer_open_path",
    "computer_reveal_path", "computer_file_metadata", "computer_copy_path",
    "computer_move_path", "computer_rename_path", "computer_create_directory",
    "computer_trash_path", "computer_take_screenshot",
    "computer_send_notification", "computer_get_system_status",
    "computer_set_system_volume", "computer_control_system",
})

READ_ONLY_MODEL_TOOLS = frozenset({
    "semantic_search", "read_file", "chunk_file", "list_tools", "ls", "repo_search",
    "repo_batch_read", "repo_batch_search", "list_files", "find_symbols", "call_graph",
    "tool_contracts", "read_skill", "github_search", "web_search",
    "git_status", "git_diff", "git_log", "git_show", "git_branch", "git_remote", "git_help",
    "document_detect", "document_read", "document_analyze", "document_query",
    "email_accounts_list", "email_search", "email_read", "email_thread_read",
    "get_media_generation_status", "automation_get", "automation_list", "automation_status",
    "canvas_get_surface", "canvas_list_surfaces", "canvas_wait_for_action",
    "api_workflow_decide", "api_docs_inspect", "api_integrations_list", "api_integration_get",
    "api_operations_search", "api_request_preview",
})

UNROUTED_SIDE_EFFECT_TOOLS = frozenset({
    "canvas_create_surface", "canvas_update_components", "canvas_update_data", "canvas_delete_surface",
    "automation_create", "automation_update", "automation_delete", "automation_enable",
    "automation_disable", "automation_run_now", "api_docs_import", "api_docs_import_semantic",
    "api_integration_update", "api_integration_delete", "generate_image", "generate_voice",
    "generate_video", "cancel_media_generation", "browser_upload", "browser_close",
    "verify_project",
})

UNROUTED_SIDE_EFFECT_PREFIXES = (
    "server_", "browser_", "mcp__",
)


class TransactionalGatewayRequired(PermissionError):
    pass


def assert_model_tool_routed(tool_name: str, metadata: dict | None = None) -> None:
    """Fail closed for known side-effect tools without a shared action adapter."""
    name = str(tool_name or "").strip()
    declared = dict(metadata or {})
    if declared.get("read_only") is True and declared.get("side_effecting") is not True:
        return
    if name.startswith("mcp__") and declared.get("transactional_adapter") == "mcp":
        return
    if (
        name in ROUTED_SIDE_EFFECT_TOOLS
        or name in READ_ONLY_MODEL_TOOLS
        or name in TRANSACTIONAL_COMPUTER_TOOLS
    ):
        return
    if name in UNROUTED_SIDE_EFFECT_TOOLS or name.startswith(UNROUTED_SIDE_EFFECT_PREFIXES):
        raise TransactionalGatewayRequired(
            f"side-effecting tool {name!r} has no registered transactional action adapter; "
            "no legacy or direct execution was performed"
        )
    raise TransactionalGatewayRequired(
        f"tool {name!r} has no explicit read-only classification or transactional adapter; "
        "default-deny policy prevented execution"
    )
