"""Computer-control settings and truthful capability matrix."""

from __future__ import annotations

import asyncio
from pathlib import Path

import streamlit as st

from mana_agent.config.user_config import save_user_config
from mana_agent.integrations.computer_control.config import ComputerControlSettings
from mana_agent.integrations.computer_control.cancellation import (
    approve_computer_action,
    decide_computer_permission,
    deny_computer_permission,
)
from mana_agent.integrations.computer_control.discovery import select_provider
from mana_agent.integrations.computer_control.models import PermissionDecision
from mana_agent.integrations.computer_control.service import default_computer_control_service


def render(_root: Path) -> None:
    st.header("Computer control")
    st.caption("Disabled by default. Desktop actions use scoped permissions, exact-action confirmations, and sanitized audit records.")
    settings = ComputerControlSettings.load()
    enabled = st.toggle("Enable computer control", value=settings.enabled)
    allow_remote = st.toggle(
        "Allow remote control",
        value=settings.allow_remote_control,
        help="Remote clients still require explicit allowlisting; private-data scopes remain disabled separately.",
    )
    require_local = st.toggle(
        "Require trusted local confirmation for high-risk remote actions",
        value=settings.require_local_confirmation_for_high_risk,
    )
    audit_enabled = st.toggle("Enable sanitized audit log", value=settings.audit_enabled)
    retention = st.number_input("Audit retention (days)", min_value=1, max_value=3650, value=settings.audit_retention_days)
    allowed_paths = st.text_area(
        "Allowed filesystem paths (one absolute path per line)",
        value="\n".join(str(path) for path in settings.allowed_paths),
    )
    st.subheader("Permissions")
    permission_values: dict[str, str] = {}
    choices = ["denied", "ask", "always"]
    for scope, current in sorted(settings.permissions.items()):
        selected = current if current in choices else "ask"
        permission_values[scope] = st.selectbox(
            scope,
            choices,
            index=choices.index(selected),
            key=f"computer-permission-{scope}",
        )
    if st.button("Save computer-control settings", type="primary"):
        payload = settings.model_dump()
        payload.update({
                "enabled": enabled,
                "allow_remote_control": allow_remote,
                "require_local_confirmation_for_high_risk": require_local,
                "audit_enabled": audit_enabled,
                "audit_retention_days": int(retention),
                "allowed_paths": [Path(line.strip()).expanduser() for line in allowed_paths.splitlines() if line.strip()],
                "permissions": permission_values,
        })
        try:
            updated = ComputerControlSettings.model_validate(payload)
        except ValueError as exc:
            st.error(f"Settings were not saved: {exc}")
            return
        save_user_config({
            "MANA_COMPUTER_CONTROL_ENABLED": updated.enabled,
            "computer_control": updated.model_dump(mode="json"),
        })
        st.success("Computer-control settings saved.")

    st.subheader("Capability matrix")
    if not settings.enabled:
        st.info("Enable and save computer control to inspect this desktop. No discovery is performed while disabled.")
        return
    try:
        report = asyncio.run(select_provider(settings).discover_capabilities())
    except Exception as exc:
        st.error(f"Capability discovery failed safely: {exc}")
        return
    st.caption(f"Provider: {report.provider} · Platform: {report.platform.value} · Headless: {report.headless}")
    st.dataframe([
        {
            "Capability": item.name,
            "Available": item.available,
            "Reason / required OS support": item.reason,
            "Permission scopes": ", ".join(sorted(item.permission_scopes)),
            "Implemented operations": ", ".join(sorted(item.operations)),
        }
        for item in report.capabilities
    ], use_container_width=True, hide_index=True)

    st.subheader("Permission requests")
    service = default_computer_control_service()
    pending_permissions = service.pending_permissions()
    if not pending_permissions:
        st.caption("No computer action is waiting for permission.")
    for item in pending_permissions:
        request_id = item["permission_request_id"]
        with st.container(border=True):
            st.write(item["preview"])
            st.caption(
                f"Scope: {item['permission_scope']} · Requested by: "
                f"{item['requesting_client']} · Expires: {item['expires_at']}"
            )
            deny_col, once_col, session_col, always_col = st.columns(4)
            if deny_col.button("Deny", key=f"deny-permission-{request_id}"):
                deny_computer_permission(request_id, client_type="dashboard")
                st.warning("Computer action denied.")
                st.rerun()
            choices = (
                (once_col, "Allow once", PermissionDecision.ALLOW_ONCE, "once"),
                (session_col, "This session", PermissionDecision.ALLOW_SESSION, "session"),
                (always_col, "Always", PermissionDecision.ALWAYS_ALLOW, "always"),
            )
            for column, label, decision, suffix in choices:
                if column.button(label, key=f"approve-permission-{suffix}-{request_id}"):
                    try:
                        result = decide_computer_permission(
                            request_id,
                            decision=decision,
                            client_type="dashboard",
                        )
                    except Exception as exc:
                        st.error(f"Permission approval or action execution failed: {exc}")
                    else:
                        st.success(f"Permission approved and action executed: {result.message}")
                        if result.data.get("artifact_path"):
                            st.code(str(result.data["artifact_path"]))

    st.subheader("Pending confirmations")
    pending = service.pending_confirmations()
    if not pending:
        st.caption("No high-risk or critical computer action is waiting.")
    for item in pending:
        with st.container(border=True):
            st.write(item["preview"])
            st.caption(f"Expires: {item['expires_at']}")
            if st.button("Approve exact action", key=f"approve-{item['confirmation_request_id']}", type="primary"):
                result = approve_computer_action(
                    item["confirmation_request_id"],
                    client_type="dashboard",
                )
                st.success(f"Approved and executed: {result.message}")
