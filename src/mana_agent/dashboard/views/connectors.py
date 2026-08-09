from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.connectors.service import ConnectorService, TelegramConnectRequest


_STATE_LABELS = {
    "healthy": "Healthy",
    "degraded": "Degraded",
    "recovering": "Recovering",
    "offline": "Offline",
    "auth_required": "Authentication Required",
    "rate_limited": "Rate Limited",
    "disabled": "Disabled",
    "unknown": "Unknown",
}


def render(root: Path | None = None) -> None:
    root = (root or Path.cwd()).resolve()
    service = ConnectorService()
    st.header("Connectors")
    st.caption(
        "Health reflects authentication, transport, ingress, and egress — "
        "not merely whether a connector process is running."
    )

    rows = service.list()
    if not rows:
        st.info("No connectors configured.")
    for row in rows:
        health = row.get("health") or {}
        health_state = str(row.get("health_state") or health.get("state") or "unknown")
        label = _STATE_LABELS.get(health_state, health_state.replace("_", " ").title())
        process_state = row.get("state", "unknown")
        title = row.get("name", "connector").title()
        if row.get("account_id"):
            title = f"{title} ({row.get('address') or row['account_id']})"
        st.subheader(title)
        cols = st.columns(4)
        cols[0].metric("Health", label)
        cols[1].metric("Process", process_state)
        cols[2].metric("Auth", str(health.get("auth") or "unknown").upper())
        cols[3].metric("Ingress", str(health.get("ingress") or "unknown").upper())
        st.write(
            f"Egress: **{str(health.get('egress') or 'unknown').upper()}** · "
            f"Transport: **{row.get('transport') or 'n/a'}** · "
            f"Last checked: `{health.get('checked_at') or 'n/a'}`"
        )
        if health.get("message"):
            st.caption(health["message"])
        if health.get("incident_id"):
            st.caption(f"Current incident: `{health['incident_id']}`")
        if health.get("latency_ms") is not None:
            st.caption(f"Probe latency: {health['latency_ms']:.0f} ms")
        # Explicit false-online guardrail in the UI copy
        if process_state in {"running", "starting"} and health_state not in {"healthy", "disabled"}:
            st.warning(
                "Process is running but connector path health is not Healthy. "
                "Do not treat this connector as online."
            )
        st.divider()

    with st.expander("Incident history"):
        try:
            from mana_agent.connectors.health import bootstrap_health_manager, reset_health_manager

            reset_health_manager()
            manager = bootstrap_health_manager()
            incidents = manager.list_incidents(limit=30)
            if not incidents:
                st.write("No incidents recorded.")
            for incident in incidents:
                status = "OPEN" if incident.open else ("RECOVERED" if incident.recovered else "CLOSED")
                st.markdown(
                    f"**{incident.connector_id}** · {status} · "
                    f"`{incident.opening_state.value}` · {incident.opening_reason.value}"
                )
                for event in incident.events[-8:]:
                    st.text(
                        f"{event.occurred_at.isoformat()} {event.event_type} "
                        f"{event.reason_code.value} {event.message}"
                    )
        except Exception as exc:
            st.error(f"Unable to load incidents: {exc}")

    with st.form("telegram_connect", clear_on_submit=True):
        st.subheader("Connect Telegram")
        token = st.text_input("Bot token", type="password", key="telegram_token_secret")
        transport = st.selectbox("Transport", ["auto", "polling", "webhook"])
        repository = st.text_input("Repository", value=str(root))
        allowed_users = st.text_input("Allowed user IDs (comma-separated)")
        allowed_chats = st.text_input("Allowed chat IDs (comma-separated)")
        webhook_url = st.text_input("Webhook public URL (when used)")
        secret_source = st.selectbox("Secret source", ["keyring", "environment"])
        submitted = st.form_submit_button("Validate and connect")
    if submitted:

        def ids(value: str) -> list[int]:
            return [int(item.strip()) for item in value.split(",") if item.strip()]

        try:
            result = service.connect_telegram(
                TelegramConnectRequest(
                    transport=transport,
                    repository=repository,
                    allowed_users=ids(allowed_users),
                    allowed_chats=ids(allowed_chats),
                    webhook_url=webhook_url,
                    secret_source="keyring"
                    if secret_source == "keyring"
                    else "environment",
                ),
                token=token,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success(result.message)
