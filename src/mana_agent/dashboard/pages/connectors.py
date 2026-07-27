from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.connectors.service import ConnectorService, TelegramConnectRequest


def render(root: Path | None = None) -> None:
    root = (root or Path.cwd()).resolve()
    service = ConnectorService()
    st.header("Connectors")
    for row in service.list():
        st.write(f"**{row['name']}** — {row['state']} · {row['transport']}")
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
                    transport=transport, repository=repository,
                    allowed_users=ids(allowed_users), allowed_chats=ids(allowed_chats),
                    webhook_url=webhook_url,
                    secret_source="keyring" if secret_source == "keyring" else "environment",
                ),
                token=token,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success(result.message)
