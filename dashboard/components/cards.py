"""Card / UI components for dashboard (expanded)."""
import streamlit as st
from typing import Any


def stat_card(label: str, value: Any, delta: str | None = None) -> None:
    """Simple metric card wrapper."""
    if delta:
        st.metric(label, value, delta)
    else:
        st.metric(label, value)


def automation_card(name: str, trigger: str, action: str, enabled: bool) -> None:
    """Compact automation summary."""
    icon = "✅" if enabled else "⏸️"
    st.markdown(f"**{icon} {name}**  \n`{trigger}` → `{action}`")


def trace_expander(trace: dict[str, Any]) -> None:
    key = f"{trace.get('_file', 'trace')} · {trace.get('kind') or trace.get('event_type') or trace.get('event', 'event')}"
    with st.expander(key):
        st.json(trace)
