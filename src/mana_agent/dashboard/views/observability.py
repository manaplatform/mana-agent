from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    get_observability_health,
    get_observability_overview,
    load_observability_spans,
    load_observability_trace,
)


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    st.header("Observability")
    st.caption("Local redacted spans from the canonical ObservabilityStore.")
    f1, f2, f3 = st.columns(3)
    status = f1.selectbox(
        "Status",
        ["", "success", "failed", "running", "queued"],
        format_func=lambda v: v or "All",
    )
    kind = f2.selectbox(
        "Span type",
        [
            "",
            "session",
            "user_request",
            "routing",
            "reasoning",
            "tool",
            "subagent",
            "response",
            "error",
        ],
        format_func=lambda v: v or "All",
    )
    agent = f3.text_input("Agent", placeholder="main or subagent id")
    spans = load_observability_spans(
        root, status=status, kind=kind, agent=agent, limit=500
    )
    overview = get_observability_overview(root)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traces", overview.get("trace_count", 0))
    c2.metric("Spans", overview.get("span_count", 0))
    c3.metric("p95 latency", f"{overview.get('p95_latency_ms', 0):,.0f} ms")
    c4.metric("Errors", overview.get("error_count", 0))
    traces = sorted({span["trace_id"] for span in spans})
    selected = st.selectbox("Trace", traces) if traces else ""
    if selected:
        for span in reversed(load_observability_trace(selected, root)):
            label = f"{span['kind']} · {span['title']} · {span['status']} · {float(span['duration_ms'] or 0):.0f}ms"
            with st.expander(label):
                st.json(span)
    st.caption("OTLP health: " + str(get_observability_health(root)))
