"""Mana Agent Web Dashboard (Grok Build - Streamlit MVP).

Entry point for the optional web UI.

Run:
    streamlit run dashboard/app.py
    # or after CLI integration:
    mana-agent dashboard

Design:
- Read-only first (safe).
- Reuses existing .mana/ artifacts, renderers, and multi-agent concepts.
- Sidebar navigation.
- "Powered by mana-agent multi-agent runtime" branding.
- Lazy / optional: core package does not require streamlit.

Grok Build rules followed:
- Inspect + small focused files.
- No changes to CLI core, multi_agent, routing, or decision layer.
- Graceful if optional deps missing (handled in CLI wrapper later).
- Model-driven philosophy preserved (dashboard visualizes decisions/traces).

Expanded: real triggers, chat embed (st.chat), reports+generate, live taskboard+traces,
metrics from telemetry, automations CRUD + sidebar triggers. Productional use via .mana/ state.
Model decisions + explicit user intent preserved for all actions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError as e:
    print("ERROR: streamlit is required to run the dashboard.", file=sys.stderr)
    print("pip install 'mana-agent[dashboard]'", file=sys.stderr)
    raise SystemExit(1) from e

# Lazy import of helpers (never at top of core)
try:
    from mana_agent.ui.streamlit_helpers import (
        append_automation_run,
        create_schedule,
        delete_schedule,
        find_mana_root,
        get_index_stats,
        get_last_analysis_summary,
        get_metrics_summary,
        get_observability_health,
        get_observability_overview,
        list_analysis_artifacts,
        load_automations,
        list_schedules,
        load_recent_traces,
        load_observability_spans,
        load_observability_trace,
        load_taskboard_state,
        safe_read_json,
        save_automations,
        run_schedule_now,
        schedule_status,
        set_schedule_enabled,
        trigger_automation,
        run_dashboard_chat,
    )
except Exception as e:  # pragma: no cover - dashboard optional
    st.error(f"Failed to load mana-agent helpers: {e}")
    st.stop()

st.set_page_config(
    page_title="Mana Agent Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Branding / Header ---
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
      <span style="font-size:28px">🧠</span>
      <div>
        <h1 style="margin:0">Mana Agent</h1>
        <div style="color:#888;font-size:0.9rem">Web Dashboard • Powered by multi-agent runtime</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

root = find_mana_root()
st.caption(f"Repository root: `{root}`")

# --- Sidebar (button navigation + automation triggers) ---
st.sidebar.markdown("## 🧠 Mana Agent")
st.sidebar.caption(f"Root: `{root.name}`")
st.markdown("""<style>
div[data-testid="stSidebar"] button[kind="secondary"] {text-align:left; border:0; border-radius:8px;}
</style>""", unsafe_allow_html=True)
pages = ["Overview", "Chat", "Reports", "Observability", "Taskboard & Traces", "Metrics", "Automations", "Cron Jobs"]
if "dashboard_page" not in st.session_state:
    st.session_state.dashboard_page = "Overview"
st.sidebar.markdown("### Navigation")
for item in pages:
    prefix = "● " if st.session_state.dashboard_page == item else "○ "
    if st.sidebar.button(prefix + item, key=f"nav_{item}", use_container_width=True):
        st.session_state.dashboard_page = item
        st.rerun()
page = st.session_state.dashboard_page

st.sidebar.divider()
st.sidebar.markdown("### ⚡ Automation Triggers")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Self-Improve", key="side_self", use_container_width=True):
        res = trigger_automation("self_improvement", root=root, limit=3)
        st.sidebar.success(f"Self-improve: {res.get('created', res)}")
        st.rerun()
    if st.button("Daily Report", key="side_daily", use_container_width=True):
        res = trigger_automation("daily_report", root=root)
        st.sidebar.info(str(res.get("note", res))[:80])
with col_b:
    if st.button("Generate Report", key="side_analyze", use_container_width=True):
        res = trigger_automation("analyze", root=root)
        st.sidebar.success("Analyze queued/executed (see Reports)")
        st.rerun()
    if st.button("Refresh All", key="side_refresh", use_container_width=True):
        st.rerun()

st.sidebar.caption("All actions respect model decision layer or explicit user intent.")
if st.sidebar.button("Refresh data", key="main_refresh"):
    st.rerun()

# --- Page content ---
if page == "Overview":
    st.header("Project Overview")

    m = get_metrics_summary(root)
    col1, col2, col3 = st.columns(3)
    idx = get_index_stats(root)
    col1.metric("Index Ready", "✅" if idx.get("ready") else "❌", idx.get("chunks", 0))
    col2.metric("Sessions / Turns", m.get("sessions", 0), f"{m.get('done_tasks', 0)} done")
    col3.metric("Success Rate", f"{m.get('success_rate', 0)}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tokens", m.get("total_tokens", 0))
    c2.metric("Avg / Turn", m.get("avg_tokens", 0))
    c3.metric("Tasks", m.get("task_count", 0))

    st.subheader("Last Analysis")
    analysis = get_last_analysis_summary(root)
    if analysis.get("type") == "md":
        st.markdown(analysis.get("preview", "")[:1500])
        st.caption(f"Source: {analysis.get('path')}")
    elif analysis.get("type") == "json":
        st.json(analysis.get("data", {}))
    else:
        st.info(analysis.get("message", "No analysis yet."))

    st.subheader("Quick Actions (safe)")
    if st.button("Run Analysis (via trigger)"):
        r = trigger_automation("analyze", root=root)
        llm_used = r.get("llm_used", False)
        status = "with LLM" if llm_used else "deterministic (no key in ~/.mana/config.toml)"
        st.success(f"Analyze -> .mana/analyze : {r.get('ok')} ({status})")
        if r.get("artifact_dir"):
            st.caption(f"Artifacts: {r.get('artifact_dir')}")
        if r.get("artifacts"):
            st.write("Created:", r["artifacts"][:5])
        st.rerun()  # refresh lists in Reports etc.

    if st.button("Open Chat in terminal"):
        st.code("mana-agent chat --root-dir " + str(root), language="bash")

elif page == "Chat":
    st.header("Chat (embedded - real)")
    st.caption("Full model routing + AskService/AskAgent like CLI chat. Uses entry router decisions and ask_with_tools for agentic answers when possible.")

    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []
        # seed from recent traces (light)
        for t in load_recent_traces(root, limit=2)[:4]:
            st.session_state.chat_msgs.append({"role": "assistant", "content": f"[trace] {t.get('event_type', t.get('kind','event'))}: {str(t)[:140]}"})

    for m in st.session_state.chat_msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about the repo (real model-routed chat)"):
        st.session_state.chat_msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # REAL routed via models, exactly like CLI chat path (AskService / router / AskAgent)
        chat_result = run_dashboard_chat(prompt, root=root)
        reply = chat_result.get("answer", "(no answer)")
        if chat_result.get("mode") == "real":
            sources = chat_result.get("sources", [])
            if sources:
                reply += "\n\n**Sources:** " + ", ".join(
                    [str(s.get("file_path", s))[:60] for s in sources[:3]]
                )
        st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

        # Persist real interaction (productional)
        try:
            chat_dir = root / ".mana" / "dashboard" / "chats"
            chat_dir.mkdir(parents=True, exist_ok=True)
            (chat_dir / "latest.jsonl").open("a", encoding="utf-8").write(
                json.dumps({
                    "ts": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                    "prompt": prompt,
                    "reply": reply,
                    "mode": chat_result.get("mode"),
                }, ensure_ascii=False) + "\n"
            )
        except Exception:
            pass

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.chat_msgs = []
        st.rerun()

elif page == "Reports":
    st.header("Reports & Artifacts")
    st.caption("Real artifacts from .mana/analyze + docs/analyze")
    st.info("LLM analysis requires OPENAI_API_KEY (or equiv) in ~/.mana/config.toml or environment. Run from CLI or here after configuring.")

    arts = list_analysis_artifacts(root)
    if st.button("🔄 Generate / Refresh Report", type="primary"):
        r = trigger_automation("analyze", root=root)
        llm_used = r.get("llm_used", False)
        status = "LLM analysis" if llm_used else "deterministic only (configure key in ~/.mana/config.toml)"
        st.success(f"Real analyze: ok={r.get('ok')} — {status}")
        if r.get("artifact_dir"):
            st.info(f"Artifacts written to: {r['artifact_dir']}")
        if r.get("artifacts"):
            st.caption("Wrote: " + ", ".join(r["artifacts"][:6]))
        arts = list_analysis_artifacts(root)
        st.rerun()

    if not arts:
        st.info("No artifacts yet. Use Generate or run `mana-agent analyze`.")
    else:
        names = [a["name"] for a in arts]
        sel = st.selectbox("Select artifact", names, index=0)
        chosen = next((a for a in arts if a["name"] == sel), None)
        if chosen:
            p = Path(chosen["path"])
            st.caption(str(p))
            if chosen["type"] == "md":
                st.markdown(p.read_text(encoding="utf-8")[:4000])
            elif chosen["type"] == "json":
                st.json(safe_read_json(p) or {})
            else:
                st.code(p.read_text(encoding="utf-8")[:3000])

    st.tabs(["Mermaid", "HTML", "JSON", "Markdown"])  # placeholders for future tabs
    st.code("graph TD\n  A[User] -->|chat| B[MainAgent]\n  B --> C[Taskboard]\n", language="mermaid")

elif page == "Taskboard & Traces":
    st.header("Live Taskboard & Traces")
    st.caption("Loaded from .mana/taskboard/state.json + traces/ (click Refresh for live updates)")

    tb = load_taskboard_state(root)
    tasks_dict = tb.get("tasks", {}) if isinstance(tb, dict) else {}
    tasks = list(tasks_dict.values()) if isinstance(tasks_dict, dict) else []

    # Rich view using real data
    if tasks:
        try:
            import pandas as pd  # optional via dashboard extra
            rows = []
            for t in tasks:
                rows.append({
                    "id": t.get("task_id", "")[:12],
                    "title": t.get("title", "")[:60],
                    "status": t.get("status"),
                    "owner": t.get("owner_agent_id"),
                    "updated": str(t.get("updated_at", ""))[:19],
                    "budget_used": t.get("budget_used_tokens", 0),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            st.json({"task_count": len(tasks), "sample": tasks[:2]})
    else:
        st.info("No tasks in state yet. Run a chat/coding session to populate.")

    with st.expander("Raw taskboard state"):
        st.json(tb, expanded=False)

    st.subheader("Recent Traces")
    traces = load_recent_traces(root, limit=5)
    if traces:
        for t in traces[:8]:
            key = f"{t.get('_file', 'trace')} - {t.get('kind', t.get('event_type', t.get('event', 'event')))}"
            with st.expander(key):
                st.json(t)
    else:
        st.write("No traces found under .mana/traces/")

elif page == "Observability":
    st.header("Observability")
    st.caption("Local, redacted trace storage. Token usage is tracked; monetary cost is intentionally unavailable until pricing is configured.")
    f1, f2, f3 = st.columns(3)
    status = f1.selectbox("Status", ["", "success", "failed", "running", "queued"], format_func=lambda value: value or "All")
    kind = f2.selectbox("Span type", ["", "session", "user_request", "routing", "reasoning", "tool", "subagent", "response", "error"], format_func=lambda value: value or "All")
    agent = f3.text_input("Agent", placeholder="main or subagent id")
    spans = load_observability_spans(root, status=status, kind=kind, agent=agent, limit=500)
    overview = get_observability_overview(root)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traces", overview.get("trace_count", 0))
    c2.metric("Spans", overview.get("span_count", 0))
    c3.metric("p95 latency", f"{overview.get('p95_latency_ms', 0):,.0f} ms")
    c4.metric("Errors", overview.get("error_count", 0))
    st.subheader("Bottlenecks")
    findings = overview.get("bottlenecks", [])
    if findings:
        for finding in findings:
            with st.expander(f"{finding['kind']} · {finding['title']} ({finding['sample_size']} spans)"):
                st.write("; ".join(finding["reasons"]))
                if st.button("Open related trace", key=f"bottleneck_{finding['trace_id']}"):
                    st.session_state.observability_trace_id = finding["trace_id"]
    else:
        st.info("No operation crosses the documented latency, error, queue, or token thresholds yet.")
    st.subheader("Trace explorer")
    traces = sorted({span["trace_id"] for span in spans})
    selected_trace = st.selectbox("Trace", traces, index=0 if traces else None, key="trace_selector") if traces else ""
    selected_trace = st.session_state.get("observability_trace_id", selected_trace)
    if selected_trace:
        trace = load_observability_trace(selected_trace, root)
        for span in reversed(trace):
            label = f"{span['kind']} · {span['title']} · {span['status']} · {float(span['duration_ms'] or 0):.0f}ms"
            with st.expander(label):
                st.json({k: span[k] for k in ("span_id", "parent_span_id", "agent_id", "subagent_id", "started_at", "ended_at", "queue_wait_ms", "token_usage", "input_summary", "output_summary", "error_summary", "attributes")})
    if spans:
        st.subheader("Filtered spans")
        st.dataframe([{k: item[k] for k in ("trace_id", "span_id", "kind", "title", "status", "agent_id", "duration_ms", "queue_wait_ms")} for item in spans], use_container_width=True, hide_index=True)
    st.caption("OTLP export health: " + json.dumps(get_observability_health(root), ensure_ascii=False))

elif page == "Metrics":
    st.header("Metrics (real telemetry)")
    m = get_metrics_summary(root)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions / Turns", m.get("sessions", 0))
    c2.metric("Avg Tokens / turn", m.get("avg_tokens", 0))
    c3.metric("Success rate", f"{m.get('success_rate', 0)}%")

    c4, c5 = st.columns(2)
    c4.metric("Tasks tracked", m.get("task_count", 0))
    c5.metric("Done", m.get("done_tasks", 0))

    overview = get_observability_overview(root)
    by_kind = overview.get("by_kind", [])
    if by_kind:
        st.bar_chart({row["kind"]: row["tokens"] for row in by_kind})
    else:
        st.info("No observability spans have been recorded yet. Start a chat session to populate real metrics.")
    st.caption("Metrics are read from `.mana/observability/telemetry.sqlite`; no synthetic series is displayed.")

elif page == "Automations":
    st.header("Automations (CRUD + real triggers)")
    st.caption("Definitions persisted to .mana/automations/config.json. Triggers execute via src layer.")

    cfg = load_automations(root)
    autos = cfg.get("automations", [])
    runs = cfg.get("runs", [])

    # Create form (CRUD)
    with st.form("new_auto", clear_on_submit=True):
        name = st.text_input("Name", value="Self-Improve on Verify")
        trigger = st.selectbox("Trigger", ["manual", "on_success", "interval"])
        action = st.selectbox("Action", ["self_improvement", "daily_report", "analyze", "noop"])
        enabled = st.checkbox("Enabled", value=True)
        submitted = st.form_submit_button("Create / Update")
        if submitted and name:
            existing = [a for a in autos if a.get("name") != name]
            existing.append({"id": name.lower().replace(" ", "_"), "name": name, "trigger": trigger, "action": action, "enabled": enabled})
            cfg["automations"] = existing
            save_automations(cfg, root)
            st.success("Automation saved.")
            st.rerun()

    st.subheader("Defined Automations")
    if autos:
        for a in autos:
            cols = st.columns([3, 1, 1, 1, 1])
            cols[0].write(f"**{a.get('name')}** — {a.get('trigger')} → {a.get('action')}")
            en = "✅" if a.get("enabled") else "⏸️"
            cols[1].write(en)
            if cols[2].button("Run", key=f"run_{a.get('id')}"):
                r = trigger_automation(a.get("action", "noop"), root=root)
                append_automation_run({"automation": a.get("name"), "result": r}, root)
                st.success(f"Ran: {r.get('action')} created={r.get('created', r.get('ok'))}")
                if r.get("detail"):
                    st.json(r["detail"][:3] if isinstance(r["detail"], list) else r["detail"])
                st.rerun()
            if cols[3].button("Toggle", key=f"tog_{a.get('id')}"):
                a["enabled"] = not a.get("enabled", True)
                save_automations(cfg, root)
                st.rerun()
            if cols[4].button("Del", key=f"del_{a.get('id')}"):
                cfg["automations"] = [x for x in autos if x.get("id") != a.get("id")]
                save_automations(cfg, root)
                st.rerun()
    else:
        st.info("No automations defined. Use form above.")

    st.subheader("Recent Runs")
    if runs:
        for r in reversed(runs[-8:]):
            st.write(f"- {r.get('ts', '')}: {r.get('action') or r.get('automation')} → {str(r.get('result', r))[:80]}")
    else:
        st.write("No runs logged yet.")

    st.divider()
    st.subheader("Scheduled jobs")
    schedules = list_schedules(root)
    if schedules:
        for schedule in schedules:
            cols = st.columns([3, 1, 1, 1])
            cols[0].write(f"**{schedule['name']}** · `{schedule['cron']}` · {', '.join(schedule['targets'])}")
            if cols[1].button("Status", key=f"auto_status_{schedule['id']}"):
                st.json(schedule_status(schedule["id"], root))
            if cols[2].button("Disable" if schedule["enabled"] else "Enable", key=f"auto_toggle_{schedule['id']}"):
                set_schedule_enabled(schedule["id"], not schedule["enabled"], root)
                st.rerun()
            if cols[3].button("Remove", key=f"auto_remove_{schedule['id']}"):
                delete_schedule(schedule["id"], root)
                st.rerun()
        st.caption("GitHub schedules use UTC and become active after the generated pull request merges to the default branch.")
    else:
        st.info("No scheduled jobs yet. Create one in Cron Jobs.")

elif page == "Cron Jobs":
    st.header("Cron Jobs")
    st.caption("Create once, deploy immediately. Local cron uses the system timezone; GitHub Actions uses UTC.")
    with st.form("create_schedule", clear_on_submit=True):
        name = st.text_input("Name", placeholder="Nightly repository analysis")
        cron = st.text_input("POSIX cron", value="0 2 * * *", help="minute hour day-of-month month day-of-week")
        action = st.selectbox("Action", ["analyze", "daily_report", "self_improvement", "custom"])
        command = st.text_input("Custom command", disabled=action != "custom", help="Single-line command; required only for custom jobs.")
        targets = st.multiselect("Deploy to", ["local", "github"], default=["local", "github"])
        submitted = st.form_submit_button("Create and deploy", type="primary")
        if submitted:
            try:
                schedule = create_schedule(name=name, action=action, cron=cron, targets=targets, command=command or None, root=root)
                st.success(f"Created {schedule['id']} and started deployment.")
                st.json(schedule.get("deployment", {}))
            except ValueError as exc:
                st.error(str(exc))

    st.subheader("Deployment status")
    schedules = list_schedules(root)
    if not schedules:
        st.info("No cron jobs deployed.")
    for schedule in schedules:
        status = schedule_status(schedule["id"], root)
        with st.expander(f"{schedule['name']} · {schedule['cron']}"):
            st.json(status)
            left, middle, right = st.columns(3)
            if left.button("Run now", key=f"cron_run_{schedule['id']}"):
                st.json(run_schedule_now(schedule["id"], root))
            if middle.button("Disable" if schedule["enabled"] else "Enable", key=f"cron_toggle_{schedule['id']}"):
                set_schedule_enabled(schedule["id"], not schedule["enabled"], root)
                st.rerun()
            if right.button("Remove job", key=f"cron_remove_{schedule['id']}"):
                delete_schedule(schedule["id"], root)
                st.rerun()

st.divider()
st.caption("© mana-agent • All decisions go through the validated model decision layer. No fallbacks.")
