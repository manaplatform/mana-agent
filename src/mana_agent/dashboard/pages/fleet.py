"""Read-mostly Fleet dashboard backed by the shared persisted service state."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.config.settings import Settings
from mana_agent.fleet import FleetConfig, FleetRegistry, FleetStore
from mana_agent.fleet.health import effective_status


def render(_root: Path | None = None) -> None:
    config = FleetConfig.from_settings(Settings())
    store = FleetStore(config.root)
    registry = FleetRegistry(store, config)
    st.header("Mana Fleet")
    st.caption(
        "Trusted cross-platform workers and immutable verification matrices. "
        "Fleet is disabled by default."
    )
    st.metric("Fleet enabled", "Yes" if config.enabled else "No")
    workers = registry.list()
    st.subheader("Workers")
    if not workers:
        st.info("No Fleet workers are registered.")
    for worker in workers:
        status = effective_status(
            worker,
            heartbeat_timeout_seconds=config.heartbeat_timeout_seconds,
            capability_ttl_seconds=config.capability_ttl_seconds,
        )
        with st.container(border=True):
            columns = st.columns(4)
            columns[0].markdown(f"**{worker.display_name or worker.worker_id}**\n\n`{worker.worker_id}`")
            columns[1].metric("Health", status.value)
            columns[2].metric(
                "Platform",
                f"{worker.capabilities.platform}/{worker.capabilities.architecture}",
            )
            columns[3].metric(
                "Jobs",
                f"{worker.health.active_job_count}/{worker.health.concurrency_limit}",
            )
            st.caption(
                "Labels: "
                + (", ".join(sorted(worker.capabilities.labels.values)) or "none")
                + " · Providers: "
                + (", ".join(sorted(worker.capabilities.execution_providers)) or "none")
            )
            with st.expander("Capabilities"):
                st.json(worker.capabilities.model_dump(mode="json"))
    st.subheader("Verification runs")
    runs = list(reversed(store.list_runs()))
    if not runs:
        st.info("No Fleet verification runs are recorded.")
    for run in runs[:100]:
        with st.container(border=True):
            outcome = run.summary.outcome.value if run.summary else "running"
            st.markdown(f"**`{run.fleet_run_id}`** · {outcome}")
            st.caption(
                f"Commit `{run.plan.repository_commit}` · "
                f"{len(run.results)}/{len(run.jobs)} jobs completed"
            )
            if run.summary:
                st.dataframe(
                    [item.model_dump(mode="json") for item in run.summary.platform_results],
                    use_container_width=True,
                    hide_index=True,
                )
            with st.expander("Jobs, logs, artifacts, and cleanup"):
                st.json({
                    "jobs": [item.model_dump(mode="json") for item in run.jobs],
                    "results": [item.model_dump(mode="json") for item in run.results],
                })
    st.caption(
        "Management actions use `/api/v1/fleet` and require the configured API "
        "mutation permission token; this read-only page never creates dashboard-only state."
    )
