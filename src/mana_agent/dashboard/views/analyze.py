from __future__ import annotations

import threading
import time
from pathlib import Path

import streamlit as st

from mana_agent.api.services.job_service import ApiJobStore
from mana_agent.services.execution_event_hub import (
    get_execution_event_hub,
    repository_id_for_root,
)
from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    get_last_analysis_summary,
    list_analysis_artifacts,
    safe_read_json,
    trigger_automation,
)
from mana_agent.workspaces.paths import repository_analysis_dir


def _run_analyze_job(root: Path, job_id: str, *, depth: str, language: str, with_llm: bool) -> None:
    from mana_agent.services.project_analyze_service import (
        ProjectAnalyzeOptions,
        ProjectAnalyzeService,
    )

    jobs = ApiJobStore()
    repo_id = repository_id_for_root(root)
    hub = get_execution_event_hub()

    def operation() -> dict:
        hub.emit(
            "step.started",
            title="Analyze started",
            conversation_id=f"analyze:{repo_id}",
            execution_id=job_id,
            repository_id=repo_id,
            message=f"depth={depth}",
            status="running",
            metadata={"tool_name": "analyze"},
        )
        artifact_dir = repository_analysis_dir(repo_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        llm_analyzer = None
        if with_llm:
            try:
                from mana_agent.commands.cli_internal import _build_project_llm_analyzer

                llm_analyzer = _build_project_llm_analyzer()
            except Exception:
                llm_analyzer = None
        result = ProjectAnalyzeService().run(
            root,
            artifact_dir,
            options=ProjectAnalyzeOptions(depth=depth, language=language, output_format="both"),
            llm_analyzer=llm_analyzer,
        )
        artifacts = list(getattr(result, "artifacts", {}) or {})
        hub.emit(
            "step.finished",
            title="Analyze finished",
            conversation_id=f"analyze:{repo_id}",
            execution_id=job_id,
            repository_id=repo_id,
            message=f"artifacts={len(artifacts)}",
            status="success",
            metadata={"tool_name": "analyze", "artifacts": artifacts[:12]},
        )
        return {
            "repository_id": repo_id,
            "artifact_dir": str(artifact_dir),
            "artifacts": artifacts[:20],
            "llm_used": llm_analyzer is not None,
            "errors": list(getattr(result, "errors", []) or []),
        }

    try:
        jobs.run(job_id, operation)
    except Exception as exc:
        hub.emit(
            "error",
            title="Analyze failed",
            conversation_id=f"analyze:{repo_id}",
            execution_id=job_id,
            repository_id=repo_id,
            message=str(exc),
            status="failed",
            metadata={"tool_name": "analyze"},
        )


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    repo_id = repository_id_for_root(root)
    st.header("Analyze")
    st.caption(
        "Start Mana-Agent repository analysis using `ProjectAnalyzeService` "
        "(same engine as CLI / chat `/analyze`). Artifacts land under the repository analysis directory."
    )

    c1, c2 = st.columns(2)
    depth = c1.selectbox("Depth", ["quick", "normal", "full"], index=1)
    language = c2.selectbox("Analysis language", ["en", "fa"], format_func=lambda value: "English" if value == "en" else "فارسی", index=0)
    with_llm = st.checkbox("Include LLM narrative when configured", value=True)

    job_id = st.session_state.get("analyze_job_id")
    if st.button("Start analysis", type="primary"):
        jobs = ApiJobStore()
        job = jobs.create(
            "repository_analyze",
            {
                "repository_id": repo_id,
                "root": str(root),
                "depth": depth,
                "language": language,
                "with_llm": with_llm,
            },
        )
        st.session_state.analyze_job_id = job["job_id"]
        thread = threading.Thread(
            target=_run_analyze_job,
            args=(root, job["job_id"]),
            kwargs={"depth": depth, "language": language, "with_llm": with_llm},
            daemon=True,
        )
        thread.start()
        st.success(f"Analyze job started: `{job['job_id']}`")
        st.rerun()

    if job_id:
        try:
            job = ApiJobStore().get(job_id)
        except FileNotFoundError:
            job = None
        if job:
            st.subheader("Job status")
            st.json(
                {
                    "job_id": job.get("job_id"),
                    "status": job.get("status"),
                    "error": job.get("error"),
                    "updated_at": job.get("updated_at"),
                    "result": job.get("result"),
                }
            )
            if job.get("status") in {"queued", "running"}:
                st.info("Analysis running…")
                time.sleep(1.2)
                st.rerun()
            elif job.get("status") == "failed":
                st.error(job.get("error") or "Analyze failed.")
            elif job.get("status") == "done":
                st.success("Analysis completed.")

    st.subheader("Latest summary")
    summary = get_last_analysis_summary(root)
    if summary.get("type") == "md":
        st.markdown(summary.get("preview", "")[:3000])
    elif summary.get("type") == "json":
        st.json(summary.get("data") or {})
    else:
        st.info(summary.get("message") or "No analysis artifacts yet.")

    st.subheader("Artifacts")
    arts = list_analysis_artifacts(root)
    if not arts:
        st.write("No artifacts found.")
    else:
        names = [a["name"] for a in arts]
        sel = st.selectbox("Select artifact", names)
        chosen = next((a for a in arts if a["name"] == sel), None)
        if chosen:
            path = Path(chosen["path"])
            st.caption(str(path))
            if chosen["type"] == "json":
                st.json(safe_read_json(path) or {})
            else:
                st.markdown(path.read_text(encoding="utf-8", errors="replace")[:6000])

    with st.expander("Synchronous fallback (legacy trigger)"):
        if st.button("Run analyze inline (blocks UI)"):
            result = trigger_automation("analyze", root=root)
            st.json(result)
            st.rerun()
