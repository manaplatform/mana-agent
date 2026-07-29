"""Experience-to-Skill proposal review dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from mana_agent.builtin_skills.skill_creator import ProposalStorage, SkillCreator


def render(root: Path) -> None:
    _ = root
    import streamlit as st

    st.title("Experience-to-Skill Workshop")
    st.caption("Proposals remain inactive until you explicitly install them.")
    storage = ProposalStorage()

    filters = st.columns(3)
    status = filters[0].selectbox(
        "Status",
        [
            "all",
            "pending_review",
            "needs_attention",
            "installed",
            "rejected",
            "quarantined",
        ],
    )
    minimum = filters[1].slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    risk = filters[2].selectbox("Risk", ["all", "low", "medium", "high", "critical"])
    proposals = storage.list(
        status=None if status == "all" else status,
        min_confidence=minimum,
        risk=None if risk == "all" else risk,
    )
    if not proposals:
        st.info("No proposals match the current filters.")
        return

    labels = {
        f"{item.display_name} · {item.status} · {item.confidence:.2f}": item.proposal_id
        for item in proposals
    }
    selected_label = st.selectbox("Proposal", list(labels))
    proposal_id = labels[selected_label]
    path, manifest, evidence, report, markdown = storage.load(proposal_id)

    summary = st.columns(4)
    summary[0].metric("Confidence", f"{manifest.confidence:.2f}")
    summary[1].metric("Risk", str(manifest.risk.get("level", "unknown")))
    summary[2].metric("Successful runs", manifest.successful_runs)
    summary[3].metric("Status", manifest.status)
    st.write(manifest.description)
    st.caption(f"Proposal: `{proposal_id}` · Path: `{path}`")

    metadata_col, editor_col = st.columns(2)
    with metadata_col:
        st.subheader("Evidence and validation")
        st.json(evidence.model_dump(mode="json"))
        st.json(report.model_dump(mode="json"))
        st.subheader("Required access")
        st.write(
            {
                "tools": manifest.required_tools,
                "permissions": manifest.required_permissions,
            }
        )
        st.subheader("Installed version history")
        active = storage.paths.skills / manifest.name
        versions = []
        for provenance in sorted((active / "versions").glob("*/provenance.json")):
            versions.append(json.loads(provenance.read_text(encoding="utf-8")))
        st.json({"versions": versions})
    with editor_col:
        st.subheader("Proposed SKILL.md")
        edited = st.text_area(
            "Review or edit", markdown, height=700, key=f"proposal-editor-{proposal_id}"
        )
        if st.button(
            "Save edit and revalidate",
            disabled=manifest.status not in {"pending_review", "needs_attention"},
            use_container_width=True,
        ):
            try:
                SkillCreator(storage=storage).edit(proposal_id, markdown=edited)
                st.success(
                    "Saved; approval state reset to needs_attention and validation reran."
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    reason = st.text_input(
        "Reject or quarantine reason", key=f"proposal-reason-{proposal_id}"
    )
    install_col, reject_col, quarantine_col = st.columns(3)
    if install_col.button(
        "Install",
        disabled=manifest.status not in {"pending_review", "needs_attention"},
        use_container_width=True,
    ):
        try:
            storage.install(proposal_id, approved=True, version=manifest.version)
            st.success("Installed after revalidation.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if reject_col.button(
        "Reject",
        disabled=manifest.status in {"installed", "quarantined"},
        use_container_width=True,
    ):
        try:
            storage.reject(proposal_id, reason)
            st.warning("Proposal rejected and retained for deduplication.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if quarantine_col.button(
        "Quarantine",
        disabled=manifest.status in {"installed", "quarantined"} or not reason.strip(),
        use_container_width=True,
    ):
        try:
            storage.quarantine(proposal_id, reason)
            st.warning("Proposal moved to quarantine.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
