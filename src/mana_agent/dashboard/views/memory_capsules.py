"""Read-only capsule visibility through the authorization-preserving API."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


def _request(path: str, *, body: dict | None = None):
    base = str(st.session_state.get("mana_api_base") or "").rstrip("/")
    if not base:
        raise ValueError("Configure Live API base in the sidebar.")
    token = str(os.getenv("MANA_API_TOKEN") or "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"{base}{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - operator-configured Mana API
        return json.loads(response.read().decode("utf-8"))


def render(_root: Path | None = None) -> None:
    st.header("Memory Capsules")
    st.caption("All results come through the authenticated capsule service; this page never reads provider storage directly.")
    st.caption("Local dashboard access can read project and user capsules. Task-private chat capsules remain available only to the gateway's authorized follow-up flow.")
    query = st.text_input("Relevance query", value="")
    scopes = st.multiselect(
        "Allowed scopes",
        ["private", "parent_child", "team", "project", "user"],
        default=["project", "user"],
    )
    max_capsules = st.number_input("Maximum capsules", min_value=1, max_value=100, value=12)
    if st.button("Load visible capsules", type="primary"):
        try:
            rows = _request("/api/v1/memory/capsules/query", body={
                "query": query,
                "allowed_scopes": scopes,
                "max_capsules": int(max_capsules),
                "max_tokens": 4000,
            })
            st.session_state["visible_memory_capsules"] = rows
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"Capsule API unavailable: {exc}")
    rows = list(st.session_state.get("visible_memory_capsules") or [])
    if rows:
        st.dataframe([
            {
                "capsule": item.get("capsule_id"),
                "scope": item.get("scope"),
                "title": item.get("title"),
                "trust": item.get("trust_state"),
                "revision": item.get("revision"),
                "expires": item.get("expires_at"),
                "provider": item.get("provider"),
            }
            for item in rows
        ], use_container_width=True, hide_index=True)
        with st.expander("Authorized capsule projections"):
            st.json(rows, expanded=False)
    else:
        st.info("No authorized capsules loaded.")

    if st.button("Load staged review queue"):
        try:
            st.session_state["staged_memory_capsules"] = _request("/api/v1/memory/capsules/staged")
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"Review queue unavailable: {exc}")
    staged = list(st.session_state.get("staged_memory_capsules") or [])
    if staged:
        st.subheader("Staged capsules")
        st.json(staged, expanded=False)
        selected_id = st.selectbox(
            "Review staged capsule",
            [str(item.get("capsule_id")) for item in staged],
        )
        strategy = st.selectbox("Merge strategy", ["append", "replace", "patch", "supersede"])
        reason = st.text_area("Review decision reason")
        target_id = st.text_input("Target capsule ID (optional)")
        expected_revision = st.number_input("Expected target revision", min_value=0, value=0)
        expected_hash = st.text_input("Expected target content hash")
        left, right = st.columns(2)
        if left.button("Approve and merge"):
            try:
                payload = {
                    "request_id": f"dashboard-{uuid.uuid4().hex}",
                    "strategy": strategy,
                    "decision_reason": reason,
                    "target_capsule_id": target_id or None,
                    "expected_target_revision": int(expected_revision) or None,
                    "expected_target_hash": expected_hash or None,
                }
                st.success(_request(f"/api/v1/memory/capsules/{selected_id}/merge", body=payload))
            except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                st.error(f"Merge failed: {exc}")
        if right.button("Reject proposal"):
            try:
                st.success(_request(f"/api/v1/memory/capsules/{selected_id}/merge", body={
                    "request_id": f"dashboard-{uuid.uuid4().hex}",
                    "strategy": "reject",
                    "decision_reason": reason,
                }))
            except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                st.error(f"Rejection failed: {exc}")
