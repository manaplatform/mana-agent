"""Authoritative durable human-in-the-loop inbox dashboard."""

from __future__ import annotations

import getpass
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from mana_agent.human_inbox import ExpectedResponseType, InboxQuery, InboxStatus, ResponseOperation, ResponseSubmission, default_human_inbox_service


def render(_root: Path) -> None:
    st.header("Human Inbox")
    st.caption(
        "Approval and clarification cards are reloaded from durable state on every render. "
        "Live events are notifications only and never construct inbox history."
    )
    service = default_human_inbox_service()
    actor = getpass.getuser()
    st.caption(f"Reviewer identity: `{actor}` (bound to the local dashboard process)")
    selected_statuses = st.multiselect(
        "Status",
        options=[status.value for status in InboxStatus],
        default=[InboxStatus.PENDING.value, InboxStatus.DELIVERED.value],
    )
    rows = service.list(
        InboxQuery(statuses={InboxStatus(value) for value in selected_statuses}),
        actor_id=actor,
    )
    if not rows:
        st.info("No inbox items are assigned to this reviewer.")
        return
    for item in rows:
        card = item.card()
        risk_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(item.risk_level.value, "⚪")
        with st.expander(f"{risk_icon} {item.title} · {item.status.value} · {item.inbox_item_id}"):
            st.write(item.summary)
            st.json({
                "risk": item.risk_level.value,
                "request_age_seconds": int(
                    (datetime.now(timezone.utc) - item.created_at).total_seconds()
                ),
                "review_queue": (
                    f"{item.assigned_reviewer_type.value}:{item.assigned_reviewer_id}"
                ),
                "task": item.task_id,
                "branch": item.branch_id,
                "other_work_continues": item.other_work_continues,
                "reversibility": item.reversibility,
                "effect_labels": item.minimal_context.get("effect_labels", {}),
                "disclosed_information": item.disclosed_fields,
                "expires_at": card["expires_at"],
                "escalation": item.escalation_policy.model_dump(mode="json"),
                "delivery_status": card["delivery_status"],
                "minimal_context": item.minimal_context,
            })
            comment = st.text_area("Reviewer comment", key=f"comment-{item.inbox_item_id}")
            if item.request_type.value == "approval" and item.status in {InboxStatus.PENDING, InboxStatus.DELIVERED}:
                approve_col, deny_col = st.columns(2)
                if approve_col.button("Approve", key=f"approve-{item.inbox_item_id}", type="primary"):
                    _respond(service, item, actor=actor, operation=ResponseOperation.APPROVE, comment=comment)
                if deny_col.button("Deny", key=f"deny-{item.inbox_item_id}"):
                    _respond(service, item, actor=actor, operation=ResponseOperation.DENY, comment=comment)
            elif item.request_type.value == "clarification" and item.status in {InboxStatus.PENDING, InboxStatus.DELIVERED}:
                answer: dict[str, object] = {}
                object_answers: dict[str, str] = {}
                for field in item.requested_fields:
                    if field.sensitive:
                        answer[field.field_id] = st.text_input(field.prompt, type="password", key=f"answer-{item.inbox_item_id}-{field.field_id}")
                    elif field.expected_type is ExpectedResponseType.MULTI_CHOICE:
                        answer[field.field_id] = st.multiselect(field.prompt, field.choices, key=f"answer-{item.inbox_item_id}-{field.field_id}")
                    elif field.choices:
                        answer[field.field_id] = st.selectbox(field.prompt, field.choices, key=f"answer-{item.inbox_item_id}-{field.field_id}")
                    elif field.expected_type is ExpectedResponseType.BOOLEAN:
                        answer[field.field_id] = st.checkbox(field.prompt, key=f"answer-{item.inbox_item_id}-{field.field_id}")
                    elif field.expected_type is ExpectedResponseType.INTEGER:
                        answer[field.field_id] = st.number_input(field.prompt, step=1, key=f"answer-{item.inbox_item_id}-{field.field_id}")
                    elif field.expected_type is ExpectedResponseType.NUMBER:
                        answer[field.field_id] = st.number_input(field.prompt, key=f"answer-{item.inbox_item_id}-{field.field_id}")
                    elif field.expected_type is ExpectedResponseType.OBJECT:
                        object_answers[field.field_id] = st.text_area(field.prompt, value="{}", key=f"answer-{item.inbox_item_id}-{field.field_id}")
                    else:
                        answer[field.field_id] = st.text_input(field.prompt, key=f"answer-{item.inbox_item_id}-{field.field_id}")
                if st.button("Submit answer", key=f"answer-submit-{item.inbox_item_id}", type="primary"):
                    try:
                        answer.update({key: json.loads(value) for key, value in object_answers.items()})
                    except ValueError:
                        st.error("Object answers must contain valid JSON objects.")
                        continue
                    _respond(service, item, actor=actor, operation=ResponseOperation.ANSWER, comment=comment, answer=answer)
            st.markdown("#### Audit timeline")
            st.json([event.model_dump(mode="json") for event in service.repository.audit_for_item(item.inbox_item_id)])


def _respond(service, item, *, actor: str, operation: ResponseOperation, comment: str, answer: dict | None = None) -> None:
    try:
        service.respond(ResponseSubmission(
            inbox_item_id=item.inbox_item_id,
            operation=operation,
            actor_id=actor,
            channel="dashboard-local",
            idempotency_key=f"dashboard:{uuid4().hex}",
            answer=answer or {},
            comment=comment,
            expected_version=item.version,
            current_action_digest=item.action_digest,
        ))
    except Exception as exc:
        st.error(str(exc))
        return
    st.rerun()
