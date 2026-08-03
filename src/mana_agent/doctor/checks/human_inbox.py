from __future__ import annotations

import stat

from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity
from mana_agent.human_inbox.models import InboxItem, InboxStatus, UNRESOLVED_STATUSES
from mana_agent.execution_supervisor.models import TaskRecord


def durable_state(context: DoctorContext) -> list[DoctorFinding]:
    root = context.home / "inbox"
    if not root.exists():
        return [DoctorFinding(
            "persistence/human-inbox",
            Severity.INFO,
            "Durable human inbox",
            "No human inbox state has been created yet.",
        )]
    findings: list[DoctorFinding] = []
    items: list[InboxItem] = []
    for path in sorted((root / "items").glob("*.json")):
        try:
            items.append(InboxItem.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            findings.append(DoctorFinding(
                "persistence/human-inbox", Severity.ERROR, "Unreadable inbox record",
                f"{path.name}: {exc}", "Restore the record from backup or inspect the durable audit history.", path=str(path),
            ))
    signing_key = root / "response-signing.key"
    if signing_key.exists() and stat.S_IMODE(signing_key.stat().st_mode) & 0o077:
        findings.append(DoctorFinding(
            "persistence/human-inbox", Severity.ERROR, "Inbox signing key permissions",
            "The response-token signing key is accessible beyond its owner.",
            "Restrict the key to owner read/write permissions.", path=str(signing_key),
        ))
    unresolved = [item for item in items if item.status in UNRESOLVED_STATUSES]
    unassigned = [item.inbox_item_id for item in unresolved if not item.eligible_reviewer_ids]
    if unassigned:
        findings.append(DoctorFinding(
            "persistence/human-inbox", Severity.WARNING, "Inbox reviewer configuration",
            f"{len(unassigned)} unresolved item(s) have no eligible reviewer.",
            "Configure identities and role/group membership in ~/.mana/inbox/identities.json.",
            details={"inbox_item_ids": unassigned},
        ))
    uncertain_executions = [
        item.inbox_item_id
        for item in items
        if item.execution_claim_id and item.execution_completed_at is None
    ]
    if uncertain_executions:
        findings.append(DoctorFinding(
            "persistence/human-inbox", Severity.WARNING, "Inbox action reconciliation",
            f"{len(uncertain_executions)} approved action(s) have a claimed but incomplete execution outcome.",
            "Inspect external state and reconcile the exact action; Mana-Agent will not execute it twice automatically.",
            details={"inbox_item_ids": uncertain_executions},
        ))
    tasks: dict[str, TaskRecord] = {}
    for path in sorted((context.home / "execution" / "tasks").glob("*.json")):
        try:
            task = TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tasks[task.task_id] = task
    item_ids = {item.inbox_item_id for item in items}
    orphaned = [item.inbox_item_id for item in unresolved if item.checkpoint_id and item.task_id not in tasks]
    waiting_mismatch = [
        item.inbox_item_id
        for item in unresolved
        if item.checkpoint_id
        and item.task_id in tasks
        and (
            tasks[item.task_id].state.value != "waiting"
            or tasks[item.task_id].waiting_inbox_item_id != item.inbox_item_id
        )
    ]
    missing_items = [
        task.task_id
        for task in tasks.values()
        if task.waiting_inbox_item_id and task.waiting_inbox_item_id not in item_ids
    ]
    resume_pending = [
        item.inbox_item_id
        for item in items
        if item.status in {
            InboxStatus.APPROVED,
            InboxStatus.DENIED,
            InboxStatus.ANSWERED,
        }
        and item.checkpoint_id
        and item.resume_completed_at is None
    ]
    if orphaned or waiting_mismatch or missing_items or resume_pending:
        findings.append(DoctorFinding(
            "persistence/human-inbox", Severity.WARNING, "Inbox and branch reconciliation",
            "Durable inbox and execution branch projections require reconciliation.",
            "Run `mana-agent inbox maintain` and inspect any reported errors.",
            details={
                "orphaned_inbox_items": orphaned,
                "waiting_projection_mismatches": waiting_mismatch,
                "branches_without_unresolved_item": missing_items,
                "terminal_responses_pending_resume": resume_pending,
            },
        ))
    if not findings:
        findings.append(DoctorFinding(
            "persistence/human-inbox", Severity.INFO, "Durable human inbox",
            f"Loaded {len(items)} durable item(s); {len(unresolved)} remain unresolved.",
        ))
    return findings
