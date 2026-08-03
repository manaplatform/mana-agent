"""CLI read and response surface for the durable inbox."""

from __future__ import annotations

import getpass
import json

import typer

from . import default_human_inbox_service
from .models import InboxQuery, InboxRequestType, InboxStatus, ResponseOperation, ResponseSubmission


inbox_app = typer.Typer(help="Review and respond to durable human-input requests.", no_args_is_help=True)


def _service():
    return default_human_inbox_service()


def _actor(actor: str) -> str:
    local = getpass.getuser()
    if actor.strip() and actor.strip() != local:
        raise typer.BadParameter("The local CLI cannot assume another reviewer identity.")
    return local


@inbox_app.command("list")
def list_items(
    status: list[InboxStatus] = typer.Option([], "--status"),
    reviewer: str = typer.Option("", "--reviewer"),
    role: str = typer.Option("", "--role"),
    group: str = typer.Option("", "--group"),
    task: str = typer.Option("", "--task"),
    branch: str = typer.Option("", "--branch"),
    request_type: InboxRequestType | None = typer.Option(None, "--request-type"),
    actor: str = typer.Option("", "--actor"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    rows = _service().list(InboxQuery(
        statuses=set(status), reviewer_id=reviewer, role=role, group=group,
        task_id=task, branch_id=branch, request_type=request_type,
    ), actor_id=_actor(actor))
    payload = [item.card() for item in rows]
    if json_output:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return
    for item in payload:
        typer.echo(
            f"{item['inbox_item_id']}  {item['status']}  {item['request_type']}  "
            f"{item['risk_level']}  {item['title']}  branch={item['branch_id']}  expires={item['expires_at']}"
        )


@inbox_app.command("show")
def show_item(
    inbox_item_id: str,
    actor: str = typer.Option("", "--actor"),
) -> None:
    service = _service()
    item = service.get(inbox_item_id, actor_id=_actor(actor))
    payload = {
        **item.card(),
        "audit": [event.model_dump(mode="json") for event in service.repository.audit_for_item(inbox_item_id)],
        "delivery_attempts": [attempt.model_dump(mode="json") for attempt in service.repository.delivery_attempts(inbox_item_id)],
    }
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _respond(inbox_item_id: str, *, operation: ResponseOperation, actor: str, comment: str, answer: dict | None = None) -> None:
    service = _service()
    item = service.repository.get(inbox_item_id)
    result = service.respond(ResponseSubmission(
        inbox_item_id=inbox_item_id,
        operation=operation,
        actor_id=_actor(actor),
        channel="cli",
        idempotency_key=f"cli:{operation.value}:{inbox_item_id}:{item.version}",
        answer=answer or {},
        comment=comment,
        expected_version=item.version,
        current_action_digest=item.action_digest,
    ))
    typer.echo(json.dumps(result.card(), indent=2, ensure_ascii=False, default=str))


@inbox_app.command("approve")
def approve_item(inbox_item_id: str, actor: str = typer.Option("", "--actor"), comment: str = typer.Option("", "--comment")) -> None:
    _respond(inbox_item_id, operation=ResponseOperation.APPROVE, actor=actor, comment=comment)


@inbox_app.command("deny")
def deny_item(inbox_item_id: str, actor: str = typer.Option("", "--actor"), comment: str = typer.Option("", "--comment")) -> None:
    _respond(inbox_item_id, operation=ResponseOperation.DENY, actor=actor, comment=comment)


@inbox_app.command("answer")
def answer_item(
    inbox_item_id: str,
    answer: str = typer.Option(..., "--answer", help="JSON object keyed by clarification field ID."),
    actor: str = typer.Option("", "--actor"),
    comment: str = typer.Option("", "--comment"),
) -> None:
    try:
        payload = json.loads(answer)
    except ValueError as exc:
        raise typer.BadParameter("--answer must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("--answer must be a JSON object")
    _respond(inbox_item_id, operation=ResponseOperation.ANSWER, actor=actor, comment=comment, answer=payload)


@inbox_app.command("maintain")
def maintain() -> None:
    """Idempotent hook for cron/automation expiry, reminders, and recovery."""
    service = _service()
    payload = {
        "expired": [item.inbox_item_id for item in service.expire_due()],
        "reminders": [attempt.delivery_attempt_id for attempt in service.send_due_reminders()],
        "reconciliation": service.reconcile().model_dump(mode="json"),
    }
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
