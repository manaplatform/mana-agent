from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import typer
from mana_agent.connectors.email.auth.credential_store import CredentialStore
from mana_agent.connectors.email.auth.oauth import gmail_authorization_flow
from mana_agent.connectors.email.config import load_accounts, remove_account, save_accounts
from mana_agent.connectors.email.models import EmailAccount, EmailAddress, EmailPermission
connector_app = typer.Typer(help="Manage optional provider-neutral connectors.")
email_app = typer.Typer(help="Manage email accounts. Gmail is the currently supported provider.")
connector_app.add_typer(email_app, name="email")
def _account(account_id: str) -> EmailAccount:
    return next((x for x in load_accounts() if x.id == account_id), (_ for _ in ()).throw(typer.BadParameter("Email account not found.")))
@email_app.command("list")
def list_accounts() -> None:
    for item in load_accounts(): typer.echo(f"{item.id}\t{item.provider}\t{item.address.address}\t{','.join(x.value for x in item.granted_permissions)}")
@email_app.command("add")
def add_account(provider: str = typer.Option("gmail", "--provider"), client_secret_file: Path = typer.Option(..., "--client-secret-file", exists=True, readable=True), permissions: str = typer.Option("email.metadata,email.read", "--permissions")) -> None:
    if provider != "gmail": raise typer.BadParameter("Only Gmail is currently functional.")
    selected = [x.strip() for x in permissions.split(",") if x.strip()]
    try: granted = {EmailPermission(x) for x in selected}
    except ValueError as exc: raise typer.BadParameter("Permissions must be email.metadata,email.read,email.compose,email.send,email.modify.") from exc
    credentials = gmail_authorization_flow(client_secret_file, [x.value for x in granted]); reference = CredentialStore().put({"token": credentials.token, "refresh_token": credentials.refresh_token, "token_uri": credentials.token_uri, "client_id": credentials.client_id, "client_secret": credentials.client_secret, "scopes": list(credentials.scopes or [])})
    account = EmailAccount(id=f"gmail-{uuid4().hex[:12]}", provider="gmail", address=EmailAddress(address="pending-profile"), granted_permissions=granted, secret_ref=reference)
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        service = build("gmail", "v1", credentials=Credentials(**CredentialStore().get(reference)), cache_discovery=False); account.address = EmailAddress(address=str(service.users().getProfile(userId="me").execute()["emailAddress"]))
    except ImportError as exc: CredentialStore().delete(reference); raise typer.BadParameter("Gmail API client is missing: install `mana-agent[email]`.") from exc
    except Exception: CredentialStore().delete(reference); raise typer.BadParameter("Gmail authorization succeeded but account profile validation failed.")
    records = load_accounts(); records.append(account); save_accounts(records); typer.echo(f"Connected Gmail account {account.id} ({account.address.address}).")
@email_app.command("status")
@email_app.command("test")
def account_status(account: str) -> None:
    item = _account(account); typer.echo(f"{item.id}: configured ({item.provider}, {item.address.address})")
@email_app.command("permissions")
def permissions(account: str) -> None: typer.echo("\n".join(sorted(x.value for x in _account(account).granted_permissions)))
@email_app.command("remove")
def remove(account: str) -> None:
    item = remove_account(account)
    if item.secret_ref: CredentialStore().delete(item.secret_ref)
    typer.echo(f"Disconnected {account}.")
@email_app.command("reconnect")
def reconnect(account: str) -> None:
    _account(account); raise typer.BadParameter("Run `mana-agent connector email add --provider gmail` to create a new OAuth connection; existing credentials are never printed or reused in CLI arguments.")
