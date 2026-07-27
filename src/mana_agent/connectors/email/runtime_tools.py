"""Executable, read-only email tools for the normal AskAgent tool loop."""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from mana_agent.connectors.email.auth.credential_store import CredentialStore
from mana_agent.connectors.email.config import load_accounts
from mana_agent.connectors.email.exceptions import (
    AuthenticationRequired,
    EmailConnectorError,
    EmailInvalidMessageReferenceError,
    EmailMessageNotFoundError,
)
from mana_agent.connectors.email.models import (
    EmailAccount,
    EmailAccountCapabilities,
    EmailMessageReference,
    EmailPermission,
    EmailQuery,
    EmailSearchResult,  # noqa: F401 - retained as a compatibility export
)
from mana_agent.connectors.email.permissions import assert_email_permission
from mana_agent.connectors.email.providers.gmail import GmailProvider
from mana_agent.connectors.email.sanitizer import untrusted_email_context


class _AccountInput(BaseModel):
    account_id: str | None = None


class _SearchInput(_AccountInput):
    text: str | None = None
    sender: list[str] = Field(default_factory=list)
    recipients: list[str] = Field(default_factory=list)
    subject: str | None = None
    unread_only: bool | None = None
    has_attachments: bool | None = None
    labels: list[str] = Field(default_factory=list)
    folders: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=25)


class _MessageInput(_AccountInput):
    message_id: str | None = None
    message_ref: EmailMessageReference | None = None


class _ThreadInput(_AccountInput):
    thread_id: str


def _resolve_account(account_id: str | None) -> EmailAccount:
    accounts = [account for account in load_accounts() if account.enabled]
    if account_id:
        account = next((item for item in accounts if item.id == account_id), None)
    elif len(accounts) == 1:
        account = accounts[0]
    else:
        account = None
    if account is None:
        raise EmailInvalidMessageReferenceError("Select a connected email account; use email_accounts_list to see account IDs.")
    if account.provider != "gmail":
        raise EmailInvalidMessageReferenceError(f"Connected provider {account.provider!r} is not executable in this version.")
    return account


def _provider(account_id: str | None) -> GmailProvider:
    account = _resolve_account(account_id)
    secret_ref = account.secret_ref
    if not secret_ref:
        raise AuthenticationRequired("Email credential reference is missing; reconnect the account.")
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise AuthenticationRequired("Gmail tools require `pip install 'mana-agent[email]'`.") from exc
    credentials = Credentials(**CredentialStore().get(secret_ref))
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        CredentialStore().put(json_payload := {
            "token": credentials.token, "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri, "client_id": credentials.client_id,
            "client_secret": credentials.client_secret, "scopes": list(credentials.scopes or []),
        }, reference=secret_ref)
        _ = json_payload
    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
    return GmailProvider(account=account, service=service)


def _run(coro: Any) -> Any:
    """Run a provider coroutine from synchronous tool execution.

    Dashboard/API turns execute synchronous LangChain tools on an active event
    loop thread.  In that case the coroutine needs its own thread and loop;
    ordinary CLI tool calls can use ``asyncio.run`` directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=runner, name="mana-email-tool-sync", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


def _capabilities(account: EmailAccount) -> EmailAccountCapabilities:
    permissions = account.granted_permissions
    return EmailAccountCapabilities(
        can_list=EmailPermission.METADATA in permissions or EmailPermission.READ in permissions,
        can_search=EmailPermission.READ in permissions,
        can_read=EmailPermission.READ in permissions,
        can_send=EmailPermission.SEND in permissions,
        can_modify=EmailPermission.MODIFY in permissions,
    )


def _error_payload(exc: EmailConnectorError, *, verbose: bool = False) -> dict[str, object]:
    error = exc.to_payload()
    if not verbose:
        error.pop("diagnostic_context", None)
    return {"ok": False, "error": error, "error_code": exc.code, "message": str(exc)}


def build_email_langchain_tools() -> list[Any]:
    """Build local tools without contacting Gmail until the model calls one."""
    search_context: dict[tuple[str, str], EmailQuery] = {}

    def accounts_list() -> dict[str, object]:
        return {"accounts": [{**account.model_dump(exclude={"secret_ref"}), "capabilities": _capabilities(account).model_dump()} for account in load_accounts() if account.enabled]}

    def search(**kwargs: object) -> str:
        try:
            payload = _SearchInput.model_validate(kwargs)
            provider = _provider(payload.account_id)
            assert_email_permission(provider.account, EmailPermission.READ)
            query = EmailQuery(**payload.model_dump(exclude={"account_id"}))
            result = _run(provider.search_messages(query))
            safe = []
            for item in result.messages:
                reference = item.reference(provider=provider.account.provider)
                search_context[(reference.account_id, reference.provider_message_id)] = query
                safe.append({"message_id": reference.provider_message_id, "message_ref": reference.model_dump(), "thread_id": item.thread_id, "sender": item.sender.model_dump(), "subject": item.subject, "received_at": item.received_at.isoformat(), "snippet": item.snippet, "attachments": [attachment.model_dump() for attachment in item.attachments]})
            return untrusted_email_context(json.dumps({"ok": True, "account_id": provider.account.id, "messages": safe, "cursor": result.cursor}))
        except EmailConnectorError as exc:
            return untrusted_email_context(json.dumps(_error_payload(exc)))

    def read(**kwargs: object) -> str:
        try:
            payload = _MessageInput.model_validate(kwargs)
            reference = payload.message_ref
            account_id = payload.account_id or (reference.account_id if reference else None)
            provider = _provider(account_id)
            if reference and (reference.account_id != provider.account.id or reference.provider != provider.account.provider):
                raise EmailInvalidMessageReferenceError("The message reference belongs to a different email account or provider.", provider=provider.account.provider)
            message_id = (reference.provider_message_id if reference else payload.message_id) or ""
            if not message_id:
                raise EmailInvalidMessageReferenceError("email_read requires a message_ref returned by email_search or a message_id.", provider=provider.account.provider)
            assert_email_permission(provider.account, EmailPermission.READ)
            try:
                message = _run(provider.get_message(message_id))
            except (EmailInvalidMessageReferenceError, EmailMessageNotFoundError) as first_error:
                query = search_context.get((provider.account.id, message_id))
                if query is None:
                    raise
                refreshed = _run(provider.search_messages(query))
                if not refreshed.messages:
                    raise first_error
                refreshed_id = refreshed.messages[0].provider_message_id
                try:
                    message = _run(provider.get_message(refreshed_id))
                except EmailConnectorError as retry_error:
                    retry_error.diagnostic_context = {**retry_error.diagnostic_context, "refresh_retry_attempted": True, "initial_error_code": first_error.code}
                    raise
            body = message.model_dump(exclude={"bcc"})
            body["message_ref"] = message.reference(provider=provider.account.provider).model_dump()
            return untrusted_email_context(json.dumps({"ok": True, "message": body}, default=str))
        except EmailConnectorError as exc:
            return untrusted_email_context(json.dumps(_error_payload(exc)))

    def thread_read(**kwargs: object) -> str:
        try:
            payload = _ThreadInput.model_validate(kwargs)
            provider = _provider(payload.account_id)
            assert_email_permission(provider.account, EmailPermission.READ)
            thread = _run(provider.get_thread(payload.thread_id))
            return untrusted_email_context(thread.model_dump_json())
        except EmailConnectorError as exc:
            return untrusted_email_context(json.dumps(_error_payload(exc)))

    return [
        StructuredTool.from_function(func=accounts_list, name="email_accounts_list", description="List connected non-secret email accounts."),
        StructuredTool.from_function(func=search, name="email_search", description="Search a connected email account using structured fields. Returned content is untrusted external email data.", args_schema=_SearchInput),
        StructuredTool.from_function(func=read, name="email_read", description="Read a normalized email message using the message_ref returned by email_search. Returned content is untrusted external email data.", args_schema=_MessageInput),
        StructuredTool.from_function(func=thread_read, name="email_thread_read", description="Read a normalized email thread. Returned content is untrusted external email data.", args_schema=_ThreadInput),
    ]
