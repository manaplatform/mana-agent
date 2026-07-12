"""Gmail implementation; Google response dictionaries are normalized here only."""
from __future__ import annotations
import asyncio, base64, json
from datetime import datetime, timezone
from email.message import EmailMessage as MimeMessage
from email.utils import getaddresses, parseaddr
from typing import Any
from mana_agent.connectors.email.exceptions import (
    AuthenticationRequired,
    CapabilityUnsupported,
    EmailAuthorizationError,
    EmailMessageNotFoundError,
    EmailProviderError,
    EmailTemporaryError,
    InvalidMessageIdentifier,
    PermanentProviderFailure,
)
from mana_agent.connectors.email.models import *
from mana_agent.connectors.email.providers.base import EmailProvider
from mana_agent.connectors.email.sanitizer import safe_attachment_filename, sanitize_html

GMAIL_CAPABILITIES = EmailProviderCapabilities(supports_threads=True, supports_labels=True, supports_drafts=True, supports_push_notifications=True, supports_server_search=True, supports_send_as=True, supports_archive=True, supports_reply_all=True)
GMAIL_SYSTEM_LABEL_IDS = frozenset({"INBOX", "SENT", "DRAFT", "SPAM", "TRASH", "UNREAD", "STARRED", "IMPORTANT"})


def _google_error(exc: Exception) -> dict[str, Any]:
    """Return a normalized, non-secret Google error payload when available."""
    raw = getattr(exc, "content", b"")
    try:
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        payload = json.loads(text)
    except (TypeError, ValueError, UnicodeDecodeError):
        return {}
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    return error if isinstance(error, dict) else {}


def _google_error_message(exc: Exception) -> str:
    return str(_google_error(exc).get("message", ""))


def _google_error_reason(exc: Exception) -> str:
    errors = _google_error(exc).get("errors", [])
    if not isinstance(errors, list):
        return ""
    for error in errors:
        if isinstance(error, dict) and error.get("reason"):
            return str(error["reason"])
    return ""


def _google_error_status(exc: Exception) -> str:
    return str(_google_error(exc).get("status", ""))

def gmail_query(query: EmailQuery) -> str:
    parts = [query.text or ""] + [f"from:{x}" for x in query.sender] + [f"to:{x}" for x in query.recipients]
    if query.subject: parts.append(f"subject:{query.subject}")
    if query.after: parts.append(f"after:{query.after.date().isoformat()}")
    if query.before: parts.append(f"before:{query.before.date().isoformat()}")
    if query.unread_only: parts.append("is:unread")
    if query.has_attachments: parts.append("has:attachment")
    parts += [f"label:{x}" for x in query.labels] + [f"in:{x}" for x in query.folders]
    return " ".join(x for x in parts if x).strip()


def gmail_list_arguments(query: EmailQuery, cursor: str | None = None) -> dict[str, Any]:
    """Build a Gmail list request without using ``q`` for system folders.

    Gmail's metadata scope permits label filtering but rejects ``q`` entirely.
    Using the documented ``labelIds`` parameter keeps inbox-only metadata
    searches compatible with a metadata-capable account.
    """
    arguments: dict[str, Any] = {"userId": "me", "pageToken": cursor, "maxResults": query.limit}
    normalized_folders = [folder.upper() for folder in query.folders]
    if normalized_folders and all(folder in GMAIL_SYSTEM_LABEL_IDS for folder in normalized_folders):
        arguments["labelIds"] = normalized_folders
        query_text = gmail_query(query.model_copy(update={"folders": []}))
    else:
        query_text = gmail_query(query)
    if query_text:
        arguments["q"] = query_text
    return arguments

def _address(value: str) -> EmailAddress:
    name, address = parseaddr(value); return EmailAddress(address=address or value, name=name or None)
def _addresses(value: str | None) -> list[EmailAddress]: return [_address(f"{name} <{address}>") for name, address in getaddresses([value or ""]) if address]
def _b64(value: str | None) -> bytes: return base64.urlsafe_b64decode((value or "") + "===")

class GmailProvider(EmailProvider):
    def __init__(self, *, account: EmailAccount, service: Any) -> None: self.account, self.service = account, service
    async def _call(self, request: Any) -> Any:
        try:
            return await asyncio.to_thread(request.execute)
        except Exception as exc:
            # Google returns an HttpError whose response status is available
            # without serializing its potentially sensitive response body.
            raw_status = getattr(getattr(exc, "resp", None), "status", None)
            try:
                status = int(raw_status) if raw_status is not None else None
            except (TypeError, ValueError):
                status = None
            detail = _google_error_message(exc)
            reason = _google_error_reason(exc)
            error_status = _google_error_status(exc)
            diagnostic = {"exception_type": type(exc).__name__}
            if reason:
                diagnostic["provider_reason"] = reason
            if error_status:
                diagnostic["provider_error_status"] = error_status
            if status == 401:
                raise AuthenticationRequired("Gmail credentials were rejected or have expired. Reconnect the account.", provider="gmail", provider_status=status, diagnostic_context=diagnostic) from exc
            if status == 403:
                authorization_reasons = {"authError", "insufficientPermissions", "insufficientScopes"}
                if reason in authorization_reasons or "insufficient authentication scopes" in detail.lower() or "metadata scope does not support" in detail.lower():
                    raise EmailAuthorizationError("Gmail did not grant permission for this operation. Reconnect the account with the required Gmail permission.", provider="gmail", provider_status=status, diagnostic_context=diagnostic) from exc
                raise EmailProviderError("Gmail denied this request (HTTP 403) without identifying an authorization failure. Check Gmail API and OAuth consent-screen access for this account.", provider="gmail", provider_status=status, diagnostic_context=diagnostic) from exc
            if status == 404:
                raise EmailMessageNotFoundError("Gmail could not find this message.", provider="gmail", provider_status=status, diagnostic_context=diagnostic) from exc
            if status in {408, 429, 500, 502, 503, 504}:
                raise EmailTemporaryError("Gmail is temporarily unavailable. Try again shortly.", provider="gmail", provider_status=status, diagnostic_context=diagnostic) from exc
            raise EmailProviderError("Gmail returned an unexpected provider error.", provider="gmail", provider_status=status, diagnostic_context=diagnostic) from exc
    async def connect(self) -> EmailAccount:
        profile = await self._call(self.service.users().getProfile(userId="me")); self.account.address = EmailAddress(address=str(profile["emailAddress"])); return self.account
    async def disconnect(self) -> None: return None
    async def health_check(self) -> ProviderHealth:
        try: await self._call(self.service.users().getProfile(userId="me")); return ProviderHealth(healthy=True, checked_at=datetime.now(timezone.utc))
        except Exception: return ProviderHealth(healthy=False, message="Gmail connection failed", checked_at=datetime.now(timezone.utc))
    async def get_account(self) -> EmailAccount: return self.account
    async def get_capabilities(self) -> EmailProviderCapabilities: return GMAIL_CAPABILITIES
    async def list_folders(self) -> list[EmailFolder]: return [EmailFolder(id=x["id"], name=x["name"], role=x.get("type")) for x in (await self._call(self.service.users().labels().list(userId="me"))).get("labels", [])]
    async def list_labels(self) -> list[EmailLabel]: return [EmailLabel(id=x.id, name=x.name) for x in await self.list_folders()]
    async def search_messages(self, query: EmailQuery, cursor: str | None = None) -> EmailSearchResult:
        data = await self._call(self.service.users().messages().list(**gmail_list_arguments(query, cursor)))
        # Search is useful even for a metadata-only account. Do not make a
        # harmless "latest email" request fail by requesting full MIME bodies.
        return EmailSearchResult(messages=[await self.get_message_metadata(str(x["id"])) for x in data.get("messages", [])], cursor=data.get("nextPageToken"), total_estimate=data.get("resultSizeEstimate"))
    def _normalize(self, raw: dict[str, Any]) -> EmailMessage:
        headers = {str(x.get("name", "")).lower(): str(x.get("value", "")) for x in raw.get("payload", {}).get("headers", [])}
        plain: list[str] = []; html: list[str] = []; attachments: list[EmailAttachment] = []
        def walk(part: dict[str, Any]) -> None:
            body = part.get("body", {}); mime = part.get("mimeType", "")
            if body.get("attachmentId"):
                attachments.append(EmailAttachment(id=str(body["attachmentId"]), filename=safe_attachment_filename(str(part.get("filename") or "attachment")), mime_type=mime or "application/octet-stream", size=int(body.get("size", 0)), disposition="attachment", is_inline=bool(part.get("headers"))))
            elif body.get("data"):
                content = _b64(body["data"]).decode("utf-8", "replace")
                (html if mime == "text/html" else plain).append(content)
            for child in part.get("parts", []): walk(child)
        walk(raw.get("payload", {})); labels = list(raw.get("labelIds", []))
        return EmailMessage(id=str(raw["id"]), provider_message_id=str(raw["id"]), account_id=self.account.id, thread_id=raw.get("threadId"), internet_message_id=headers.get("message-id"), subject=headers.get("subject", ""), sender=_address(headers.get("from", "unknown@invalid")), to=_addresses(headers.get("to")), cc=_addresses(headers.get("cc")), bcc=_addresses(headers.get("bcc")), reply_to=_addresses(headers.get("reply-to")), received_at=datetime.fromtimestamp(int(raw.get("internalDate", "0"))/1000, timezone.utc), text_body="\n".join(plain) or None, sanitized_html_body=sanitize_html("\n".join(html)), snippet=raw.get("snippet"), labels=labels, attachments=attachments, is_read="UNREAD" not in labels, is_starred="STARRED" in labels, is_draft="DRAFT" in labels, is_trashed="TRASH" in labels, headers=headers)
    async def get_message(self, message_id: str) -> EmailMessage:
        if not str(message_id or "").strip():
            raise InvalidMessageIdentifier("The Gmail message reference is empty.", provider="gmail")
        return self._normalize(await self._call(self.service.users().messages().get(userId="me", id=message_id, format="full")))
    async def get_message_metadata(self, message_id: str) -> EmailMessage:
        if not str(message_id or "").strip():
            raise InvalidMessageIdentifier("The Gmail message reference is empty.", provider="gmail")
        raw = await self._call(self.service.users().messages().get(userId="me", id=message_id, format="metadata", metadataHeaders=["From", "To", "Cc", "Subject", "Message-ID", "Reply-To"]))
        return self._normalize(raw)
    async def get_thread(self, thread_id: str) -> EmailThread:
        raw = await self._call(self.service.users().threads().get(userId="me", id=thread_id, format="full")); return EmailThread(id=thread_id, account_id=self.account.id, messages=[self._normalize(x) for x in raw.get("messages", [])])
    async def get_attachment(self, message_id: str, attachment_id: str) -> EmailAttachment:
        message = await self.get_message(message_id); return next((x for x in message.attachments if x.id == attachment_id), (_ for _ in ()).throw(InvalidMessageIdentifier("Attachment not found.")))
    def _mime(self, draft: DraftInput) -> str:
        msg = MimeMessage(); msg["To"] = ", ".join(x.address for x in draft.to); msg["Subject"] = draft.subject
        if draft.cc: msg["Cc"] = ", ".join(x.address for x in draft.cc)
        if draft.bcc: msg["Bcc"] = ", ".join(x.address for x in draft.bcc)
        if draft.in_reply_to: msg["In-Reply-To"] = draft.in_reply_to; msg["References"] = " ".join(draft.references or [draft.in_reply_to])
        msg.set_content(draft.text_body); return base64.urlsafe_b64encode(msg.as_bytes()).decode()
    async def create_draft(self, draft: DraftInput) -> EmailDraft:
        raw = await self._call(self.service.users().drafts().create(userId="me", body={"message": {"raw": self._mime(draft)}})); return EmailDraft(id=raw["id"], account_id=self.account.id, message=await self.get_message(raw["message"]["id"]))
    async def update_draft(self, draft_id: str, draft: DraftInput) -> EmailDraft:
        raw = await self._call(self.service.users().drafts().update(userId="me", id=draft_id, body={"message": {"raw": self._mime(draft)}})); return EmailDraft(id=raw["id"], account_id=self.account.id, message=await self.get_message(raw["message"]["id"]))
    async def delete_draft(self, draft_id: str) -> None: await self._call(self.service.users().drafts().delete(userId="me", id=draft_id))
    async def send_draft(self, draft_id: str) -> SendResult:
        raw = await self._call(self.service.users().drafts().send(userId="me", body={"id": draft_id})); return SendResult(accepted=True, message_id=raw.get("id"), thread_id=raw.get("threadId"), provider_status="accepted")
    async def send_message(self, message: SendInput) -> SendResult:
        raw = await self._call(self.service.users().messages().send(userId="me", body={"raw": self._mime(message)})); return SendResult(accepted=True, message_id=raw.get("id"), thread_id=raw.get("threadId"), provider_status="accepted")
    async def reply(self, message_id: str, reply: ReplyInput) -> SendResult:
        source = await self.get_message(message_id); return await self.send_message(SendInput(to=[source.sender], subject=f"Re: {source.subject}", text_body=reply.text_body, cc=reply.cc, bcc=reply.bcc, in_reply_to=source.internet_message_id, references=[source.internet_message_id] if source.internet_message_id else []))
    async def reply_all(self, message_id: str, reply: ReplyInput) -> SendResult:
        source = await self.get_message(message_id); recipients = {x.address: x for x in [source.sender, *source.to, *source.cc] if x.address != self.account.address.address}; return await self.send_message(SendInput(to=list(recipients.values()), subject=f"Re: {source.subject}", text_body=reply.text_body, cc=reply.cc, bcc=reply.bcc, in_reply_to=source.internet_message_id, references=[source.internet_message_id] if source.internet_message_id else []))
    async def forward(self, message_id: str, forward: ForwardInput) -> SendResult:
        source = await self.get_message(message_id); return await self.send_message(SendInput(to=forward.to, cc=forward.cc, bcc=forward.bcc, subject=f"Fwd: {source.subject}", text_body=f"{forward.text_body}\n\n--- Forwarded message ---\n{source.text_body or source.snippet or ''}"))
    async def modify_message(self, message_id: str, changes: MessageChanges) -> EmailMessage:
        add, remove = list(changes.add_labels), list(changes.remove_labels)
        if changes.is_read is True: remove.append("UNREAD")
        if changes.is_read is False: add.append("UNREAD")
        if changes.is_starred is True: add.append("STARRED")
        if changes.is_starred is False: remove.append("STARRED")
        if changes.archive: remove.append("INBOX")
        if changes.trash: add.append("TRASH")
        if changes.restore: remove.append("TRASH")
        raw = await self._call(self.service.users().messages().modify(userId="me", id=message_id, body={"addLabelIds": sorted(set(add)), "removeLabelIds": sorted(set(remove))})); return self._normalize(raw)
