"""Provider-independent email value objects.

Provider response objects deliberately never leave ``providers``.  These
models are the only data shape exposed to tools, memory, or audit events.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EmailPermission(str, Enum):
    METADATA = "email.metadata"
    READ = "email.read"
    COMPOSE = "email.compose"
    SEND = "email.send"
    MODIFY = "email.modify"


class EmailAddress(BaseModel):
    model_config = ConfigDict(frozen=True)
    address: str
    name: str | None = None


class EmailProviderCapabilities(BaseModel):
    supports_threads: bool = False
    supports_labels: bool = False
    supports_folders: bool = True
    supports_drafts: bool = False
    supports_push_notifications: bool = False
    supports_shared_mailboxes: bool = False
    supports_server_search: bool = True
    supports_send_as: bool = False
    supports_archive: bool = False
    supports_reply_all: bool = False


class EmailAccount(BaseModel):
    id: str
    provider: str
    address: EmailAddress
    display_name: str | None = None
    granted_permissions: set[EmailPermission] = Field(default_factory=set)
    enabled: bool = True
    secret_ref: str | None = None
    last_synced_at: datetime | None = None
    last_error: str | None = None


class ProviderHealth(BaseModel):
    healthy: bool
    message: str | None = None
    checked_at: datetime


class EmailFolder(BaseModel):
    id: str
    name: str
    role: str | None = None


class EmailLabel(BaseModel):
    id: str
    name: str


class EmailAttachment(BaseModel):
    id: str
    filename: str
    mime_type: str
    size: int
    disposition: str | None = None
    is_inline: bool = False
    sha256: str | None = None


class EmailMessage(BaseModel):
    id: str
    provider_message_id: str
    account_id: str
    thread_id: str | None = None
    internet_message_id: str | None = None
    subject: str = ""
    sender: EmailAddress
    to: list[EmailAddress] = Field(default_factory=list)
    cc: list[EmailAddress] = Field(default_factory=list)
    bcc: list[EmailAddress] = Field(default_factory=list)
    reply_to: list[EmailAddress] = Field(default_factory=list)
    received_at: datetime
    sent_at: datetime | None = None
    text_body: str | None = None
    sanitized_html_body: str | None = None
    snippet: str | None = None
    labels: list[str] = Field(default_factory=list)
    folder: str | None = None
    attachments: list[EmailAttachment] = Field(default_factory=list)
    is_read: bool = False
    is_starred: bool = False
    is_draft: bool = False
    is_trashed: bool = False
    headers: dict[str, str] = Field(default_factory=dict)


class EmailThread(BaseModel):
    id: str
    account_id: str
    messages: list[EmailMessage]


class EmailQuery(BaseModel):
    text: str | None = None
    sender: list[str] = Field(default_factory=list)
    recipients: list[str] = Field(default_factory=list)
    subject: str | None = None
    after: datetime | None = None
    before: datetime | None = None
    unread_only: bool | None = None
    has_attachments: bool | None = None
    labels: list[str] = Field(default_factory=list)
    folders: list[str] = Field(default_factory=list)
    thread_id: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class EmailSearchResult(BaseModel):
    messages: list[EmailMessage] = Field(default_factory=list)
    cursor: str | None = None
    total_estimate: int | None = None


class DraftInput(BaseModel):
    to: list[EmailAddress]
    subject: str
    text_body: str
    cc: list[EmailAddress] = Field(default_factory=list)
    bcc: list[EmailAddress] = Field(default_factory=list)
    in_reply_to: str | None = None
    references: list[str] = Field(default_factory=list)


class EmailDraft(BaseModel):
    id: str
    account_id: str
    message: EmailMessage


class SendInput(DraftInput):
    pass


class ReplyInput(BaseModel):
    text_body: str
    cc: list[EmailAddress] = Field(default_factory=list)
    bcc: list[EmailAddress] = Field(default_factory=list)


class ForwardInput(BaseModel):
    to: list[EmailAddress]
    text_body: str = ""
    cc: list[EmailAddress] = Field(default_factory=list)
    bcc: list[EmailAddress] = Field(default_factory=list)


class MessageChanges(BaseModel):
    is_read: bool | None = None
    is_starred: bool | None = None
    archive: bool | None = None
    trash: bool | None = None
    restore: bool | None = None
    add_labels: list[str] = Field(default_factory=list)
    remove_labels: list[str] = Field(default_factory=list)


class SendResult(BaseModel):
    accepted: bool
    message_id: str | None = None
    thread_id: str | None = None
    provider_status: str | None = None


class EmailEvent(BaseModel):
    id: str
    account_id: str
    provider: str
    type: str
    message_id: str | None = None
    thread_id: str | None = None
    occurred_at: datetime
    deduplication_key: str
    metadata: dict[str, Any] = Field(default_factory=dict)
