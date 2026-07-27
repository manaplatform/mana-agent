import pytest
from mana_agent.connectors.email.approval import ApprovalBinding, approval_for
from mana_agent.connectors.email.exceptions import ApprovalRequired, AuthenticationRequired, EmailAuthorizationError, EmailMessageNotFoundError, EmailProviderError, EmailTemporaryError
from mana_agent.connectors.email.models import EmailQuery, EmailMessage
from mana_agent.connectors.email.auth.oauth import GMAIL_SCOPES, gmail_scopes_for_permissions
from mana_agent.connectors.email.providers.gmail import GMAIL_CAPABILITIES, gmail_list_arguments, gmail_query
from mana_agent.connectors.email.sanitizer import safe_attachment_filename, sanitize_html, untrusted_email_context
from mana_agent.connectors.email.tools import email_tool_contracts
from mana_agent.connectors.email.runtime_tools import build_email_langchain_tools
from mana_agent.connectors.email.providers.gmail import GmailProvider
from mana_agent.connectors.email.models import EmailAccount, EmailAddress
def test_gmail_query_is_structured():
    assert gmail_query(EmailQuery(sender=["a@example.com"], unread_only=True)) == "from:a@example.com is:unread"; assert GMAIL_CAPABILITIES.supports_threads


def test_gmail_read_scope_replaces_metadata_scope_for_searchable_tokens():
    assert gmail_scopes_for_permissions(["email.metadata", "email.read"]) == [GMAIL_SCOPES["email.read"]]


def test_gmail_inbox_only_search_uses_label_ids_not_metadata_blocked_query():
    arguments = gmail_list_arguments(EmailQuery(folders=["INBOX"], limit=1))
    assert arguments["labelIds"] == ["INBOX"]
    assert "q" not in arguments
def test_html_and_attachment_safety():
    clean = sanitize_html('<script>x()</script><img src="https://track"><a href="javascript:x">bad</a><b>ok</b>'); assert "script" not in clean and "img" not in clean and "javascript" not in clean and "<b>ok</b>" in clean; assert safe_attachment_filename("../../evil.txt") == "evil.txt"; assert untrusted_email_context("x").startswith("UNTRUSTED")
def test_approval_is_bound_to_exact_content():
    binding = ApprovalBinding(account_id="a", provider="gmail", action="send", recipients=["a@example.com"], body_hash="one"); approval = approval_for(binding, "approval-1"); approval.assert_valid_for(binding)
    with pytest.raises(ApprovalRequired): approval.assert_valid_for(binding.model_copy(update={"body_hash": "two"}))
def test_explicit_email_tools_expose_permissions():
    tools = {x.name: x for x in email_tool_contracts()}; assert "email_search" in tools and "email_send" in tools and "approval_id" in tools["email_send"].input_schema["required"]

def test_email_runtime_tools_are_available_without_connecting_to_gmail():
    assert {tool.name for tool in build_email_langchain_tools()} >= {"email_accounts_list", "email_search", "email_read", "email_thread_read"}

def test_gmail_search_reads_metadata_not_full_mime():
    calls = []
    class Request:
        def execute(self): return {"messages": [{"id": "one"}]}
    class Messages:
        def list(self, **kwargs): calls.append(("list", kwargs)); return Request()
        def get(self, **kwargs): calls.append(("get", kwargs)); return type("Metadata", (), {"execute": lambda self: {"id": "one", "internalDate": "0", "payload": {"headers": [{"name": "From", "value": "a@example.com"}]}}})()
    class Users:
        def messages(self): return Messages()
    provider = GmailProvider(account=EmailAccount(id="a", provider="gmail", address=EmailAddress(address="me@example.com")), service=type("Service", (), {"users": lambda self: Users()})())
    import asyncio
    asyncio.run(provider.search_messages(EmailQuery(limit=1)))
    assert calls[1][1]["format"] == "metadata"

def test_gmail_403_without_authorization_evidence_is_not_a_reconnect_error():
    class Request:
        def execute(self):
            error = RuntimeError("forbidden")
            error.resp = type("Response", (), {"status": 403})()
            raise error
    class Messages:
        def list(self, **kwargs): return Request()
    class Users:
        def messages(self): return Messages()
    provider = GmailProvider(account=EmailAccount(id="a", provider="gmail", address=EmailAddress(address="me@example.com")), service=type("Service", (), {"users": lambda self: Users()})())
    import asyncio
    with pytest.raises(EmailProviderError) as raised:
        asyncio.run(provider.search_messages(EmailQuery(limit=1)))
    assert raised.value.reconnect_required is False
    assert raised.value.provider_status == 403


def test_gmail_403_missing_scope_is_an_authorization_error():
    class Request:
        def execute(self):
            error = RuntimeError("forbidden")
            error.resp = type("Response", (), {"status": 403})()
            error.content = b'{"error":{"message":"Metadata scope does not support \'q\' parameter"}}'
            raise error
    class Messages:
        def list(self, **kwargs): return Request()
    class Users:
        def messages(self): return Messages()
    provider = GmailProvider(account=EmailAccount(id="a", provider="gmail", address=EmailAddress(address="me@example.com")), service=type("Service", (), {"users": lambda self: Users()})())
    import asyncio
    with pytest.raises(EmailAuthorizationError) as raised:
        asyncio.run(provider.search_messages(EmailQuery(limit=1)))
    assert raised.value.reconnect_required is True
    assert raised.value.provider_status == 403


def test_gmail_403_decodes_string_content_and_provider_status():
    class Request:
        def execute(self):
            error = RuntimeError("forbidden")
            error.resp = type("Response", (), {"status": "403"})()
            error.content = '{"error":{"status":"PERMISSION_DENIED","errors":[{"reason":"insufficientScopes"}]}}'
            raise error
    class Messages:
        def get(self, **kwargs): return Request()
    class Users:
        def messages(self): return Messages()
    provider = GmailProvider(account=EmailAccount(id="a", provider="gmail", address=EmailAddress(address="me@example.com")), service=type("Service", (), {"users": lambda self: Users()})())
    import asyncio
    with pytest.raises(EmailAuthorizationError) as raised:
        asyncio.run(provider.get_message("gmail-id"))
    assert raised.value.provider_status == 403
    assert raised.value.diagnostic_context["provider_error_status"] == "PERMISSION_DENIED"


@pytest.mark.parametrize(
    ("status", "content", "error_type", "reconnect"),
    [
        (401, b"", AuthenticationRequired, True),
        (404, b"", EmailMessageNotFoundError, False),
        (500, b"", EmailTemporaryError, False),
        (403, b'{"error":{"errors":[{"reason":"insufficientPermissions"}]}}', EmailAuthorizationError, True),
    ],
)
def test_gmail_http_errors_are_typed_without_inventing_reconnect(status, content, error_type, reconnect):
    class Request:
        def execute(self):
            error = RuntimeError("provider response")
            error.resp = type("Response", (), {"status": status})()
            error.content = content
            raise error
    class Messages:
        def get(self, **kwargs): return Request()
    class Users:
        def messages(self): return Messages()
    provider = GmailProvider(account=EmailAccount(id="a", provider="gmail", address=EmailAddress(address="me@example.com")), service=type("Service", (), {"users": lambda self: Users()})())
    import asyncio
    with pytest.raises(error_type) as raised:
        asyncio.run(provider.get_message("gmail-id"))
    assert raised.value.provider_status == status
    assert raised.value.reconnect_required is reconnect


def test_runtime_search_reference_is_accepted_by_read_and_keeps_account(monkeypatch):
    import json
    from mana_agent.connectors.email import runtime_tools
    account = EmailAccount(id="gmail-a", provider="gmail", address=EmailAddress(address="me@example.com"), granted_permissions={runtime_tools.EmailPermission.READ})
    message = EmailMessage(id="id-1", provider_message_id="id-1", account_id=account.id, sender=EmailAddress(address="sender@example.com"), received_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc), subject="subject")
    class Provider:
        def __init__(self): self.account = account; self.read_ids = []
        async def search_messages(self, query): return runtime_tools.EmailSearchResult(messages=[message])
        async def get_message(self, message_id): self.read_ids.append(message_id); return message
    provider = Provider()
    monkeypatch.setattr(runtime_tools, "_provider", lambda account_id: provider)
    tools = {tool.name: tool for tool in runtime_tools.build_email_langchain_tools()}
    searched = json.loads(tools["email_search"].invoke({"account_id": account.id, "limit": 1}).split("\n", 1)[1])
    reference = searched["messages"][0]["message_ref"]
    read = json.loads(tools["email_read"].invoke({"message_ref": reference}).split("\n", 1)[1])
    assert read["ok"] is True
    assert provider.read_ids == ["id-1"]
    assert read["message"]["message_ref"] == reference


def test_runtime_search_executes_inside_active_event_loop(monkeypatch):
    import asyncio
    import json
    import threading
    from mana_agent.connectors.email import runtime_tools

    account = EmailAccount(
        id="gmail-a",
        provider="gmail",
        address=EmailAddress(address="me@example.com"),
        granted_permissions={runtime_tools.EmailPermission.READ},
    )
    message = EmailMessage(
        id="id-1",
        provider_message_id="id-1",
        account_id=account.id,
        sender=EmailAddress(address="sender@example.com"),
        received_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        subject="subject",
    )
    provider_thread_ids = []

    class Provider:
        def __init__(self):
            self.account = account

        async def search_messages(self, query):
            provider_thread_ids.append(threading.get_ident())
            return runtime_tools.EmailSearchResult(messages=[message])

    monkeypatch.setattr(runtime_tools, "_provider", lambda account_id: Provider())
    tool = {item.name: item for item in runtime_tools.build_email_langchain_tools()}["email_search"]
    event_loop_thread_id = threading.get_ident()

    async def invoke_from_dashboard_loop():
        return tool.invoke({"account_id": account.id, "limit": 1})

    payload = json.loads(asyncio.run(invoke_from_dashboard_loop()).split("\n", 1)[1])

    assert payload["ok"] is True
    assert provider_thread_ids
    assert provider_thread_ids[0] != event_loop_thread_id


def test_runtime_stale_reference_refreshes_once_but_authentication_does_not_retry(monkeypatch):
    import json
    from mana_agent.connectors.email import runtime_tools
    account = EmailAccount(id="gmail-a", provider="gmail", address=EmailAddress(address="me@example.com"), granted_permissions={runtime_tools.EmailPermission.READ})
    stale = EmailMessage(id="stale", provider_message_id="stale", account_id=account.id, sender=EmailAddress(address="sender@example.com"), received_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    fresh = EmailMessage(id="fresh", provider_message_id="fresh", account_id=account.id, sender=EmailAddress(address="sender@example.com"), received_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    class Provider:
        def __init__(self): self.account = account; self.searches = 0; self.reads = []
        async def search_messages(self, query):
            self.searches += 1
            return runtime_tools.EmailSearchResult(messages=[stale if self.searches == 1 else fresh])
        async def get_message(self, message_id):
            self.reads.append(message_id)
            if message_id == "stale": raise EmailMessageNotFoundError("not found", provider="gmail", provider_status=404)
            return fresh
    provider = Provider()
    monkeypatch.setattr(runtime_tools, "_provider", lambda account_id: provider)
    tools = {tool.name: tool for tool in runtime_tools.build_email_langchain_tools()}
    searched = json.loads(tools["email_search"].invoke({"account_id": account.id, "limit": 1}).split("\n", 1)[1])
    runtime_tools_result = tools["email_read"].invoke({"message_ref": searched["messages"][0]["message_ref"]})
    assert '"ok": true' in runtime_tools_result
    assert provider.searches == 2
    # An authentication failure is returned directly and is never refreshed/retried.
    async def auth_failure(message_id): raise AuthenticationRequired("expired", provider="gmail", provider_status=401)
    provider.get_message = auth_failure
    failure = json.loads(tools["email_read"].invoke({"message_ref": {"account_id": account.id, "provider": "gmail", "provider_message_id": "fresh"}}).split("\n", 1)[1])
    assert failure["error"]["reconnect_required"] is True
    assert provider.searches == 2
