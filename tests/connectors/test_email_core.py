import pytest
from mana_agent.connectors.email.approval import ApprovalBinding, approval_for
from mana_agent.connectors.email.exceptions import ApprovalRequired
from mana_agent.connectors.email.models import EmailQuery
from mana_agent.connectors.email.providers.gmail import GMAIL_CAPABILITIES, gmail_query
from mana_agent.connectors.email.sanitizer import safe_attachment_filename, sanitize_html, untrusted_email_context
from mana_agent.connectors.email.tools import email_tool_contracts
from mana_agent.connectors.email.runtime_tools import build_email_langchain_tools
from mana_agent.connectors.email.providers.gmail import GmailProvider
from mana_agent.connectors.email.models import EmailAccount, EmailAddress
def test_gmail_query_is_structured():
    assert gmail_query(EmailQuery(sender=["a@example.com"], unread_only=True)) == "from:a@example.com is:unread"; assert GMAIL_CAPABILITIES.supports_threads
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
