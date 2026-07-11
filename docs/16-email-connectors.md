# Email connectors

Mana-Agent's email connector is provider-neutral. Gmail is the only currently enabled provider; Outlook, IMAP/SMTP, and JMAP are intentionally not shown as usable accounts until their provider implementations exist.

## Install and connect Gmail

Install the optional dependency group:

```bash
pip install "mana-agent[email]"
```

Create a Desktop OAuth client in Google Cloud, enable the Gmail API, and download its client JSON outside the repository. Then connect with only the capabilities required by the account:

```bash
mana-agent connector email add --provider gmail --client-secret-file ~/Downloads/google-client.json --permissions email.metadata,email.read,email.compose
```

The local OAuth callback opens a browser. Tokens and the OAuth client secret are stored in the operating-system keyring; account metadata stores only a keyring reference. Do not pass credentials as command-line flags or place them in `.env` files.

Use `mana-agent connector email list`, `status ACCOUNT`, `permissions ACCOUNT`, and `remove ACCOUNT` to manage accounts.

## Safety model

All email content is untrusted external data. Sanitized HTML excludes scripts, forms, frames, event handlers, unsafe URLs, and remote images. Attachment names are normalized to prevent traversal. Email content cannot authorize sending, modifying a mailbox, exposing credentials, or running unrelated tools.

The tool surface is explicit and structured; it is selected through the model decision layer, never keyword matching. Sending, forwarding, replying, deleting drafts, trashing, and configured mailbox mutations require an approval bound to the exact action. Any change invalidates it.
