from __future__ import annotations
from pathlib import Path
from typing import Sequence
from mana_agent.connectors.email.exceptions import AuthenticationRequired
GMAIL_SCOPES = {"email.metadata": "https://www.googleapis.com/auth/gmail.metadata", "email.read": "https://www.googleapis.com/auth/gmail.readonly", "email.compose": "https://www.googleapis.com/auth/gmail.compose", "email.send": "https://www.googleapis.com/auth/gmail.send", "email.modify": "https://www.googleapis.com/auth/gmail.modify"}
def gmail_authorization_flow(client_secret_file: Path, permissions: Sequence[str], port: int = 0):
    try: from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc: raise AuthenticationRequired("Gmail OAuth requires `pip install 'mana-agent[email]'`.") from exc
    if not client_secret_file.is_file(): raise AuthenticationRequired("Google OAuth client JSON file was not found.")
    scopes = [GMAIL_SCOPES[x] for x in permissions if x in GMAIL_SCOPES]
    if not scopes: raise AuthenticationRequired("Select at least one email capability before connecting Gmail.")
    return InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes=scopes).run_local_server(port=port, open_browser=True)
