from __future__ import annotations
import json
from pathlib import Path
from mana_agent.config.settings import mana_home
from mana_agent.connectors.email.models import EmailAccount
def accounts_path() -> Path: return mana_home() / "email_accounts.json"
def load_accounts() -> list[EmailAccount]:
    path = accounts_path()
    if not path.exists(): return []
    try: return [EmailAccount.model_validate(x) for x in json.loads(path.read_text())]
    except (OSError, ValueError, TypeError): return []
def save_accounts(accounts: list[EmailAccount]) -> None:
    path = accounts_path(); path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps([x.model_dump(mode="json") for x in accounts], indent=2) + "\n"); path.chmod(0o600)
def remove_account(account_id: str) -> EmailAccount:
    accounts = load_accounts()
    for account in accounts:
        if account.id == account_id: save_accounts([x for x in accounts if x.id != account_id]); return account
    raise ValueError(f"Email account not found: {account_id}")
