from mana_agent.connectors.email.exceptions import EmailPermissionDenied
from mana_agent.connectors.email.models import EmailAccount, EmailPermission

def assert_email_permission(account: EmailAccount, required: EmailPermission) -> None:
    if required not in account.granted_permissions:
        raise EmailPermissionDenied(f"Account {account.id} does not grant {required.value}.")
