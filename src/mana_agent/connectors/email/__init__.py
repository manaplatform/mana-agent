from .registry import registry
from .providers.gmail import GmailProvider
registry.register("gmail", GmailProvider)
__all__ = ["registry", "GmailProvider"]
