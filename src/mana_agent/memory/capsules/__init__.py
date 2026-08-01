"""Scoped shared-memory capsule API."""

from mana_agent.memory.capsules.models import *  # noqa: F403
from mana_agent.memory.capsules.policy import CapsuleAuthorizationPolicy
from mana_agent.memory.capsules.service import CapsuleService

__all__ = ["CapsuleAuthorizationPolicy", "CapsuleService"]
