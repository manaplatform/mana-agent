"""Firewall plans that preserve the active management connection."""

from __future__ import annotations

from pydantic import Field

from .models import StrictModel


class FirewallRule(StrictModel):
    port: int = Field(ge=1, le=65535)
    protocol: str = "tcp"
    source: str | None = None


class FirewallPlan(StrictModel):
    manager: str
    allow: list[FirewallRule]
    management_port: int = Field(ge=1, le=65535)
    require_second_connection: bool = True

    def validate_management_access(self) -> "FirewallPlan":
        if not any(rule.port == self.management_port and rule.protocol == "tcp" for rule in self.allow):
            raise ValueError("Firewall plan does not preserve the enrolled SSH management port.")
        if not self.require_second_connection:
            raise ValueError("Firewall changes require verification through a second SSH connection.")
        return self


def ufw_apply_argv(rule: FirewallRule) -> list[str]:
    target = f"{rule.port}/{rule.protocol}"
    if rule.source:
        return ["sudo", "ufw", "allow", "from", rule.source, "to", "any", "port", str(rule.port), "proto", rule.protocol]
    return ["sudo", "ufw", "allow", target]
