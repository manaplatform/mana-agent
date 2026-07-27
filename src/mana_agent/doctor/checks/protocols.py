from __future__ import annotations

import importlib.metadata
import importlib.util

from mana_agent.config.user_config import load_effective_settings
from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity
from mana_agent.protocols.common.config import validate_protocol_configuration


def acp_sdk(context: DoctorContext) -> list[DoctorFinding]:
    del context
    if importlib.util.find_spec("acp") is None:
        return [DoctorFinding("protocols/acp-sdk", Severity.WARNING, "ACP SDK not installed", "ACP commands are unavailable.", "Install mana-agent with the `acp` extra.")]
    version = importlib.metadata.version("agent-client-protocol")
    return [DoctorFinding("protocols/acp-sdk", Severity.INFO, "ACP SDK", f"agent-client-protocol {version}; protocol v1.")]


def a2a_sdk(context: DoctorContext) -> list[DoctorFinding]:
    del context
    if importlib.util.find_spec("a2a") is None:
        return [DoctorFinding("protocols/a2a-sdk", Severity.WARNING, "A2A SDK not installed", "A2A commands are unavailable.", "Install mana-agent with the `a2a` extra.")]
    version = importlib.metadata.version("a2a-sdk")
    return [DoctorFinding("protocols/a2a-sdk", Severity.INFO, "A2A SDK", f"a2a-sdk {version}; protocol 1.0.")]


def configuration(context: DoctorContext) -> list[DoctorFinding]:
    del context
    values = load_effective_settings(include_env=True)
    try:
        validate_protocol_configuration(values)
    except ValueError as exc:
        return [DoctorFinding("protocols/security-config", Severity.ERROR, "Unsafe protocol configuration", str(exc), "Correct the protocol settings before serving ACP or A2A.")]
    public_url = str(values.get("MANA_A2A_PUBLIC_BASE_URL") or "")
    if bool(values.get("MANA_A2A_SERVER_ENABLED")) and public_url.startswith("http://"):
        return [DoctorFinding("protocols/security-config", Severity.WARNING, "A2A TLS termination required", "The configured public A2A URL is not HTTPS.", "Use HTTPS for every externally exposed A2A deployment.")]
    return [DoctorFinding("protocols/security-config", Severity.INFO, "Protocol security configuration", "ACP/A2A settings passed fail-closed validation.")]
