from __future__ import annotations

from mana_agent.config import user_config
from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity


def mcp_health(context: DoctorContext) -> list[DoctorFinding]:
    """Basic checks for MCP configuration and server tokens."""
    config = user_config.load_effective_settings()
    
    # We can check if MCP is enabled and if tokens are set
    acp_enabled = config.get("mana_acp_enabled", True)
    if not acp_enabled:
        return [DoctorFinding("mcp/status", Severity.INFO, "MCP Status", "MCP/ACP is disabled.", code="MCP_DISABLED")]
        
    findings = [DoctorFinding("mcp/status", Severity.INFO, "MCP Status", "MCP/ACP is enabled.", code="MCP_ENABLED")]
    
    server_token = config.get("mana_mcp_server_token")
    if server_token:
        findings.append(DoctorFinding("mcp/token", Severity.INFO, "MCP Server Token", "Server token is configured.", code="MCP_TOKEN_OK"))
    else:
        findings.append(DoctorFinding("mcp/token", Severity.INFO, "MCP Server Token", "No server token configured. Open access if port is exposed.", code="MCP_TOKEN_MISSING"))
        
    return findings
