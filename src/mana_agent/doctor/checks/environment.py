from __future__ import annotations

import os
import platform
import socket
from pathlib import Path

from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity


def os_info(context: DoctorContext) -> list[DoctorFinding]:
    """Report OS and platform information."""
    os_name = platform.system()
    release = platform.release()
    return [DoctorFinding("environment/os", Severity.INFO, "Operating System", f"{os_name} {release}", code="OS_INFO")]


def network_dns(context: DoctorContext) -> list[DoctorFinding]:
    """Check basic DNS resolution."""
    try:
        socket.gethostbyname("api.openai.com")
        return [DoctorFinding("environment/network-dns", Severity.INFO, "DNS Resolution", "DNS resolution is working.", code="DNS_OK")]
    except OSError as exc:
        return [DoctorFinding("environment/network-dns", Severity.WARNING, "DNS Resolution Failed", str(exc), "Check your network connection and DNS settings.", code="DNS_FAILED")]


def docker_env(context: DoctorContext) -> list[DoctorFinding]:
    """Detect if running inside Docker and check common localhost issues."""
    if Path("/.dockerenv").exists() or os.environ.get("KUBERNETES_SERVICE_HOST"):
        findings = [DoctorFinding("environment/docker", Severity.INFO, "Container Environment", "Running inside a container.", code="DOCKER_DETECTED")]
        
        # Check if mana-agent configuration uses 'localhost' or '127.0.0.1' which might be problematic in docker
        from mana_agent.config import user_config
        config = user_config.load_user_config()
        localhost_issues = []
        for key, value in config.items():
            if isinstance(value, str) and ("localhost" in value or "127.0.0.1" in value):
                localhost_issues.append(key)
                
        if localhost_issues:
            findings.append(
                DoctorFinding(
                    "environment/docker-localhost",
                    Severity.WARNING,
                    "Localhost used in container",
                    f"Configuration keys {', '.join(localhost_issues)} use localhost/127.0.0.1 which resolves to the container itself.",
                    "Use host.docker.internal or the specific service hostname instead.",
                    code="DOCKER_LOCALHOST"
                )
            )
        return findings
    return [DoctorFinding("environment/docker", Severity.INFO, "Container Environment", "Not running in a container.", code="NO_DOCKER")]

