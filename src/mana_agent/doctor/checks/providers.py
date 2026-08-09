from __future__ import annotations

import urllib.request
import urllib.error
import json
import socket

from mana_agent.config import user_config
from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity


def connectivity(context: DoctorContext) -> list[DoctorFinding]:
    """Test connectivity and authentication for configured AI providers."""
    config = user_config.load_effective_settings()
    providers = config.get("MANA_CONFIGURED_PROVIDERS", ["openai"])
    if isinstance(providers, str):
        providers = [p.strip() for p in providers.split(",")]
        
    findings = []
    
    for provider in providers:
        if provider == "openai":
            base_url = config.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            api_key = config.get("OPENAI_API_KEY")
            
            if not api_key:
                findings.append(DoctorFinding("providers/openai-auth", Severity.ERROR, "OpenAI API Key Missing", "OPENAI_API_KEY is not set.", "Set OPENAI_API_KEY.", code="AUTH_MISSING"))
                continue
                
            try:
                # Test connectivity
                hostname = urllib.parse.urlparse(base_url).hostname
                if hostname:
                    socket.gethostbyname(hostname)
                    
                # Test auth/metadata (list models)
                req = urllib.request.Request(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        findings.append(DoctorFinding("providers/openai", Severity.INFO, "OpenAI Provider", "Successfully connected and authenticated.", code="PROVIDER_OK"))
            except urllib.error.HTTPError as e:
                findings.append(DoctorFinding("providers/openai-auth", Severity.ERROR, "OpenAI Authentication Failed", f"HTTP {e.code}: {e.reason}", "Check your OPENAI_API_KEY.", code="AUTH_FAILED"))
            except Exception as e:
                findings.append(DoctorFinding("providers/openai-connect", Severity.ERROR, "OpenAI Connectivity Failed", str(e), "Check your network and base URL.", code="CONNECT_FAILED"))

        elif provider == "openrouter":
            base_url = config.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
            api_key = config.get("OPENROUTER_API_KEY")
            
            if not api_key:
                findings.append(DoctorFinding("providers/openrouter-auth", Severity.ERROR, "OpenRouter API Key Missing", "OPENROUTER_API_KEY is not set.", "Set OPENROUTER_API_KEY.", code="AUTH_MISSING"))
                continue
                
            try:
                req = urllib.request.Request(f"{base_url}/auth/key", headers={"Authorization": f"Bearer {api_key}"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        findings.append(DoctorFinding("providers/openrouter", Severity.INFO, "OpenRouter Provider", "Successfully connected and authenticated.", code="PROVIDER_OK"))
            except urllib.error.HTTPError as e:
                findings.append(DoctorFinding("providers/openrouter-auth", Severity.ERROR, "OpenRouter Authentication Failed", f"HTTP {e.code}: {e.reason}", "Check your OPENROUTER_API_KEY.", code="AUTH_FAILED"))
            except Exception as e:
                findings.append(DoctorFinding("providers/openrouter-connect", Severity.ERROR, "OpenRouter Connectivity Failed", str(e), "Check your network and base URL.", code="CONNECT_FAILED"))

    return findings

