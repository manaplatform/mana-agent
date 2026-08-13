from __future__ import annotations

import os

from mana_agent.config import user_config
from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity


def check_secrets(context: DoctorContext) -> list[DoctorFinding]:
    """Validate that referenced secrets exist without exposing their values."""
    config = user_config.load_effective_settings()
    findings = []
    
    # We only check keys that are required based on other settings.
    # For now, just check a few essential ones if features are enabled.
    
    if config.get("mana_github_metadata_enabled") and config.get("mana_github_credential_source") == "env":
        if not os.environ.get("MANA_GITHUB_TOKEN") and not config.get("mana_github_token"):
            findings.append(DoctorFinding("secrets/github", Severity.ERROR, "GitHub Token Missing", "GitHub metadata is enabled but no token is provided.", "Set MANA_GITHUB_TOKEN.", code="SECRET_MISSING"))
        else:
            findings.append(DoctorFinding("secrets/github", Severity.INFO, "GitHub Token", "Reference exists.", code="SECRET_OK"))

    providers = config.get("mana_configured_providers", ["openai"])
    if isinstance(providers, str):
        providers = [p.strip() for p in providers.split(",")]
        
    for p in providers:
        key = f"{p.upper()}_API_KEY"
        if not config.get(key.lower()) and not os.environ.get(key):
            findings.append(DoctorFinding(f"secrets/{p}", Severity.ERROR, f"{p} API Key Missing", f"Provider {p} is enabled but {key} is missing.", f"Set {key}.", code="SECRET_MISSING"))
        else:
            findings.append(DoctorFinding(f"secrets/{p}", Severity.INFO, f"{p} API Key", "Reference exists.", code="SECRET_OK"))
            
    return findings
