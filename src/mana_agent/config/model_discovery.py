"""Configuration exports for the single native capability registry."""
from mana_agent.llm.capabilities import ModelProfile
from mana_agent.llm.registry import GLOBAL_CAPABILITY_REGISTRY, ModelCapabilityRegistry
from mana_agent.llm.discovery import ModelDiscoveryService, ModelMetadataResolver

__all__ = ["ModelProfile", "GLOBAL_CAPABILITY_REGISTRY", "ModelCapabilityRegistry", "ModelDiscoveryService", "ModelMetadataResolver"]
