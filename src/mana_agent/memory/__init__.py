"""Public provider-neutral memory API."""

from mana_agent.memory.config import MemoryConfig, MemorySecretStore
from mana_agent.memory.contracts import MemoryBackend
from mana_agent.memory.errors import (
    MemoryAuthenticationError,
    MemoryConfigurationError,
    MemoryDependencyError,
    MemoryError,
    MemoryNetworkError,
    MemoryNotFoundError,
    MemoryProviderError,
    MemoryStorageError,
)
from mana_agent.memory.models import (
    MemoryContent,
    MemoryHealth,
    MemoryHealthStatus,
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemoryUpdateRequest,
    MemoryWriteRequest,
)
from mana_agent.memory.service import (
    CodingMemoryService,
    EvidenceMemory,
    MemoryService,
    MultiAgentMemoryService,
)
# Legacy data shapes remain public through this package while storage classes
# themselves stay isolated behind MemoryService.
from mana_agent.services.coding_memory_service import FlowSummary
from mana_agent.services.memory_service import (
    AgentDecisionMemoryRecord,
    FileReadMemoryRecord,
    ReadMode,
    ScopedMemoryBundle,
    TaskMemoryRecord,
    ToolExecutionMemoryRecord,
    VerificationMemoryRecord,
    normalize_file_path,
    normalize_text,
    normalize_tool_name,
    stable_hash,
    task_fingerprint,
    utc_iso,
)

__all__ = [
    "AgentDecisionMemoryRecord",
    "CodingMemoryService",
    "EvidenceMemory",
    "FileReadMemoryRecord",
    "FlowSummary",
    "MemoryAuthenticationError",
    "MemoryBackend",
    "MemoryConfig",
    "MemoryConfigurationError",
    "MemoryContent",
    "MemoryDependencyError",
    "MemoryError",
    "MemoryHealth",
    "MemoryHealthStatus",
    "MemoryNetworkError",
    "MemoryNotFoundError",
    "MemoryProviderError",
    "MemoryRecord",
    "MemoryScope",
    "MemorySearchRequest",
    "MemorySecretStore",
    "MemoryService",
    "MemoryStorageError",
    "MemoryUpdateRequest",
    "MemoryWriteRequest",
    "MultiAgentMemoryService",
    "ReadMode",
    "ScopedMemoryBundle",
    "TaskMemoryRecord",
    "ToolExecutionMemoryRecord",
    "VerificationMemoryRecord",
    "normalize_file_path",
    "normalize_text",
    "normalize_tool_name",
    "stable_hash",
    "task_fingerprint",
    "utc_iso",
]
