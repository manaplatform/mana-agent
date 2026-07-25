"""Remote execution policy configuration, separate from OpenSSH global config."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mana_agent.remote_execution.target_policy import TargetPolicyMode


class RemoteSSHConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_policy: TargetPolicyMode = TargetPolicyMode.PROMPT_EACH_TIME
    strict_host_key_checking: bool = True
    allow_interactive_shell: bool = False
    allow_port_forwarding: bool = False
    max_output_bytes: int = Field(default=10_485_760, gt=0)
    default_timeout_seconds: int = Field(default=60, gt=0)


class RemoteExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_provider: str = "auto"
    fallback_to_external_worker: bool = True
    default_worker: str = ""
    ssh: RemoteSSHConfig = Field(default_factory=RemoteSSHConfig)
