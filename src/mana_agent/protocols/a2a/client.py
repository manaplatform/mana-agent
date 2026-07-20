"""Official-SDK A2A discovery and invocation client."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from google.protobuf.json_format import MessageToDict, ParseDict

from mana_agent.protocols.common.exceptions import OptionalProtocolDependencyError, ProtocolPolicyError
from mana_agent.protocols.common.security import ProtocolSecurityPolicy

from .registry import RemoteAgentRegistry


class ManaA2AClient:
    def __init__(self, *, allow_local_urls: bool = False) -> None:
        self.policy = ProtocolSecurityPolicy.for_workspace(".", allow_local_urls=allow_local_urls)
        self.registry = RemoteAgentRegistry()

    async def discover(self, name_or_id: str, *, refresh: bool = False) -> object:
        try:
            import httpx
            from a2a.client import A2ACardResolver
        except ImportError as exc:
            raise OptionalProtocolDependencyError.for_protocol("a2a") from exc
        remote = self.registry.get(name_or_id)
        if not refresh and remote.cached_card and remote.card_expires_at:
            try:
                expires = datetime.fromisoformat(remote.card_expires_at.replace("Z", "+00:00"))
            except ValueError:
                expires = datetime.min.replace(tzinfo=timezone.utc)
            if expires > datetime.now(timezone.utc):
                from a2a.types.a2a_pb2 import AgentCard

                return ParseDict(remote.cached_card, AgentCard())
        validated = self.policy.validate_remote_url(remote.card_url, production=not self.policy.allow_local_urls)
        parsed = urlsplit(validated)
        base = f"{parsed.scheme}://{parsed.netloc}"
        card_path = parsed.path.lstrip("/")
        async with httpx.AsyncClient(timeout=remote.timeout_seconds, follow_redirects=False) as http:
            card = await A2ACardResolver(http, base, agent_card_path=card_path).get_agent_card()
        if not any(interface.protocol_version == "1.0" for interface in card.supported_interfaces):
            raise ProtocolPolicyError("Remote Agent Card does not expose an A2A 1.0 interface.")
        advertised = {skill.id for skill in card.skills}
        if remote.allowed_skills and not set(remote.allowed_skills).issubset(advertised):
            raise ProtocolPolicyError("Remote Agent Card no longer advertises an allowed skill.")
        remote.cached_card = MessageToDict(card, preserving_proto_field_name=True)
        remote.last_discovered_at = datetime.now(timezone.utc).isoformat()
        remote.card_expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self.registry.update(remote)
        return card

    async def delegate(self, name_or_id: str, task: str, *, bearer_token: str = "") -> list[object]:
        """Send only the explicitly supplied task text and return streamed SDK events."""
        try:
            import httpx
            from a2a.client import ClientConfig, ClientFactory
            from a2a.helpers.proto_helpers import new_text_message
            from a2a.types.a2a_pb2 import Role, SendMessageRequest
        except ImportError as exc:
            raise OptionalProtocolDependencyError.for_protocol("a2a") from exc
        remote = self.registry.get(name_or_id)
        card = await self.discover(name_or_id)
        headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
        async with httpx.AsyncClient(headers=headers, timeout=remote.timeout_seconds, follow_redirects=False) as http:
            client = ClientFactory(ClientConfig(streaming=True, httpx_client=http)).create(card)
            try:
                request = SendMessageRequest(message=new_text_message(str(task), role=Role.ROLE_USER))
                return [event async for event in client.send_message(request)]
            finally:
                await client.close()
