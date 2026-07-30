"""Approval-gated provider lifecycle orchestration."""

from __future__ import annotations

from .models import CreateServerRequest, ProvisionedServer
from .providers import ServerProvider


class ProvisioningApprovalRequired(RuntimeError):
    pass


class ServerProvisioner:
    def __init__(self, providers: dict[str, ServerProvider]) -> None:
        self.providers = providers

    def provider(self, name: str) -> ServerProvider:
        try:
            return self.providers[name]
        except KeyError as exc:
            raise LookupError(f"Server provider {name!r} is not configured; no provider fallback was used.") from exc

    async def create(self, provider_name: str, request: CreateServerRequest) -> ProvisionedServer:
        if not request.cost_approval_id:
            raise ProvisioningApprovalRequired(
                f"Paid resource approval is required for provider={provider_name}, region={request.region}, "
                f"size={request.size}, image={request.image}."
            )
        return await self.provider(provider_name).create_server(request)

    async def delete(self, provider_name: str, server_id: str, *, confirmed_server_id: str, snapshot_name: str | None) -> None:
        if confirmed_server_id != server_id:
            raise ProvisioningApprovalRequired("Exact provider server identity confirmation is required.")
        provider = self.provider(provider_name)
        if snapshot_name:
            await provider.create_snapshot(server_id, snapshot_name)
        else:
            raise ProvisioningApprovalRequired("A provider snapshot name is required before deletion.")
        await provider.delete_server(server_id)
