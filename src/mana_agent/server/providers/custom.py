"""Generic HTTP provider contract with injected authenticated transport."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mana_agent.server.models import CreateServerRequest, ProviderServer, ProvisionedServer, Region, ServerImage, ServerSize


class CustomHTTPProvider:
    name = "custom"

    def __init__(self, request: Callable[[str, str, dict[str, Any] | None], Awaitable[Any]]) -> None:
        self._request = request

    async def list_regions(self) -> list[Region]:
        return [Region.model_validate(item) for item in await self._request("GET", "/regions", None)]

    async def list_sizes(self) -> list[ServerSize]:
        return [ServerSize.model_validate(item) for item in await self._request("GET", "/sizes", None)]

    async def list_images(self) -> list[ServerImage]:
        return [ServerImage.model_validate(item) for item in await self._request("GET", "/images", None)]

    async def create_server(self, request: CreateServerRequest) -> ProvisionedServer:
        return ProvisionedServer.model_validate(await self._request("POST", "/servers", request.model_dump(mode="json")))

    async def inspect_server(self, server_id: str) -> ProviderServer:
        return ProviderServer.model_validate(await self._request("GET", f"/servers/{server_id}", None))

    async def start_server(self, server_id: str) -> None:
        await self._action(server_id, "start")

    async def stop_server(self, server_id: str) -> None:
        await self._action(server_id, "stop")

    async def reboot_server(self, server_id: str) -> None:
        await self._action(server_id, "reboot")

    async def resize_server(self, server_id: str, size: str) -> None:
        await self._request("POST", f"/servers/{server_id}/resize", {"size": size})

    async def delete_server(self, server_id: str) -> None:
        await self._request("DELETE", f"/servers/{server_id}", None)

    async def create_snapshot(self, server_id: str, name: str) -> str:
        payload = await self._request("POST", f"/servers/{server_id}/snapshots", {"name": name})
        return str(payload["id"])

    async def _action(self, server_id: str, action: str) -> None:
        await self._request("POST", f"/servers/{server_id}/{action}", {})
