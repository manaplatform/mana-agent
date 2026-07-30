"""Hetzner adapter registration point; HTTP transport is injected by configuration."""

from .custom import CustomHTTPProvider


class HetznerProvider(CustomHTTPProvider):
    name = "hetzner"
