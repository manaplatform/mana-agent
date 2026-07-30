"""DigitalOcean adapter registration point; HTTP transport is injected by configuration."""

from .custom import CustomHTTPProvider


class DigitalOceanProvider(CustomHTTPProvider):
    name = "digitalocean"
